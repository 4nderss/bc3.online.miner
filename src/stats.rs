//! Hashrate reporting and estimated time to block.

use crate::consensus::difficulty_of_bits;
use crate::shared::Shared;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// When each NVIDIA card's power and clocks are logged (`format_gpu_report`):
/// a minute in, when the cards mine and have warmed up, and then every ten
/// minutes - often enough to see a host change its limits, rarely enough
/// that six cards' lines do not bury the rest.
const GPU_REPORT_FIRST: Duration = Duration::from_secs(60);
const GPU_REPORT_EVERY: Duration = Duration::from_secs(600);

pub fn run_reporter(shared: Arc<Shared>, interval_secs: u64) {
    let mut last_hashes = 0u64;
    let mut last_gpu = 0u64;
    let mut last_cpu = 0u64;
    let mut last_t = Instant::now();
    let telemetry = crate::telemetry::Telemetry::open();
    let mut next_gpu_report = Instant::now() + GPU_REPORT_FIRST;
    loop {
        std::thread::sleep(Duration::from_secs(interval_secs));
        if Instant::now() >= next_gpu_report {
            next_gpu_report += GPU_REPORT_EVERY;
            for report in telemetry.gpu_reports() {
                crate::human!("[gpu] {}", format_gpu_report(&report));
            }
        }
        let hashes = shared.stats.hashes.load(Ordering::Relaxed);
        let now = Instant::now();
        let dt = now.duration_since(last_t).as_secs_f64();
        let rate = (hashes - last_hashes) as f64 / dt.max(0.001);
        last_hashes = hashes;

        // Per backend, so dual mode can show what each one contributes.
        let gpu_total = shared.stats.gpu_hashes.load(Ordering::Relaxed);
        let cpu_total = shared.stats.cpu_hashes.load(Ordering::Relaxed);
        let gpu_rate = (gpu_total - last_gpu) as f64 / dt.max(0.001);
        let cpu_rate = (cpu_total - last_cpu) as f64 / dt.max(0.001);
        last_gpu = gpu_total;
        last_cpu = cpu_total;
        last_t = now;

        let accepted = shared.stats.accepted.load(Ordering::Relaxed);
        let rejected = shared.stats.rejected.load(Ordering::Relaxed);
        // Read together: the GUI shows them as a pair, the block and its
        // difficulty.
        let bits = shared.stats.network_bits.load(Ordering::Relaxed);
        let job_height = shared.stats.job_height.load(Ordering::Relaxed);
        let network_difficulty = if bits != 0 { difficulty_of_bits(bits) } else { 0.0 };
        let eta_secs = (rate > 0.0 && bits != 0)
            .then(|| network_difficulty * 4_294_967_296.0 / rate);

        let temps = telemetry.read(0);
        let pool_rtt_ms = shared.pool_rtt_ms();
        let best = shared.best_share();
        let (best_share, best_needed) =
            best.map_or((0.0, 0.0), |b| (b.difficulty, b.network_difficulty));
        crate::ipc::emit(&crate::ipc::Event::Stats {
            hashrate: rate,
            hashrate_gpu: gpu_rate,
            hashrate_cpu: cpu_rate,
            accepted,
            rejected,
            best_share,
            best_share_network_difficulty: best_needed,
            best_share_age_secs: best.map(|b| b.found.elapsed().as_secs_f64()),
            blocks: shared.stats.blocks.load(Ordering::Relaxed),
            eta_secs,
            network_difficulty,
            job_height,
            pool_rtt_ms,
            telemetry: temps,
        });
        let temp_text = match (temps.gpu_temp_c, temps.cpu_temp_c) {
            (Some(g), Some(c)) => format!(" | GPU {g}C CPU {c}C"),
            (Some(g), None) => format!(" | GPU {g}C"),
            (None, Some(c)) => format!(" | CPU {c}C"),
            (None, None) => String::new(),
        };
        let blocks = shared.stats.blocks.load(Ordering::Relaxed);
        crate::human!(
            "[miner] {} | shares {accepted} ok, {rejected} rejected | {} | best {} | blocks {blocks} | est. block: {}{}",
            format_hashrate(rate),
            format_pool_rtt(pool_rtt_ms),
            format_best_share(best_share, best_needed),
            eta_secs.map(format_duration).unwrap_or_else(|| "unknown".into()),
            temp_text
        );
    }
}

/// "NVML 0 NVIDIA GeForce RTX 5090: 448 of 450 W (default 575, max 600), SM
/// 2280 of 3090 MHz, PCIe gen 2 x8, held back by power limit": a card at its
/// power limit hashes what that limit allows, and one held back by nothing
/// is limited by the kernel. Fields NVML did not answer are left out.
pub fn format_gpu_report(r: &crate::telemetry::GpuReport) -> String {
    let mut parts = Vec::new();
    match (r.power_w, r.limit_w) {
        (Some(p), Some(l)) => parts.push(format!("{p:.0} of {l:.0} W")),
        (Some(p), None) => parts.push(format!("{p:.0} W")),
        (None, Some(l)) => parts.push(format!("limit {l:.0} W")),
        (None, None) => {}
    }
    let range: Vec<String> = [("default", r.default_limit_w), ("max", r.max_limit_w)]
        .iter()
        .filter_map(|(what, w)| w.map(|w| format!("{what} {w:.0}")))
        .collect();
    if !range.is_empty() && r.limit_w.is_some() {
        let last = parts.pop().unwrap_or_default();
        parts.push(format!("{last} ({})", range.join(", ")));
    }
    match (r.sm_mhz, r.max_sm_mhz) {
        (Some(c), Some(m)) => parts.push(format!("SM {c} of {m} MHz")),
        (Some(c), None) => parts.push(format!("SM {c} MHz")),
        _ => {}
    }
    if let Some((gen, width)) = r.pcie {
        parts.push(format!("PCIe gen {gen} x{width}"));
    }
    parts.push(if r.held_back_by.is_empty() {
        "clock held back by nothing".into()
    } else {
        format!("held back by {}", r.held_back_by.join(", "))
    });
    format!("NVML {} {}: {}", r.index, r.name, parts.join(", "))
}

/// "1.460 of 11.280 (12.94%)": the best share against the block difficulty
/// of the job it was found on - what it would have needed to be a block.
/// "-" before the first share.
pub fn format_best_share(best: f64, needed: f64) -> String {
    if best <= 0.0 {
        return "-".into();
    }
    if needed <= 0.0 {
        return format_difficulty(best);
    }
    let pct = best / needed * 100.0;
    // Two decimals stop saying anything below 0.01 %, which is where a
    // BC3Hash share at the pool's start difficulty sits.
    let pct = if pct >= 0.01 { format!("{pct:.2}%") } else { format!("{pct:.1e}%") };
    format!("{} of {} ({pct})", format_difficulty(best), format_difficulty(needed))
}

/// "pool 34 ms": the latest round trip to the pool, "pool -" before one.
pub fn format_pool_rtt(ms: Option<f64>) -> String {
    match ms {
        Some(ms) if ms >= 10.0 => format!("pool {ms:.0} ms"),
        Some(ms) => format!("pool {ms:.1} ms"),
        None => "pool -".into(),
    }
}

/// A share difficulty for the stats line. Three decimals were enough for
/// SHA3-256t, whose shares are worth 0.1 and up; a BC3Hash share at the
/// pool's start difficulty is worth 2e-6, and "best 0.000" said nothing.
pub fn format_difficulty(d: f64) -> String {
    if d == 0.0 || d >= 0.1 {
        format!("{d:.3}")
    } else {
        format!("{d:.2e}")
    }
}

pub fn format_hashrate(rate: f64) -> String {
    const UNITS: &[(&str, f64)] = &[
        ("TH/s", 1e12),
        ("GH/s", 1e9),
        ("MH/s", 1e6),
        ("kH/s", 1e3),
    ];
    for (unit, factor) in UNITS {
        if rate >= *factor {
            return format!("{:.2} {unit}", rate / factor);
        }
    }
    format!("{rate:.0} H/s")
}

pub fn format_duration(secs: f64) -> String {
    if !secs.is_finite() {
        return "never".into();
    }
    let s = secs as u64;
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m {}s", s / 60, s % 60),
        3600..=86_399 => format!("{}h {}m", s / 3600, (s % 3600) / 60),
        _ => format!("{}d {}h", s / 86_400, (s % 86_400) / 3600),
    }
}

#[cfg(test)]
mod tests {
    /// Log output must be pure ASCII.
    ///
    /// Not style: a container host truncated the startup line at an em dash
    /// and then dropped every line after it, so the miner looked dead while it
    /// was mining normally. Every line we emit passes through log pipelines we
    /// do not control, and some of them are not UTF-8 clean.
    #[test]
    fn log_formatters_are_ascii() {
        let mut samples = vec![
            format_duration(f64::INFINITY),
            format_duration(f64::NAN),
            format_duration(0.0),
            format_duration(42.0),
            format_duration(4242.0),
            format_duration(424242.0),
            format_duration(4.2e9),
        ];
        for r in [0.0, 1.0, 1.5e3, 1.5e6, 1.5e9, 1.5e12] {
            samples.push(format_hashrate(r));
        }
        for d in [0.0, 2e-6, 0.0213, 1.133, 5e3] {
            samples.push(format_difficulty(d));
        }
        for ms in [None, Some(0.4), Some(34.4), Some(1234.5)] {
            samples.push(format_pool_rtt(ms));
        }
        for (best, needed) in [(0.0, 0.0), (2e-6, 11.28), (1.46, 11.28), (5e3, 0.0)] {
            samples.push(format_best_share(best, needed));
        }
        for s in samples {
            assert!(s.is_ascii(), "non-ASCII in log output: {s:?}");
        }
    }

    use super::*;

    #[test]
    fn formats() {
        assert_eq!(format_hashrate(1234.0), "1.23 kH/s");
        assert_eq!(format_hashrate(2.5e9), "2.50 GH/s");
        assert_eq!(format_hashrate(12.0), "12 H/s");
        assert_eq!(format_duration(45.0), "45s");
        assert_eq!(format_duration(3900.0), "1h 5m");
        assert_eq!(format_duration(200_000.0), "2d 7h");
        // BC3Hash shares are worth millionths; SHA3-256t's as before.
        assert_eq!(format_difficulty(0.0), "0.000");
        assert_eq!(format_difficulty(1.1334), "1.133");
        assert_eq!(format_difficulty(0.0213), "2.13e-2");
        assert_eq!(format_difficulty(2.5e-6), "2.50e-6");
        assert_eq!(format_pool_rtt(None), "pool -");
        assert_eq!(format_pool_rtt(Some(2.34)), "pool 2.3 ms");
        assert_eq!(format_pool_rtt(Some(34.4)), "pool 34 ms");
    }

    /// The best share is shown against the difficulty its block needed, as
    /// a percentage - readable from a BC3Hash share at the pool's start
    /// difficulty up to one that was a block.
    #[test]
    fn best_share_against_its_block() {
        assert_eq!(format_best_share(0.0, 11.28), "-");
        assert_eq!(format_best_share(1.46, 11.28), "1.460 of 11.280 (12.94%)");
        assert_eq!(format_best_share(14.0, 11.28), "14.000 of 11.280 (124.11%)");
        assert_eq!(format_best_share(2.5e-6, 11.28), "2.50e-6 of 11.280 (2.2e-5%)");
        // No block difficulty to measure against: the share alone.
        assert_eq!(format_best_share(1.46, 0.0), "1.460");
    }

    #[test]
    fn a_gpu_report_says_what_holds_the_card_back() {
        let r = crate::telemetry::GpuReport {
            index: 3,
            name: "NVIDIA GeForce RTX 5090".into(),
            power_w: Some(447.6),
            limit_w: Some(450.0),
            default_limit_w: Some(575.0),
            max_limit_w: Some(600.0),
            sm_mhz: Some(2280),
            max_sm_mhz: Some(3090),
            pcie: Some((2, 8)),
            held_back_by: vec!["power limit"],
        };
        assert_eq!(
            format_gpu_report(&r),
            "NVML 3 NVIDIA GeForce RTX 5090: 448 of 450 W (default 575, max 600), \
             SM 2280 of 3090 MHz, PCIe gen 2 x8, held back by power limit"
        );
        // What NVML did not answer is left out, and nothing in the way says so.
        let bare = crate::telemetry::GpuReport { index: 0, name: "GPU".into(), ..Default::default() };
        assert_eq!(format_gpu_report(&bare), "NVML 0 GPU: clock held back by nothing");
    }
}
