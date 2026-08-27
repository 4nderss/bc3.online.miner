//! Hashrate-rapportering och est. time-to-block.

use crate::consensus::difficulty_of_bits;
use crate::shared::Shared;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub fn run_reporter(shared: Arc<Shared>, interval_secs: u64) {
    let mut last_hashes = 0u64;
    let mut last_gpu = 0u64;
    let mut last_cpu = 0u64;
    let mut last_t = Instant::now();
    let telemetry = crate::telemetry::Telemetry::open();
    loop {
        std::thread::sleep(Duration::from_secs(interval_secs));
        let hashes = shared.stats.hashes.load(Ordering::Relaxed);
        let now = Instant::now();
        let dt = now.duration_since(last_t).as_secs_f64();
        let rate = (hashes - last_hashes) as f64 / dt.max(0.001);
        last_hashes = hashes;

        // Per backend, så dual-läget kan visa vad var och en bidrar med.
        let gpu_total = shared.stats.gpu_hashes.load(Ordering::Relaxed);
        let cpu_total = shared.stats.cpu_hashes.load(Ordering::Relaxed);
        let gpu_rate = (gpu_total - last_gpu) as f64 / dt.max(0.001);
        let cpu_rate = (cpu_total - last_cpu) as f64 / dt.max(0.001);
        last_gpu = gpu_total;
        last_cpu = cpu_total;
        last_t = now;

        let accepted = shared.stats.accepted.load(Ordering::Relaxed);
        let rejected = shared.stats.rejected.load(Ordering::Relaxed);
        let bits = shared.stats.network_bits.load(Ordering::Relaxed);
        let network_difficulty = if bits != 0 { difficulty_of_bits(bits) } else { 0.0 };
        let eta_secs = (rate > 0.0 && bits != 0)
            .then(|| network_difficulty * 4_294_967_296.0 / rate);

        let temps = telemetry.read(0);
        crate::ipc::emit(&crate::ipc::Event::Stats {
            hashrate: rate,
            hashrate_gpu: gpu_rate,
            hashrate_cpu: cpu_rate,
            accepted,
            rejected,
            best_share: shared.best_share(),
            blocks: shared.stats.blocks.load(Ordering::Relaxed),
            eta_secs,
            network_difficulty,
            job_height: shared.stats.job_height.load(Ordering::Relaxed),
            telemetry: temps,
        });
        let temp_text = match (temps.gpu_temp_c, temps.cpu_temp_c) {
            (Some(g), Some(c)) => format!(" | GPU {g}°C CPU {c}°C"),
            (Some(g), None) => format!(" | GPU {g}°C"),
            (None, Some(c)) => format!(" | CPU {c}°C"),
            (None, None) => String::new(),
        };
        let best = shared.best_share();
        let blocks = shared.stats.blocks.load(Ordering::Relaxed);
        crate::human!(
            "[miner] {} | shares {accepted}✓ {rejected}✗ | best {:.3} | blocks {blocks} | est. block: {}{}",
            format_hashrate(rate),
            best,
            eta_secs.map(format_duration).unwrap_or_else(|| "—".into()),
            temp_text
        );
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
        return "∞".into();
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
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(format_hashrate(1234.0), "1.23 kH/s");
        assert_eq!(format_hashrate(2.5e9), "2.50 GH/s");
        assert_eq!(format_hashrate(12.0), "12 H/s");
        assert_eq!(format_duration(45.0), "45s");
        assert_eq!(format_duration(3900.0), "1h 5m");
        assert_eq!(format_duration(200_000.0), "2d 7h");
    }
}
