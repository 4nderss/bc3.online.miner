//! `--benchmark`: mining without a pool or a network.
//!
//! An in-process job source stands where the stratum client would: it
//! publishes random jobs through `Shared::publish_job` to the ordinary
//! workers, so the benchmark runs exactly the code a pool job runs - job
//! dispatch, the GPU kernels and their launch queue, the CPU re-check of
//! every GPU hit - and then checks every share that comes out once more
//! against the CPU reference, the way a pool would. Per-device hash rates are
//! read from the same counters the miner keeps for its statistics, and so is
//! how busy each GPU was: the device time of its launches (CUDA events,
//! OpenCL profiling) over the wall time between them.
//!
//! On GPUs with a BC3Hash kernel it compares the table windows (16, 18, 19,
//! 20, 22 and 24 bits) unless `--bc3-window` names one - the bigger ones need
//! 0.9, 1.8, 3.3, 8 and 28 GiB of GPU memory, and a card fails those it has
//! not the memory for. A release does not compare: each card is tuned while
//! it mines (src/tuning.rs).

use crate::consensus::{
    hash_meets_target, root_from_steps, sha256d, target_for_difficulty, BlockHeader, PowAlgo,
};
use crate::shared::{FoundShare, MinerJob, Shared, BC3_SLOT_FAILED};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The PoW the benchmark measures - the one the miner mines.
const ALGO: PowAlgo = PowAlgo::Bc3Hash;

/// The version getblocktemplate hands out from the fork height on: bit 11,
/// BC3Hash, and bit 12, which stays set.
const VERSION: u32 = 0x2000_1800;

pub struct Config {
    /// Seconds to measure, per window.
    pub seconds: u64,
    /// Fixed share difficulty; None sets it from the measured rate.
    pub difficulty: Option<f64>,
    /// BC3Hash table windows to measure the GPUs at, in order.
    pub windows: Vec<u32>,
    /// Per GPU slot: its description, and whether it has a BC3Hash kernel.
    pub gpu_names: Vec<String>,
    pub gpu_bc3: Vec<bool>,
    pub cpu_threads: usize,
}

/// Seconds between progress lines.
const REPORT_EVERY: Duration = Duration::from_secs(5);
/// A new "block" - a fresh random job - this often.
const NEW_BLOCK_EVERY: Duration = Duration::from_secs(30);
/// Seconds between shares the automatic difficulty aims for.
const SHARE_EVERY: f64 = 3.0;
/// Hashing before measuring, for the GPU autotune to settle its launch size.
const SETTLE: Duration = Duration::from_secs(3);
/// How long a GPU may take to get its BC3Hash kernel ready: table build,
/// first-time JIT of a 2.8 MB PTX, self-test.
const READY_TIMEOUT: Duration = Duration::from_secs(300);

/// Random jobs with BC3Hash's version bits, as a pool would hand them out.
struct JobSource {
    rng: u64,
    count: u64,
    extranonce1: Vec<u8>,
    template: Option<MinerJob>,
}

impl JobSource {
    fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x0bc3_be4c)
            | 1;
        let mut s = Self { rng: seed, count: 0, extranonce1: vec![], template: None };
        s.extranonce1 = s.bytes(4);
        s
    }

    fn next(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }

    /// A new block: fresh previous hash, coinbase and branches.
    fn new_block(&mut self, difficulty: f64) -> MinerJob {
        let mut coinb1 = vec![2, 0, 0, 0, 1];
        coinb1.extend_from_slice(&[0u8; 32]);
        coinb1.extend_from_slice(&[0xff; 4]);
        coinb1.push(24);
        // No BIP34 height push: the benchmark is not mining a block.
        coinb1.extend_from_slice(b"/bc3-miner benchmark/");
        let coinb2 = self.bytes(60);
        let prev: [u8; 32] = self.bytes(32).try_into().unwrap();
        let merkle_steps: Vec<[u8; 32]> =
            (0..3).map(|_| self.bytes(32).try_into().unwrap()).collect();
        let template = MinerJob {
            job_id: String::new(),
            extranonce1: self.extranonce1.clone(),
            extranonce2_size: 4,
            coinb1,
            coinb2,
            merkle_steps,
            version: VERSION,
            prev_hash: prev,
            // A hard network target: no share is a block candidate.
            bits: 0x1a00_ffff,
            ntime: 1_790_000_000 + (self.count as u32) * 30,
            share_target: [0; 32],
            height: None,
            algo: ALGO,
        };
        self.template = Some(template);
        self.retarget(difficulty)
    }

    /// The current block at another share difficulty - what a pool's vardiff
    /// sends. Same header space, so the workers carry on where they were.
    fn retarget(&mut self, difficulty: f64) -> MinerJob {
        self.count += 1;
        let mut job = self.template.clone().expect("a block first");
        job.job_id = format!("bench-{}", self.count);
        job.share_target = target_for_difficulty(difficulty);
        job
    }
}

/// Shares counted by the job source.
#[derive(Default, Clone, Copy)]
struct Shares {
    ok: u64,
    invalid: u64,
    duplicate: u64,
}

/// Check one share the way a pool does: rebuild its header from the job and
/// the share alone and hash it with the CPU reference.
fn check(job: &MinerJob, share: &FoundShare) -> bool {
    let mut coinbase = job.coinb1.clone();
    coinbase.extend_from_slice(&job.extranonce1);
    coinbase.extend_from_slice(&share.extranonce2);
    coinbase.extend_from_slice(&job.coinb2);
    let header = BlockHeader {
        version: job.version,
        prev_hash: job.prev_hash,
        merkle_root: root_from_steps(&sha256d(&coinbase), &job.merkle_steps),
        time: share.ntime,
        bits: job.bits,
        nonce: share.nonce,
    }
    .serialize();
    match job.algo.hash_fn() {
        Some(pow) => hash_meets_target(&pow(&header), &job.share_target),
        None => false,
    }
}

/// The share difficulty that makes a share come every `SHARE_EVERY` seconds
/// at `rate` H/s.
fn difficulty_for(rate: f64) -> f64 {
    (rate * SHARE_EVERY / 4_294_967_296.0).max(1.0 / (1u64 << 31) as f64)
}

/// A first guess at the rate, before anything is measured.
fn guess_rate(gpus: usize, cpu_threads: usize) -> f64 {
    if gpus > 0 {
        gpus as f64 * 100e3
    } else {
        cpu_threads.max(1) as f64 * 1e3
    }
}

pub fn format_rate(rate: f64) -> String {
    crate::stats::format_hashrate(rate)
}

struct Run<'a> {
    shared: &'a Shared,
    rx: &'a Receiver<FoundShare>,
    cfg: &'a Config,
    source: JobSource,
    jobs: HashMap<String, MinerJob>,
    order: Vec<String>,
    seen: HashSet<(String, Vec<u8>, u32)>,
    shares: Shares,
    difficulty: f64,
    block_started: Instant,
}

impl Run<'_> {
    fn publish(&mut self, job: MinerJob) {
        self.order.push(job.job_id.clone());
        self.jobs.insert(job.job_id.clone(), job.clone());
        // Shares can trail a job change by a launch; keep the last few jobs.
        while self.order.len() > 16 {
            let old = self.order.remove(0);
            self.jobs.remove(&old);
        }
        self.shared.publish_job(job);
    }

    fn new_block(&mut self) {
        let job = self.source.new_block(self.difficulty);
        self.block_started = Instant::now();
        self.publish(job);
    }

    fn drain(&mut self) {
        while let Ok(share) = self.rx.try_recv() {
            let key = (share.job_id.clone(), share.extranonce2.clone(), share.nonce);
            let ok = self.jobs.get(&share.job_id).is_some_and(|job| check(job, &share));
            if !self.seen.insert(key) {
                self.shares.duplicate += 1;
            } else if ok {
                self.shares.ok += 1;
            } else {
                self.shares.invalid += 1;
                println!(
                    "[bench] INVALID share: job {} nonce {:08x} - the CPU reference does not agree",
                    share.job_id, share.nonce
                );
            }
        }
    }

    /// Hash counters: per GPU slot with the moment (us) it last moved and its
    /// device-busy total, then the CPU threads together with now.
    fn counters(&self) -> Counters {
        let s = self.shared;
        let gpus = s
            .gpu_slot_hashes
            .iter()
            .zip(&s.gpu_slot_stamp)
            .zip(&s.gpu_slot_busy_us)
            .map(|((h, t), b)| GpuCount {
                hashes: h.load(Ordering::Acquire),
                stamp_us: t.load(Ordering::Relaxed),
                busy_us: b.load(Ordering::Relaxed),
            })
            .collect();
        Counters {
            gpus,
            cpu: s.stats.cpu_hashes.load(Ordering::Relaxed),
            now: s.started.elapsed().as_micros() as u64,
        }
    }

    /// Sleep `d` in small steps, checking shares and starting new blocks.
    fn tick(&mut self, d: Duration) {
        let until = Instant::now() + d;
        while Instant::now() < until {
            std::thread::sleep(Duration::from_millis(100));
            self.drain();
            if self.block_started.elapsed() >= NEW_BLOCK_EVERY {
                self.new_block();
            }
        }
    }

    /// Wait for every GPU with a BC3Hash kernel to have it ready at `window`
    /// (or to have given up on it). Ok(per-slot failed flags).
    fn wait_ready(&mut self, window: u32) -> Vec<bool> {
        let started = Instant::now();
        let mut last_note = Instant::now();
        loop {
            let states: Vec<u32> =
                self.shared.gpu_slot_bc3.iter().map(|s| s.load(Ordering::Relaxed)).collect();
            let pending: Vec<usize> = (0..states.len())
                .filter(|&i| self.cfg.gpu_bc3[i] && states[i] != window && states[i] != BC3_SLOT_FAILED)
                .collect();
            if pending.is_empty() {
                return states.iter().map(|&s| s == BC3_SLOT_FAILED).collect();
            }
            if started.elapsed() > READY_TIMEOUT {
                println!("[bench] gave up waiting for GPU(s) {pending:?} to load window {window}");
                return (0..states.len()).map(|i| pending.contains(&i)).collect();
            }
            if last_note.elapsed() >= Duration::from_secs(10) {
                println!(
                    "[bench] still preparing BC3Hash window {window} on GPU(s) {pending:?} \
                     ({:.0} s) - the first JIT of a kernel on a card takes a while",
                    started.elapsed().as_secs_f64()
                );
                last_note = Instant::now();
            }
            self.tick(Duration::from_millis(200));
        }
    }

    /// Measure one run: rates per device, shares.
    fn measure(&mut self, label: &str, telemetry: &crate::telemetry::Telemetry) -> Phase {
        self.tick(SETTLE);
        let shares0 = self.shares;
        let first = self.counters();
        let t0 = Instant::now();
        let mut last = first.clone();
        let end = t0 + Duration::from_secs(self.cfg.seconds);
        while Instant::now() < end {
            let step = REPORT_EVERY.min(end.saturating_duration_since(Instant::now()));
            self.tick(step);
            let now = self.counters();
            let (gpu_rates, cpu_rate) = now.rates_since(&last);
            let busy = now.busy_since(&last);
            let total: f64 = gpu_rates.iter().sum::<f64>() + cpu_rate;
            let mut line = format!("[bench] {label} {:>5.0} s |", t0.elapsed().as_secs_f64());
            for (i, r) in gpu_rates.iter().enumerate() {
                line += &format!(" GPU{i} {} busy {} |", format_rate(*r), format_busy(busy[i]));
            }
            if self.cfg.cpu_threads > 0 {
                line += &format!(" CPU {} |", format_rate(cpu_rate));
            }
            line += &format!(
                " total {} | shares {} ok, {} invalid | diff {:.3e}",
                format_rate(total),
                self.shares.ok - shares0.ok,
                self.shares.invalid - shares0.invalid,
                self.difficulty
            );
            let t = telemetry.read(0);
            if let Some(temp) = t.gpu_temp_c {
                line += &format!(" | {temp} C");
            }
            if let Some(w) = t.gpu_power_w {
                line += &format!(" {w:.0} W");
            }
            println!("{line}");
            // A pool's vardiff, made simple: aim for a share every few
            // seconds once there is a rate to aim with.
            if self.cfg.difficulty.is_none() && total > 0.0 {
                let want = difficulty_for(total);
                if !(0.5..=2.0).contains(&(want / self.difficulty)) {
                    self.difficulty = want;
                    let job = self.source.retarget(want);
                    self.publish(job);
                }
            }
            last = now;
        }
        let end = self.counters();
        let (gpu, cpu) = end.rates_since(&first);
        let busy = end.busy_since(&first);
        let shares = self.shares;
        Phase {
            window: 0,
            gpu,
            busy,
            cpu,
            ok: shares.ok - shares0.ok,
            invalid: shares.invalid - shares0.invalid,
            failed: vec![],
        }
    }
}

/// One GPU's counters at a reading.
#[derive(Clone, Copy, Default)]
struct GpuCount {
    hashes: u64,
    /// When the count last moved - the end of a launch (us).
    stamp_us: u64,
    /// Device time of its launches so far (us).
    busy_us: u64,
}

/// A reading of the hash counters - see `Run::counters`.
#[derive(Clone)]
struct Counters {
    gpus: Vec<GpuCount>,
    cpu: u64,
    now: u64,
}

impl Counters {
    /// H/s per GPU and for the CPU since an earlier reading. A GPU's rate
    /// runs between the ends of launches, so it counts whole launches only.
    fn rates_since(&self, earlier: &Counters) -> (Vec<f64>, f64) {
        let gpus = self
            .gpus
            .iter()
            .zip(&earlier.gpus)
            .map(|(g1, g0)| {
                if g1.stamp_us > g0.stamp_us {
                    (g1.hashes - g0.hashes) as f64 / ((g1.stamp_us - g0.stamp_us) as f64 / 1e6)
                } else {
                    0.0
                }
            })
            .collect();
        let dt = (self.now.saturating_sub(earlier.now)) as f64 / 1e6;
        let cpu = if dt > 0.0 { (self.cpu - earlier.cpu) as f64 / dt } else { 0.0 };
        (gpus, cpu)
    }

    /// Per GPU: the fraction of the time between the same two launch ends
    /// that the device spent running launches. Below 1 by whatever the card
    /// waited for the host between them. None where no launch ended, or the
    /// device kept no time.
    fn busy_since(&self, earlier: &Counters) -> Vec<Option<f64>> {
        self.gpus
            .iter()
            .zip(&earlier.gpus)
            .map(|(g1, g0)| {
                let wall = g1.stamp_us.saturating_sub(g0.stamp_us);
                let busy = g1.busy_us.saturating_sub(g0.busy_us);
                (wall > 0 && busy > 0).then(|| busy as f64 / wall as f64)
            })
            .collect()
    }
}

/// A busy fraction as the benchmark prints it. It can come out a hair above
/// 100 %: the device's clock and the host's stamps are not the same clock.
fn format_busy(busy: Option<f64>) -> String {
    match busy {
        Some(b) => format!("{:.1} %", 100.0 * b),
        None => "n/a".into(),
    }
}

struct Phase {
    /// The BC3Hash window; 0 for a run without one.
    window: u32,
    gpu: Vec<f64>,
    /// Per GPU, see `Counters::busy_since`.
    busy: Vec<Option<f64>>,
    cpu: f64,
    ok: u64,
    invalid: u64,
    /// Per GPU slot: its BC3Hash kernel was not usable at this window.
    failed: Vec<bool>,
}

/// The runs at one window folded into one: mean rates, summed shares, and a
/// GPU counts as failed there if it failed in any run.
fn average(phases: &[Phase], window: u32) -> Phase {
    let runs: Vec<&Phase> = phases.iter().filter(|p| p.window == window).collect();
    let n = runs.len().max(1) as f64;
    let gpus = runs.first().map_or(0, |p| p.gpu.len());
    Phase {
        window,
        gpu: (0..gpus).map(|i| runs.iter().map(|p| p.gpu[i]).sum::<f64>() / n).collect(),
        busy: (0..gpus)
            .map(|i| {
                let known: Vec<f64> = runs.iter().filter_map(|p| p.busy.get(i).copied().flatten()).collect();
                (!known.is_empty()).then(|| known.iter().sum::<f64>() / known.len() as f64)
            })
            .collect(),
        cpu: runs.iter().map(|p| p.cpu).sum::<f64>() / n,
        ok: runs.iter().map(|p| p.ok).sum(),
        invalid: runs.iter().map(|p| p.invalid).sum(),
        failed: (0..gpus)
            .map(|i| runs.iter().any(|p| p.failed.get(i).copied().unwrap_or(false)))
            .collect(),
    }
}

/// Run the benchmark; the process exit code: 0 good, 1 if any share failed
/// the CPU check, 2 if nothing hashed at all.
pub fn run(shared: Arc<Shared>, rx: Receiver<FoundShare>, cfg: Config) -> i32 {
    let telemetry = crate::telemetry::Telemetry::open();
    println!(
        "[bench] bc3-miner {} benchmark - {}, {} s per run{}",
        env!("CARGO_PKG_VERSION"),
        ALGO.name(),
        cfg.seconds,
        match &cfg.difficulty {
            Some(d) => format!(", share difficulty {d}"),
            None => ", share difficulty set from the rate".into(),
        }
    );
    for (i, name) in cfg.gpu_names.iter().enumerate() {
        println!("[bench] GPU{i}: {name}");
    }
    if let Some(v) = telemetry.driver_version() {
        println!("[bench] NVIDIA driver {v}");
    }
    if !cfg.gpu_names.is_empty() {
        println!("[bench] kernel launches queued per GPU: {}", shared.gpu_queue());
    }
    if cfg.cpu_threads > 0 {
        println!("[bench] CPU: {} thread(s)", cfg.cpu_threads);
    }
    let gpu_bc3 = cfg.gpu_bc3.iter().any(|&b| b);
    let windows = if gpu_bc3 { cfg.windows.clone() } else { vec![0] };
    // Comparing windows: up and back down again, and the two runs of each
    // window averaged. A GPU's clock drifts as it warms up - on a laptop by
    // more than the windows differ - and whatever drifts steadily cancels
    // out of an up-and-down sweep.
    let schedule: Vec<u32> = if windows.len() > 1 {
        windows.iter().chain(windows.iter().rev()).copied().collect()
    } else {
        windows.clone()
    };
    if schedule.len() > 1 {
        println!(
            "[bench] comparing GPU table windows {windows:?}: each runs twice, up and back \
             down, {} s a run",
            cfg.seconds
        );
    }

    let mut run = Run {
        shared: &shared,
        rx: &rx,
        cfg: &cfg,
        source: JobSource::new(),
        jobs: HashMap::new(),
        order: vec![],
        seen: HashSet::new(),
        shares: Shares::default(),
        difficulty: cfg
            .difficulty
            .unwrap_or_else(|| difficulty_for(guess_rate(cfg.gpu_names.len(), cfg.cpu_threads))),
        block_started: Instant::now(),
    };

    // BC3Hash on the CPU, and the check of every share, need the CPU tables:
    // the fast ones take ~11 s to build, which is not hashing and must not be
    // measured as if it were.
    let t0 = Instant::now();
    match crate::worker::cpu_unavailable(ALGO, true) {
        None => println!(
            "[bench] CPU reference ready and self-tested ({:.1} s)",
            t0.elapsed().as_secs_f64()
        ),
        Some(why) => println!("[bench] no CPU reference: {why}"),
    }

    let mut phases = Vec::new();
    for &window in &schedule {
        let label = if window > 0 { format!("w{window}") } else { "run".to_string() };
        let mut failed = vec![false; cfg.gpu_names.len()];
        if window > 0 {
            for slot in &shared.gpu_slot_bc3 {
                slot.store(0, Ordering::Relaxed);
            }
            shared.set_bc3_window(window);
            println!("[bench] BC3Hash GPU table window {window} bits");
            run.new_block();
            failed = run.wait_ready(window);
        } else {
            run.new_block();
        }
        let mut phase = run.measure(&label, &telemetry);
        phase.window = window;
        phase.failed = failed;
        phases.push(phase);
    }

    // The summary: one line per window, its runs averaged.
    println!();
    println!("==== bc3-miner benchmark: {} ====", ALGO.name());
    for (i, name) in cfg.gpu_names.iter().enumerate() {
        println!("GPU{i}  {name}");
    }
    if let Some(v) = telemetry.driver_version() {
        println!("NVIDIA driver {v}");
    }
    let mut header = format!("{:<8}", if gpu_bc3 { "window" } else { "" });
    for i in 0..cfg.gpu_names.len() {
        header += &format!("{:>16}{:>9}", format!("GPU{i}"), "busy");
    }
    if cfg.cpu_threads > 0 {
        header += &format!("{:>14}", "CPU");
    }
    header += &format!("{:>16}{:>18}", "total", "shares ok/bad");
    println!("{header}");
    let rows: Vec<Phase> = windows.iter().map(|&w| average(&phases, w)).collect();
    let mut any = false;
    for p in &rows {
        let mut line =
            format!("{:<8}", if p.window > 0 { p.window.to_string() } else { String::new() });
        for (i, r) in p.gpu.iter().enumerate() {
            let failed = p.failed.get(i).copied().unwrap_or(false);
            let cell = if failed { "FAILED".to_string() } else { format_rate(*r) };
            let busy = if failed { String::new() } else { format_busy(p.busy.get(i).copied().flatten()) };
            line += &format!("{cell:>16}{busy:>9}");
        }
        if cfg.cpu_threads > 0 {
            line += &format!("{:>14}", format_rate(p.cpu));
        }
        let total = p.gpu.iter().sum::<f64>() + p.cpu;
        any |= total > 0.0;
        line += &format!("{:>16}{:>18}", format_rate(total), format!("{}/{}", p.ok, p.invalid));
        println!("{line}");
    }
    if schedule.len() > rows.len() {
        println!("(each window: the mean of its run going up and its run coming back down)");
    }
    println!(
        "(busy: the share of the time the GPU ran kernels, by its own clock - the rest it \
         waited for the host)"
    );
    if gpu_bc3 && rows.len() > 1 {
        for i in 0..cfg.gpu_names.len() {
            let best = rows
                .iter()
                .filter(|p| !p.failed.get(i).copied().unwrap_or(false))
                .max_by(|a, b| a.gpu[i].total_cmp(&b.gpu[i]));
            if let Some(best) = best {
                println!(
                    "GPU{i}: fastest at window {} ({}) - mine with --bc3-window {}",
                    best.window,
                    format_rate(best.gpu[i]),
                    best.window
                );
            }
        }
    }
    let invalid: u64 = phases.iter().map(|p| p.invalid).sum();
    let all = run.shares;
    println!(
        "shares checked against the CPU reference: {} ok, {} invalid, {} duplicate",
        all.ok, all.invalid, all.duplicate
    );
    if invalid > 0 || all.invalid > 0 {
        println!("RESULT: INVALID SHARES - this build or this GPU computes wrong hashes. Report it.");
        1
    } else if !any {
        println!("RESULT: nothing hashed");
        2
    } else {
        println!("RESULT: ok");
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GPU's rate runs from launch end to launch end, so a reading taken
    /// in the middle of a launch neither gains nor loses part of one.
    #[test]
    fn rates_count_whole_launches() {
        let g = |hashes, stamp_us, busy_us| GpuCount { hashes, stamp_us, busy_us };
        let a = Counters { gpus: vec![g(1_000, 1_000_000, 0)], cpu: 0, now: 1_100_000 };
        // 4000 more hashes over launches that ended 2 s apart; read 2.3 s later.
        let b = Counters { gpus: vec![g(5_000, 3_000_000, 1_900_000)], cpu: 2_300, now: 3_400_000 };
        let (gpus, cpu) = b.rates_since(&a);
        assert_eq!(gpus, vec![2_000.0]);
        assert!((cpu - 1_000.0).abs() < 1e-6);
        // No launch finished in between: no rate rather than a division by 0.
        assert_eq!(a.rates_since(&a).0, vec![0.0]);
        // 1.9 s of device time between launch ends 2 s apart: 95 % busy.
        assert_eq!(b.busy_since(&a), vec![Some(0.95)]);
        // No launch ended, or no device clock: no figure rather than 0 %.
        assert_eq!(a.busy_since(&a), vec![None]);
        let untimed = Counters { gpus: vec![g(5_000, 3_000_000, 0)], cpu: 0, now: 0 };
        assert_eq!(untimed.busy_since(&a), vec![None]);
        assert_eq!(format_busy(Some(0.9991)), "99.9 %");
        assert_eq!(format_busy(None), "n/a");
    }

    /// Each window's line is the mean of its up and down runs; a failure in
    /// either marks it.
    #[test]
    fn windows_average_their_runs() {
        let run = |window, rate, busy, failed| Phase {
            window,
            gpu: vec![rate],
            busy: vec![busy],
            cpu: 0.0,
            ok: 2,
            invalid: 0,
            failed: vec![failed],
        };
        let phases = [
            run(10, 100.0, Some(0.9), false),
            run(16, 130.0, None, false),
            run(16, 120.0, Some(0.8), true),
            run(10, 110.0, Some(1.0), false),
        ];
        let w10 = average(&phases, 10);
        assert_eq!((w10.gpu[0], w10.ok, w10.failed[0]), (105.0, 4, false));
        assert!((w10.busy[0].unwrap() - 0.95).abs() < 1e-12);
        let w16 = average(&phases, 16);
        assert_eq!((w16.gpu[0], w16.failed[0]), (125.0, true));
        // A run without a device clock does not drag the mean to 0.
        assert_eq!(w16.busy[0], Some(0.8));
    }

    /// The automatic difficulty puts a share every few seconds at the
    /// measured rate, and never below what a target can express.
    #[test]
    fn difficulty_follows_the_rate() {
        let d = difficulty_for(100e3);
        assert!((d * 4_294_967_296.0 / 100e3 - SHARE_EVERY).abs() < 1e-6);
        assert_eq!(difficulty_for(0.0), 1.0 / (1u64 << 31) as f64);
    }

    /// The job source's jobs: BC3Hash's version bits, a target for the
    /// difficulty, and a retarget that keeps the header space.
    #[test]
    fn jobs_carry_the_pow_and_keep_their_space_on_retarget() {
        let mut s = JobSource::new();
        let a = s.new_block(1e-5);
        assert_eq!(PowAlgo::for_version(a.version), PowAlgo::Bc3Hash);
        assert_eq!(a.algo, PowAlgo::Bc3Hash);
        assert_eq!(a.share_target, target_for_difficulty(1e-5));
        let b = s.retarget(1e-4);
        assert!(a.same_search_space(&b));
        assert_ne!(a.job_id, b.job_id);
        let c = s.new_block(1e-4);
        assert!(!b.same_search_space(&c));
    }

    /// The check a share gets is the pool's: right header, right PoW, under
    /// the target - and a share for another nonce does not pass.
    #[cfg(feature = "bc3hash")]
    #[test]
    fn shares_are_checked_like_a_pool_does() {
        let mut s = JobSource::new();
        // Easy enough that a few nonces find one: about every other hash.
        let job = s.new_block(1.0 / (1u64 << 31) as f64);
        let mut share = FoundShare {
            job_id: job.job_id.clone(),
            extranonce2: vec![0, 0, 0, 7],
            ntime: job.ntime,
            nonce: 0,
            hash_display: String::new(),
            is_block_candidate: false,
        };
        let good = (0..10_000u32).find(|&n| {
            share.nonce = n;
            check(&job, &share)
        });
        let good = good.expect("an easy target finds a share in 10000 nonces");
        share.nonce = good;
        assert!(check(&job, &share));
        // The same nonce under a harder target, or on another job, fails.
        let mut hard = job.clone();
        hard.share_target = [0; 32];
        assert!(!check(&hard, &share));
    }
}
