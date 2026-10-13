//! State shared between the stratum client and the worker threads.

use crate::consensus::{difficulty_of_bits, PowAlgo, Target};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
#[cfg(test)]
use std::sync::Arc;
use std::time::Instant;

/// An active mining job from the pool (all the worker threads need).
#[derive(Clone)]
pub struct MinerJob {
    pub job_id: String,
    pub extranonce1: Vec<u8>,
    pub extranonce2_size: usize,
    pub coinb1: Vec<u8>,
    pub coinb2: Vec<u8>,
    pub merkle_steps: Vec<[u8; 32]>,
    pub version: u32,
    pub prev_hash: [u8; 32],
    pub bits: u32,
    pub ntime: u32,
    /// Share target for the difficulty that was in effect at notify time.
    pub share_target: Target,
    /// Block height from the coinbase (BIP34); None if it could not be read.
    pub height: Option<u32>,
    /// The PoW this job must be hashed with, decided once at notify time from
    /// the version bits.
    pub algo: PowAlgo,
}

/// How many found shares may wait for the stratum thread.
///
/// The queue used to be unbounded. The pool starts every connection at
/// difficulty 0.1 and only retargets after a dozen shares, so a fast rig
/// produces shares far quicker than one socket write per share can drain
/// them - at 1 PH/s that is 2.3 million per second against a writer doing
/// tens of thousands. The queue grew by hundreds of MB per second and the
/// miner died before vardiff ever caught up.
///
/// Bounded, and never blocking: a worker that stalls waiting for the socket
/// stops hashing, which is worse than losing a share. Shares are cheap and
/// replaceable; hashing time is not.
pub const SUBMIT_QUEUE: usize = 4096;

impl MinerJob {
    /// Do two jobs cover the same header space?
    ///
    /// The pool sends a fresh job whenever it retargets our difficulty, and
    /// that job is built from the same template: identical in every field that
    /// feeds the hash. Restarting the search there re-covers ground we already
    /// hashed, and the pool rejects the shares we re-find as duplicates -
    /// which is correct of it, and pure waste for us.
    ///
    /// job_id and share_target are deliberately excluded: neither goes into
    /// the header, so neither changes what there is to search.
    pub fn same_search_space(&self, other: &MinerJob) -> bool {
        self.extranonce1 == other.extranonce1
            && self.extranonce2_size == other.extranonce2_size
            && self.coinb1 == other.coinb1
            && self.coinb2 == other.coinb2
            && self.merkle_steps == other.merkle_steps
            && self.version == other.version
            && self.prev_hash == other.prev_hash
            && self.bits == other.bits
            && self.ntime == other.ntime
    }
}

/// A found share on its way to the pool.
pub struct FoundShare {
    pub job_id: String,
    pub extranonce2: Vec<u8>,
    pub ntime: u32,
    pub nonce: u32,
    /// Hash in display format (for logging) + whether it also meets the
    /// block target.
    pub hash_display: String,
    pub is_block_candidate: bool,
}

#[derive(Default)]
pub struct Stats {
    /// Total (GPU + CPU) - the sum of the two below.
    pub hashes: AtomicU64,
    /// Split per backend so dual mode can show each of them separately.
    pub gpu_hashes: AtomicU64,
    pub cpu_hashes: AtomicU64,
    pub accepted: AtomicU64,
    pub rejected: AtomicU64,
    pub blocks: AtomicU64,
    /// nBits of the latest job: the block the miner is working on, for its
    /// difficulty and the ETA.
    pub network_bits: AtomicU32,
    /// Height of the block we are mining on right now (from the job's
    /// coinbase, BIP34). 0 = unknown (no job yet).
    pub job_height: AtomicU32,
    /// Shares dropped because the submit queue was full. Non-zero means the
    /// pool difficulty is too low for this hashrate - see `Shared::submit`.
    pub dropped_submits: AtomicU64,
    /// GPU worker threads that are still able to hash. Counted so the
    /// watchdog in main can tell "mining" from "connected but idle" - see
    /// [`GpuWorkerAlive`].
    pub gpu_workers_alive: AtomicU32,
    /// Latest round trip of a request to the pool, in microseconds. 0 until
    /// the pool has answered something - see `Shared::record_pool_rtt`.
    pub pool_rtt_us: AtomicU64,
}

/// The highest difficulty a share of the current PoW reached ("best share"),
/// and what it would have needed to be a block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BestShare {
    pub difficulty: f64,
    /// The block difficulty of the job the share was found on, from that
    /// job's nBits. Recorded with the share, not looked up when it is shown:
    /// with ASERT (since the BC3Hash fork) the difficulty moves every block,
    /// and the current block's is not what an hour-old share was up against.
    pub network_difficulty: f64,
    pub found: Instant,
}

/// The PoW of the latest job and the best share found with it, under one
/// lock: a change of PoW starts the best share over, and a share of the old
/// PoW still draining from a launch must not land between the check and the
/// update.
#[derive(Default)]
struct PowState {
    algo: Option<PowAlgo>,
    best: Option<BestShare>,
}

/// Holds one live GPU worker in the count for as long as the worker can hash.
///
/// A GPU worker can stop for reasons that leave the process perfectly
/// healthy: the driver refused the kernel, the card fell off the bus, a TDR
/// killed the context. When that happened the miner stayed connected to the
/// pool, kept saying "Mining" in the GUI, and hashed nothing - which on a
/// rented GPU host is billed at GPU prices, and at home just looks like bad
/// luck. Drop runs on every exit path, panics included, so the count cannot
/// drift.
///
/// The workers are counted UP by main before the threads start, never here:
/// a thread that has not been scheduled yet must not look the same as one
/// that has died.
pub struct GpuWorkerAlive(std::sync::Arc<Shared>);

impl GpuWorkerAlive {
    /// The caller must already be counted in `gpu_workers_alive`.
    pub fn claim(shared: &std::sync::Arc<Shared>) -> Self {
        Self(shared.clone())
    }
}

impl Drop for GpuWorkerAlive {
    fn drop(&mut self) {
        self.0.stats.gpu_workers_alive.fetch_sub(1, Ordering::Relaxed);
    }
}

pub struct Shared {
    job: Mutex<Option<MinerJob>>,
    job_cv: Condvar,
    /// Intensity 1-100 (%). 100 = full speed; lower values insert idle time
    /// between work passes so the card/CPU does not run hot and the machine
    /// stays usable. Read by the workers on every batch.
    intensity: AtomicU32,
    /// Kernel launches a GPU keeps queued (`--gpu-queue`), at most
    /// `backend::QUEUE_DEPTH`.
    gpu_queue: AtomicU32,
    /// Bumped on every new job - the worker threads poll it cheaply.
    pub generation: AtomicU64,
    pub stats: Stats,
    /// Bounded on purpose - see `Shared::submit`.
    pub submit_tx: std::sync::mpsc::SyncSender<FoundShare>,
    /// Which devices were started for what; fixed before the workers start.
    pub plan: MiningPlan,
    /// The PoW of the latest job - so a change of PoW is announced once, not
    /// once per job - and the best share found with it.
    pow: Mutex<PowState>,
    /// GPU workers able to mine each PoW (by `PowAlgo::index`): raised when a
    /// worker's backend has a kernel for it, lowered when that kernel fails
    /// its self-test or the worker stops. The stand-in CPU threads go by it.
    gpu_capable: [AtomicU32; 3],
    /// GPU workers past opening their backend, whether that worked or not.
    /// Until all are, a PoW the GPUs have a kernel for but do not take yet
    /// is still starting up - not failed.
    gpus_opened: AtomicU32,
    /// The secp256k1 table window the GPUs' BC3Hash kernels should use; the
    /// GPU workers switch to it before their next BC3Hash job.
    bc3_window: AtomicU32,
    /// Hashes per GPU (by the GPU's slot in the detected list), for the
    /// benchmark's per-device rates.
    pub gpu_slot_hashes: Vec<AtomicU64>,
    /// Per GPU slot: the window its BC3Hash kernel is ready at, 0 while none
    /// is (loading, or never needed), `BC3_SLOT_FAILED` after a failure.
    pub gpu_slot_bc3: Vec<AtomicU32>,
    /// Per GPU slot: when its hash counter last moved, in microseconds since
    /// `started` - the end of a launch. Rates taken between two of these count
    /// whole launches only, instead of whichever launches happened to land
    /// inside a reporting interval (with 0.4 s launches, +-4 % per 10 s).
    pub gpu_slot_stamp: Vec<AtomicU64>,
    /// Per GPU slot: microseconds its launches ran on the device, by the
    /// device's own clock. Against the stamps above it says how much of the
    /// time the card was actually working.
    pub gpu_slot_busy_us: Vec<AtomicU64>,
    pub started: std::time::Instant,
}

/// The BC3Hash table window the GPUs use unless told otherwise (`--bc3-window`).
pub const DEFAULT_BC3_WINDOW: u32 = 16;

/// `Shared::gpu_slot_bc3` value for a GPU whose BC3Hash kernel failed to load
/// or failed its self-test at the requested window.
pub const BC3_SLOT_FAILED: u32 = 1;

/// A GPU worker's claim on `Shared::gpu_capable`, released when dropped - on
/// every exit path, like `GpuWorkerAlive`.
pub struct GpuCapability {
    shared: std::sync::Arc<Shared>,
    held: [bool; 3],
}

impl GpuCapability {
    pub fn new(shared: &std::sync::Arc<Shared>) -> Self {
        Self { shared: shared.clone(), held: [false; 3] }
    }

    /// Claim or release `algo`; tells the user where the current PoW is
    /// mined now if that changed anything.
    pub fn set(&mut self, algo: PowAlgo, capable: bool) {
        let i = algo.index();
        if self.held[i] == capable {
            return;
        }
        self.held[i] = capable;
        if capable {
            self.shared.gpu_capable[i].fetch_add(1, Ordering::Relaxed);
        } else {
            self.shared.gpu_capable[i].fetch_sub(1, Ordering::Relaxed);
        }
        self.shared.capability_changed(algo);
    }
}

impl Drop for GpuCapability {
    fn drop(&mut self) {
        for algo in PowAlgo::ALL {
            self.set(algo, false);
        }
    }
}

/// Which devices were started for what. Decided once in main from the
/// devices found; whether a GPU actually takes a PoW is decided at run time
/// (`Shared::gpus_mine`), since a kernel can fail its self-test.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MiningPlan {
    /// GPU workers started.
    pub gpus: usize,
    /// CPU threads that mine every job the CPU can hash.
    pub cpu_threads: usize,
    /// CPU threads that mine only what no GPU kernel can - see
    /// `worker::CpuRole::GpuFallback`.
    pub cpu_fallback_threads: usize,
    /// Per PoW (`PowAlgo::index`): do the started GPUs have a kernel for it?
    /// Only for telling "no kernel" apart from "not ready" in the status.
    pub gpu_kernel: [bool; 3],
    /// GPUs mining SHA3-256t compile their BC3Hash kernel in the background,
    /// ready for the fork. Off for the benchmark, which loads it at once.
    pub compile_ahead: bool,
}

/// Where the jobs of one PoW are hashed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowDevices {
    pub gpu: bool,
    pub cpu: bool,
}

impl PowDevices {
    /// For the GUI: "gpu", "cpu", "gpu+cpu" or "none".
    pub fn label(&self) -> &'static str {
        match (self.gpu, self.cpu) {
            (true, true) => "gpu+cpu",
            (true, false) => "gpu",
            (false, true) => "cpu",
            (false, false) => "none",
        }
    }
}

impl MiningPlan {
    /// Where jobs of `algo` are hashed, given whether the GPUs take them
    /// (`Shared::gpus_mine`). Never waits: this runs on the stratum thread,
    /// and a BC3Hash still building its tables counts as mining until it is
    /// known not to.
    pub fn devices_for(&self, algo: PowAlgo, gpu: bool) -> PowDevices {
        let cpu = crate::worker::cpu_may_support(algo)
            && (self.cpu_threads > 0 || (self.cpu_fallback_threads > 0 && !gpu));
        PowDevices { gpu, cpu }
    }

    /// What the user is told when jobs of this PoW start arriving: whether
    /// anything is mining them, and where.
    pub fn status(&self, algo: PowAlgo, gpu: bool) -> (bool, String) {
        let name = algo.name();
        let d = self.devices_for(algo, gpu);
        match (d.gpu, d.cpu) {
            (true, true) => (true, format!("{name}: mining on GPU and CPU")),
            (true, false) => (true, format!("{name}: mining on GPU")),
            (false, true) if self.gpu_kernel[algo.index()] => (
                true,
                format!("{name}: mining on CPU - the GPU is not ready for it, or failed its self-test"),
            ),
            (false, true) if self.gpus > 0 => {
                (true, format!("{name}: mining on CPU (no GPU kernel yet)"))
            }
            (false, true) => (true, format!("{name}: mining on CPU")),
            (false, false) if !crate::worker::cpu_may_support(algo) => {
                let why = crate::worker::cpu_unavailable(algo, false).unwrap_or_default();
                (false, format!("{name}: NOT mining - {why}"))
            }
            (false, false) => (
                false,
                format!(
                    "{name}: NOT mining - there is no GPU kernel for it yet and no CPU threads \
                     are running. Use --backend auto, or add --threads N"
                ),
            ),
        }
    }
}

/// A worker's gate in front of its hashing loop: may it hash this PoW, or
/// does it sit the job out?
///
/// Sitting out is the whole point. Grinding a job with the wrong PoW still
/// "finds" shares, every one of which the pool rejects - so the rig looks
/// busy, burns power, and earns nothing.
#[derive(Default)]
pub struct AlgoGate {
    /// The algorithm this worker last said it cannot hash, so it says it once.
    warned: Option<PowAlgo>,
}

impl AlgoGate {
    /// `worker` names the worker in the warning; None keeps it quiet, so
    /// twenty CPU threads do not say the same thing twenty times.
    pub fn check(
        &mut self,
        algo: PowAlgo,
        worker: Option<&str>,
        supports: impl Fn(PowAlgo) -> bool,
    ) -> Option<PowAlgo> {
        if supports(algo) {
            self.warned = None;
            return Some(algo);
        }
        if self.warned != Some(algo) {
            if let Some(worker) = worker {
                eprintln!(
                    "[miner] {worker} has no {} implementation - idle until a job it can hash",
                    algo.name()
                );
            }
            self.warned = Some(algo);
        }
        None
    }
}

impl Shared {
    pub fn new(submit_tx: std::sync::mpsc::SyncSender<FoundShare>, plan: MiningPlan) -> Self {
        Self {
            job: Mutex::new(None),
            job_cv: Condvar::new(),
            intensity: AtomicU32::new(100),
            gpu_queue: AtomicU32::new(u32::MAX),
            generation: AtomicU64::new(0),
            stats: Stats::default(),
            submit_tx,
            plan,
            pow: Mutex::new(PowState::default()),
            gpu_capable: Default::default(),
            gpus_opened: AtomicU32::new(0),
            bc3_window: AtomicU32::new(DEFAULT_BC3_WINDOW),
            gpu_slot_hashes: (0..plan.gpus).map(|_| AtomicU64::new(0)).collect(),
            gpu_slot_bc3: (0..plan.gpus).map(|_| AtomicU32::new(0)).collect(),
            gpu_slot_stamp: (0..plan.gpus).map(|_| AtomicU64::new(0)).collect(),
            gpu_slot_busy_us: (0..plan.gpus).map(|_| AtomicU64::new(0)).collect(),
            started: std::time::Instant::now(),
        }
    }

    /// Do GPUs take jobs of `algo` right now? A GPU counts from the moment
    /// its backend has a kernel for it until that kernel fails its self-test
    /// or the worker stops; a GPU that dies outright is the watchdog's
    /// business in main, not a reason to move its work to the CPU.
    pub fn gpus_mine(&self, algo: PowAlgo) -> bool {
        self.gpu_capable[algo.index()].load(Ordering::Relaxed) > 0
    }

    pub fn devices_for(&self, algo: PowAlgo) -> PowDevices {
        self.plan.devices_for(algo, self.gpus_mine(algo))
    }

    pub fn status(&self, algo: PowAlgo) -> (bool, String) {
        let gpu = self.gpus_mine(algo);
        // The pool's first job usually comes before the GPU has loaded its
        // kernels. That is a start, not a failed self-test.
        if !gpu && self.plan.gpu_kernel[algo.index()] && self.gpus_starting() {
            let name = algo.name();
            return if self.plan.devices_for(algo, false).cpu {
                (true, format!("{name}: mining on CPU until the GPU is ready"))
            } else {
                (true, format!("{name}: waiting for the GPU to start"))
            };
        }
        self.plan.status(algo, gpu)
    }

    fn gpus_starting(&self) -> bool {
        (self.gpus_opened.load(Ordering::Relaxed) as usize) < self.plan.gpus
    }

    /// A GPU worker is past opening its backend (after claiming what it can
    /// mine). If the GPUs still do not take the current PoW, say where it is
    /// mined now; if they do, the claim has already said so.
    pub fn gpu_opened(&self) {
        self.gpus_opened.fetch_add(1, Ordering::Relaxed);
        if let Some(algo) = self.current_algo() {
            if !self.gpus_mine(algo) {
                self.announce(algo);
            }
        }
    }

    /// The BC3Hash table window the GPUs should use.
    pub fn bc3_window(&self) -> u32 {
        self.bc3_window.load(Ordering::Relaxed)
    }

    /// Ask the GPUs for another BC3Hash window; they switch before their next
    /// BC3Hash job, so publish a job after this.
    pub fn set_bc3_window(&self, window: u32) {
        self.bc3_window.store(window, Ordering::Relaxed);
    }

    /// A GPU took on or gave up a PoW: if it is the current one, say where it
    /// is mined now, and let every worker re-decide whether the job is theirs.
    /// A stand-in CPU thread must not wait for the next job to take over from
    /// a GPU whose kernel just failed.
    fn capability_changed(&self, algo: PowAlgo) {
        if self.current_algo() == Some(algo) {
            self.announce(algo);
            self.nudge();
        }
    }

    /// Wake the workers as if a job had arrived, with the job they already
    /// have. Their place in it carries over (`MinerJob::same_search_space`).
    fn nudge(&self) {
        {
            let slot = self.job.lock().unwrap();
            if slot.is_none() {
                return;
            }
            self.generation.fetch_add(1, Ordering::Release);
        }
        self.job_cv.notify_all();
    }

    /// Record a submitted share, found on a job with nBits `bits`: updates
    /// the best share (and the block counter when the share also met the
    /// block's target) - for a share of the current PoW only. The
    /// difficulties of two PoWs do not compare: a SHA3-256t best share kept
    /// past the fork read as hundreds of percent of a BC3Hash block, and a
    /// launch still draining at the switch would have put one back.
    pub fn record_share(&self, algo: PowAlgo, difficulty: f64, bits: u32, is_block: bool) {
        let mut pow = self.pow.lock().unwrap();
        if pow.algo.is_some_and(|current| current != algo) {
            return;
        }
        if is_block {
            self.stats.blocks.fetch_add(1, Ordering::Relaxed);
        }
        if difficulty > pow.best.map_or(0.0, |b| b.difficulty) {
            pow.best = Some(BestShare {
                difficulty,
                network_difficulty: difficulty_of_bits(bits),
                found: Instant::now(),
            });
        }
    }

    /// None until a share of the current PoW has been found.
    pub fn best_share(&self) -> Option<BestShare> {
        self.pow.lock().unwrap().best
    }

    /// The pool answered a request `rtt` after it was sent. Only the latest
    /// value is kept: it is a "ping", not a statistic, and the next share
    /// replaces it within seconds anyway.
    pub fn record_pool_rtt(&self, rtt: std::time::Duration) {
        // Floored at 1 us so a measured round trip can never read as "none".
        let us = rtt.as_micros().clamp(1, u64::MAX as u128) as u64;
        self.stats.pool_rtt_us.store(us, Ordering::Relaxed);
    }

    /// Latest pool round trip in milliseconds; None before the first answer.
    pub fn pool_rtt_ms(&self) -> Option<f64> {
        match self.stats.pool_rtt_us.load(Ordering::Relaxed) {
            0 => None,
            us => Some(us as f64 / 1000.0),
        }
    }

    pub fn set_intensity(&self, percent: u32) {
        self.intensity.store(percent.clamp(1, 100), Ordering::Relaxed);
    }

    pub fn intensity(&self) -> u32 {
        self.intensity.load(Ordering::Relaxed)
    }

    /// Cap the launches a GPU keeps queued; the default is as many as the
    /// backends allow.
    pub fn set_gpu_queue(&self, launches: u32) {
        self.gpu_queue.store(launches.max(1), Ordering::Relaxed);
    }

    /// Launches a GPU worker should keep queued: one below 100 % intensity,
    /// where the card has to idle between launches and a queued one would
    /// not let it.
    pub fn gpu_queue(&self) -> usize {
        if self.intensity() < 100 {
            return 1;
        }
        (self.gpu_queue.load(Ordering::Relaxed) as usize).clamp(1, crate::backend::QUEUE_DEPTH)
    }

    /// Idle in proportion to the last work pass so that the duty cycle ends
    /// up at roughly `intensity`%. At 100% there is no idling at all.
    pub fn throttle(&self, worked: std::time::Duration) {
        let pct = self.intensity();
        if pct >= 100 {
            return;
        }
        // work/(work+idle) = pct/100  ->  idle = work * (100-pct)/pct
        let idle = worked.mul_f64((100 - pct) as f64 / pct as f64);
        // Take it in small pieces so a new job need not wait out the whole
        // idle period.
        let deadline = std::time::Instant::now() + idle;
        let gen = self.generation.load(Ordering::Acquire);
        while std::time::Instant::now() < deadline {
            if self.generation.load(Ordering::Acquire) != gen {
                return; // new job - stop idling right away
            }
            std::thread::sleep(std::time::Duration::from_millis(5).min(idle));
        }
    }

    pub fn publish_job(&self, job: MinerJob) {
        self.stats.network_bits.store(job.bits, Ordering::Relaxed);
        if let Some(h) = job.height {
            // Only report on an actual height change - the pool sends new
            // jobs within the same block too (new transactions, new ntime).
            let prev = self.stats.job_height.swap(h, Ordering::Relaxed);
            if prev != h {
                // With its difficulty: since the BC3Hash fork (ASERT) a new
                // figure every block.
                let difficulty = difficulty_of_bits(job.bits);
                crate::ipc::emit(&crate::ipc::Event::NewBlockHeight {
                    height: h,
                    network_difficulty: difficulty,
                });
                crate::human!(
                    "[pool] now working on block #{h}, difficulty {}",
                    crate::stats::format_difficulty(difficulty)
                );
            }
        }
        self.note_pow(job.algo);
        {
            // The bump happens under the lock so that `wait_for_job_after`,
            // which checks the generation under the same lock, cannot see the
            // old generation and then sleep through this notify.
            let mut slot = self.job.lock().unwrap();
            *slot = Some(job);
            self.generation.fetch_add(1, Ordering::Release);
        }
        self.job_cv.notify_all();
    }

    /// Tell the user where the hashing happens whenever the pool's jobs move
    /// to another PoW - at the fork, that is the moment a GPU rig switches to
    /// its CPU. Only the stratum thread publishes jobs, so this is said once
    /// per change, not once per worker.
    fn note_pow(&self, algo: PowAlgo) {
        {
            let mut pow = self.pow.lock().unwrap();
            if pow.algo == Some(algo) {
                return;
            }
            // A new PoW starts the best share over - see `record_share`.
            if pow.algo.is_some() {
                pow.best = None;
            }
            pow.algo = Some(algo);
        }
        self.announce(algo);
    }

    fn announce(&self, algo: PowAlgo) {
        let (mining, message) = self.status(algo);
        if mining {
            crate::human!("[miner] {message}");
        } else {
            eprintln!("[miner] {message}");
        }
        crate::ipc::emit(&crate::ipc::Event::Pow {
            algo: algo.name().into(),
            devices: self.devices_for(algo).label().into(),
        });
        crate::ipc::emit(&crate::ipc::Event::Status {
            state: if mining {
                crate::ipc::StatusState::Mining
            } else {
                crate::ipc::StatusState::Error
            },
            message,
        });
    }

    /// The PoW of the latest job; None before the first one.
    pub fn current_algo(&self) -> Option<PowAlgo> {
        self.pow.lock().unwrap().algo
    }

    /// Like `wait_for_job`, but only for a job published after generation
    /// `seen`. A worker that sat a job out waits here instead of spinning on
    /// the job it cannot hash.
    pub fn wait_for_job_after(&self, seen: u64) -> (u64, MinerJob) {
        let mut guard = self.job.lock().unwrap();
        loop {
            let generation = self.generation.load(Ordering::Acquire);
            if generation != seen {
                if let Some(j) = guard.as_ref() {
                    return (generation, j.clone());
                }
            }
            // publish_job bumps the generation under this lock, so it cannot
            // slip in between the check above and this wait.
            guard = self.job_cv.wait(guard).unwrap();
        }
    }

    /// Queue a share for the pool. Never blocks; counts what it drops.
    pub fn submit(&self, share: FoundShare) {
        if self.submit_tx.try_send(share).is_err() {
            let n = self.stats.dropped_submits.fetch_add(1, Ordering::Relaxed) + 1;
            // Only shout the first one: if the queue is full it is full for a
            // while, and a line per share would make it worse.
            if n == 1 {
                eprintln!(
                    "[miner] submit queue full - dropping shares. The pool difficulty is \
                     too low for this hashrate; it should retarget shortly."
                );
            }
        }
    }

    pub fn clear_job(&self) {
        *self.job.lock().unwrap() = None;
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// Block until a job exists; return (generation, job).
    pub fn wait_for_job(&self) -> (u64, MinerJob) {
        let mut guard = self.job.lock().unwrap();
        loop {
            if let Some(j) = guard.as_ref() {
                return (self.generation.load(Ordering::Acquire), j.clone());
            }
            guard = self.job_cv.wait(guard).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    fn a_job() -> MinerJob {
        MinerJob {
            job_id: "a".into(),
            extranonce1: vec![1, 2, 3, 4],
            extranonce2_size: 4,
            coinb1: vec![9; 20],
            coinb2: vec![8; 20],
            merkle_steps: vec![[7u8; 32]],
            version: 0x2000_1000,
            prev_hash: [6u8; 32],
            bits: 0x1d00_ffff,
            ntime: 1_700_000_000,
            share_target: [0xFFu8; 32],
            height: Some(59_342),
            algo: PowAlgo::Sha3t,
        }
    }

    /// A retarget must not look like new ground to search.
    ///
    /// The pool issues a fresh job every time it changes our difficulty, built
    /// from the same template. Treating that as a new space made the miner
    /// re-hash what it had already covered, and the pool rejects the shares it
    /// re-finds as duplicates - correctly, and at our expense.
    #[test]
    fn a_retarget_is_the_same_search_space() {
        let base = a_job();

        // Only the job name and the target changed: same space.
        let mut retarget = a_job();
        retarget.job_id = "b".into();
        retarget.share_target = [0x0Fu8; 32];
        assert!(base.same_search_space(&retarget));

        // Anything that feeds the header makes it a different space.
        for mutate in [
            (|j: &mut MinerJob| j.ntime += 1) as fn(&mut MinerJob),
            |j: &mut MinerJob| j.prev_hash[0] ^= 1,
            |j: &mut MinerJob| j.version ^= 1,
            |j: &mut MinerJob| j.bits ^= 1,
            |j: &mut MinerJob| j.coinb1.push(0),
            |j: &mut MinerJob| j.coinb2.push(0),
            |j: &mut MinerJob| j.extranonce1[0] ^= 1,
            |j: &mut MinerJob| j.merkle_steps.push([0u8; 32]),
            |j: &mut MinerJob| j.extranonce2_size = 8,
        ] {
            let mut other = a_job();
            mutate(&mut other);
            assert!(
                !base.same_search_space(&other),
                "a change that feeds the header must count as a new space"
            );
        }
    }

    use super::*;
    use std::time::{Duration, Instant};

    fn shared() -> Shared {
        let (tx, _rx) = std::sync::mpsc::sync_channel(SUBMIT_QUEUE);
        Shared::new(tx, MiningPlan::default())
    }

    /// The GPU kernels only do SHA3-256t: a GPU worker sits out BC3Hash (and
    /// SHA256d) instead of grinding it, and says so once.
    #[test]
    fn the_gate_lets_through_only_what_the_backend_supports() {
        let gpu = crate::backend::kernel_supports;
        let mut gate = AlgoGate::default();
        assert_eq!(gate.check(PowAlgo::Sha3t, Some("gpu"), gpu), Some(PowAlgo::Sha3t));
        assert_eq!(gate.check(PowAlgo::Bc3Hash, Some("gpu"), gpu), None);
        assert_eq!(gate.warned, Some(PowAlgo::Bc3Hash), "said once, remembered");
        assert_eq!(gate.check(PowAlgo::Bc3Hash, Some("gpu"), gpu), None);
        assert_eq!(gate.check(PowAlgo::Sha256d, Some("gpu"), gpu), None);
        assert_eq!(gate.check(PowAlgo::Sha3t, Some("gpu"), gpu), Some(PowAlgo::Sha3t));
        assert_eq!(gate.warned, None, "back to normal");
        // The CPU hashes whatever this build implements.
        let cpu = crate::worker::cpu_supports;
        assert_eq!(gate.check(PowAlgo::Sha3t, Some("cpu"), cpu), Some(PowAlgo::Sha3t));
        assert_eq!(
            gate.check(PowAlgo::Bc3Hash, Some("cpu"), cpu).is_some(),
            cfg!(feature = "bc3hash")
        );
    }

    /// Where each PoW is mined, for every way main can start the workers -
    /// and what the user is told. `gpu` is whether a GPU takes the PoW right
    /// now (`Shared::gpus_mine`).
    #[cfg(feature = "bc3hash")]
    #[test]
    fn the_plan_says_who_mines_each_pow() {
        let devices = |plan: MiningPlan, algo, gpu| {
            let d = plan.devices_for(algo, gpu);
            (d.gpu, d.cpu)
        };
        let sha3_only = [false, true, false];
        let both = [false, true, true];
        // --backend auto with an OpenCL GPU: no BC3Hash kernel, so the CPU
        // threads stand in for it.
        let auto = MiningPlan { gpus: 1, cpu_threads: 0, cpu_fallback_threads: 20, gpu_kernel: sha3_only, compile_ahead: true };
        assert_eq!(devices(auto, PowAlgo::Sha3t, true), (true, false));
        assert_eq!(devices(auto, PowAlgo::Bc3Hash, false), (false, true));
        assert_eq!(auto.status(PowAlgo::Sha3t, true), (true, "SHA3-256t: mining on GPU".into()));
        assert_eq!(
            auto.status(PowAlgo::Bc3Hash, false),
            (true, "BC3Hash: mining on CPU (no GPU kernel yet)".into())
        );
        // With CUDA the GPU takes BC3Hash too, and the CPU stays out - unless
        // the kernel is not ready or failed its self-test.
        let cuda = MiningPlan { gpu_kernel: both, ..auto };
        assert_eq!(devices(cuda, PowAlgo::Bc3Hash, true), (true, false));
        assert_eq!(cuda.status(PowAlgo::Bc3Hash, true), (true, "BC3Hash: mining on GPU".into()));
        assert_eq!(devices(cuda, PowAlgo::Bc3Hash, false), (false, true));
        assert!(cuda.status(PowAlgo::Bc3Hash, false).1.contains("self-test"));
        // A forced OpenCL backend has no CPU threads: BC3Hash is not mined,
        // and that must read as an error, not as "Mining".
        let forced = MiningPlan { gpus: 2, cpu_threads: 0, cpu_fallback_threads: 0, gpu_kernel: sha3_only, compile_ahead: true };
        assert_eq!(devices(forced, PowAlgo::Sha3t, true), (true, false));
        assert_eq!(devices(forced, PowAlgo::Bc3Hash, false), (false, false));
        let (mining, message) = forced.status(PowAlgo::Bc3Hash, false);
        assert!(!mining && message.contains("NOT mining"), "{message}");
        // GPU plus --threads: both on SHA3-256t, the CPU alone on BC3Hash.
        let dual = MiningPlan { gpus: 1, cpu_threads: 4, cpu_fallback_threads: 0, gpu_kernel: sha3_only, compile_ahead: true };
        assert_eq!(devices(dual, PowAlgo::Sha3t, true), (true, true));
        assert_eq!(devices(dual, PowAlgo::Bc3Hash, false), (false, true));
        // CPU only.
        let cpu = MiningPlan { gpus: 0, cpu_threads: 8, cpu_fallback_threads: 0, gpu_kernel: [false; 3], compile_ahead: true };
        assert_eq!(devices(cpu, PowAlgo::Bc3Hash, false), (false, true));
        assert_eq!(cpu.status(PowAlgo::Bc3Hash, false), (true, "BC3Hash: mining on CPU".into()));
        assert_eq!(PowDevices { gpu: true, cpu: true }.label(), "gpu+cpu");
    }

    /// The pool's first job comes before the GPU has opened its backend:
    /// the CPU stands in, and that is reported as a start, not as a failed
    /// self-test. Once the GPU is open, the real reason shows.
    #[test]
    fn a_starting_gpu_is_not_reported_as_failed() {
        let (tx, _rx) = std::sync::mpsc::sync_channel(SUBMIT_QUEUE);
        let plan = MiningPlan {
            gpus: 1,
            cpu_threads: 0,
            cpu_fallback_threads: 20,
            gpu_kernel: [false, true, true],
            compile_ahead: true,
        };
        let s = Shared::new(tx, plan);
        assert_eq!(
            s.status(PowAlgo::Sha3t),
            (true, "SHA3-256t: mining on CPU until the GPU is ready".into())
        );
        s.gpu_opened();
        assert!(s.status(PowAlgo::Sha3t).1.contains("self-test"));
        // No GPU kernel for SHA256d at all: never "until the GPU is ready".
        let (tx, _rx) = std::sync::mpsc::sync_channel(SUBMIT_QUEUE);
        let s = Shared::new(tx, plan);
        assert!(!s.status(PowAlgo::Sha256d).1.contains("until the GPU"));
    }

    /// The stand-in CPU threads follow what the GPUs can do now: a GPU that
    /// claims a PoW takes it from them, and gives it back when it lets go -
    /// including when its worker stops.
    #[test]
    fn gpu_capability_comes_and_goes() {
        let s = Arc::new(shared());
        assert!(!s.gpus_mine(PowAlgo::Bc3Hash));
        let mut a = GpuCapability::new(&s);
        let mut b = GpuCapability::new(&s);
        a.set(PowAlgo::Bc3Hash, true);
        a.set(PowAlgo::Bc3Hash, true); // idempotent
        b.set(PowAlgo::Bc3Hash, true);
        assert!(s.gpus_mine(PowAlgo::Bc3Hash));
        a.set(PowAlgo::Bc3Hash, false); // a failed its self-test
        assert!(s.gpus_mine(PowAlgo::Bc3Hash), "b still mines it");
        drop(b); // b's worker stopped
        assert!(!s.gpus_mine(PowAlgo::Bc3Hash));
        assert!(!s.gpus_mine(PowAlgo::Sha3t));
        assert_eq!(s.bc3_window(), DEFAULT_BC3_WINDOW);
        s.set_bc3_window(12);
        assert_eq!(s.bc3_window(), 12);
    }

    /// A change of PoW is announced once, not once per job.
    #[test]
    fn publish_announces_each_change_of_pow_once() {
        let s = shared();
        assert_eq!(s.current_algo(), None);
        s.publish_job(a_job());
        assert_eq!(s.current_algo(), Some(PowAlgo::Sha3t));
        let mut fork = a_job();
        fork.algo = PowAlgo::Bc3Hash;
        s.publish_job(fork.clone());
        s.publish_job(fork);
        assert_eq!(s.current_algo(), Some(PowAlgo::Bc3Hash));
        s.publish_job(a_job());
        assert_eq!(s.current_algo(), Some(PowAlgo::Sha3t));
    }

    /// A worker that sat a job out wakes for the next job, not before.
    #[test]
    fn wait_for_job_after_waits_for_a_newer_job() {
        let s = Arc::new(shared());
        s.publish_job(a_job());
        let (seen, _) = s.wait_for_job();
        let s2 = s.clone();
        let waiter = std::thread::spawn(move || s2.wait_for_job_after(seen));
        std::thread::sleep(Duration::from_millis(50));
        assert!(!waiter.is_finished(), "returned the job it was told to skip");
        let mut next = a_job();
        next.job_id = "next".into();
        s.publish_job(next);
        let (generation, job) = waiter.join().unwrap();
        assert_ne!(generation, seen);
        assert_eq!(job.job_id, "next");
    }

    #[test]
    fn intensity_defaults_to_full_and_clamps() {
        let s = shared();
        assert_eq!(s.intensity(), 100);
        s.set_intensity(50);
        assert_eq!(s.intensity(), 50);
        // Out-of-range values are clamped instead of turning mining off.
        s.set_intensity(0);
        assert_eq!(s.intensity(), 1);
        s.set_intensity(9000);
        assert_eq!(s.intensity(), 100);
    }

    /// nBits of difficulty 1 and of difficulty 16 (diff1 >> 4).
    const DIFF_1: u32 = 0x1d00_ffff;
    const DIFF_16: u32 = 0x1c0f_fff0;

    fn best(s: &Shared) -> Option<f64> {
        s.best_share().map(|b| b.difficulty)
    }

    #[test]
    fn best_share_keeps_the_maximum_and_counts_blocks() {
        let s = shared();
        assert_eq!(best(&s), None);
        s.record_share(PowAlgo::Sha3t, 12.5, DIFF_1, false);
        s.record_share(PowAlgo::Sha3t, 3.0, DIFF_1, false); // lower - must not overwrite
        assert_eq!(best(&s), Some(12.5));
        s.record_share(PowAlgo::Sha3t, 900.0, DIFF_1, true); // a block is a share as well
        assert_eq!(best(&s), Some(900.0));
        assert_eq!(s.stats.blocks.load(Ordering::Relaxed), 1);
    }

    /// With ASERT every block has a difficulty of its own. The best share
    /// keeps the one of the job it was found on, and when it was found -
    /// the latest job's would turn an old share's percentage into fiction.
    #[test]
    fn the_best_share_keeps_the_difficulty_of_its_own_block() {
        let s = shared();
        s.publish_job(a_job()); // difficulty 1
        let before = Instant::now();
        s.record_share(PowAlgo::Sha3t, 0.5, DIFF_1, false);
        let first = s.best_share().expect("recorded");
        assert_eq!((first.difficulty, first.network_difficulty), (0.5, 1.0));
        assert!(first.found >= before && first.found <= Instant::now());

        // The next block is harder. The block being mined moves on to its
        // difficulty; a lower share found there changes nothing, the time
        // included.
        let mut next = a_job();
        next.height = Some(59_343);
        next.bits = DIFF_16;
        s.publish_job(next);
        assert_eq!(s.stats.network_bits.load(Ordering::Relaxed), DIFF_16);
        s.record_share(PowAlgo::Sha3t, 0.25, DIFF_16, false);
        assert_eq!(s.best_share(), Some(first), "a lower share replaced the best one");

        // A higher one takes over, with its own block's difficulty.
        s.record_share(PowAlgo::Sha3t, 3.0, DIFF_16, false);
        let second = s.best_share().expect("recorded");
        assert_eq!((second.difficulty, second.network_difficulty), (3.0, 16.0));
        assert!(second.found >= first.found);
    }

    /// The best share is per PoW: the fork starts it over, with the
    /// difficulty it needed and when it was found, and a SHA3-256t share
    /// from a launch still draining after the switch does not put the old
    /// figure back - nor count as a block.
    #[test]
    fn the_best_share_starts_over_with_a_new_pow() {
        let s = shared();
        let mut sha3t = a_job();
        sha3t.bits = DIFF_16;
        s.publish_job(sha3t);
        s.record_share(PowAlgo::Sha3t, 5.0, DIFF_16, false);
        assert_eq!(best(&s), Some(5.0));
        let mut fork = a_job();
        fork.algo = PowAlgo::Bc3Hash;
        s.publish_job(fork.clone());
        assert_eq!(s.best_share(), None, "the SHA3-256t best share outlived the fork");
        s.record_share(PowAlgo::Sha3t, 9_000.0, DIFF_16, true);
        assert_eq!(s.best_share(), None, "a late SHA3-256t share counted");
        assert_eq!(s.stats.blocks.load(Ordering::Relaxed), 0);
        s.record_share(PowAlgo::Bc3Hash, 0.002, DIFF_1, false);
        let after = s.best_share().expect("recorded");
        assert_eq!((after.difficulty, after.network_difficulty), (0.002, 1.0));
        // A new job of the same PoW keeps it.
        s.publish_job(fork);
        assert_eq!(s.best_share(), Some(after));
    }

    #[test]
    fn throttle_is_free_at_full_intensity() {
        let s = shared();
        let t0 = Instant::now();
        s.throttle(Duration::from_millis(50));
        assert!(t0.elapsed() < Duration::from_millis(20), "no idling at 100 %");
    }

    #[test]
    fn throttle_sleeps_proportionally() {
        let s = shared();
        s.set_intensity(50); // 50% -> idle about as long as the work
        let t0 = Instant::now();
        s.throttle(Duration::from_millis(40));
        let waited = t0.elapsed();
        assert!(waited >= Duration::from_millis(25), "idled only {waited:?}");
        assert!(waited < Duration::from_millis(120), "idled too long: {waited:?}");
    }

    #[test]
    fn throttle_aborts_on_new_job() {
        let s = Arc::new(shared());
        s.set_intensity(10); // -> long idle (9x the work)
        let s2 = s.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            s2.generation.fetch_add(1, Ordering::Release);
        });
        let t0 = Instant::now();
        s.throttle(Duration::from_millis(100)); // would otherwise idle ~900 ms
        assert!(t0.elapsed() < Duration::from_millis(300), "the idle was not interrupted");
    }
}
