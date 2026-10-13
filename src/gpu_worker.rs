//! GPU worker thread: the same job pipeline as the CPU worker, but the nonce
//! grinding happens on the GPU in batches. Coinbase -> merkle root is built on
//! the CPU per extranonce2 (cheap), and every GPU hit is verified on the CPU
//! before submit.
//!
//! Launches are queued `QUEUE_DEPTH` deep: the next one waits on the device
//! behind the one running, and the host re-checks a launch's hits while the
//! card is already on the next. One at a time, the card sat idle for every
//! round trip - collect, re-check, queue - and a BC3Hash re-check costs about
//! a millisecond a hit on the CPU.

use crate::backend::{
    open_backend_with_fallback, GpuDevice, HitBudget, Launch, MiningBackend, Scan,
};
use crate::consensus::{
    compact_to_target, hash_meets_target, hash_to_display, root_from_steps, sha256d, BlockHeader,
    PowAlgo, PowFn, Target,
};
use crate::shared::{AlgoGate, FoundShare, GpuCapability, MinerJob, Shared, BC3_SLOT_FAILED};
use crate::worker::en2_fits;
use crate::tuning::{Action, Tuner, START_WINDOW};
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Autotune limits for the batch size (nonces per kernel launch) and the time
/// a launch aims for. BC3Hash is about 110 kH/s on an RTX 3050 Ti. Launches
/// start at the floor: 256 nonces are 2 ms there, and still far from the
/// Windows TDR watchdog (2 s) on an integrated GPU through OpenCL that may
/// manage a few hundred hashes a second - a 16k start would have been seconds
/// there. From the floor the autotune doubles its way up, in whole waves of
/// the card (`launch_granule`; on CUDA the floor is one wave). Short launches
/// cost more than their launch overhead: the first waves of every launch run
/// in lockstep and are slower. Interleaved in one process on an RTX 3050 Ti
/// at window 16, the device time per hash was 8.42 us at 1 and 2 waves a
/// launch, 8.18 at 4 and 7.95 at 8, no better at 16 - so 400 ms, which
/// settles at 8 waves there (and on an RTX 4090), and not shorter for the
/// sake of a faster switch to a new job.
const BC3_MIN_BATCH: u32 = 1 << 8;
const BC3_MAX_BATCH: u32 = 1 << 24;
const BC3_START_BATCH: u32 = 1 << 8;
const BC3_TARGET_LAUNCH: Duration = Duration::from_millis(400);

/// How long to wait before trying to re-open a device that stopped answering.
/// A driver reset (TDR) or a hung card usually clears in seconds.
const REOPEN_DELAY: Duration = Duration::from_secs(5);

/// How many times a device may be reopened before the worker gives up.
///
/// The reopen is only a cure for a context that died once. A card that keeps
/// accepting new contexts and then failing every launch on them - a TDR loop, a
/// wedged OpenCL runtime, an ECC fault - would otherwise cycle for ever: five
/// failed launches, sleep, reopen, five more. The worker never exits, so the
/// watchdog in main never sees the count reach zero, and the miner reports
/// itself healthy while hashing nothing. That is the exact symptom the reopen
/// was written to fix, just with a fresh handle each time round.
const MAX_REOPENS: u32 = 5;

/// How many launches in a row may fail before the device counts as gone.
const MAX_SCAN_ERRORS: u32 = 5;

/// This card's BC3Hash window tuning (src/tuning.rs), as the worker drives
/// it: the window the card should be at, the tuner while it runs, and the
/// compile ahead of the kernel it is to switch to. Its tables the card
/// builds itself, at the switch.
struct Tune {
    want: u32,
    tuner: Option<Tuner>,
    /// Whether the tuning was started - or ruled out - for this card.
    begun: bool,
    key: String,
    name: String,
    slot: usize,
    /// The tuner waits for the target to be ready (`Action::Build`).
    preparing: bool,
    /// The compile of the target's kernel ahead, while it runs.
    jit: Option<Receiver<()>>,
    /// A switch the launch loop is to stop for; the main loop makes it.
    switch_to: Option<u32>,
}

impl Tune {
    fn new(shared: &Shared, slot: usize) -> Self {
        Tune {
            want: shared.bc3_window(),
            tuner: None,
            begun: false,
            key: String::new(),
            name: String::new(),
            slot,
            preparing: false,
            jit: None,
            switch_to: None,
        }
    }

    /// The window the card should be at: its tuning's, or one set by hand -
    /// the internal tools' - which every card follows and none tunes. The
    /// benchmark sets one window after another while the workers run.
    fn want(&self, shared: &Shared) -> u32 {
        if shared.bc3_window_set() {
            shared.bc3_window()
        } else {
            self.want
        }
    }

    /// Once the card mines at the start window: decide what to try, from its
    /// free memory, or recall what was chosen before.
    fn begin(&mut self, shared: &Shared, backend: &dyn MiningBackend) {
        if self.begun || shared.bc3_window_set() || backend.bc3hash_window() != Some(START_WINDOW) {
            return;
        }
        self.begun = true;
        self.name = backend.name();
        let Some((free, total)) = backend.memory() else {
            self.announce(false);
            return;
        };
        let driver = crate::telemetry::Telemetry::open().driver_version();
        self.key = crate::tuning::key(&self.name, total, driver.as_deref());
        // The start window's tables go before the bigger ones come.
        let fitting = crate::tuning::largest_fitting(free + crate::tuning::table_bytes(START_WINDOW));
        let recalled = crate::tuning::recall(&self.key);
        let Some((tuner, first)) = Tuner::start(fitting, recalled) else {
            match recalled {
                Some(c) if c.window > START_WINDOW => crate::human!(
                    "[gpu] {}: BC3Hash window {} was chosen for this card, but its tables do not fit now - \
                     staying at {START_WINDOW}",
                    self.name,
                    c.window
                ),
                Some(_) => crate::human!("[gpu] {}: BC3Hash window {START_WINDOW} - remembered for this card", self.name),
                None => {}
            }
            self.announce(false);
            return;
        };
        let target = tuner.target();
        let gib = crate::tuning::table_bytes(target) as f64 / f64::from(1u32 << 30);
        if recalled.is_some() {
            crate::human!(
                "[gpu] {}: BC3Hash window {target} ({gib:.1} GiB of tables) was chosen for this card - \
                 switching to it",
                self.name
            );
        } else {
            crate::human!(
                "[gpu] {}: tuning the BC3Hash window - measuring window {target} ({gib:.1} GiB of tables, \
                 built on the card) against {START_WINDOW} over the next few minutes",
                self.name
            );
        }
        self.tuner = Some(tuner);
        self.announce(recalled.is_none());
        if let Action::Build(w) = first {
            self.jit = backend.compile_bc3hash_ahead(w);
            self.preparing = true;
        }
    }

    /// A launch came back at `window`.
    fn launched(&mut self, window: u32, hashes: u64, busy: Duration) {
        let Some(tuner) = self.tuner.as_mut() else { return };
        let mut action = Action::Wait;
        // Ready once the compile ahead is over, however it went: a switch
        // without its result compiles the kernel itself.
        if self.preparing && !self.jit.as_ref().is_some_and(|rx| matches!(rx.try_recv(), Err(TryRecvError::Empty))) {
            self.preparing = false;
            self.jit = None;
            action = tuner.ready();
        }
        if action == Action::Wait {
            action = tuner.launched(window, hashes, busy);
        }
        self.act(action);
    }

    /// The card switched to `window` (`ok`), or could not.
    fn switched(&mut self, window: u32, ok: bool) {
        let Some(tuner) = self.tuner.as_mut() else { return };
        if !ok {
            self.want = START_WINDOW;
        }
        let action = tuner.switched(window, ok);
        match action {
            Action::Switch(next) if !ok => crate::human!("[gpu] {}: trying BC3Hash window {next} instead", self.name),
            Action::Done { .. } if !ok => crate::human!("[gpu] {}: staying at BC3Hash window {START_WINDOW}", self.name),
            _ => {}
        }
        self.act(action);
    }

    fn act(&mut self, action: Action) {
        match action {
            Action::Wait | Action::Build(_) => {}
            Action::Switch(w) => {
                self.want = w;
                self.switch_to = Some(w);
            }
            Action::Done { choice, remember } => {
                self.tuner = None;
                self.preparing = false;
                self.jit = None;
                let mut measured = None;
                if choice.start_rate > 0.0 && choice.tried_rate > 0.0 {
                    let gain = (choice.tried_rate / choice.start_rate - 1.0) * 100.0;
                    measured = Some((choice.tried, gain));
                    crate::human!(
                        "[gpu] {}: BC3Hash window {} measured {gain:+.1} % against {START_WINDOW} - {}{}",
                        self.name,
                        choice.tried,
                        if choice.window == choice.tried { "kept" } else { "staying at 16" },
                        if remember {
                            ""
                        } else {
                            " for now: 16 measured differently before and after, so the next start measures again"
                        }
                    );
                }
                if remember {
                    if let Err(e) = crate::tuning::remember(&self.key, choice) {
                        eprintln!(
                            "[gpu] {}: could not remember the tuned window ({e}) - it measures again next time",
                            self.name
                        );
                    }
                }
                self.emit(false, measured);
            }
        }
    }

    /// Tell the GUI the window this card mines at, and whether it is tuning.
    fn announce(&self, tuning: bool) {
        self.emit(tuning, None);
    }

    /// As `announce`, with what the measurement found: the window tried and
    /// its gain over 16 in percent.
    fn emit(&self, tuning: bool, measured: Option<(u32, f64)>) {
        crate::ipc::emit(&crate::ipc::Event::Tables {
            gpu: self.slot,
            window: self.want,
            mib: crate::tuning::table_bytes(self.want) >> 20,
            tuning,
            tried: measured.map(|m| m.0),
            gain_pct: measured.map(|m| m.1),
        });
    }
}

/// `slot` is the GPU's place in the detected list (per-device statistics);
/// `worker_index`/`total_workers` partition the extranonce2 space across all
/// workers, CPU threads included, exactly as in worker.rs.
pub fn run_gpu_worker(
    shared: Arc<Shared>,
    device: GpuDevice,
    slot: usize,
    worker_index: usize,
    total_workers: usize,
) {
    // Counted up by main before the thread started; this counts it back down
    // on EVERY exit, so the watchdog there can tell that nothing is mining.
    let _alive = crate::shared::GpuWorkerAlive::claim(&shared);

    let mut backend = match open_backend_with_fallback(&device) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[gpu] could not open {}: {e}", device.describe());
            shared.gpu_opened();
            return;
        }
    };
    crate::human!("[gpu] started: {}", backend.name());

    // What this card can mine, as far as anyone can tell before trying. A
    // kernel that later fails its self-test withdraws its claim; dropping
    // this on any exit withdraws them all. Not "once it has passed": the
    // stand-in CPU threads mining meanwhile slow the table build and the
    // self-test down more than they mine - window 16 took 11.2 s to get
    // ready instead of 4.6 with twenty of them busy beside it.
    let mut caps = GpuCapability::new(&shared);
    let claim = |caps: &mut GpuCapability, backend: &dyn MiningBackend| {
        for algo in PowAlgo::ALL {
            caps.set(algo, backend.has_kernel(algo) && crate::worker::cpu_may_support(algo));
        }
    };
    claim(&mut caps, &*backend);
    shared.gpu_opened();
    // The BC3Hash window that last failed here, so it is not retried on
    // every job; a different window still gets its chance.
    let mut bc3_failed: Option<u32> = None;

    let mut w = Worker::new(&shared, slot, worker_index, total_workers);
    let mut reopens: u32 = 0;
    let mut gate = AlgoGate::default();
    // Generation of a job this worker sat out; wait for the next one instead
    // of spinning on it.
    let mut sat_out: Option<u64> = None;
    loop {
        let (generation, job) = match sat_out.take() {
            Some(seen) => shared.wait_for_job_after(seen),
            None => shared.wait_for_job(),
        };
        if job.algo == PowAlgo::Bc3Hash && backend.has_kernel(PowAlgo::Bc3Hash) {
            // A switch that fails may make way for another window at once.
            while let Some(window) = w.tune.switch_to.take() {
                let ok = switch_bc3(&shared, &mut *backend, slot, window);
                w.tune.switched(window, ok);
            }
            prepare_bc3(&shared, &mut *backend, slot, &mut caps, &mut bc3_failed, w.tune.want(&shared));
            w.tune.begin(&shared, &*backend);
        }
        // Decided per job, so the launch loop below carries no dispatch. A
        // kernel without a CPU reference to re-check its hits against does
        // not mine at all - see AGENTS.md.
        let supported = |a: PowAlgo| backend.supports(a) && a.hash_fn().is_some();
        let Some((algo, verify)) = gate
            .check(job.algo, Some(backend.name().as_str()), supported)
            .and_then(|a| Some((a, a.hash_fn()?)))
        else {
            // A job without BC3Hash's version bit, which nothing mines, or
            // BC3Hash on a card whose kernel is not usable, which in
            // --backend auto the CPU threads take over: this worker sleeps
            // until a job it can do.
            sat_out = Some(generation);
            continue;
        };
        let end = w.mine_job(&mut *backend, &job, generation, algo, verify);
        // Launches came back, so the card works again and the reopen budget is
        // spent, not consumed. A job change alone proves nothing: a card that
        // fails every launch still sees jobs come and go.
        if std::mem::take(&mut w.collected) > 0 {
            reopens = 0;
        }
        match end {
            JobEnd::Changed => {}
            JobEnd::Exhausted => {
                eprintln!(
                    "[gpu] {}: searched all {}-byte extranonce2 values of this job - idle \
                     until the next one",
                    backend.name(),
                    job.extranonce2_size
                );
                sat_out = Some(generation);
            }
            JobEnd::DeviceGone => {
                // The device stopped answering. Re-open it - after a driver
                // reset the old context is dead for good, and every later
                // launch on it fails the same way. The old loop slept and
                // retried on that dead handle for ever: no hashrate, no exit,
                // and nothing in the logs after the first few lines.
                reopens += 1;
                if reopens > MAX_REOPENS {
                    eprintln!(
                        "[gpu] {} has failed {reopens} times after reopening - giving up on it. \
                         The card or its driver needs attention.",
                        device.describe()
                    );
                    return;
                }
                eprintln!(
                    "[gpu] {} stopped answering - reopening in {} s ({reopens}/{MAX_REOPENS})",
                    device.describe(),
                    REOPEN_DELAY.as_secs()
                );
                // Its kernels went with the context; the CPU stands in until
                // the new one has passed its self-test again.
                caps.set(PowAlgo::Bc3Hash, false);
                std::thread::sleep(REOPEN_DELAY);
                // The old backend goes first. CUDA hands every opener the
                // device's primary context, and only resets it when the last
                // reference goes: opened while the old one still held it, the
                // "new" backend got the same broken context back - a TDR
                // loop the reopen could never leave.
                drop(std::mem::replace(&mut backend, Box::new(Closed)));
                match open_backend_with_fallback(&device) {
                    Ok(b) => {
                        backend = b;
                        crate::human!("[gpu] reopened: {}", backend.name());
                        // A fresh context loads its BC3Hash kernel afresh.
                        bc3_failed = None;
                        shared.gpu_slot_bc3[slot].store(0, Ordering::Relaxed);
                        claim(&mut caps, &*backend);
                    }
                    Err(e) => {
                        eprintln!(
                            "[gpu] {} could not be reopened: {e} - this worker stops",
                            device.describe()
                        );
                        return;
                    }
                }
                // A new context means a new search space position is fine, and
                // the job may well have moved on while we waited.
                w.prev = None;
            }
        }
    }
}

/// How a job ended for this worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobEnd {
    /// A new job, or the same one again after a capability change.
    Changed,
    /// Every extranonce2 the job's size allows has been searched.
    Exhausted,
    /// The device stopped answering; the caller must re-open it.
    DeviceGone,
}

/// Where the next launch starts in this worker's part of the search space:
/// the extranonce2 counter (this worker's index, then every
/// `total_workers`-th) and the nonce within it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cursor {
    en2: u64,
    nonce: u32,
}

impl Cursor {
    /// The next launch of at most `max` nonces (`max` > 0): (extranonce2
    /// counter, first nonce, count). Moves past it - to the next
    /// extranonce2 of this worker when the nonce space runs out. A launch
    /// never crosses the end of the nonce space: the kernel's nonces wrap,
    /// but the header's extranonce2 cannot change in the middle of one.
    fn take(&mut self, max: u32, stride: u64) -> (u64, u32, u32) {
        let left = (1u64 << 32) - u64::from(self.nonce);
        let count = u64::from(max.max(1)).min(left) as u32;
        let taken = (self.en2, self.nonce, count);
        let (next, wrapped) = self.nonce.overflowing_add(count);
        if wrapped {
            self.en2 = self.en2.saturating_add(stride);
            self.nonce = 0;
        } else {
            self.nonce = next;
        }
        taken
    }
}

/// A launch in flight, with what its hits need to become shares.
struct InFlight {
    en2: u64,
    extranonce2: Vec<u8>,
    /// The header with nonce 0.
    header: BlockHeader,
    start: u32,
    count: u32,
    queued: Instant,
}

/// One GPU's state across jobs.
struct Worker<'a> {
    shared: &'a Shared,
    slot: usize,
    worker_index: usize,
    stride: u64,
    /// The job last mined, and where it stopped: a job that covers the same
    /// space continues there. See MinerJob::same_search_space.
    prev: Option<MinerJob>,
    cursor: Cursor,
    /// Autotuned nonces per launch.
    batch: u32,
    /// Keeps the launches inside the kernel's hit buffer.
    budget: HitBudget,
    errors: u32,
    /// Launches that came back since the caller last looked.
    collected: u64,
    /// When the last launch came back, for timing launches by the host
    /// clock where the device has none.
    last_collect: Option<Instant>,
    /// The BC3Hash window tuning of this card.
    tune: Tune,
}

/// The header of one extranonce2, built once and used by all its launches.
struct En2Header {
    en2: u64,
    extranonce2: Vec<u8>,
    header: BlockHeader,
    header76: [u8; 76],
}

fn en2_header(job: &MinerJob, en2: u64) -> En2Header {
    let extranonce2 = crate::worker::encode_extranonce2(en2, job.extranonce2_size);
    // Coinbase -> txid -> merkle root (once per extranonce2).
    let mut coinbase = Vec::with_capacity(
        job.coinb1.len() + job.extranonce1.len() + extranonce2.len() + job.coinb2.len(),
    );
    coinbase.extend_from_slice(&job.coinb1);
    coinbase.extend_from_slice(&job.extranonce1);
    coinbase.extend_from_slice(&extranonce2);
    coinbase.extend_from_slice(&job.coinb2);
    let merkle_root = root_from_steps(&sha256d(&coinbase), &job.merkle_steps);
    let header = BlockHeader {
        version: job.version,
        prev_hash: job.prev_hash,
        merkle_root,
        time: job.ntime,
        bits: job.bits,
        nonce: 0,
    };
    let header76 = header.serialize()[..76].try_into().unwrap();
    En2Header { en2, extranonce2, header, header76 }
}

/// One job on this GPU, while it runs.
struct JobRun<'j> {
    job: &'j MinerJob,
    generation: u64,
    algo: PowAlgo,
    verify: PowFn,
    block_target: Option<Target>,
    depth: usize,
    granule: u32,
    min_batch: u32,
    max_batch: u32,
    inflight: VecDeque<InFlight>,
    /// Set once no more launches may be queued for the job.
    ending: Option<JobEnd>,
    header: Option<En2Header>,
    /// The BC3Hash window the launches hash at (0: none), for the tuning.
    window: u32,
}

impl<'a> Worker<'a> {
    fn new(shared: &'a Shared, slot: usize, worker_index: usize, total_workers: usize) -> Self {
        Self {
            shared,
            slot,
            worker_index,
            stride: total_workers.max(1) as u64,
            prev: None,
            cursor: Cursor { en2: worker_index as u64, nonce: 0 },
            batch: BC3_START_BATCH,
            budget: HitBudget::default(),
            errors: 0,
            collected: 0,
            last_collect: None,
            tune: Tune::new(shared, slot),
        }
    }

    /// Grind `job` until it changes, its extranonce2 space runs out, or the
    /// device fails. Whatever is in flight when the job changes is collected
    /// and its hits submitted before this returns.
    fn mine_job(
        &mut self,
        backend: &mut dyn MiningBackend,
        job: &MinerJob,
        generation: u64,
        algo: PowAlgo,
        // The CPU reference for `algo`, resolved once per job.
        verify: PowFn,
    ) -> JobEnd {
        if self.prev.as_ref().is_none_or(|p| !p.same_search_space(job)) {
            self.cursor = Cursor { en2: self.worker_index as u64, nonce: 0 };
        }
        // Whole waves of the card only (see `launch_granule`); the autotune
        // below keeps doubling and halving within them.
        let granule = backend.launch_granule(algo).max(1);
        let min_batch = (BC3_MIN_BATCH.max(granule) / granule) * granule;
        let max_batch = BC3_MAX_BATCH.max(min_batch);
        self.batch = (self.batch.clamp(min_batch, max_batch) / granule).max(1) * granule;
        let mut run = JobRun {
            job,
            generation,
            algo,
            verify,
            block_target: compact_to_target(job.bits),
            depth: self.shared.gpu_queue(),
            granule,
            min_batch,
            max_batch,
            inflight: VecDeque::new(),
            ending: None,
            header: None,
            window: backend.bc3hash_window().unwrap_or(0),
        };
        let end = self.pump(backend, &mut run);
        self.prev = Some(job.clone());
        end
    }

    /// The launch loop of one job.
    fn pump(&mut self, backend: &mut dyn MiningBackend, run: &mut JobRun) -> JobEnd {
        loop {
            if let Err(gone) = self.top_up(backend, run) {
                return gone;
            }
            let Some(done) = run.inflight.pop_front() else {
                return run.ending.unwrap_or(JobEnd::Changed);
            };
            let scan = match backend.collect() {
                Ok(scan) => scan,
                Err(e) => {
                    let restart = Cursor { en2: done.en2, nonce: done.start };
                    if let Err(gone) = self.device_error(backend, run, &e, restart) {
                        return gone;
                    }
                    continue;
                }
            };
            self.errors = 0;
            // The next launch goes in behind the running one before the CPU
            // spends any time on this one's hits.
            if run.depth > 1 {
                if let Err(gone) = self.top_up(backend, run) {
                    return gone;
                }
            }
            let elapsed = self.finish(run, done, scan);
            // Intensity < 100 % -> idle proportionally (keeps heat down and
            // the machine usable while mining).
            self.shared.throttle(elapsed);
        }
    }

    /// Queue launches until `run.depth` are in flight, unless the job is
    /// over. Err only when the device is gone.
    fn top_up(&mut self, backend: &mut dyn MiningBackend, run: &mut JobRun) -> Result<(), JobEnd> {
        while run.ending.is_none() && run.inflight.len() < run.depth {
            // A window switch ends the run like a new job: what is in flight
            // is collected, and the main loop switches and picks the same job
            // up again where the cursor stopped.
            if self.shared.generation.load(Ordering::Acquire) != run.generation || self.tune.switch_to.is_some() {
                run.ending = Some(JobEnd::Changed);
                break;
            }
            if !en2_fits(self.cursor.en2, run.job.extranonce2_size) {
                run.ending = Some(JobEnd::Exhausted);
                break;
            }
            let max = self.budget.chunk(self.batch).max(1);
            let restart = self.cursor;
            let (en2, start, count) = self.cursor.take(max, self.stride);
            if run.header.as_ref().is_none_or(|h| h.en2 != en2) {
                run.header = Some(en2_header(run.job, en2));
            }
            let h = run.header.as_ref().expect("built above");
            let launch = Launch {
                algo: run.algo,
                header76: &h.header76,
                start,
                count,
                target: &run.job.share_target,
            };
            match backend.enqueue(&launch) {
                Ok(()) => run.inflight.push_back(InFlight {
                    en2,
                    extranonce2: h.extranonce2.clone(),
                    header: h.header,
                    start,
                    count,
                    queued: Instant::now(),
                }),
                Err(e) => {
                    // The oldest launch not done is where to start again.
                    let restart = run
                        .inflight
                        .front()
                        .map(|l| Cursor { en2: l.en2, nonce: l.start })
                        .unwrap_or(restart);
                    self.device_error(backend, run, &e, restart)?;
                }
            }
        }
        Ok(())
    }

    /// A launch failed to queue or to come back. Everything in flight is
    /// given up and done again from `restart`, so no range is skipped - and
    /// since none of it was collected, none of it was submitted either. Err
    /// once the device counts as gone.
    fn device_error(
        &mut self,
        backend: &mut dyn MiningBackend,
        run: &mut JobRun,
        e: &str,
        restart: Cursor,
    ) -> Result<(), JobEnd> {
        self.errors += 1;
        eprintln!("[gpu] {}: scan error: {e}", backend.name());
        backend.abandon();
        run.inflight.clear();
        self.cursor = restart;
        self.last_collect = None;
        if self.errors >= MAX_SCAN_ERRORS {
            eprintln!(
                "[gpu] {}: {} launches in a row failed - treating the device as gone",
                backend.name(),
                self.errors
            );
            self.errors = 0;
            return Err(JobEnd::DeviceGone);
        }
        std::thread::sleep(Duration::from_secs(1));
        Ok(())
    }

    /// Everything a launch that came back is owed: its hits re-checked and
    /// submitted, the counters, the launch size. Returns how long it ran.
    fn finish(&mut self, run: &JobRun, done: InFlight, scan: Scan) -> Duration {
        let now = Instant::now();
        // The device's own clock where it has one. Otherwise the host's: from
        // when the launch could start - queued, and the one before it back -
        // to now.
        let host = now.duration_since(self.last_collect.map_or(done.queued, |t| t.max(done.queued)));
        let elapsed = scan.busy.unwrap_or(host);
        self.last_collect = Some(now);
        self.collected += 1;

        if let Some(warning) = self.budget.overflowed(done.count, scan.reported) {
            eprintln!("{warning}");
        }
        // Verify every hit on the CPU before submit - guards against kernel
        // bugs. A nonce reported twice would be a duplicate share.
        let mut hits = scan.hits;
        hits.sort_unstable();
        hits.dedup();
        let mut header = done.header;
        for &nonce in &hits {
            // A nonce outside the launch's range is a kernel bug, whatever
            // the hash says: it would be a share for some other launch.
            if nonce.wrapping_sub(done.start) >= done.count {
                eprintln!("[gpu] WARNING: the kernel reported nonce {nonce:08x} outside its range - ignored");
                continue;
            }
            header.nonce = nonce;
            let hash = (run.verify)(&header.serialize());
            if !hash_meets_target(&hash, &run.job.share_target) {
                eprintln!("[gpu] WARNING: false hit from the kernel, nonce={nonce:08x} - ignored");
                continue;
            }
            let is_block = run.block_target.is_some_and(|t| hash_meets_target(&hash, &t));
            let difficulty = crate::consensus::difficulty_of_hash(&hash);
            self.shared.record_share(run.algo, difficulty, run.job.bits, is_block);
            self.shared.submit(FoundShare {
                job_id: run.job.job_id.clone(),
                extranonce2: done.extranonce2.clone(),
                ntime: run.job.ntime,
                nonce,
                hash_display: hash_to_display(&hash),
                is_block_candidate: is_block,
            });
        }

        let s = self.shared;
        s.stats.hashes.fetch_add(u64::from(done.count), Ordering::Relaxed);
        s.stats.gpu_hashes.fetch_add(u64::from(done.count), Ordering::Relaxed);
        if let (Some(hashes), Some(stamp), Some(busy)) = (
            s.gpu_slot_hashes.get(self.slot),
            s.gpu_slot_stamp.get(self.slot),
            s.gpu_slot_busy_us.get(self.slot),
        ) {
            // Stamp and busy time first: a reader that sees the new count
            // also sees a stamp and a busy total no older than it.
            stamp.store(s.started.elapsed().as_micros() as u64, Ordering::Relaxed);
            if let Some(b) = scan.busy {
                busy.fetch_add(b.as_micros() as u64, Ordering::Relaxed);
            }
            hashes.fetch_add(u64::from(done.count), Ordering::Release);
        }

        // Autotune toward the target time per launch, in whole waves. Judged
        // by what a full-size launch would take, so the short launch at the
        // end of an extranonce2 does not look like a fast card.
        let projected = elapsed.mul_f64(f64::from(self.batch) / f64::from(done.count.max(1)));
        self.batch =
            autotune(self.batch, projected, BC3_TARGET_LAUNCH, run.granule, run.min_batch, run.max_batch);
        if run.algo == PowAlgo::Bc3Hash && run.window != 0 {
            self.tune.launched(run.window, u64::from(done.count), elapsed);
        }
        elapsed
    }
}

/// The next batch size: double below half the target time, halve above two
/// and a half times it, in whole multiples of `granule`, within [min, max].
fn autotune(batch: u32, took: Duration, target: Duration, granule: u32, min: u32, max: u32) -> u32 {
    let granule = granule.max(1);
    let next = if took < target / 2 && batch < max {
        batch.saturating_mul(2).min(max)
    } else if took > target * 5 / 2 && batch > min {
        (batch / 2).max(min)
    } else {
        batch
    };
    ((next / granule).max(1) * granule).clamp(min.min(max), max)
}

/// The backend a worker holds for the moment between dropping a failed one
/// and opening its replacement.
struct Closed;

impl MiningBackend for Closed {
    fn name(&self) -> String {
        "closed".into()
    }
    fn supports(&self, _: PowAlgo) -> bool {
        false
    }
    fn enqueue(&mut self, _: &Launch) -> Result<(), String> {
        Err("the device is closed".into())
    }
    fn collect(&mut self) -> Result<Scan, String> {
        Err("the device is closed".into())
    }
    fn in_flight(&self) -> usize {
        0
    }
    fn abandon(&mut self) {}
}

/// Switch the card to BC3Hash window `window` for its tuning: the kernel was
/// compiled ahead where the card has no machine code for it, so the card
/// stops for its tables - built on the card, about a second even at window
/// 24 - and the self-test. A failure is no failure of the card - the tuning
/// tries a smaller window or goes back to the start window, which worked -
/// so unlike `prepare_bc3` it neither hands BC3Hash to the CPU nor says so in
/// red.
fn switch_bc3(shared: &Shared, backend: &mut dyn MiningBackend, slot: usize, window: u32) -> bool {
    if backend.bc3hash_window() == Some(window) {
        return true;
    }
    shared.gpu_slot_bc3[slot].store(0, Ordering::Relaxed);
    match backend.prepare_bc3hash(window) {
        Ok(took) => {
            shared.gpu_slot_bc3[slot].store(window, Ordering::Relaxed);
            crate::human!(
                "[gpu] {}: BC3Hash window {window} ready, self-test passed ({:.1} s)",
                backend.name(),
                took.as_secs_f64()
            );
            true
        }
        Err(e) => {
            eprintln!("[gpu] {}: switching to BC3Hash window {window} failed: {e}", backend.name());
            false
        }
    }
}

/// Have the card's BC3Hash kernel at window `want` - where the card starts,
/// `Shared::bc3_window`, or where its tuning moved it -
/// self-tested, before it mines a BC3Hash job - or know that it cannot, and
/// let the CPU have the job. Loading means building the tables (0.1-3 s), the
/// driver's JIT of the PTX (slow the first time on a card, cached after) and
/// the self-test against the CPU reference.
fn prepare_bc3(
    shared: &Shared,
    backend: &mut dyn MiningBackend,
    slot: usize,
    caps: &mut GpuCapability,
    failed: &mut Option<u32>,
    want: u32,
) {
    // Already there, or already known not to work: say so again, for a reader
    // (the benchmark) that has reset the slot.
    if backend.bc3hash_window() == Some(want) {
        shared.gpu_slot_bc3[slot].store(want, Ordering::Relaxed);
        return;
    }
    if *failed == Some(want) {
        shared.gpu_slot_bc3[slot].store(BC3_SLOT_FAILED, Ordering::Relaxed);
        return;
    }
    shared.gpu_slot_bc3[slot].store(0, Ordering::Relaxed);
    // The self-test needs the CPU reference; so does every hit later. A CPU
    // build that failed its own vectors leaves the GPU nothing to check
    // against.
    let result = if let Some(why) = crate::worker::cpu_unavailable(PowAlgo::Bc3Hash, true) {
        Err(format!("no CPU reference to check it against: {why}"))
    } else {
        backend.prepare_bc3hash(want)
    };
    match result {
        Ok(took) => {
            *failed = None;
            shared.gpu_slot_bc3[slot].store(want, Ordering::Relaxed);
            crate::human!(
                "[gpu] {}: BC3Hash kernel ready - window {want}, self-test passed ({:.1} s)",
                backend.name(),
                took.as_secs_f64()
            );
            caps.set(PowAlgo::Bc3Hash, true);
        }
        Err(e) => {
            *failed = Some(want);
            shared.gpu_slot_bc3[slot].store(BC3_SLOT_FAILED, Ordering::Relaxed);
            let message = format!(
                "{}: BC3Hash with window {want} is NOT usable on this GPU - {e}. It mines no \
                 BC3Hash there; in --backend auto the CPU does instead. Report it.",
                backend.name()
            );
            eprintln!("[gpu] {message}");
            crate::ipc::emit(&crate::ipc::Event::Status {
                state: crate::ipc::StatusState::Error,
                message,
            });
            // The card has held BC3Hash since it opened, so the stand-ins
            // stay out of its way while it loads. Without giving that up
            // here they never took over: nothing mined BC3Hash, and the
            // message above said the CPU did.
            caps.set(PowAlgo::Bc3Hash, false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::QUEUE_DEPTH;
    use crate::shared::{MiningPlan, SUBMIT_QUEUE};
    use std::collections::HashSet;

    /// Launches taken from a cursor tile this worker's space exactly: no
    /// nonce twice, none skipped, never across the end of a nonce space, and
    /// on to the worker's next extranonce2 after it.
    #[test]
    fn launches_tile_the_space_without_gaps_or_overlap() {
        let stride = 3;
        let mut c = Cursor { en2: 1, nonce: u32::MAX - 1000 };
        let mut covered: Vec<(u64, u64, u64)> = Vec::new();
        for max in [300u32, 300, 300, 300, 300, 7, 1 << 20, 1] {
            let (en2, start, count) = c.take(max, stride);
            assert!(count >= 1 && count <= max);
            assert!(u64::from(start) + u64::from(count) <= 1 << 32, "crossed the end");
            covered.push((en2, u64::from(start), u64::from(count)));
        }
        // 1001 nonces left in en2 1: 300, 300, 300, 101 - then en2 4.
        assert_eq!(covered[3], (1, u64::from(u32::MAX - 100), 101));
        assert_eq!(covered[4], (4, 0, 300));
        assert_eq!(covered[5], (4, 300, 7));
        assert_eq!(covered[6], (4, 307, 1 << 20));
        assert_eq!(covered[7], (4, 307 + (1 << 20), 1));
        for pair in covered.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if a.0 == b.0 {
                assert_eq!(a.1 + a.2, b.1, "gap or overlap within an extranonce2");
            } else {
                assert_eq!(a.1 + a.2, 1 << 32, "left an extranonce2 before its end");
                assert_eq!((b.0 - a.0, b.1), (stride, 0));
            }
        }
    }

    /// A launch that ends exactly at 2^32 moves the cursor on, and a full
    /// 2^32 space in one launch is impossible (counts are u32) but a cursor
    /// at 0 with a huge max takes what fits.
    #[test]
    fn the_cursor_wraps_exactly_at_the_end() {
        let mut c = Cursor { en2: 5, nonce: u32::MAX };
        assert_eq!(c.take(10, 2), (5, u32::MAX, 1));
        assert_eq!(c, Cursor { en2: 7, nonce: 0 });
        let mut c = Cursor { en2: 0, nonce: 0 };
        assert_eq!(c.take(u32::MAX, 1), (0, 0, u32::MAX));
        assert_eq!(c.take(u32::MAX, 1), (0, u32::MAX, 1));
        assert_eq!(c, Cursor { en2: 1, nonce: 0 });
        // max 0 is treated as 1 rather than a launch of nothing.
        let mut c = Cursor { en2: 0, nonce: 9 };
        assert_eq!(c.take(0, 1), (0, 9, 1));
    }

    /// The extranonce2 counter must stay inside what its encoding can hold.
    #[test]
    fn extranonce2_counters_fit_their_size() {
        assert!(en2_fits(255, 1));
        assert!(!en2_fits(256, 1));
        assert!(en2_fits(65_535, 2));
        assert!(!en2_fits(65_536, 2));
        assert!(en2_fits(u64::MAX, 8));
        assert!(en2_fits(u64::MAX, 16));
        // What the encoding does past the limit: it wraps - the reason for
        // the check.
        assert_eq!(
            crate::worker::encode_extranonce2(256, 1),
            crate::worker::encode_extranonce2(0, 1)
        );
    }

    /// The autotune doubles towards the target, halves well above it, stays
    /// in whole granules and inside its limits.
    #[test]
    fn the_autotune_moves_in_whole_granules() {
        let t = BC3_TARGET_LAUNCH;
        let ms = Duration::from_millis;
        // An RTX 3050 Ti at window 16: one wave of 5120 is 44 ms, and the
        // autotune settles at eight.
        assert_eq!(autotune(5120, ms(44), t, 5120, 5120, 1 << 24), 10_240);
        assert_eq!(autotune(20_480, ms(178), t, 5120, 5120, 1 << 24), 40_960);
        assert_eq!(autotune(40_960, ms(356), t, 5120, 5120, 1 << 24), 40_960);
        assert_eq!(autotune(40_960, ms(1100), t, 5120, 5120, 1 << 24), 20_480);
        // Never below one granule, never past the ceiling.
        assert_eq!(autotune(5120, ms(5000), t, 5120, 5120, 1 << 24), 5120);
        assert_eq!(autotune(1 << 24, ms(1), t, 32_768, 32_768, 1 << 24), 1 << 24);
        // A granule that does not divide the ceiling.
        assert_eq!(autotune(3 * 1000, ms(1), t, 1000, 1000, 5500), 5000);
    }

    /// A scripted backend: every launch "finds" the nonces at `hits_at`
    /// (offsets into the launch) and took `BC3_TARGET_LAUNCH` of device time,
    /// so the autotune leaves the launch size alone. It records what was
    /// queued, can fail one collect, and moves the job on (bumps the
    /// generation, as a new job from the pool would) after `bump_after`
    /// collects.
    struct Scripted<'a> {
        queue: VecDeque<(u32, u32)>,
        log: Vec<(u32, u32)>,
        hits_at: Vec<u32>,
        fail_collect_at: Option<usize>,
        bump_after: (usize, &'a Shared),
        collects: usize,
    }

    impl<'a> Scripted<'a> {
        fn new(shared: &'a Shared, hits_at: Vec<u32>, fail_collect_at: Option<usize>, bump_after: usize) -> Self {
            Self {
                queue: VecDeque::new(),
                log: vec![],
                hits_at,
                fail_collect_at,
                bump_after: (bump_after, shared),
                collects: 0,
            }
        }
    }

    impl MiningBackend for Scripted<'_> {
        fn name(&self) -> String {
            "scripted".into()
        }
        fn supports(&self, algo: PowAlgo) -> bool {
            algo == PowAlgo::Bc3Hash
        }
        fn enqueue(&mut self, l: &Launch) -> Result<(), String> {
            assert!(self.queue.len() < QUEUE_DEPTH, "more than QUEUE_DEPTH in flight");
            self.queue.push_back((l.start, l.count));
            self.log.push((l.start, l.count));
            Ok(())
        }
        fn collect(&mut self) -> Result<Scan, String> {
            self.collects += 1;
            if self.collects == self.bump_after.0 {
                self.bump_after.1.generation.fetch_add(1, Ordering::Release);
            }
            let (start, count) = self.queue.pop_front().ok_or("empty")?;
            if self.fail_collect_at == Some(self.collects) {
                return Err("scripted failure".into());
            }
            let hits: Vec<u32> = self.hits_at.iter().filter(|&&o| o < count).map(|o| start + o).collect();
            Ok(Scan { reported: hits.len(), hits, busy: Some(BC3_TARGET_LAUNCH) })
        }
        fn in_flight(&self) -> usize {
            self.queue.len()
        }
        fn abandon(&mut self) {
            self.queue.clear();
        }
    }

    fn job(target: Target) -> MinerJob {
        MinerJob {
            job_id: "j".into(),
            extranonce1: vec![1, 2, 3, 4],
            extranonce2_size: 4,
            coinb1: vec![9; 50],
            coinb2: vec![8; 20],
            merkle_steps: vec![],
            version: 0x2000_1800,
            prev_hash: [6u8; 32],
            bits: 0x1d00_ffff,
            ntime: 1_700_000_000,
            share_target: target,
            height: None,
            algo: PowAlgo::Bc3Hash,
        }
    }

    /// The CPU re-check these tests run hits through. Not BC3Hash's own: the
    /// pipeline does the same with any PoW, the tests only ask whether a hash
    /// meets an all-ones target (every hash does) or an all-zero one (none
    /// does), and SHA256d needs no tables and no `bc3hash` feature.
    fn check_fn() -> PowFn {
        |h: &[u8; 80]| sha256d(h)
    }

    fn shared_with_job(j: &MinerJob) -> (Shared, std::sync::mpsc::Receiver<FoundShare>, u64) {
        let (tx, rx) = std::sync::mpsc::sync_channel(SUBMIT_QUEUE);
        let shared = Shared::new(tx, MiningPlan { gpus: 1, ..Default::default() });
        shared.publish_job(j.clone());
        let (generation, _) = shared.wait_for_job();
        (shared, rx, generation)
    }

    /// The pipeline end to end on a scripted device: launches queue two deep
    /// and tile the space, hits are re-checked and submitted, and a job
    /// change stops the queueing, drains what is in flight and leaves the
    /// cursor just past it.
    #[test]
    fn the_pipeline_queues_ahead_and_drains_on_a_job_change() {
        // Every hash meets an all-ones target, so each reported hit is real.
        let j = job([0xff; 32]);
        let (shared, rx, generation) = shared_with_job(&j);
        let mut backend = Scripted::new(&shared, vec![0, 3], None, 5);
        let mut w = Worker::new(&shared, 0, 0, 1);
        let verify = check_fn();
        assert_eq!(w.mine_job(&mut backend, &j, generation, PowAlgo::Bc3Hash, verify), JobEnd::Changed);
        assert_eq!(backend.queue.len(), 0, "everything in flight was collected");
        // Five collects, then the job changed: the sixth launch was already
        // queued behind the fifth and is the last.
        assert_eq!(backend.log.len(), 6);
        let mut next = 0u32;
        for &(start, count) in &backend.log {
            assert_eq!(start, next, "launches must be contiguous");
            next = start + count;
        }
        assert_eq!(w.cursor, Cursor { en2: 0, nonce: next });
        let shares: Vec<FoundShare> = rx.try_iter().collect();
        assert_eq!(shares.len(), 2 * backend.log.len());
        let unique: HashSet<u32> = shares.iter().map(|s| s.nonce).collect();
        assert_eq!(unique.len(), shares.len(), "a nonce submitted twice");
        assert!(shares.iter().all(|s| s.job_id == "j" && s.extranonce2 == vec![0, 0, 0, 0]));
        assert_eq!(shared.stats.gpu_hashes.load(Ordering::Relaxed), u64::from(next));
        assert_eq!(
            shared.gpu_slot_busy_us[0].load(Ordering::Relaxed),
            6 * BC3_TARGET_LAUNCH.as_micros() as u64
        );
        assert_eq!(std::mem::take(&mut w.collected), 6);

        // The same space again (a retarget): the next launch starts right
        // where the last one ended.
        let mut retarget = j.clone();
        retarget.job_id = "k".into();
        shared.publish_job(retarget.clone());
        let (generation, _) = shared.wait_for_job();
        let mut backend = Scripted::new(&shared, vec![], None, 1);
        w.mine_job(&mut backend, &retarget, generation, PowAlgo::Bc3Hash, verify);
        assert_eq!(backend.log[0].0, next);
    }

    /// A hit the CPU does not confirm is dropped.
    #[test]
    fn hits_are_checked_before_they_are_submitted() {
        // An impossible target: no hash meets it, so every hit is false.
        let j = job([0; 32]);
        let (shared, rx, generation) = shared_with_job(&j);
        let mut backend = Scripted::new(&shared, vec![0, 1, 2], None, 3);
        let mut w = Worker::new(&shared, 0, 0, 1);
        w.mine_job(&mut backend, &j, generation, PowAlgo::Bc3Hash, check_fn());
        assert_eq!(rx.try_iter().count(), 0, "a false hit was submitted");
    }

    /// A nonce the kernel reports twice is still one share.
    #[test]
    fn a_hit_reported_twice_is_submitted_once() {
        let j = job([0xff; 32]);
        let (shared, rx, generation) = shared_with_job(&j);
        let mut backend = Scripted::new(&shared, vec![5, 5, 9], None, 1);
        let mut w = Worker::new(&shared, 0, 0, 1);
        w.mine_job(&mut backend, &j, generation, PowAlgo::Bc3Hash, check_fn());
        let nonces: Vec<u32> = rx.try_iter().map(|s| s.nonce).collect();
        // Two launches (one queued behind the first): 5 and 9 from each.
        let (a, b) = (backend.log[0].0, backend.log[1].0);
        assert_eq!(nonces, vec![a + 5, a + 9, b + 5, b + 9]);
    }

    /// A failed collect gives up what is in flight and starts again from the
    /// oldest launch that did not come back: nothing skipped, nothing
    /// submitted twice.
    #[test]
    fn a_failed_launch_is_done_again() {
        let j = job([0xff; 32]);
        let (shared, rx, generation) = shared_with_job(&j);
        let mut backend = Scripted::new(&shared, vec![1], Some(3), 6);
        let mut w = Worker::new(&shared, 0, 0, 1);
        let verify = check_fn();
        assert_eq!(w.mine_job(&mut backend, &j, generation, PowAlgo::Bc3Hash, verify), JobEnd::Changed);
        // Launches 1 and 2 came back; 3 failed with 4 queued behind it, and
        // both were queued again from 3's start.
        let starts: Vec<u32> = backend.log.iter().map(|l| l.0).collect();
        assert_eq!(starts[4], starts[2], "launch 3 was not redone: {starts:?}");
        assert_eq!(starts[5], starts[3], "launch 4 was not redone: {starts:?}");
        let nonces: Vec<u32> = rx.try_iter().map(|s| s.nonce).collect();
        let unique: HashSet<u32> = nonces.iter().copied().collect();
        assert_eq!(unique.len(), nonces.len(), "a nonce submitted twice: {nonces:?}");
        assert!(!nonces.is_empty());
    }

    /// Five failures in a row and the device counts as gone.
    #[test]
    fn a_device_that_keeps_failing_is_given_up() {
        struct Broken;
        impl MiningBackend for Broken {
            fn name(&self) -> String {
                "broken".into()
            }
            fn supports(&self, _: PowAlgo) -> bool {
                true
            }
            fn enqueue(&mut self, _: &Launch) -> Result<(), String> {
                Err("lost".into())
            }
            fn collect(&mut self) -> Result<Scan, String> {
                Err("lost".into())
            }
            fn in_flight(&self) -> usize {
                0
            }
            fn abandon(&mut self) {}
        }
        let j = job([0xff; 32]);
        let (shared, _rx, generation) = shared_with_job(&j);
        let mut w = Worker::new(&shared, 0, 0, 1);
        let t0 = Instant::now();
        let end = w.mine_job(&mut Broken, &j, generation, PowAlgo::Bc3Hash, check_fn());
        assert_eq!(end, JobEnd::DeviceGone);
        assert!(t0.elapsed() >= Duration::from_secs(4), "it must retry, with a pause, before giving up");
        assert_eq!(w.cursor, Cursor { en2: 0, nonce: 0 }, "nothing was done, nothing skipped");
        // Nothing came back, so nothing may reset the reopen budget.
        assert_eq!(w.collected, 0);
    }

    /// A job whose extranonce2 space this worker has used up is not ground
    /// on through wrapped values.
    #[test]
    fn an_exhausted_extranonce2_space_ends_the_job() {
        let mut j = job([0xff; 32]);
        j.extranonce2_size = 1;
        let (shared, _rx, generation) = shared_with_job(&j);
        let mut backend = Scripted::new(&shared, vec![], None, 1000);
        let mut w = Worker::new(&shared, 0, 0, 1);
        w.prev = Some(j.clone());
        w.cursor = Cursor { en2: 255, nonce: u32::MAX - 10 };
        let end = w.mine_job(&mut backend, &j, generation, PowAlgo::Bc3Hash, check_fn());
        assert_eq!(end, JobEnd::Exhausted);
        assert_eq!(backend.log, vec![(u32::MAX - 10, 11)]);
    }

    /// A hit buffer that overflows shrinks the next launches until the hits
    /// fit, and says so.
    #[test]
    fn an_overflowing_hit_buffer_shrinks_the_launches() {
        let mut budget = HitBudget::default();
        assert_eq!(budget.chunk(1 << 20), 1 << 20);
        assert!(budget.overflowed(1 << 20, crate::backend::MAX_HITS + 10).is_some());
        assert_eq!(budget.chunk(1 << 20), 1 << 19);
    }

    /// The backend that stands in while a device is reopened can do nothing.
    #[test]
    fn a_closed_backend_refuses_everything() {
        let mut c = Closed;
        let t = [0u8; 32];
        let h = [0u8; 76];
        assert!(c.enqueue(&Launch { algo: PowAlgo::Bc3Hash, header76: &h, start: 0, count: 1, target: &t }).is_err());
        assert!(c.collect().is_err());
        assert!(!c.supports(PowAlgo::Bc3Hash));
    }

    /// A card whose BC3Hash kernel will not load or fails its self-test, the
    /// way `run_gpu_worker` meets it at its first job: the card claimed
    /// BC3Hash when it opened (so the stand-ins keep out of its table build
    /// and self-test), and a failure has to give that claim up - or the
    /// stand-in CPU threads of --backend auto never take over, nothing mines
    /// at all, and the status line says the CPU does.
    #[cfg(feature = "bc3hash")]
    #[test]
    fn a_failed_bc3hash_kernel_hands_the_pow_to_the_cpu() {
        struct NoBc3;
        impl MiningBackend for NoBc3 {
            fn name(&self) -> String {
                "no-bc3".into()
            }
            fn has_kernel(&self, _: PowAlgo) -> bool {
                true
            }
            fn supports(&self, _: PowAlgo) -> bool {
                false
            }
            fn prepare_bc3hash(&mut self, _: u32) -> Result<Duration, String> {
                Err("self-test failed: scripted".into())
            }
            fn enqueue(&mut self, _: &Launch) -> Result<(), String> {
                Err("not in this test".into())
            }
            fn collect(&mut self) -> Result<Scan, String> {
                Err("not in this test".into())
            }
            fn in_flight(&self) -> usize {
                0
            }
            fn abandon(&mut self) {}
        }
        let (tx, rx) = std::sync::mpsc::sync_channel(SUBMIT_QUEUE);
        let plan = MiningPlan {
            gpus: 1,
            cpu_threads: 0,
            cpu_fallback_threads: 1,
            gpu_kernel: [false, false, true],
        };
        let shared = Arc::new(Shared::new(tx, plan));
        // What run_gpu_worker claims when the card opens.
        let mut caps = GpuCapability::new(&shared);
        caps.set(PowAlgo::Bc3Hash, true);
        shared.gpu_opened();
        let s = shared.clone();
        std::thread::spawn(move || {
            crate::worker::run_worker(s, 1, 2, crate::worker::CpuRole::GpuFallback)
        });
        // A BC3Hash job with bit 11 alone, as at the fork, and a target every
        // hash meets.
        let mut fork = job([0xff; 32]);
        fork.version = 0x2000_0800;
        shared.publish_job(fork);
        std::thread::sleep(Duration::from_millis(300));
        assert!(rx.try_recv().is_err(), "the stand-in mined what the card had claimed");

        let mut failed = None;
        prepare_bc3(&shared, &mut NoBc3, 0, &mut caps, &mut failed, shared.bc3_window());
        assert_eq!(failed, Some(shared.bc3_window()));
        assert_eq!(shared.gpu_slot_bc3[0].load(Ordering::Relaxed), BC3_SLOT_FAILED);
        assert!(!shared.gpus_mine(PowAlgo::Bc3Hash), "the failed card still claims BC3Hash");
        assert!(shared.devices_for(PowAlgo::Bc3Hash).cpu);
        let share = rx
            .recv_timeout(Duration::from_secs(30))
            .expect("nothing mines BC3Hash after the card's kernel failed");
        assert_eq!(share.job_id, "j");
        // The next job looks again and finds the same failure: no claim.
        prepare_bc3(&shared, &mut NoBc3, 0, &mut caps, &mut failed, shared.bc3_window());
        assert!(!shared.gpus_mine(PowAlgo::Bc3Hash));
        shared.clear_job();
    }
}
