//! GPU worker thread: the same job pipeline as the CPU worker, but the nonce
//! grinding happens on the GPU in batches. Coinbase -> merkle root is built on
//! the CPU per extranonce2 (cheap), and every GPU hit is verified on the CPU
//! before submit.

use crate::backend::{open_backend_with_fallback, GpuDevice, MiningBackend};
use crate::consensus::{
    compact_to_target, hash_meets_target, hash_to_display, root_from_steps, sha256d, BlockHeader,
    PowAlgo, PowFn,
};
use crate::shared::{AlgoGate, FoundShare, GpuCapability, MinerJob, Shared, BC3_SLOT_FAILED};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Autotune limits for the batch size (nonces per kernel launch).
///
/// The ceiling has to sit high enough that the autotune actually reaches
/// TARGET_LAUNCH even on a fast card. With the old ceiling of 1 << 24 a launch
/// took only ~12 ms on an RTX 4090 - the autotune hit the ceiling and every
/// launch paid the full sync overhead (two H2D, one D2H, cuCtxSynchronize) for
/// far too little work. It showed up as ~96 % GPU usage instead of ~99 %.
///
/// 1 << 29 leaves room up to ~5 GH/s before the ceiling bites again. The
/// autotune still settles around TARGET_LAUNCH, so slow cards are unaffected -
/// and no launch gets long enough to approach the Windows TDR watchdog (2 s).
const MIN_BATCH: u32 = 1 << 18;
const MAX_BATCH: u32 = 1 << 29;
const START_BATCH: u32 = 1 << 20;
/// Target time per launch - short enough for fast job switching, long enough
/// to keep launch overhead down.
const TARGET_LAUNCH: Duration = Duration::from_millis(100);

/// The same for BC3Hash, which is ~1000 times SHA3-256t's work per hash -
/// about 110 kH/s on an RTX 3050 Ti. Launches start at the floor: 256 nonces
/// are 2 ms there, and still far from the Windows TDR watchdog (2 s) on an
/// integrated GPU through OpenCL that may manage a few hundred hashes a
/// second - a 16k start would have been seconds there. From the floor the
/// autotune doubles its way up, in whole waves of the card (`launch_granule`;
/// on CUDA the floor is one wave). 400 ms per launch is several waves - what
/// runs short at the end of a launch is a small part of it.
const BC3_MIN_BATCH: u32 = 1 << 8;
const BC3_MAX_BATCH: u32 = 1 << 24;
const BC3_START_BATCH: u32 = 1 << 8;
const BC3_TARGET_LAUNCH: Duration = Duration::from_millis(400);

/// Launch sizing per PoW: (min, max, start, target time per launch).
fn tuning(algo: PowAlgo) -> (u32, u32, u32, Duration) {
    match algo {
        PowAlgo::Sha256d | PowAlgo::Sha3t => (MIN_BATCH, MAX_BATCH, START_BATCH, TARGET_LAUNCH),
        PowAlgo::Bc3Hash => (BC3_MIN_BATCH, BC3_MAX_BATCH, BC3_START_BATCH, BC3_TARGET_LAUNCH),
    }
}

/// `worker_index`/`total_workers` partition the extranonce2 space disjointly
/// across all workers (CPU threads + GPUs), exactly as in worker.rs.
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
            return;
        }
    };
    crate::human!("[gpu] started: {}", backend.name());

    // What this card can mine, as far as anyone can tell before trying. A
    // kernel that later fails its self-test withdraws its claim; dropping
    // this on any exit withdraws them all.
    let mut caps = GpuCapability::new(&shared);
    for algo in PowAlgo::ALL {
        caps.set(algo, backend.has_kernel(algo) && crate::worker::cpu_may_support(algo));
    }
    // The BC3Hash window that last failed here, so it is not retried on
    // every job; a different window still gets its chance.
    let mut bc3_failed: Option<u32> = None;

    let mut batch: [u32; 3] = PowAlgo::ALL.map(|a| tuning(a).2);
    // Position in the search space, carried across jobs that cover the same
    // space. See MinerJob::same_search_space.
    let mut prev: Option<MinerJob> = None;
    let mut en2 = worker_index as u64;
    let mut nonce: u32 = 0;
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
            prepare_bc3(&shared, &mut *backend, slot, &mut caps, &mut bc3_failed);
        }
        // Decided per job, so the launch loop below carries no dispatch. A
        // kernel without a CPU reference to re-check its hits against does
        // not mine at all - see AGENTS.md.
        let supported = |a: PowAlgo| backend.supports(a) && a.hash_fn().is_some();
        let Some((algo, verify)) = gate
            .check(job.algo, Some(backend.name().as_str()), supported)
            .and_then(|a| Some((a, a.hash_fn()?)))
        else {
            // BC3Hash since the fork: in --backend auto the CPU threads take
            // it over, and this worker sleeps until SHA3-256t comes back.
            sat_out = Some(generation);
            continue;
        };
        if prev.as_ref().is_none_or(|p| !p.same_search_space(&job)) {
            en2 = worker_index as u64;
            nonce = 0;
        }
        let alive = mine_job(
            &shared, &job, algo, verify, generation, slot, total_workers, &mut backend,
            &mut batch[algo.index()], &mut en2, &mut nonce,
        );
        if !alive {
            // The device stopped answering. Re-open it - after a driver reset
            // the old context is dead for good, and every later launch on it
            // fails the same way. The old loop slept and retried on that dead
            // handle for ever: no hashrate, no exit, and nothing in the logs
            // after the first few lines.
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
            std::thread::sleep(REOPEN_DELAY);
            match open_backend_with_fallback(&device) {
                Ok(b) => {
                    backend = b;
                    crate::human!("[gpu] reopened: {}", backend.name());
                    // A fresh context loads its BC3Hash kernel afresh.
                    bc3_failed = None;
                    shared.gpu_slot_bc3[slot].store(0, Ordering::Relaxed);
                }
                Err(e) => {
                    eprintln!(
                        "[gpu] {} could not be reopened: {e} - this worker stops",
                        device.describe()
                    );
                    return;
                }
            }
            // A new context means a new search space position is fine, and the
            // job may well have moved on while we waited.
            prev = None;
            continue;
        }
        // `mine_job` returning true means at least one scan came back, so the
        // card is working again and the budget above is spent, not consumed.
        reopens = 0;
        prev = Some(job);
    }
}

/// Have the card's BC3Hash kernel at the window `Shared::bc3_window` asks for,
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
) {
    let want = shared.bc3_window();
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
            caps.set(PowAlgo::Bc3Hash, false);
        }
    }
}

/// How many launches in a row may fail before the device counts as gone.
const MAX_SCAN_ERRORS: u32 = 5;

/// Grind until the job changes. `false` = the device stopped answering and the
/// caller must re-open it; `true` = ordinary job change.
#[allow(clippy::too_many_arguments)]
fn mine_job(
    shared: &Shared,
    job: &MinerJob,
    algo: PowAlgo,
    // The CPU reference for `algo`, resolved once per job.
    verify: PowFn,
    generation: u64,
    slot: usize,
    total_workers: usize,
    backend: &mut Box<dyn MiningBackend>,
    batch: &mut u32,
    en2_counter: &mut u64,
    nonce_start: &mut u32,
) -> bool {
    let block_target = compact_to_target(job.bits);
    let (min_batch, max_batch, _, target_launch) = tuning(algo);
    // Whole waves of the card only (see `launch_granule`); the autotune below
    // keeps doubling and halving within them.
    let granule = backend.launch_granule(algo).max(1);
    let min_batch = min_batch.max(granule) / granule * granule;
    *batch = ((*batch).clamp(min_batch, max_batch.max(min_batch)) / granule).max(1) * granule;
    // Starting point comes from the caller so it can survive a job that
    // covers the same space.
    loop {
        let extranonce2 =
            crate::worker::encode_extranonce2(*en2_counter, job.extranonce2_size);

        // Coinbase -> txid -> merkle root (once per extranonce2).
        let mut coinbase = Vec::with_capacity(
            job.coinb1.len() + job.extranonce1.len() + extranonce2.len() + job.coinb2.len(),
        );
        coinbase.extend_from_slice(&job.coinb1);
        coinbase.extend_from_slice(&job.extranonce1);
        coinbase.extend_from_slice(&extranonce2);
        coinbase.extend_from_slice(&job.coinb2);
        let cb_txid = sha256d(&coinbase);
        let merkle_root = root_from_steps(&cb_txid, &job.merkle_steps);

        let mut header = BlockHeader {
            version: job.version,
            prev_hash: job.prev_hash,
            merkle_root,
            time: job.ntime,
            bits: job.bits,
            nonce: 0,
        };
        let header76: [u8; 76] = header.serialize()[..76].try_into().unwrap();

        // Grind the whole nonce space in batches.
        let mut start: u32 = *nonce_start;
        let mut errors = 0u32;
        loop {
            let remaining = (u32::MAX - start).saturating_add(1).max(1);
            let count = (*batch).min(remaining);

            let t0 = Instant::now();
            let hits = match backend.scan_range(algo, &header76, start, count, &job.share_target) {
                Ok(h) => h,
                Err(e) => {
                    errors += 1;
                    eprintln!("[gpu] {}: scan error: {e}", backend.name());
                    if errors >= MAX_SCAN_ERRORS {
                        eprintln!(
                            "[gpu] {}: {errors} launches in a row failed - treating the device as gone",
                            backend.name()
                        );
                        return false;
                    }
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
            };
            let elapsed = t0.elapsed();
            errors = 0;

            // Verify every hit on the CPU before submit - guards against
            // kernel bugs.
            for nonce in hits {
                header.nonce = nonce;
                let hash = verify(&header.serialize());
                if !hash_meets_target(&hash, &job.share_target) {
                    eprintln!(
                        "[gpu] WARNING: {} reported a false hit, nonce={nonce:08x} - ignored",
                        backend.name()
                    );
                    continue;
                }
                let is_block = block_target
                    .map(|t| hash_meets_target(&hash, &t))
                    .unwrap_or(false);
                shared.record_share(crate::consensus::difficulty_of_hash(&hash), is_block);
                shared.submit(FoundShare {
                    job_id: job.job_id.clone(),
                    extranonce2: extranonce2.clone(),
                    ntime: job.ntime,
                    nonce,
                    hash_display: hash_to_display(&hash),
                    is_block_candidate: is_block,
                });
            }

            shared.stats.hashes.fetch_add(count as u64, Ordering::Relaxed);
            shared.stats.gpu_hashes.fetch_add(count as u64, Ordering::Relaxed);
            if let (Some(hashes), Some(stamp)) =
                (shared.gpu_slot_hashes.get(slot), shared.gpu_slot_stamp.get(slot))
            {
                // Stamp first: a reader that sees the new count also sees a
                // stamp no older than it.
                stamp.store(shared.started.elapsed().as_micros() as u64, Ordering::Relaxed);
                hashes.fetch_add(count as u64, Ordering::Release);
            }

            // Intensity < 100 % -> idle proportionally (keeps heat down and
            // the machine usable while mining).
            shared.throttle(elapsed);

            // Simple autotune towards the PoW's target time per launch, in
            // whole waves.
            if elapsed < target_launch / 2 && *batch < max_batch {
                *batch = (*batch * 2).min(max_batch.max(granule)) / granule * granule;
            } else if elapsed > target_launch * 5 / 2 && *batch > min_batch {
                *batch = ((*batch / 2).max(min_batch) / granule).max(1) * granule;
            }

            // Where the batch we just finished ends - not where it began.
            // Saving `start` re-hashed the whole batch on the next job, up to
            // 2^29 nonces, and the shares found in it came back as duplicates.
            // That is the very waste this bookkeeping exists to avoid.
            let (next, wrapped) = start.overflowing_add(count);

            if shared.generation.load(Ordering::Acquire) != generation {
                // Remember where we stopped: if the next job covers the same
                // space we continue here instead of re-hashing it.
                let (e, n) = crate::worker::resume_point(
                    *en2_counter,
                    next,
                    wrapped,
                    total_workers as u64,
                );
                *en2_counter = e;
                *nonce_start = n;
                return true;
            }
            if wrapped {
                break; // nonce space exhausted for this extranonce2
            }
            start = next;
        }
        // Done with this extranonce2 - next one, from the top of its nonces.
        *en2_counter += total_workers as u64;
        *nonce_start = 0;
    }
}
