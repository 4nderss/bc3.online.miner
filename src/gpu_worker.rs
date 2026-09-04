//! GPU worker thread: the same job pipeline as the CPU worker, but the nonce
//! grinding happens on the GPU in batches. Coinbase -> merkle root is built on
//! the CPU per extranonce2 (cheap), and every GPU hit is verified on the CPU
//! before submit.

use crate::backend::{open_backend_with_fallback, GpuDevice, MiningBackend};
use crate::consensus::{
    compact_to_target, hash_meets_target, root_from_steps, sha256d, BlockHeader,
};
use crate::shared::{FoundShare, MinerJob, Shared};
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

pub fn run_gpu_worker(
    shared: Arc<Shared>,
    device: GpuDevice,
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

    let mut batch = START_BATCH;
    // Position in the search space, carried across jobs that cover the same
    // space. See MinerJob::same_search_space.
    let mut prev: Option<MinerJob> = None;
    let mut en2 = worker_index as u64;
    let mut nonce: u32 = 0;
    let mut reopens: u32 = 0;
    loop {
        let (generation, job) = shared.wait_for_job();
        if prev.as_ref().is_none_or(|p| !p.same_search_space(&job)) {
            en2 = worker_index as u64;
            nonce = 0;
        }
        let alive = mine_job(
            &shared, &job, generation, total_workers, &mut backend, &mut batch, &mut en2,
            &mut nonce,
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

/// How many launches in a row may fail before the device counts as gone.
const MAX_SCAN_ERRORS: u32 = 5;

/// Grind until the job changes. `false` = the device stopped answering and the
/// caller must re-open it; `true` = ordinary job change.
#[allow(clippy::too_many_arguments)]
fn mine_job(
    shared: &Shared,
    job: &MinerJob,
    generation: u64,
    total_workers: usize,
    backend: &mut Box<dyn MiningBackend>,
    batch: &mut u32,
    en2_counter: &mut u64,
    nonce_start: &mut u32,
) -> bool {
    let block_target = compact_to_target(job.bits);
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
            let hits = match backend.scan_range(&header76, start, count, &job.share_target) {
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
                let hash = header.hash();
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
                    hash_display: header.hash_display(),
                    is_block_candidate: is_block,
                });
            }

            shared.stats.hashes.fetch_add(count as u64, Ordering::Relaxed);
            shared.stats.gpu_hashes.fetch_add(count as u64, Ordering::Relaxed);

            // Intensity < 100 % -> idle proportionally (keeps heat down and
            // the machine usable while mining).
            shared.throttle(elapsed);

            // Simple autotune towards ~TARGET_LAUNCH per launch.
            if elapsed < TARGET_LAUNCH / 2 && *batch < MAX_BATCH {
                *batch = (*batch * 2).min(MAX_BATCH);
            } else if elapsed > TARGET_LAUNCH * 5 / 2 && *batch > MIN_BATCH {
                *batch = (*batch / 2).max(MIN_BATCH);
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
