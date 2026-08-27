//! GPU-arbetstråd: samma jobbpipeline som CPU-workern, men nonce-grindningen
//! sker på GPU i batchar. Coinbase→merklerot byggs på CPU per extranonce2
//! (billigt), och varje GPU-träff CPU-verifieras innan submit.

use crate::backend::{open_backend, GpuDevice, MiningBackend};
use crate::consensus::{
    compact_to_target, hash_meets_target, root_from_steps, sha256d, BlockHeader,
};
use crate::shared::{FoundShare, MinerJob, Shared};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Autotune-gränser för batchstorlek (noncer per kernel-launch).
const MIN_BATCH: u32 = 1 << 18;
const MAX_BATCH: u32 = 1 << 24;
const START_BATCH: u32 = 1 << 20;
/// Måltid per launch — lagom för snabb jobbväxling utan launch-overhead.
const TARGET_LAUNCH: Duration = Duration::from_millis(100);

/// `worker_index`/`total_workers` partitionerar extranonce2-rymden disjunkt
/// mellan alla arbetare (CPU-trådar + GPU:er), precis som i worker.rs.
pub fn run_gpu_worker(
    shared: Arc<Shared>,
    device: GpuDevice,
    worker_index: usize,
    total_workers: usize,
) {
    let mut backend = match open_backend(&device) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[gpu] kunde inte öppna {}: {e}", device.describe());
            return;
        }
    };
    crate::human!("[gpu] startad: {}", backend.name());

    let mut batch = START_BATCH;
    loop {
        let (generation, job) = shared.wait_for_job();
        mine_job(&shared, &job, generation, worker_index, total_workers, &mut backend, &mut batch);
    }
}

fn mine_job(
    shared: &Shared,
    job: &MinerJob,
    generation: u64,
    worker_index: usize,
    total_workers: usize,
    backend: &mut Box<dyn MiningBackend>,
    batch: &mut u32,
) {
    let block_target = compact_to_target(job.bits);
    let mut en2_counter = worker_index as u64;

    loop {
        let extranonce2 = crate::worker::encode_extranonce2(en2_counter, job.extranonce2_size);

        // Coinbase → txid → merklerot (en gång per extranonce2).
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

        // Grinda hela nonce-rymden i batchar.
        let mut start: u32 = 0;
        let mut errors = 0u32;
        loop {
            let remaining = (u32::MAX - start).saturating_add(1).max(1);
            let count = (*batch).min(remaining);

            let t0 = Instant::now();
            let hits = match backend.scan_range(&header76, start, count, &job.share_target) {
                Ok(h) => h,
                Err(e) => {
                    errors += 1;
                    eprintln!("[gpu] {}: scan-fel: {e}", backend.name());
                    if errors >= 5 {
                        eprintln!("[gpu] {}: ger upp efter {errors} fel", backend.name());
                        std::thread::sleep(Duration::from_secs(30));
                    }
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
            };
            let elapsed = t0.elapsed();
            errors = 0;

            // CPU-verifiera varje träff innan submit — skydd mot kernelbuggar.
            for nonce in hits {
                header.nonce = nonce;
                let hash = header.hash();
                if !hash_meets_target(&hash, &job.share_target) {
                    eprintln!(
                        "[gpu] VARNING: {} rapporterade falsk träff nonce={nonce:08x} — ignorerad",
                        backend.name()
                    );
                    continue;
                }
                let is_block = block_target
                    .map(|t| hash_meets_target(&hash, &t))
                    .unwrap_or(false);
                let _ = shared.submit_tx.send(FoundShare {
                    job_id: job.job_id.clone(),
                    extranonce2: extranonce2.clone(),
                    ntime: job.ntime,
                    nonce,
                    hash_display: header.hash_display(),
                    is_block_candidate: is_block,
                });
            }

            shared.stats.hashes.fetch_add(count as u64, Ordering::Relaxed);

            // Enkel autotune mot ~TARGET_LAUNCH per launch.
            if elapsed < TARGET_LAUNCH / 2 && *batch < MAX_BATCH {
                *batch = (*batch * 2).min(MAX_BATCH);
            } else if elapsed > TARGET_LAUNCH * 5 / 2 && *batch > MIN_BATCH {
                *batch = (*batch / 2).max(MIN_BATCH);
            }

            if shared.generation.load(Ordering::Acquire) != generation {
                return; // nytt jobb — släpp det gamla direkt
            }
            let (next, wrapped) = start.overflowing_add(count);
            if wrapped {
                break; // nonce-rymden uttömd för denna extranonce2
            }
            start = next;
        }
        en2_counter += total_workers as u64;
    }
}
