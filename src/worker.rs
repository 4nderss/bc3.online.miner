//! Worker threads: build the coinbase per extranonce2, grind the nonce space
//! with SHA3-256t and report shares.

use crate::consensus::{
    compact_to_target, hash_meets_target, root_from_steps, sha256d, BlockHeader,
};
use crate::shared::{FoundShare, MinerJob, Shared};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// How often the threads check for new jobs (counted in hashes).
const CHECK_INTERVAL: u32 = 4096;

pub fn run_worker(shared: Arc<Shared>, thread_id: usize, num_threads: usize) {
    // Position in the search space, carried across jobs that cover the same
    // space. See MinerJob::same_search_space.
    let mut prev: Option<MinerJob> = None;
    let mut en2 = thread_id as u64;
    let mut nonce: u32 = 0;
    loop {
        let (generation, job) = shared.wait_for_job();
        if prev.as_ref().is_none_or(|p| !p.same_search_space(&job)) {
            en2 = thread_id as u64;
            nonce = 0;
        }
        mine_job(&shared, &job, generation, num_threads, &mut en2, &mut nonce);
        prev = Some(job);
    }
}

fn mine_job(
    shared: &Shared,
    job: &MinerJob,
    generation: u64,
    num_threads: usize,
    en2_counter: &mut u64,
    nonce_start: &mut u32,
) {
    let block_target = compact_to_target(job.bits);
    // Each thread takes every Nth extranonce2 - disjoint search spaces with
    // no coordination. The starting point comes from the caller so it can
    // survive a job that covers the same space.
    loop {
        let extranonce2 = encode_extranonce2(*en2_counter, job.extranonce2_size);

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

        let mut nonce: u32 = *nonce_start;
        loop {
            let batch_start = std::time::Instant::now();
            // Count iterations rather than compare against an end value.
            //
            // `nonce.saturating_add(CHECK_INTERVAL)` saturates at u32::MAX, so
            // `nonce < batch_end` is false there: the inner loop runs zero
            // times, the nonce never wraps to 0, and the outer loop spins
            // forever - while adding 4096 to stats every turn without hashing
            // anything. A burned core and an invented hashrate. Nonce
            // 0xFFFFFFFF was never hashed either. The GPU path already handled
            // this boundary with overflowing_add; now this one does too.
            let mut done: u32 = 0;
            while done < CHECK_INTERVAL {
                header.nonce = nonce;
                let hash = header.hash();
                if hash_meets_target(&hash, &job.share_target) {
                    let is_block = block_target
                        .map(|t| hash_meets_target(&hash, &t))
                        .unwrap_or(false);
                    shared.record_share(crate::consensus::difficulty_of_hash(&hash), is_block);
                    let _ = shared.submit_tx.send(FoundShare {
                        job_id: job.job_id.clone(),
                        extranonce2: extranonce2.clone(),
                        ntime: job.ntime,
                        nonce,
                        hash_display: header.hash_display(),
                        is_block_candidate: is_block,
                    });
                }
                done += 1;
                let (next, wrapped) = nonce.overflowing_add(1);
                nonce = next;
                if wrapped {
                    break; // nonce space exhausted for this extranonce2
                }
            }
            // `done`, not CHECK_INTERVAL: the last batch before the wrap is
            // shorter, and counting the whole thing inflated the hashrate.
            shared.stats.hashes.fetch_add(done as u64, Ordering::Relaxed);
            shared.stats.cpu_hashes.fetch_add(done as u64, Ordering::Relaxed);
            shared.throttle(batch_start.elapsed());
            if shared.generation.load(Ordering::Acquire) != generation {
                // Remember where we stopped: if the next job covers the same
                // space we continue here instead of re-hashing it.
                *nonce_start = nonce;
                return;
            }
            if nonce == 0 {
                break;
            }
        }
        // Done with this extranonce2 - next one, from the top of its nonces.
        *en2_counter += num_threads as u64;
        *nonce_start = 0;
    }
}

/// Shared with the GPU worker (same partitioning of the extranonce2 space).
pub fn encode_extranonce2(counter: u64, size: usize) -> Vec<u8> {
    let bytes = counter.to_be_bytes();
    if size >= 8 {
        let mut v = vec![0u8; size - 8];
        v.extend_from_slice(&bytes);
        v
    } else {
        bytes[8 - size..].to_vec()
    }
}

#[cfg(test)]
mod tests {
    /// The nonce loop must get past u32::MAX instead of livelocking.
    ///
    /// With `batch_end = nonce.saturating_add(CHECK_INTERVAL)` the condition
    /// `nonce < batch_end` was false at MAX: zero iterations, no wrap, and the
    /// outer loop span forever. This runs the same arithmetic as mine_job,
    /// without the hashing.
    #[test]
    fn the_nonce_loop_terminates_at_the_boundary() {
        let mut nonce: u32 = u32::MAX - 2;
        let mut hashed: Vec<u32> = Vec::new();
        let mut batches = 0;
        loop {
            batches += 1;
            assert!(batches < 10, "loopen avslutas inte - livelock");
            let mut done: u32 = 0;
            let mut wrapped_out = false;
            while done < CHECK_INTERVAL {
                hashed.push(nonce);
                done += 1;
                let (next, wrapped) = nonce.overflowing_add(1);
                nonce = next;
                if wrapped {
                    wrapped_out = true;
                    break;
                }
            }
            // The batch before the wrap is shorter than CHECK_INTERVAL.
            assert!(done <= CHECK_INTERVAL);
            if wrapped_out {
                assert_eq!(nonce, 0);
                break;
            }
        }
        // All three remaining nonces were hashed, MAX included.
        assert_eq!(hashed, vec![u32::MAX - 2, u32::MAX - 1, u32::MAX]);
    }

    use super::*;
    use std::collections::HashSet;

    /// The workers must NEVER grind on the same thing. Every worker starts at
    /// its own index and steps by the number of workers, so the extranonce2
    /// series are disjoint. If they overlapped, a rig with N threads would do
    /// the same work N times and the hashrate would be an illusion.
    #[test]
    fn workers_never_share_an_extranonce2() {
        let total_workers = 8usize;
        let per_worker = 500u64;
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        for worker_index in 0..total_workers {
            let mut counter = worker_index as u64;
            for _ in 0..per_worker {
                let en2 = encode_extranonce2(counter, 4);
                assert!(
                    seen.insert(en2.clone()),
                    "arbetare {worker_index} fick en extranonce2 nagon annan redan hade: {en2:02x?}"
                );
                counter += total_workers as u64;
            }
        }
        assert_eq!(seen.len(), total_workers * per_worker as usize);
    }

    /// The same partitioning must hold for any number of workers - even 1.
    #[test]
    fn partitioning_holds_for_any_worker_count() {
        for total in [1usize, 2, 3, 20] {
            let mut seen: HashSet<Vec<u8>> = HashSet::new();
            for w in 0..total {
                let mut c = w as u64;
                for _ in 0..50 {
                    assert!(seen.insert(encode_extranonce2(c, 4)));
                    c += total as u64;
                }
            }
            assert_eq!(seen.len(), total * 50);
        }
    }

    #[test]
    fn extranonce2_encoding() {
        assert_eq!(encode_extranonce2(1, 4), vec![0, 0, 0, 1]);
        assert_eq!(encode_extranonce2(0x0102_0304, 4), vec![1, 2, 3, 4]);
        assert_eq!(encode_extranonce2(7, 8), vec![0, 0, 0, 0, 0, 0, 0, 7]);
        assert_eq!(encode_extranonce2(0x0aff, 2), vec![0x0a, 0xff]);
    }
}
