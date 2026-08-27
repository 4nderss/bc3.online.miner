//! Arbetstrådar: bygger coinbase per extranonce2, malar nonce-rymden med
//! SHA3-256t och rapporterar shares.

use crate::consensus::{
    compact_to_target, hash_meets_target, root_from_steps, sha256d, BlockHeader,
};
use crate::shared::{FoundShare, MinerJob, Shared};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Hur ofta trådarna kollar efter nya jobb (i antal hashar).
const CHECK_INTERVAL: u32 = 4096;

pub fn run_worker(shared: Arc<Shared>, thread_id: usize, num_threads: usize) {
    loop {
        let (generation, job) = shared.wait_for_job();
        mine_job(&shared, &job, generation, thread_id, num_threads);
    }
}

fn mine_job(shared: &Shared, job: &MinerJob, generation: u64, thread_id: usize, num_threads: usize) {
    let block_target = compact_to_target(job.bits);
    // Varje tråd tar var N:te extranonce2 — disjunkta sökrymder utan samordning.
    let mut en2_counter = thread_id as u64;

    loop {
        let extranonce2 = encode_extranonce2(en2_counter, job.extranonce2_size);

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

        let mut nonce: u32 = 0;
        loop {
            let batch_end = nonce.saturating_add(CHECK_INTERVAL);
            while nonce < batch_end {
                header.nonce = nonce;
                let hash = header.hash();
                if hash_meets_target(&hash, &job.share_target) {
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
                nonce = nonce.wrapping_add(1);
                if nonce == 0 {
                    break; // nonce-rymden uttömd för denna extranonce2
                }
            }
            shared
                .stats
                .hashes
                .fetch_add(CHECK_INTERVAL as u64, Ordering::Relaxed);
            if shared.generation.load(Ordering::Acquire) != generation {
                return; // nytt jobb — släpp det gamla direkt
            }
            if nonce == 0 {
                break;
            }
        }
        en2_counter += num_threads as u64;
    }
}

/// Delas med GPU-workern (samma partitionering av extranonce2-rymden).
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
    use super::*;

    #[test]
    fn extranonce2_encoding() {
        assert_eq!(encode_extranonce2(1, 4), vec![0, 0, 0, 1]);
        assert_eq!(encode_extranonce2(0x0102_0304, 4), vec![1, 2, 3, 4]);
        assert_eq!(encode_extranonce2(7, 8), vec![0, 0, 0, 0, 0, 0, 0, 7]);
        assert_eq!(encode_extranonce2(0x0aff, 2), vec![0x0a, 0xff]);
    }
}
