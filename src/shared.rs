//! Delat tillstånd mellan stratum-klienten och arbetstrådarna.

use crate::consensus::Target;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

/// Ett aktivt mining-jobb från poolen (allt arbetstrådarna behöver).
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
    /// Share-target enligt svårigheten som gällde vid notify.
    pub share_target: Target,
}

/// En funnen share på väg till poolen.
pub struct FoundShare {
    pub job_id: String,
    pub extranonce2: Vec<u8>,
    pub ntime: u32,
    pub nonce: u32,
    /// Hash i display-format (för loggning) + om den även når block-target.
    pub hash_display: String,
    pub is_block_candidate: bool,
}

#[derive(Default)]
pub struct Stats {
    pub hashes: AtomicU64,
    pub accepted: AtomicU64,
    pub rejected: AtomicU64,
    pub blocks: AtomicU64,
    /// Senaste nätverks-nBits (för ETA-beräkning).
    pub network_bits: AtomicU32,
}

pub struct Shared {
    job: Mutex<Option<MinerJob>>,
    job_cv: Condvar,
    /// Bumpas vid varje nytt jobb — arbetstrådarna pollar den billigt.
    pub generation: AtomicU64,
    pub stats: Stats,
    pub submit_tx: std::sync::mpsc::Sender<FoundShare>,
}

impl Shared {
    pub fn new(submit_tx: std::sync::mpsc::Sender<FoundShare>) -> Self {
        Self {
            job: Mutex::new(None),
            job_cv: Condvar::new(),
            generation: AtomicU64::new(0),
            stats: Stats::default(),
            submit_tx,
        }
    }

    pub fn publish_job(&self, job: MinerJob) {
        self.stats.network_bits.store(job.bits, Ordering::Relaxed);
        *self.job.lock().unwrap() = Some(job);
        self.generation.fetch_add(1, Ordering::Release);
        self.job_cv.notify_all();
    }

    pub fn clear_job(&self) {
        *self.job.lock().unwrap() = None;
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// Blockera tills ett jobb finns; returnera (generation, jobb).
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
