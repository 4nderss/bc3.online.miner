//! State shared between the stratum client and the worker threads.

use crate::consensus::Target;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
#[cfg(test)]
use std::sync::Arc;

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
}

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
    /// Highest share difficulty reached (f64 via `to_bits`) - "best share".
    pub best_share_bits: AtomicU64,
    /// Latest network nBits (for the ETA calculation).
    pub network_bits: AtomicU32,
    /// Height of the block we are mining on right now (from the job's
    /// coinbase, BIP34). 0 = unknown (no job yet).
    pub job_height: AtomicU32,
}

pub struct Shared {
    job: Mutex<Option<MinerJob>>,
    job_cv: Condvar,
    /// Intensity 1-100 (%). 100 = full speed; lower values insert idle time
    /// between work passes so the card/CPU does not run hot and the machine
    /// stays usable. Read by the workers on every batch.
    intensity: AtomicU32,
    /// Bumped on every new job - the worker threads poll it cheaply.
    pub generation: AtomicU64,
    pub stats: Stats,
    pub submit_tx: std::sync::mpsc::Sender<FoundShare>,
}

impl Shared {
    pub fn new(submit_tx: std::sync::mpsc::Sender<FoundShare>) -> Self {
        Self {
            job: Mutex::new(None),
            job_cv: Condvar::new(),
            intensity: AtomicU32::new(100),
            generation: AtomicU64::new(0),
            stats: Stats::default(),
            submit_tx,
        }
    }

    /// Record a submitted share: updates the best share (and the block
    /// counter when the share also met the block's target).
    pub fn record_share(&self, difficulty: f64, is_block: bool) {
        if is_block {
            self.stats.blocks.fetch_add(1, Ordering::Relaxed);
        }
        let bits = difficulty.to_bits();
        let mut cur = self.stats.best_share_bits.load(Ordering::Relaxed);
        while difficulty > f64::from_bits(cur) {
            match self.stats.best_share_bits.compare_exchange_weak(
                cur,
                bits,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }

    pub fn best_share(&self) -> f64 {
        f64::from_bits(self.stats.best_share_bits.load(Ordering::Relaxed))
    }

    pub fn set_intensity(&self, percent: u32) {
        self.intensity.store(percent.clamp(1, 100), Ordering::Relaxed);
    }

    pub fn intensity(&self) -> u32 {
        self.intensity.load(Ordering::Relaxed)
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
        if let Some(h) = crate::consensus::bip34_height(&job.coinb1) {
            // Only report on an actual height change - the pool sends new
            // jobs within the same block too (new transactions, new ntime).
            let prev = self.stats.job_height.swap(h, Ordering::Relaxed);
            if prev != h {
                crate::ipc::emit(&crate::ipc::Event::NewBlockHeight { height: h });
                crate::human!("[pool] now working on block #{h}");
            }
        }
        *self.job.lock().unwrap() = Some(job);
        self.generation.fetch_add(1, Ordering::Release);
        self.job_cv.notify_all();
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
        let (tx, _rx) = std::sync::mpsc::channel();
        Shared::new(tx)
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

    #[test]
    fn best_share_keeps_the_maximum_and_counts_blocks() {
        let s = shared();
        assert_eq!(s.best_share(), 0.0);
        s.record_share(12.5, false);
        s.record_share(3.0, false); // lower - must not overwrite
        assert_eq!(s.best_share(), 12.5);
        s.record_share(900.0, true); // a block is a share as well
        assert_eq!(s.best_share(), 900.0);
        assert_eq!(s.stats.blocks.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn throttle_is_free_at_full_intensity() {
        let s = shared();
        let t0 = Instant::now();
        s.throttle(Duration::from_millis(50));
        assert!(t0.elapsed() < Duration::from_millis(20), "100% ska inte vila");
    }

    #[test]
    fn throttle_sleeps_proportionally() {
        let s = shared();
        s.set_intensity(50); // 50% -> idle about as long as the work
        let t0 = Instant::now();
        s.throttle(Duration::from_millis(40));
        let waited = t0.elapsed();
        assert!(waited >= Duration::from_millis(25), "vilade bara {waited:?}");
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
