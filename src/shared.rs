//! Delat tillstånd mellan stratum-klienten och arbetstrådarna.

use crate::consensus::Target;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
#[cfg(test)]
use std::sync::Arc;

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
    /// Totalt (GPU + CPU) — summan av de två nedan.
    pub hashes: AtomicU64,
    /// Uppdelat per backend så dual-läget kan visa båda var för sig.
    pub gpu_hashes: AtomicU64,
    pub cpu_hashes: AtomicU64,
    pub accepted: AtomicU64,
    pub rejected: AtomicU64,
    pub blocks: AtomicU64,
    /// Högsta uppnådda share-svårighet (f64 via `to_bits`) — "best share".
    pub best_share_bits: AtomicU64,
    /// Senaste nätverks-nBits (för ETA-beräkning).
    pub network_bits: AtomicU32,
}

pub struct Shared {
    job: Mutex<Option<MinerJob>>,
    job_cv: Condvar,
    /// Intensitet 1–100 (%). 100 = full fart; lägre värden lägger in vila
    /// mellan arbetspass så att kort/CPU inte går varma och datorn förblir
    /// användbar. Läses av arbetarna varje batch.
    intensity: AtomicU32,
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
            intensity: AtomicU32::new(100),
            generation: AtomicU64::new(0),
            stats: Stats::default(),
            submit_tx,
        }
    }

    /// Registrera en inskickad share: uppdaterar best share (och blockräknaren
    /// när sharen också uppfyllde blockets target).
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

    /// Vila proportionellt mot senaste arbetspasset så att arbetscykeln
    /// blir ungefär `intensity`%. Vid 100 % blir det ingen vila alls.
    pub fn throttle(&self, worked: std::time::Duration) {
        let pct = self.intensity();
        if pct >= 100 {
            return;
        }
        // arbete/(arbete+vila) = pct/100  ⇒  vila = arbete * (100-pct)/pct
        let idle = worked.mul_f64((100 - pct) as f64 / pct as f64);
        // Ta i småbitar så ett nytt jobb inte behöver vänta ut hela vilan.
        let deadline = std::time::Instant::now() + idle;
        let gen = self.generation.load(Ordering::Acquire);
        while std::time::Instant::now() < deadline {
            if self.generation.load(Ordering::Acquire) != gen {
                return; // nytt jobb — sluta vila direkt
            }
            std::thread::sleep(std::time::Duration::from_millis(5).min(idle));
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

#[cfg(test)]
mod tests {
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
        // Utanför intervallet klampas i stället för att stänga av mining.
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
        s.record_share(3.0, false); // lägre — ska inte skriva över
        assert_eq!(s.best_share(), 12.5);
        s.record_share(900.0, true); // ett block är också en share
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
        s.set_intensity(50); // 50% ⇒ vila ≈ lika länge som arbetet
        let t0 = Instant::now();
        s.throttle(Duration::from_millis(40));
        let waited = t0.elapsed();
        assert!(waited >= Duration::from_millis(25), "vilade bara {waited:?}");
        assert!(waited < Duration::from_millis(120), "vilade för länge: {waited:?}");
    }

    #[test]
    fn throttle_aborts_on_new_job() {
        let s = Arc::new(shared());
        s.set_intensity(10); // ⇒ lång vila (9× arbetet)
        let s2 = s.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            s2.generation.fetch_add(1, Ordering::Release);
        });
        let t0 = Instant::now();
        s.throttle(Duration::from_millis(100)); // skulle annars vila ~900 ms
        assert!(t0.elapsed() < Duration::from_millis(300), "vilan avbröts inte");
    }
}
