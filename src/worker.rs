//! Worker threads: build the coinbase per extranonce2, grind the nonce space
//! with the job's PoW and report shares.

use crate::consensus::{
    compact_to_target, hash_meets_target, hash_to_display, root_from_steps, sha256d,
    BlockHeader, PowAlgo, PowFn,
};
use crate::shared::{AlgoGate, FoundShare, MinerJob, Shared};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// How often the threads check for new jobs, counted in SHA3-256t hashes.
const CHECK_INTERVAL: u32 = 4096;

/// Hashes per batch: the job check, the stats and the throttle run once per
/// batch, so a batch should take a few milliseconds whatever the PoW - long
/// enough that the check is free, short enough that a new block is picked up
/// at once. 4096 SHA3-256t hashes are ~5 ms on one thread; 4096 BC3Hash
/// hashes would leave a thread ~4 s on a stale job after every block, so it
/// checks every 4 (~4 ms at ~1100 H/s per thread).
fn batch_size(algo: PowAlgo) -> u32 {
    match algo {
        PowAlgo::Sha256d | PowAlgo::Sha3t => CHECK_INTERVAL,
        PowAlgo::Bc3Hash => 4,
    }
}

/// Can this process hash `algo` on the CPU? BC3Hash needs the node's C++
/// linked in, its tables built, and the build's self-test passed - the first
/// call waits for that (the table thread in main has normally done it).
pub fn cpu_supports(algo: PowAlgo) -> bool {
    algo.hash_fn().is_some() && cpu_unavailable(algo, true).is_none()
}

/// `cpu_supports` without waiting: true until BC3Hash is known to be broken.
/// For the stratum thread, which must never stall on the table build.
pub fn cpu_may_support(algo: PowAlgo) -> bool {
    algo.hash_fn().is_some() && cpu_unavailable(algo, false).is_none()
}

/// Why the CPU cannot hash `algo`, once that is known; None while it can, or
/// (without `wait`) while BC3Hash is still being set up.
pub fn cpu_unavailable(algo: PowAlgo, wait: bool) -> Option<String> {
    match algo {
        PowAlgo::Sha256d | PowAlgo::Sha3t => None,
        PowAlgo::Bc3Hash => bc3hash_unavailable(wait),
    }
}

#[cfg(feature = "bc3hash")]
fn bc3hash_unavailable(wait: bool) -> Option<String> {
    let status = if wait { Some(::bc3hash::initialize()) } else { ::bc3hash::status() };
    match status {
        Some(Err(e)) => Some(e.to_string()),
        Some(Ok(())) | None => None,
    }
}

#[cfg(not(feature = "bc3hash"))]
fn bc3hash_unavailable(_wait: bool) -> Option<String> {
    Some("this build of the miner has no BC3Hash - update the miner".into())
}

/// What a CPU thread mines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuRole {
    /// Every job the CPU can hash: CPU mining, or `--threads` next to a GPU.
    All,
    /// A stand-in for the GPUs: only jobs whose PoW no GPU kernel implements,
    /// which since the fork is every job - BC3Hash has no GPU kernel. While
    /// the GPUs have work, these threads sleep on the job condvar and cost
    /// nothing.
    GpuFallback,
}

impl CpuRole {
    /// Does a thread in this role take a job, given whether the GPUs take its
    /// PoW right now (`Shared::gpus_mine`)?
    pub fn takes(&self, gpus_mine: bool) -> bool {
        match self {
            CpuRole::All => true,
            CpuRole::GpuFallback => !gpus_mine,
        }
    }
}

pub fn run_worker(shared: Arc<Shared>, thread_id: usize, num_threads: usize, role: CpuRole) {
    // Position in the search space, carried across jobs that cover the same
    // space. See MinerJob::same_search_space.
    let mut prev: Option<MinerJob> = None;
    let mut en2 = thread_id as u64;
    let mut nonce: u32 = 0;
    let mut gate = AlgoGate::default();
    // Only the first thread says why it sits a job out; the rest would repeat it.
    let label = format!("CPU thread {thread_id}");
    let label = (thread_id == 0).then_some(label.as_str());
    // Generation of a job this thread sat out; wait for the next one instead
    // of spinning on it.
    let mut sat_out: Option<u64> = None;
    loop {
        let (generation, job) = match sat_out.take() {
            Some(seen) => shared.wait_for_job_after(seen),
            None => shared.wait_for_job(),
        };
        // A stand-in leaves the GPUs' work to the GPUs, silently: that is
        // the normal case, not something to warn about.
        if !role.takes(shared.gpus_mine(job.algo)) {
            sat_out = Some(generation);
            continue;
        }
        let Some(pow) = gate.check(job.algo, label, cpu_supports).and_then(|a| a.hash_fn())
        else {
            sat_out = Some(generation);
            continue;
        };
        if prev.as_ref().is_none_or(|p| !p.same_search_space(&job)) {
            en2 = thread_id as u64;
            nonce = 0;
        }
        let batch = batch_size(job.algo);
        mine_job(&shared, &job, pow, batch, generation, num_threads, &mut en2, &mut nonce);
        prev = Some(job);
    }
}

#[allow(clippy::too_many_arguments)]
fn mine_job(
    shared: &Shared,
    job: &MinerJob,
    // Resolved once per job, so the loop below calls a plain function with no
    // match per nonce. Next to a triple keccak, let alone a BC3Hash, the
    // indirect call is noise.
    pow: PowFn,
    batch: u32,
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

        // Serialized once per extranonce2; only the nonce bytes change below.
        let mut header = BlockHeader {
            version: job.version,
            prev_hash: job.prev_hash,
            merkle_root,
            time: job.ntime,
            bits: job.bits,
            nonce: 0,
        }
        .serialize();

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
            while done < batch {
                header[76..80].copy_from_slice(&nonce.to_le_bytes());
                let hash = pow(&header);
                if hash_meets_target(&hash, &job.share_target) {
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
                done += 1;
                let (next, wrapped) = nonce.overflowing_add(1);
                nonce = next;
                if wrapped {
                    break; // nonce space exhausted for this extranonce2
                }
            }
            // `done`, not the batch size: the last batch before the wrap is
            // shorter, and counting the whole thing inflated the hashrate.
            shared.stats.hashes.fetch_add(done as u64, Ordering::Relaxed);
            shared.stats.cpu_hashes.fetch_add(done as u64, Ordering::Relaxed);
            shared.throttle(batch_start.elapsed());
            if shared.generation.load(Ordering::Acquire) != generation {
                // Remember where we stopped: if the next job covers the same
                // space we continue here instead of re-hashing it. `nonce == 0`
                // here means the space wrapped in this very batch.
                let (e, n) = resume_point(*en2_counter, nonce, nonce == 0, num_threads as u64);
                *en2_counter = e;
                *nonce_start = n;
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

/// Where to pick up again when a job change interrupts a worker mid-sweep:
/// the point just past the work that was actually finished.
///
/// `wrapped` means the nonce space of the current extranonce2 ran out in that
/// same step. Then the resume point is not nonce 0 of the SAME extranonce2 —
/// that whole space is already hashed — but nonce 0 of the next one this
/// worker owns. Getting that wrong re-hashed a full 2^32 nonces on the next
/// job, and every share found in them came back from the pool as a duplicate.
pub fn resume_point(en2: u64, next_nonce: u32, wrapped: bool, stride: u64) -> (u64, u32) {
    if wrapped {
        (en2.saturating_add(stride), 0)
    } else {
        (en2, next_nonce)
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
    use super::resume_point;

    /// The point saved on a job change must be past the work already done.
    ///
    /// Saving the START of the finished batch re-hashed up to 2^29 nonces on
    /// every job change, and the pool answered the resulting shares with
    /// "duplicate share".
    #[test]
    fn a_job_change_resumes_after_the_finished_batch() {
        // Worker 2 of 8, halfway through extranonce2 number 2.
        assert_eq!(resume_point(2, 1_000_000, false, 8), (2, 1_000_000));
    }

    /// And when the nonce space ran out in the very same step, nonce 0 of the
    /// SAME extranonce2 is the one place we must not resume: it is fully
    /// hashed. Resuming there re-mined all 2^32 of them.
    #[test]
    fn a_wrap_moves_on_to_the_next_extranonce2() {
        assert_eq!(resume_point(2, 0, true, 8), (10, 0));
        // The stride is the worker count, so workers never collide.
        assert_eq!(resume_point(7, 0, true, 8), (15, 0));
        assert_eq!(resume_point(0, 0, true, 1), (1, 0));
    }

    /// A counter at the top must not wrap around onto another worker's ground.
    #[test]
    fn the_extranonce2_counter_saturates() {
        assert_eq!(resume_point(u64::MAX, 0, true, 8), (u64::MAX, 0));
    }

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
    #[cfg(feature = "bc3hash")]
    use std::time::Duration;

    #[cfg(feature = "bc3hash")]
    use crate::shared::MiningPlan;
    #[cfg(feature = "bc3hash")]
    use crate::shared::SUBMIT_QUEUE;

    /// A job that yields a share about every 16 hashes, so a worker that
    /// hashes it at all shows up on the submit channel within a batch or two -
    /// even at BC3Hash speed.
    #[cfg(feature = "bc3hash")]
    fn easy_job(job_id: &str, algo: PowAlgo) -> MinerJob {
        let mut share_target = [0xffu8; 32];
        share_target[0] = 0x0f;
        let version = match algo {
            PowAlgo::Sha256d => 0x2000_0000,
            PowAlgo::Sha3t => 0x2000_1000,
            PowAlgo::Bc3Hash => 0x2000_1800,
        };
        MinerJob {
            job_id: job_id.into(),
            extranonce1: vec![1, 2, 3, 4],
            extranonce2_size: 4,
            coinb1: vec![9; 50],
            coinb2: vec![8; 20],
            merkle_steps: vec![[5u8; 32]],
            version,
            prev_hash: [6u8; 32],
            bits: 0x1d00_ffff,
            ntime: 1_700_000_000,
            share_target,
            height: Some(64_360),
            algo,
        }
    }

    /// The header a share stands for, rebuilt from the job and the share
    /// alone - the way the pool rebuilds it.
    #[cfg(feature = "bc3hash")]
    fn header_of(job: &MinerJob, extranonce2: &[u8], nonce: u32) -> [u8; 80] {
        let mut coinbase = job.coinb1.clone();
        coinbase.extend_from_slice(&job.extranonce1);
        coinbase.extend_from_slice(extranonce2);
        coinbase.extend_from_slice(&job.coinb2);
        BlockHeader {
            version: job.version,
            prev_hash: job.prev_hash,
            merkle_root: root_from_steps(&sha256d(&coinbase), &job.merkle_steps),
            time: job.ntime,
            bits: job.bits,
            nonce,
        }
        .serialize()
    }

    /// Wait until a worker has stopped hashing - it leaves a job at the end of
    /// its batch, which in a debug build can take a good part of a second -
    /// and drain what it submitted on the way out.
    #[cfg(feature = "bc3hash")]
    fn wait_until_idle(shared: &Shared, rx: &std::sync::mpsc::Receiver<FoundShare>) {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        loop {
            let before = shared.stats.hashes.load(Ordering::Relaxed);
            std::thread::sleep(Duration::from_secs(1));
            while rx.try_recv().is_ok() {}
            if shared.stats.hashes.load(Ordering::Relaxed) == before {
                return;
            }
            assert!(std::time::Instant::now() < deadline, "the worker never went idle");
        }
    }

    /// The fork as a GPU rig in `--backend auto` sees it. The stand-in CPU
    /// thread leaves to the GPU whatever the GPU claims - not one hash - and
    /// takes over the moment it does not: a PoW without a GPU kernel, or a
    /// GPU kernel that drops out (failed self-test) in the middle of a job,
    /// without waiting for the next one.
    #[cfg(feature = "bc3hash")]
    #[test]
    fn a_stand_in_thread_mines_only_what_the_gpus_cannot() {
        let (tx, rx) = std::sync::mpsc::sync_channel(SUBMIT_QUEUE);
        let plan = MiningPlan {
            gpus: 1,
            cpu_threads: 0,
            cpu_fallback_threads: 1,
            gpu_kernel: [false, true, true],
        };
        let shared = Arc::new(Shared::new(tx, plan));
        // The GPU worker's claim: SHA3-256t, for now.
        let mut gpu = crate::shared::GpuCapability::new(&shared);
        gpu.set(PowAlgo::Sha3t, true);
        shared.publish_job(easy_job("sha3t", PowAlgo::Sha3t));
        let s = shared.clone();
        std::thread::spawn(move || run_worker(s, 1, 2, CpuRole::GpuFallback));

        std::thread::sleep(Duration::from_millis(300));
        assert!(rx.try_recv().is_err(), "the stand-in submitted a share for the GPU's job");
        assert_eq!(shared.stats.hashes.load(Ordering::Relaxed), 0, "the stand-in hashed the GPU's job");

        // No GPU kernel for BC3Hash (an OpenCL card): the CPU takes it.
        shared.publish_job(easy_job("fork", PowAlgo::Bc3Hash));
        let share = rx
            .recv_timeout(Duration::from_secs(30))
            .expect("the stand-in did not take over the BC3Hash job");
        assert_eq!(share.job_id, "fork");

        shared.publish_job(easy_job("sha3t-again", PowAlgo::Sha3t));
        wait_until_idle(&shared, &rx);
        std::thread::sleep(Duration::from_millis(300));
        assert!(rx.try_recv().is_err(), "a share was submitted after the GPU took over again");

        // A CUDA card claims BC3Hash: the stand-in leaves it alone...
        gpu.set(PowAlgo::Bc3Hash, true);
        shared.publish_job(easy_job("fork-gpu", PowAlgo::Bc3Hash));
        std::thread::sleep(Duration::from_millis(300));
        let hashed = shared.stats.hashes.load(Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(shared.stats.hashes.load(Ordering::Relaxed), hashed, "the stand-in hashed the GPU's BC3Hash");
        assert!(rx.try_recv().is_err());
        // ...until its kernel fails the self-test: no new job, and the CPU
        // still takes over at once.
        gpu.set(PowAlgo::Bc3Hash, false);
        let share = rx
            .recv_timeout(Duration::from_secs(30))
            .expect("the stand-in did not take over when the GPU dropped BC3Hash");
        assert_eq!(share.job_id, "fork-gpu");
        shared.clear_job();
    }

    /// A BC3Hash job produces shares the pool will accept: the header rebuilt
    /// from the share alone hashes - with the node's code - to a value under
    /// the share target, and the hash in the log is that hash.
    #[cfg(feature = "bc3hash")]
    #[test]
    fn a_bc3hash_job_yields_valid_shares() {
        let (tx, rx) = std::sync::mpsc::sync_channel(SUBMIT_QUEUE);
        let plan = MiningPlan { cpu_threads: 1, ..Default::default() };
        let shared = Arc::new(Shared::new(tx, plan));
        let job = easy_job("fork", PowAlgo::Bc3Hash);
        shared.publish_job(job.clone());
        let s = shared.clone();
        std::thread::spawn(move || run_worker(s, 0, 1, CpuRole::All));
        for _ in 0..3 {
            let share = rx
                .recv_timeout(Duration::from_secs(30))
                .expect("no share from an easy BC3Hash job");
            assert_eq!(share.job_id, "fork");
            let hash = ::bc3hash::bc3hash(&header_of(&job, &share.extranonce2, share.nonce));
            assert!(hash_meets_target(&hash, &job.share_target), "the share does not meet its target");
            assert_ne!(hash, sha3t_of(&job, &share), "hashed with SHA3-256t, not BC3Hash");
            assert_eq!(share.hash_display, hash_to_display(&hash));
        }
        shared.clear_job();
    }

    #[cfg(feature = "bc3hash")]
    fn sha3t_of(job: &MinerJob, share: &FoundShare) -> [u8; 32] {
        crate::consensus::sha3t(&header_of(job, &share.extranonce2, share.nonce))
    }

    /// BC3Hash rate through the real worker loop, to hold against
    /// `bc3hash/examples/bench.rs`: the gap is the worker's own overhead.
    ///
    ///     cargo test --release -- --ignored bc3hash_worker_rate --nocapture
    #[cfg(feature = "bc3hash")]
    #[test]
    #[ignore]
    fn bc3hash_worker_rate() {
        ::bc3hash::configure(::bc3hash::Mode::Fast);
        ::bc3hash::initialize().unwrap();
        let (tx, _rx) = std::sync::mpsc::sync_channel(SUBMIT_QUEUE);
        let plan = MiningPlan { cpu_threads: 1, ..Default::default() };
        let shared = Arc::new(Shared::new(tx, plan));
        let mut job = easy_job("rate", PowAlgo::Bc3Hash);
        job.share_target = [0u8; 32]; // never a share: measure hashing only
        shared.publish_job(job);
        let s = shared.clone();
        std::thread::spawn(move || run_worker(s, 0, 1, CpuRole::All));
        std::thread::sleep(Duration::from_secs(2));
        let (h0, t0) = (shared.stats.cpu_hashes.load(Ordering::Relaxed), std::time::Instant::now());
        std::thread::sleep(Duration::from_secs(10));
        let hashes = shared.stats.cpu_hashes.load(Ordering::Relaxed) - h0;
        println!("worker loop: {:.1} H/s on one thread", hashes as f64 / t0.elapsed().as_secs_f64());
        shared.clear_job();
    }

    /// A stand-in takes exactly what no GPU takes; an `All` thread takes
    /// everything.
    #[test]
    fn roles_take_the_right_jobs() {
        assert!(!CpuRole::GpuFallback.takes(true));
        assert!(CpuRole::GpuFallback.takes(false));
        assert!(CpuRole::All.takes(true));
        assert!(CpuRole::All.takes(false));
    }

    /// A batch stays a few milliseconds long whatever the PoW, so a new block
    /// is picked up at once - BC3Hash is about 1000 times slower per hash.
    #[test]
    fn batches_are_sized_per_pow() {
        assert_eq!(batch_size(PowAlgo::Sha3t), CHECK_INTERVAL);
        assert!(batch_size(PowAlgo::Bc3Hash) * 1000 <= CHECK_INTERVAL * 2);
        assert!(batch_size(PowAlgo::Bc3Hash) >= 1);
    }

    /// The CPU path hashes exactly what the GPU re-check and consensus.rs
    /// would: the serialized-once header with the nonce patched in.
    #[test]
    fn cpu_hash_matches_the_header_reference() {
        let mut h = BlockHeader {
            version: 0x2000_1000,
            prev_hash: [1u8; 32],
            merkle_root: [2u8; 32],
            time: 1_700_000_000,
            bits: 0x1d00_ffff,
            nonce: 0,
        };
        let mut ser = h.serialize();
        for nonce in [0u32, 1, 0xdead_beef, u32::MAX] {
            ser[76..80].copy_from_slice(&nonce.to_le_bytes());
            h.nonce = nonce;
            assert_eq!((PowAlgo::Sha3t.hash_fn().unwrap())(&ser), h.hash());
        }
        assert!(cpu_supports(PowAlgo::Sha3t) && cpu_supports(PowAlgo::Sha256d));
        assert_eq!(cpu_supports(PowAlgo::Bc3Hash), cfg!(feature = "bc3hash"));
    }

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
