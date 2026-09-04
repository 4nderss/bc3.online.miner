//! Mining backends: an abstraction over CUDA/OpenCL GPUs (and the CPU
//! fallback).
//!
//! A backend grinds a nonce range for a fixed 76-byte header prefix and
//! returns the nonces whose SHA3-256t hash is <= the share target. Every hit
//! is re-checked on the CPU in gpu_worker before submit - a kernel bug can
//! never produce a bad share, only a missed one.

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "opencl")]
pub mod cl_sys;
#[cfg(feature = "opencl")]
pub mod opencl;

use crate::consensus::Target;

/// Shared kernel source (CUDA via NVRTC + OpenCL). See that file for the
/// lane layout.
#[allow(dead_code)]
pub const KERNEL_SOURCE: &str = include_str!("../kernels/sha3t.cl");

/// Max number of hits per launch that the kernel reports (more than this in
/// a single batch never happens in practice at sane share difficulties).
#[allow(dead_code)]
pub const MAX_HITS: usize = 64;

/// Keeps a backend's launches inside what the kernel's hit buffer can hold.
///
/// The kernel writes at most `MAX_HITS` nonces plus a count. Anything past
/// that is already gone by the time the buffer is read, and it used to be
/// dropped with no log and no change of behaviour - so a pool that set the
/// share difficulty low enough made the miner lose shares on every launch,
/// with nothing anywhere to say so.
///
/// "Never happens in practice" is not a bound: the pool picks the difficulty,
/// and the batch size grows with the card.
#[derive(Default)]
#[allow(dead_code)]
pub struct HitBudget {
    /// `None` until an overflow has been seen. Until then a scan runs exactly
    /// the range the worker asked for, in one launch - the fast path is not
    /// paying for a case that normally never occurs.
    limit: Option<u32>,
    logged: bool,
    /// Launches since the last overflow. The limit only ever went DOWN, so a
    /// single burst at a low starting difficulty pinned the backend to small
    /// launches for the rest of the process - paying per-launch overhead long
    /// after vardiff had raised the difficulty out of the problem.
    clean_runs: u32,
}

/// How many launches WITH HEADROOM before the budget doubles again.
///
/// Low enough that a difficulty raise is followed within seconds rather than
/// minutes. What keeps it from oscillating is the headroom rule in
/// `overflowed`, not this number.
const RECOVER_AFTER: u32 = 64;

#[cfg(all(test, feature = "cuda", feature = "opencl"))]
mod fallback_tests {
    use super::*;

    fn cu(index: usize, name: &str) -> GpuDevice {
        GpuDevice::Cuda { index, name: name.into() }
    }
    fn cl(device: usize, name: &str, vendor: &str) -> GpuDevice {
        GpuDevice::Opencl { platform: 0, device, name: name.into(), vendor: vendor.into() }
    }

    /// The case that shipped broken: a rig of identical cards. Every worker
    /// must land on its OWN device, or one card is oversubscribed N times
    /// while the rest idle - and every worker still reports itself alive.
    #[test]
    fn identical_cards_each_get_their_own_device() {
        let cuda = vec![cu(0, "NVIDIA RTX 3060"), cu(1, "NVIDIA RTX 3060"), cu(2, "NVIDIA RTX 3060")];
        let ocl = vec![
            cl(0, "NVIDIA RTX 3060", "NVIDIA"),
            cl(1, "NVIDIA RTX 3060", "NVIDIA"),
            cl(2, "NVIDIA RTX 3060", "NVIDIA"),
        ];
        let picks: Vec<Option<usize>> = (0..3)
            .map(|i| pick_opencl_for_cuda(&cuda, &ocl, i, "NVIDIA RTX 3060"))
            .collect();
        assert_eq!(picks, vec![Some(0), Some(1), Some(2)]);
    }

    /// A MIXED rig. The CUDA index is a global ordinal while the name-matched
    /// list is per model, so indexing one by the other dropped a card - and
    /// which card depended on how CUDA happened to order them.
    #[test]
    fn a_mixed_rig_does_not_lose_a_card() {
        // CUDA orders by performance: the 4090 first, then the two 3060s.
        let cuda = vec![
            cu(0, "NVIDIA RTX 4090"),
            cu(1, "NVIDIA RTX 3060"),
            cu(2, "NVIDIA RTX 3060"),
        ];
        let ocl = vec![
            cl(0, "NVIDIA RTX 4090", "NVIDIA"),
            cl(1, "NVIDIA RTX 3060", "NVIDIA"),
            cl(2, "NVIDIA RTX 3060", "NVIDIA"),
        ];
        assert_eq!(pick_opencl_for_cuda(&cuda, &ocl, 0, "NVIDIA RTX 4090"), Some(0));
        assert_eq!(pick_opencl_for_cuda(&cuda, &ocl, 1, "NVIDIA RTX 3060"), Some(1));
        // This one returned None before: rank 2 in a two-element list.
        assert_eq!(pick_opencl_for_cuda(&cuda, &ocl, 2, "NVIDIA RTX 3060"), Some(2));

        // And no two workers may land on the same device.
        let picks: Vec<usize> = [(0, "NVIDIA RTX 4090"), (1, "NVIDIA RTX 3060"), (2, "NVIDIA RTX 3060")]
            .iter()
            .filter_map(|(i, n)| pick_opencl_for_cuda(&cuda, &ocl, *i, n))
            .collect();
        let unique: std::collections::HashSet<usize> = picks.iter().copied().collect();
        assert_eq!(unique.len(), picks.len(), "two workers landed on one device");
    }

    /// The population this fallback actually serves: an old driver, whose
    /// OpenCL reports the card WITHOUT the "NVIDIA " prefix that CUDA uses,
    /// next to an Intel iGPU. Refusing here would decline to help the exact
    /// machines the fallback exists for.
    #[test]
    fn an_old_driver_with_an_igpu_beside_it_still_matches() {
        let cuda = vec![cu(0, "NVIDIA GeForce GTX 1080")];
        let ocl = vec![
            cl(0, "Intel(R) UHD Graphics 630", "Intel(R) Corporation"),
            cl(1, "GeForce GTX 1080", "NVIDIA Corporation"),
        ];
        assert_eq!(
            pick_opencl_for_cuda(&cuda, &ocl, 0, "NVIDIA GeForce GTX 1080"),
            Some(1),
            "the vendor identifies the card when the names have drifted"
        );
    }

    /// But two cards of the same vendor and no name match is a GUESS, and
    /// guessing is what put every worker on one device to begin with.
    #[test]
    fn it_refuses_rather_than_guess_between_two_cards() {
        let cuda = vec![cu(0, "NVIDIA A"), cu(1, "NVIDIA B")];
        let ocl = vec![cl(0, "Something Else", "NVIDIA"), cl(1, "Another", "NVIDIA")];
        assert_eq!(pick_opencl_for_cuda(&cuda, &ocl, 0, "NVIDIA A"), None);
        assert_eq!(pick_opencl_for_cuda(&cuda, &ocl, 1, "NVIDIA B"), None);
    }

    /// A card that has VANISHED from CUDA enumeration must not take a healthy
    /// sibling's device.
    ///
    /// `cuda::list_devices` drops any device whose context fails to open, so
    /// the list goes sparse exactly when a card is failing - which is when
    /// this fallback runs. Guessing the rank from the global ordinal there
    /// put two workers on one device: the vanished card's worker (no
    /// position, ordinal 0) and the first survivor (position 0).
    #[test]
    fn a_vanished_cuda_device_does_not_collide_with_a_healthy_one() {
        // Three identical cards at startup; #0 has since fallen off the bus,
        // so it is missing from the list this call re-enumerates.
        let cuda = vec![cu(1, "NVIDIA RTX 3060"), cu(2, "NVIDIA RTX 3060")];
        let ocl = vec![
            cl(0, "NVIDIA RTX 3060", "NVIDIA"),
            cl(1, "NVIDIA RTX 3060", "NVIDIA"),
            cl(2, "NVIDIA RTX 3060", "NVIDIA"),
        ];
        let gone = pick_opencl_for_cuda(&cuda, &ocl, 0, "NVIDIA RTX 3060");
        let healthy = pick_opencl_for_cuda(&cuda, &ocl, 1, "NVIDIA RTX 3060");
        assert_eq!(gone, None, "a vanished device must not get any device at all");
        assert_eq!(healthy, Some(0));
        assert_ne!(gone, healthy, "two workers landed on the same device");
    }

    /// No OpenCL at all: nothing to fall back to.
    #[test]
    fn no_opencl_devices_means_no_pick() {
        let cuda = vec![cu(0, "NVIDIA RTX 3060")];
        assert_eq!(pick_opencl_for_cuda(&cuda, &[], 0, "NVIDIA RTX 3060"), None);
    }
}

#[cfg(test)]
mod hit_budget_tests {
    use super::*;

    /// The budget has to come back up. A pool starts a new connection at a low
    /// difficulty and raises it within a minute; without recovery, one burst of
    /// overflows during that first minute left the backend running tiny
    /// launches - and paying their overhead - for the rest of the process.
    #[test]
    fn the_budget_recovers_after_the_difficulty_rises() {
        let mut b = HitBudget::default();
        // No overflow seen yet: the whole range in one launch.
        assert_eq!(b.chunk(1_000_000), 1_000_000);

        assert!(b.overflowed(1_000_000, MAX_HITS + 1).is_some());
        let shrunk = b.chunk(1_000_000);
        assert!(shrunk < 1_000_000, "an overflow must shrink the launch");

        // Clean launches below the threshold change nothing.
        for _ in 0..RECOVER_AFTER - 1 {
            assert!(b.overflowed(shrunk, 0).is_none());
        }
        assert_eq!(b.chunk(1_000_000), shrunk);

        // The one that crosses it doubles the budget.
        assert!(b.overflowed(shrunk, 0).is_none());
        assert_eq!(b.chunk(1_000_000), shrunk * 2);
    }

    /// A launch that lands JUST UNDER the buffer must not count as clean.
    ///
    /// Doubling from there is near-certain to overflow, and the overflowing
    /// launch drops its excess hits before halving back - a permanent cycle of
    /// 64 good launches and one lossy one. The recovery has to probe upward
    /// only when there is room for the probe to succeed.
    #[test]
    fn a_launch_without_headroom_does_not_count_as_recovery() {
        let mut b = HitBudget::default();
        b.overflowed(1_000_000, MAX_HITS + 1);
        let shrunk = b.chunk(1_000_000);
        // Just under the cap: no overflow, but no room to double into either.
        for _ in 0..RECOVER_AFTER * 2 {
            assert!(b.overflowed(shrunk, MAX_HITS).is_none());
        }
        assert_eq!(b.chunk(1_000_000), shrunk, "must not have doubled");

        // With real headroom it recovers as before.
        for _ in 0..RECOVER_AFTER {
            b.overflowed(shrunk, MAX_HITS / 4);
        }
        assert_eq!(b.chunk(1_000_000), shrunk * 2);
    }

    /// An overflow resets the run of clean launches - otherwise a budget that
    /// overflows every other launch would still creep upwards.
    #[test]
    fn an_overflow_restarts_the_count() {
        let mut b = HitBudget::default();
        b.overflowed(1_000_000, MAX_HITS + 1);
        let shrunk = b.chunk(1_000_000);
        for _ in 0..RECOVER_AFTER - 1 {
            b.overflowed(shrunk, 0);
        }
        // One more overflow, and the near-complete run is discarded.
        b.overflowed(shrunk, MAX_HITS + 1);
        let smaller = b.chunk(1_000_000);
        assert!(smaller < shrunk);
        b.overflowed(smaller, 0);
        assert_eq!(b.chunk(1_000_000), smaller, "the run must have restarted");
    }

    /// Only the first overflow is logged; the condition repeats until the
    /// halving catches up, and a line per launch would bury the log.
    #[test]
    fn only_the_first_overflow_is_logged() {
        let mut b = HitBudget::default();
        assert!(b.overflowed(1_000_000, MAX_HITS + 1).is_some());
        assert!(b.overflowed(1_000, MAX_HITS + 1).is_none());
    }
}

#[allow(dead_code)]
impl HitBudget {
    /// How many nonces the next launch may cover, of `remaining` left to do.
    pub fn chunk(&self, remaining: u32) -> u32 {
        match self.limit {
            Some(limit) => limit.min(remaining),
            None => remaining,
        }
    }

    /// Record a launch over `chunk` nonces that reported `reported` hits.
    ///
    /// Returns a message to log the FIRST time it overflows - once, not per
    /// launch, because the condition repeats until the halving has caught up
    /// and a line per launch would bury everything else in the log.
    pub fn overflowed(&mut self, chunk: u32, reported: usize) -> Option<String> {
        if reported <= MAX_HITS {
            // A launch only counts as clean when it had real HEADROOM.
            //
            // Counting every `reported <= MAX_HITS` made the probe blind: once
            // the budget settles at a size whose hit count sits just under the
            // boundary, doubling is near-certain to overflow, and the
            // overflowing launch drops its excess hits before halving back.
            // Steady state was 64 good launches, one lossy one, for ever.
            // RECOVER_AFTER set the period of that oscillation, not whether it
            // happened. A quarter of the buffer means doubling lands at half,
            // so the probe is near-free.
            //
            // It also stops a scan's trailing partial chunk - often a handful
            // of nonces - from earning a clean run on equal terms with a
            // full-size launch it was never at risk of matching.
            if let Some(limit) = self.limit {
                if reported * 4 <= MAX_HITS {
                    self.clean_runs += 1;
                    if self.clean_runs >= RECOVER_AFTER {
                        self.clean_runs = 0;
                        self.limit = limit.checked_mul(2);
                    }
                } else {
                    self.clean_runs = 0;
                }
            }
            return None;
        }
        self.clean_runs = 0;
        // Halve, so the next launch expects half as many hits. Floor at 1:
        // a batch of 0 would make no progress at all.
        self.limit = Some(self.chunk(chunk).div_ceil(2).max(1));
        if self.logged {
            return None;
        }
        self.logged = true;
        Some(format!(
            "[gpu] kernel reported {reported} hits but only {MAX_HITS} fit in the buffer - \
             {} nonce(s) dropped. Halving the batch to {} nonces per launch. \
             This means the share difficulty is far below what this device needs.",
            reported - MAX_HITS,
            self.limit.unwrap_or(1),
        ))
    }
}

pub trait MiningBackend {
    /// Human-readable name ("CUDA: NVIDIA GeForce RTX 3050 Ti ...").
    fn name(&self) -> String;

    /// Grind [start_nonce, start_nonce+count) for the header (nonce field
    /// excluded) and return the nonces whose hash is <= target.
    fn scan_range(
        &mut self,
        header76: &[u8; 76],
        start_nonce: u32,
        count: u32,
        target: &Target,
    ) -> Result<Vec<u32>, String>;
}

/// A discovered GPU - data only (Send); the backend itself is opened in the
/// worker thread.
#[derive(Clone, Debug)]
pub enum GpuDevice {
    #[cfg(feature = "cuda")]
    Cuda { index: usize, name: String },
    #[cfg(feature = "opencl")]
    Opencl {
        platform: usize,
        device: usize,
        name: String,
        /// CL_DEVICE_VENDOR. Not shown to the user - it is half the key that
        /// recognises one physical card exposed by two platforms. See
        /// `dedup_opencl`.
        vendor: String,
    },
}

impl GpuDevice {
    pub fn describe(&self) -> String {
        match self {
            #[cfg(feature = "cuda")]
            GpuDevice::Cuda { index, name } => format!("CUDA #{index}: {name}"),
            #[cfg(feature = "opencl")]
            GpuDevice::Opencl { platform, device, name, .. } => {
                format!("OpenCL {platform}.{device}: {name}")
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

/// Which backends the user asked for.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum BackendKind {
    /// CUDA if possible, otherwise OpenCL, otherwise CPU.
    Auto,
    Cuda,
    Opencl,
    Cpu,
}

/// List GPUs for the requested backend. `gpu_id` narrows it down to a single
/// device (the CUDA index, or the position in the OpenCL list).
pub fn detect_gpus(kind: BackendKind, gpu_id: Option<usize>) -> Vec<GpuDevice> {
    let mut found: Vec<GpuDevice> = Vec::new();

    #[cfg(feature = "cuda")]
    if matches!(kind, BackendKind::Auto | BackendKind::Cuda) {
        found.extend(cuda::list_devices());
    }
    #[cfg(feature = "opencl")]
    if matches!(kind, BackendKind::Opencl)
        || (matches!(kind, BackendKind::Auto) && found.is_empty())
    {
        found.extend(dedup_opencl(opencl::list_devices()));
    }
    let _ = kind; // (in case no GPU features are enabled)

    // Exits rather than returning an Err: by the time an empty list reaches
    // the caller it is indistinguishable from "this machine has no GPU", and
    // that path starts CPU mining instead. The rule itself lives in
    // `select_gpu` so it stays testable without a GPU.
    match select_gpu(found, gpu_id) {
        Ok(picked) => picked,
        Err(msg) => {
            eprintln!("bc3-miner: {msg}");
            std::process::exit(2);
        }
    }
}

/// Narrow a detected list down to the one device `--gpu-id` names.
///
/// An id past the end is an error. It used to produce an empty list, which
/// under the default `Auto` backend (no `--require-gpu`) means "no GPU found -
/// mining on CPU": a typo in `BC3_GPU_ID` then looked exactly like a card that
/// had fallen off the bus, and on a host rented by the hour it was billed at
/// GPU prices the whole time.
///
/// An empty input is left alone deliberately - there is no valid range to name
/// then, and the caller already reports "no GPU found" for that case.
fn select_gpu(found: Vec<GpuDevice>, gpu_id: Option<usize>) -> Result<Vec<GpuDevice>, String> {
    let Some(id) = gpu_id else { return Ok(found) };
    if found.is_empty() {
        return Ok(found);
    }
    if id >= found.len() {
        return Err(format!(
            "--gpu-id {id} is out of range - {} GPU(s) detected, valid ids are 0..={}",
            found.len(),
            found.len() - 1
        ));
    }
    Ok(found.into_iter().skip(id).take(1).collect())
}

/// Drop the OpenCL devices that are a second platform's view of a card another
/// platform already listed.
///
/// Two runtimes routinely expose the same physical GPU: Mesa rusticl alongside
/// ROCm, or an Intel Compute Runtime alongside the Intel legacy one. Without
/// this the miner starts two workers on one card. They split the card's
/// throughput while each reports a full hashrate, so the totals look right and
/// the pool sees roughly half the shares it should.
///
/// The key is vendor+name, and multiplicity is per platform, not summed: two
/// identical cards on one platform are two cards, while the same card seen
/// through two platforms is one. Counting occurrences instead of keying on the
/// name alone is what keeps a dual-RTX-3060 box from losing a card here.
///
/// If the two runtimes spell the vendor or the name differently, nothing
/// matches and the result is what it is today - a duplicate, not a lost card.
#[cfg(feature = "opencl")]
pub fn dedup_opencl(devices: Vec<GpuDevice>) -> Vec<GpuDevice> {
    use std::collections::HashMap;

    type Key = (String, String);
    fn key_of(dev: &GpuDevice) -> Option<(Key, usize)> {
        #[allow(irrefutable_let_patterns)]
        if let GpuDevice::Opencl { platform, name, vendor, .. } = dev {
            let key = (vendor.trim().to_lowercase(), name.trim().to_lowercase());
            return Some((key, *platform));
        }
        None
    }

    let mut per_platform: HashMap<Key, HashMap<usize, usize>> = HashMap::new();
    for dev in &devices {
        if let Some((key, platform)) = key_of(dev) {
            *per_platform.entry(key).or_default().entry(platform).or_default() += 1;
        }
    }

    // For each card, keep the platform that exposes the most of it; ties go to
    // the lowest platform index so the choice is stable across runs.
    let winner: HashMap<Key, usize> = per_platform
        .into_iter()
        .filter_map(|(key, counts)| {
            counts
                .into_iter()
                .max_by_key(|&(platform, n)| (n, std::cmp::Reverse(platform)))
                .map(|(platform, _)| (key, platform))
        })
        .collect();

    devices
        .into_iter()
        .filter(|dev| match key_of(dev) {
            Some((key, platform)) => winner.get(&key) == Some(&platform),
            None => true,
        })
        .collect()
}

/// Open a backend for a discovered device (runs in the GPU worker thread).
pub fn open_backend(dev: &GpuDevice) -> Result<Box<dyn MiningBackend>, String> {
    match dev {
        #[cfg(feature = "cuda")]
        GpuDevice::Cuda { index, .. } => Ok(Box::new(cuda::CudaBackend::new(*index)?)),
        #[cfg(feature = "opencl")]
        GpuDevice::Opencl { platform, device, .. } => {
            Ok(Box::new(opencl::OpenClBackend::new(*platform, *device)?))
        }
        #[allow(unreachable_patterns)]
        _ => unreachable!(),
    }
}

/// Open a backend for `dev`, and if a CUDA device refuses to open, try the
/// same card again through OpenCL.
///
/// The failure this exists for is the driver refusing our PTX. The kernel is
/// precompiled to a fixed PTX ISA version, and a driver can only JIT versions
/// it knows - build the PTX with a newer toolkit than the user's driver and
/// `cuModuleLoadData` fails with CUDA_ERROR_UNSUPPORTED_PTX_VERSION. Detection
/// has already succeeded at that point (libcuda is there, the card is there),
/// so `detect_gpus` never looked at OpenCL, and without this fallback the
/// miner sat connected to the pool with no worker at all: no hashrate, no
/// error the user could act on, and `--require-gpu` blind to it because a GPU
/// *was* found.
///
/// NVIDIA's own OpenCL runtime ships with the same driver, so the card is
/// almost always reachable that way instead.
pub fn open_backend_with_fallback(dev: &GpuDevice) -> Result<Box<dyn MiningBackend>, String> {
    let first = match open_backend(dev) {
        Ok(b) => return Ok(b),
        Err(e) => e,
    };

    #[cfg(all(feature = "cuda", feature = "opencl"))]
    if let GpuDevice::Cuda { index, name } = dev {
        let cuda = cuda::list_devices();
        let candidates = opencl::list_devices();
        match pick_opencl_for_cuda(&cuda, &candidates, *index, name) {
            Some(i) => {
                // No claim about WHICH card this is. The last-resort branch
                // can land on a different one - an Intel iGPU beside a card
                // whose NVIDIA ICD is missing - and describe() already names
                // it, so the sentence must not contradict the evidence.
                eprintln!(
                    "[gpu] CUDA could not open {name}: {first}\n\
                     [gpu] falling back to {} through OpenCL",
                    candidates[i].describe()
                );
                return open_backend(&candidates[i]);
            }
            None => eprintln!(
                "[gpu] CUDA could not open {name}: {first}\n\
                 [gpu] no OpenCL device could be matched to CUDA #{index} \
                 ({} OpenCL GPU(s) found) - not guessing",
                candidates.len()
            ),
        }
    }

    Err(first)
}

/// Which OpenCL device, if any, is the same physical card as CUDA `index`?
///
/// Pure and testable ON PURPOSE. Every other device-selection rule here -
/// `select_gpu`, `dedup_opencl` - is a pure function with a test matrix, and
/// this one was not: it was inline, it called `list_devices()` itself, and the
/// bug below therefore could not be reached from any test. It only runs on
/// hardware we cannot reach, which is exactly why it needs the coverage most.
///
/// The bug it had: the CUDA index is a GLOBAL ordinal while the name-matched
/// list is per model. On a rig of identical cards the two line up and all is
/// well; on a mixed rig - one 4090 and two 3060s - CUDA #2 indexes past the
/// end of a two-element list and the card is dropped, with one line on stderr
/// and a watchdog that stays quiet because other workers still live.
///
/// So the rank is computed WITHIN the same-named CUDA devices. That mapping is
/// injective: each worker passes its own rank, so no two land on one device.
/// If CUDA and OpenCL happen to enumerate the same model in a different order
/// the result is a permutation - every card still mines at full rate, only a
/// log line names the wrong one. Extranonce2 partitioning comes from the
/// worker index, not the device, so there is no duplicated work either.
#[cfg(all(feature = "cuda", feature = "opencl"))]
fn pick_opencl_for_cuda(
    cuda: &[GpuDevice],
    opencl: &[GpuDevice],
    index: usize,
    name: &str,
) -> Option<usize> {
    let same_name = |n: &str| n.eq_ignore_ascii_case(name);

    // Position of this card among the CUDA devices of the SAME model.
    //
    // `None` when the card is no longer in the list, and then we must NOT
    // guess. `cuda::list_devices` drops any device whose context fails to
    // open, so the list goes sparse at exactly the moment a card is failing -
    // which is the moment this fallback runs. Falling back to the global
    // ordinal there re-created the collision this function exists to prevent:
    // on three identical cards where #0 has vanished, worker 0 (no position,
    // ordinal 0) and worker 1 (position 0 in the surviving list) both resolve
    // to the same device, one card idles, and both workers report alive.
    //
    // The branches below are all gated on `cuda.len() == 1`, so at most one
    // worker can reach them. This is the only shared path, and skipping it
    // when the rank is unknown keeps it injective.
    let rank = cuda
        .iter()
        .filter(|d| matches!(d, GpuDevice::Cuda { name: n, .. } if same_name(n)))
        .position(|d| matches!(d, GpuDevice::Cuda { index: i, .. } if *i == index));

    let by_name: Vec<usize> = opencl
        .iter()
        .enumerate()
        .filter(|(_, c)| matches!(c, GpuDevice::Opencl { name: n, .. } if same_name(n)))
        .map(|(i, _)| i)
        .collect();
    if let Some(i) = rank.and_then(|r| by_name.get(r)) {
        return Some(*i);
    }

    // No name match. That is the COMMON case for the population this fallback
    // serves: a driver old enough to refuse ISA 8.0 is from 2022 or earlier,
    // and NVIDIA's OpenCL of that era reported "GeForce GTX 1080" where CUDA
    // says "NVIDIA GeForce GTX 1080". Refusing here would decline to help
    // precisely the machines the fallback was written for.
    //
    // The vendor is the safe discriminator: it is already on the device from
    // the dedup work, and it separates the NVIDIA card from the Intel or AMD
    // iGPU sitting next to it in a laptop. Take it only when it leaves exactly
    // one candidate - one card of a vendor is an identification, two is a
    // guess, and guessing is what put every worker on one device.
    let nvidia: Vec<usize> = opencl
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            matches!(c, GpuDevice::Opencl { vendor: v, .. } if v.to_lowercase().contains("nvidia"))
        })
        .map(|(i, _)| i)
        .collect();
    if nvidia.len() == 1 && cuda.len() == 1 {
        return Some(nvidia[0]);
    }
    // Last resort: a single OpenCL GPU and a single CUDA device must be the
    // same card, whatever either runtime chose to call it.
    if opencl.len() == 1 && cuda.len() == 1 {
        return Some(0);
    }
    None
}

/// Pack the 80-byte header (nonce = 0) as ten LE u64 lanes for the kernel.
#[allow(dead_code)]
pub fn header_lanes(header76: &[u8; 76]) -> [u64; 10] {
    let mut buf = [0u8; 80];
    buf[..76].copy_from_slice(header76);
    let mut lanes = [0u64; 10];
    for (i, lane) in lanes.iter_mut().enumerate() {
        *lane = u64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
    }
    lanes
}

/// Target ([u8;32] big-endian) -> four u64 limbs [t0..t3], t3 most significant.
/// Matches how the kernel reads them: hash limb k = LE u64 of bytes 8k..8k+7.
#[allow(dead_code)]
pub fn target_limbs(target: &Target) -> [u64; 4] {
    let mut t = [0u64; 4];
    for (k, limb) in t.iter_mut().enumerate() {
        let off = (3 - k) * 8;
        *limb = u64::from_be_bytes(target[off..off + 8].try_into().unwrap());
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::{
        compact_to_target, hash_meets_target, sha3t, target_for_difficulty, BlockHeader, SHA3_VBIT,
    };

    // ------------------------------------------------------------------
    // Rust mirror of the kernel algorithm (same lane layout, padding and
    // comparison as src/kernels/sha3t.cl). Verifies the kernel's math
    // against the CPU reference without a GPU - bit-exactness on a real GPU
    // is then pinned down by the #[ignore]d tests in the cuda backend.
    // ------------------------------------------------------------------

    const RC: [u64; 24] = [
        0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
        0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
        0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
    ];
    const ROTC: [u32; 24] = [
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
    ];
    const PILN: [usize; 24] = [
        10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
    ];

    fn keccakf(st: &mut [u64; 25]) {
        for round in 0..24 {
            let mut bc = [0u64; 5];
            for i in 0..5 {
                bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
            }
            for i in 0..5 {
                let t = bc[(i + 4) % 5] ^ bc[(i + 1) % 5].rotate_left(1);
                for j in (0..25).step_by(5) {
                    st[j + i] ^= t;
                }
            }
            let mut t = st[1];
            for i in 0..24 {
                let j = PILN[i];
                let tmp = st[j];
                st[j] = t.rotate_left(ROTC[i]);
                t = tmp;
            }
            for j in (0..25).step_by(5) {
                let mut bc = [0u64; 5];
                bc.copy_from_slice(&st[j..j + 5]);
                for i in 0..5 {
                    st[j + i] ^= (!bc[(i + 1) % 5]) & bc[(i + 2) % 5];
                }
            }
            st[0] ^= RC[round];
        }
    }

    fn sha3_256_32(input: [u64; 4]) -> [u64; 4] {
        let mut st = [0u64; 25];
        st[..4].copy_from_slice(&input);
        st[4] = 0x06;
        st[16] = 0x8000_0000_0000_0000;
        keccakf(&mut st);
        [st[0], st[1], st[2], st[3]]
    }

    /// Exactly what the kernel does per nonce, in Rust.
    fn kernel_mirror_hash(header76: &[u8; 76], nonce: u32) -> [u64; 4] {
        let lanes = header_lanes(header76);
        let mut st = [0u64; 25];
        st[..10].copy_from_slice(&lanes);
        st[9] |= (nonce as u64) << 32;
        st[10] = 0x06;
        st[16] = 0x8000_0000_0000_0000;
        keccakf(&mut st);
        let h = [st[0], st[1], st[2], st[3]];
        sha3_256_32(sha3_256_32(h))
    }

    fn limbs_of_hash(hash: &[u8; 32]) -> [u64; 4] {
        let mut l = [0u64; 4];
        for (k, limb) in l.iter_mut().enumerate() {
            *limb = u64::from_le_bytes(hash[k * 8..k * 8 + 8].try_into().unwrap());
        }
        l
    }

    fn genesis_header() -> BlockHeader {
        let mut merkle: [u8; 32] =
            hex::decode("8e1df52fddd25c460304ff8ea7bcb570850bf0b0c027eecf8ebf8ab17d3e93b1")
                .unwrap()
                .try_into()
                .unwrap();
        merkle.reverse();
        BlockHeader {
            version: 1,
            prev_hash: [0u8; 32],
            merkle_root: merkle,
            time: 1_777_245_555,
            bits: 0x1d00ffff,
            nonce: 2_442_659_435,
        }
    }

    /// Simple deterministic PRNG for test headers (no rand dependency).
    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn kernel_mirror_matches_cpu_reference_genesis() {
        let g = genesis_header();
        let ser = g.serialize();
        let header76: [u8; 76] = ser[..76].try_into().unwrap();
        let expected = limbs_of_hash(&sha3t(&ser));
        assert_eq!(kernel_mirror_hash(&header76, g.nonce), expected);
    }

    #[test]
    fn kernel_mirror_matches_cpu_reference_random() {
        let mut seed = 0xbc3_0001u64;
        for _ in 0..50 {
            let mut header80 = [0u8; 80];
            for b in header80.iter_mut() {
                *b = xorshift(&mut seed) as u8;
            }
            let header76: [u8; 76] = header80[..76].try_into().unwrap();
            let nonce = u32::from_le_bytes(header80[76..80].try_into().unwrap());
            let expected = limbs_of_hash(&sha3t(&header80));
            assert_eq!(kernel_mirror_hash(&header76, nonce), expected, "header {header80:02x?}");
        }
    }

    #[test]
    fn target_limbs_comparison_matches_hash_meets_target() {
        // The limb comparison (the one the kernel does) must give the same
        // answer as consensus::hash_meets_target for random hashes/targets.
        let mut seed = 0xbc3_0002u64;
        let targets = [
            compact_to_target(0x1d00ffff).unwrap(),
            target_for_difficulty(16.0),
            target_for_difficulty(0.001), // high target - many hits
        ];
        for target in targets {
            let t = target_limbs(&target);
            for _ in 0..2000 {
                let mut hash = [0u8; 32];
                for b in hash.iter_mut() {
                    *b = xorshift(&mut seed) as u8;
                }
                // Make some hashes small so both branches get exercised.
                if seed % 3 == 0 {
                    for b in hash[4..].iter_mut() {
                        *b = 0;
                    }
                }
                let h = limbs_of_hash(&hash);
                let kernel_ok = if h[3] != t[3] {
                    h[3] < t[3]
                } else if h[2] != t[2] {
                    h[2] < t[2]
                } else if h[1] != t[1] {
                    h[1] < t[1]
                } else if h[0] != t[0] {
                    h[0] < t[0]
                } else {
                    true
                };
                assert_eq!(kernel_ok, hash_meets_target(&hash, &target));
            }
        }
    }

    #[test]
    fn header_lanes_roundtrip() {
        let mut header76 = [0u8; 76];
        for (i, b) in header76.iter_mut().enumerate() {
            *b = i as u8;
        }
        let lanes = header_lanes(&header76);
        // lane 9 = bytes 72..75 + a zeroed nonce.
        assert_eq!(lanes[9], u64::from_le_bytes([72, 73, 74, 75, 0, 0, 0, 0]));
        assert_eq!(lanes[0], u64::from_le_bytes([0, 1, 2, 3, 4, 5, 6, 7]));
    }

    // ------------------------------------------------------------------
    // Device selection.
    // ------------------------------------------------------------------

    #[cfg(feature = "opencl")]
    fn ocl(platform: usize, device: usize, vendor: &str, name: &str) -> GpuDevice {
        GpuDevice::Opencl {
            platform,
            device,
            name: name.into(),
            vendor: vendor.into(),
        }
    }

    #[cfg(feature = "opencl")]
    fn described(devices: &[GpuDevice]) -> Vec<String> {
        devices.iter().map(|d| d.describe()).collect()
    }

    /// The case this exists for: rusticl and ROCm both listing one card.
    #[cfg(feature = "opencl")]
    #[test]
    fn one_card_on_two_platforms_is_listed_once() {
        let out = dedup_opencl(vec![
            ocl(0, 0, "Advanced Micro Devices, Inc.", "gfx1030"),
            ocl(1, 0, "Advanced Micro Devices, Inc.", "gfx1030"),
        ]);
        assert_eq!(described(&out), ["OpenCL 0.0: gfx1030"]);
    }

    /// ...and the case that must NOT be broken by fixing it: two of the same
    /// card on one platform are two cards. Keying on the name alone, without
    /// counting per platform, would silently drop half the rig.
    #[cfg(feature = "opencl")]
    #[test]
    fn two_identical_cards_on_one_platform_are_both_kept() {
        let out = dedup_opencl(vec![
            ocl(0, 0, "NVIDIA Corporation", "NVIDIA GeForce RTX 3060"),
            ocl(0, 1, "NVIDIA Corporation", "NVIDIA GeForce RTX 3060"),
        ]);
        assert_eq!(out.len(), 2);
    }

    /// Two cards seen through two runtimes: still two cards, from whichever
    /// platform exposes both.
    #[cfg(feature = "opencl")]
    #[test]
    fn duplicated_pairs_collapse_to_the_pair() {
        let out = dedup_opencl(vec![
            ocl(0, 0, "AMD", "gfx1030"),
            ocl(0, 1, "AMD", "gfx1030"),
            ocl(1, 0, "AMD", "gfx1030"),
            ocl(1, 1, "AMD", "gfx1030"),
        ]);
        assert_eq!(described(&out), ["OpenCL 0.0: gfx1030", "OpenCL 0.1: gfx1030"]);
    }

    /// A platform that exposes MORE of a card than the first one wins, so a
    /// runtime that only sees one of two identical cards cannot hide the other.
    #[cfg(feature = "opencl")]
    #[test]
    fn the_platform_that_sees_the_most_of_a_card_wins() {
        let out = dedup_opencl(vec![
            ocl(0, 0, "Intel", "Arc A770"),
            ocl(1, 0, "Intel", "Arc A770"),
            ocl(1, 1, "Intel", "Arc A770"),
        ]);
        assert_eq!(described(&out), ["OpenCL 1.0: Arc A770", "OpenCL 1.1: Arc A770"]);
    }

    /// Different cards are never merged, whichever platform they sit on.
    #[cfg(feature = "opencl")]
    #[test]
    fn different_cards_are_left_alone() {
        let out = dedup_opencl(vec![
            ocl(0, 0, "AMD", "gfx1030"),
            ocl(0, 1, "Intel", "Arc A770"),
            ocl(1, 0, "AMD", "gfx900"),
        ]);
        assert_eq!(out.len(), 3);
    }

    /// Vendor is half the key: same model name from two vendors is two cards.
    #[cfg(feature = "opencl")]
    #[test]
    fn the_vendor_is_part_of_the_key() {
        let out = dedup_opencl(vec![
            ocl(0, 0, "Mesa", "Graphics Device"),
            ocl(1, 0, "Intel", "Graphics Device"),
        ]);
        assert_eq!(out.len(), 2);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_id_selects_one_device() {
        let found = vec![ocl(0, 0, "AMD", "a"), ocl(0, 1, "AMD", "b")];
        let out = select_gpu(found.clone(), Some(1)).unwrap();
        assert_eq!(described(&out), ["OpenCL 0.1: b"]);
        // No --gpu-id: everything, untouched.
        assert_eq!(select_gpu(found.clone(), None).unwrap().len(), 2);
    }

    /// An out-of-range id must fail loudly and say what the range is. It used
    /// to produce an empty list, which under the default backend means
    /// "mining on CPU" - a typo that costs a rented GPU host its whole run.
    #[cfg(feature = "opencl")]
    #[test]
    fn an_out_of_range_gpu_id_is_an_error_naming_the_range() {
        let found = vec![ocl(0, 0, "AMD", "a"), ocl(0, 1, "AMD", "b")];
        let err = select_gpu(found, Some(2)).unwrap_err();
        assert!(err.contains("--gpu-id 2"), "{err}");
        assert!(err.contains("0..=1"), "{err}");
        assert!(err.is_ascii(), "non-ASCII in log output: {err:?}");
    }

    /// With no GPU at all there is no range to name, and the caller already
    /// reports that case - so this stays an empty list, not an error.
    #[test]
    fn gpu_id_without_any_device_is_not_an_error() {
        assert!(select_gpu(vec![], Some(3)).unwrap().is_empty());
    }

    // ------------------------------------------------------------------
    // Hit buffer overflow.
    // ------------------------------------------------------------------

    /// Until the kernel overflows, a scan is one launch over the whole range.
    #[test]
    fn the_hit_budget_does_not_split_anything_by_default() {
        let mut b = HitBudget::default();
        assert_eq!(b.chunk(1_000_000), 1_000_000);
        assert_eq!(b.overflowed(1_000_000, MAX_HITS), None);
        assert_eq!(b.chunk(1_000_000), 1_000_000);
    }

    /// The first overflow is reported and halves the batch; later ones halve
    /// it further but stay quiet, or the log fills with one line per launch.
    #[test]
    fn an_overflow_is_logged_once_and_halves_the_batch() {
        let mut b = HitBudget::default();
        let msg = b.overflowed(1000, MAX_HITS + 5).expect("first overflow must be logged");
        assert!(msg.contains("5 nonce(s) dropped"), "{msg}");
        assert!(msg.is_ascii(), "non-ASCII in log output: {msg:?}");
        assert_eq!(b.chunk(1000), 500);

        assert_eq!(b.overflowed(500, MAX_HITS + 1), None, "must not log per launch");
        assert_eq!(b.chunk(1000), 250);
    }

    /// Halving must never reach a batch of 0 - that would make no progress at
    /// all and hang the worker on an endless loop of empty launches.
    #[test]
    fn the_hit_budget_never_halves_to_zero() {
        let mut b = HitBudget::default();
        for _ in 0..64 {
            b.overflowed(b.chunk(4096), MAX_HITS + 1);
            assert!(b.chunk(4096) >= 1);
        }
        assert_eq!(b.chunk(4096), 1);
    }

    #[test]
    fn sha3_vbit_headers_use_sha3t() {
        // Sanity: jobs from the pool always have the version bit set, so
        // the GPU path (which always runs sha3t) matches BlockHeader::hash.
        let mut h = genesis_header();
        h.version |= SHA3_VBIT;
        let ser = h.serialize();
        assert_eq!(h.hash(), sha3t(&ser));
    }
}
