//! CUDA backend via `cudarc` (driver API).
//!
//! The kernel is PRECOMPILED to PTX at build time (see build.rs) and embedded
//! in the binary. At runtime we therefore only need `nvcuda.dll`/`libcuda.so`
//! - that is, the graphics driver - which JITs the PTX into the card's
//! machine code. NVRTC is NOT used: it ships with the CUDA Toolkit, which end
//! users do not have installed. libcuda is loaded dynamically, so the binary
//! starts even with no NVIDIA driver at all (detection then returns an empty
//! list).
//!
//! Launches are queued, not run one at a time (see `QUEUE_DEPTH`): every
//! launch has its own buffers, a pinned buffer its hits are read back into,
//! and two events - one before the kernel, one after the readback - which
//! time it on the device and let the host sleep until it is done.
//!
//! All on one stream. A stream per launch, so the next kernel could start on
//! SMs the previous one has finished with, was measured to change nothing
//! (RTX 3050 Ti, Linux and Windows): what a short BC3Hash launch loses is
//! its first waves running in lockstep, not a tail - and it put cudarc into
//! its multi-stream event tracking for nothing.

use super::{
    kernel_lanes, kernel_supports, kernel_target, Launch, MiningBackend, Scan, MAX_HITS,
    QUEUE_DEPTH,
};
use crate::consensus::{PowAlgo, Target};
use cudarc::driver::{
    sys, CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, LaunchConfig,
    PinnedHostSlice, PushKernelArg,
};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

const BLOCK_DIM: u32 = 256;

/// Precompiled kernel PTX (built by build.rs).
const SHA3T_PTX: &str = include_str!(env!("BC3_SHA3T_PTX"));

/// The BC3Hash kernel, one PTX per secp256k1 table window: the best window
/// depends on the card's L2 - on an RTX 3050 Ti (2 MB of L2) the lab measured
/// 110 kH/s at 10 bits and 128 at 16 - so the choice is made at run time.
#[cfg(feature = "bc3hash")]
const BC3HASH_PTX: [(u32, &str); 4] = [
    (10, include_str!(env!("BC3_BC3HASH_W10_PTX"))),
    (12, include_str!(env!("BC3_BC3HASH_W12_PTX"))),
    (14, include_str!(env!("BC3_BC3HASH_W14_PTX"))),
    (16, include_str!(env!("BC3_BC3HASH_W16_PTX"))),
];

/// Threads per block for BC3Hash. 32 to 256 measured the same; a small
/// block keeps the tail of a launch short.
#[cfg(feature = "bc3hash")]
const BC3_BLOCK_DIM: u32 = 64;

/// One loaded BC3Hash kernel: its window, the two entry points, and the
/// tables they read (uploaded once).
#[cfg(feature = "bc3hash")]
struct Bc3Kernel {
    window: u32,
    scan: CudaFunction,
    batch: CudaFunction,
    d_tables: CudaSlice<u64>,
    /// Nonces the card runs at once: SMs x resident blocks x block size.
    /// At 255 registers a thread only 4 blocks of 64 fit on an SM, so a
    /// launch that is not a whole number of these waves leaves most of the
    /// card idle for its last one - 16384 nonces on an RTX 3050 Ti is 3.2
    /// waves, which cost 14 %.
    wave: u32,
}

/// What one launch in flight owns, so that the next can be queued while it
/// runs: the per-launch inputs, the hit buffer and the pinned memory it is
/// read back into, and its two events.
struct Slot {
    /// SHA3-256t: the header as ten interleaved lanes.
    lanes: CudaSlice<u64>,
    /// BC3Hash: the header and the target limbs.
    #[cfg_attr(not(feature = "bc3hash"), allow(dead_code))]
    header: CudaSlice<u8>,
    #[cfg_attr(not(feature = "bc3hash"), allow(dead_code))]
    target: CudaSlice<u64>,
    /// [count, nonce0, nonce1, ...]
    hits: CudaSlice<u32>,
    hits_host: PinnedHostSlice<u32>,
    /// Recorded just before the kernel, and after the readback: the time
    /// between them is the launch's time on the device.
    start: CudaEvent,
    /// Blocking: the host thread sleeps on it instead of spinning a core,
    /// which on a laptop is power the GPU then does not get.
    done: CudaEvent,
    /// What `lanes`, `header` and `target` hold, so the copies that would
    /// change nothing - nearly all of them - are skipped.
    lanes_of: Option<[u64; 10]>,
    header_of: Option<[u8; 76]>,
    target_of: Option<[u64; 4]>,
}

impl Slot {
    fn new(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Result<Self, String> {
        let alloc = |what: &str, e| format!("alloc {what}: {e}");
        Ok(Self {
            lanes: stream.alloc_zeros::<u64>(10).map_err(|e| alloc("lanes", e))?,
            header: stream.alloc_zeros::<u8>(76).map_err(|e| alloc("header", e))?,
            target: stream.alloc_zeros::<u64>(4).map_err(|e| alloc("target", e))?,
            hits: stream.alloc_zeros::<u32>(1 + MAX_HITS).map_err(|e| alloc("hits", e))?,
            // SAFETY: only ever read after a readback into it has completed.
            hits_host: unsafe { ctx.alloc_pinned::<u32>(1 + MAX_HITS) }
                .map_err(|e| alloc("pinned hits", e))?,
            start: ctx
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| alloc("event", e))?,
            done: ctx
                .new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .map_err(|e| alloc("event", e))?,
            lanes_of: None,
            header_of: None,
            target_of: None,
        })
    }

    /// Forget what the device buffers hold - after a failed or abandoned
    /// launch nobody knows.
    fn forget(&mut self) {
        self.lanes_of = None;
        self.header_of = None;
        self.target_of = None;
    }
}

/// List CUDA devices; empty list if the driver or the devices are missing.
///
/// cudarc's dynamic loader PANICS if libcuda/nvcuda is not present on the
/// system, so the first contact is made behind catch_unwind (with a silenced
/// panic hook) - with no driver the result is an empty list instead of a
/// crash. Requires that panic = "abort" is NOT used (see Cargo.toml).
pub fn list_devices() -> Vec<super::GpuDevice> {
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let probed = std::panic::catch_unwind(CudaContext::device_count);
    std::panic::set_hook(prev_hook);

    let count = match probed {
        Ok(Ok(n)) if n > 0 => n as usize,
        _ => return vec![],
    };
    (0..count)
        .filter_map(|i| {
            let ctx = CudaContext::new(i).ok()?;
            let name = ctx.name().unwrap_or_else(|_| format!("CUDA device {i}"));
            Some(super::GpuDevice::Cuda { index: i, name })
        })
        .collect()
}

/// Compile the BC3Hash kernel for `window` into the driver's JIT cache, so
/// that loading it later - at the fork, when the card switches - comes from
/// the cache in about a second instead of 10-60 s of JIT.
///
/// On a context of its own: a JIT on the mining context blocks every launch
/// on it until it is done (window 16 on an i9-12900HK: 16.7 s, and one
/// SHA3-256t launch waited all of it), while on a second context nothing
/// waits (1234 launches meanwhile, the slowest 28 ms). The context goes when
/// this returns, and the memory it held with it.
#[cfg(feature = "bc3hash")]
pub fn compile_bc3hash_ahead(index: usize, window: u32) -> Result<std::time::Duration, String> {
    let t0 = std::time::Instant::now();
    let ptx = BC3HASH_PTX
        .iter()
        .find(|(w, _)| *w == window)
        .map(|(_, ptx)| *ptx)
        .ok_or_else(|| format!("no BC3Hash kernel for a window of {window} bits"))?;
    let ctx = CudaContext::new_non_primary(index, 0).map_err(|e| format!("context: {e}"))?;
    ctx.load_module(cudarc::nvrtc::Ptx::from_src(ptx))
        .map_err(|e| format!("could not load the BC3Hash kernel (window {window}): {e}"))?;
    Ok(t0.elapsed())
}

pub struct CudaBackend {
    /// The device's CUDA ordinal.
    #[cfg(feature = "bc3hash")]
    index: usize,
    #[cfg(feature = "bc3hash")]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    func: CudaFunction,
    name: String,
    /// One per launch that may be in flight, taken in turn.
    slots: Vec<Slot>,
    /// Slots of the launches in flight, oldest first.
    queued: VecDeque<usize>,
    /// The slot the next launch takes.
    next_slot: usize,
    /// The BC3Hash kernel in use - loaded on demand, for one window at a time.
    #[cfg(feature = "bc3hash")]
    bc3: Option<Bc3Kernel>,
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // A readback still in flight writes into pinned memory that the slots
        // free when they drop; let the device finish first.
        let _ = self.stream.synchronize();
    }
}

impl CudaBackend {
    pub fn new(index: usize) -> Result<Self, String> {
        let ctx = CudaContext::new(index).map_err(|e| format!("CudaContext::new: {e}"))?;
        let name = ctx.name().unwrap_or_else(|_| format!("CUDA device {index}"));
        let stream = ctx.default_stream();

        // Load the embedded PTX; the driver JITs it for the card.
        let ptx = cudarc::nvrtc::Ptx::from_src(SHA3T_PTX);
        let module = ctx.load_module(ptx).map_err(|e| {
            format!("could not load the kernel PTX (driver too old?): {e}")
        })?;
        let func = module
            .load_function("sha3t_scan")
            .map_err(|e| format!("load_function: {e}"))?;

        let slots = (0..QUEUE_DEPTH).map(|_| Slot::new(&ctx, &stream)).collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            #[cfg(feature = "bc3hash")]
            index,
            #[cfg(feature = "bc3hash")]
            ctx,
            stream,
            func,
            name,
            slots,
            queued: VecDeque::new(),
            next_slot: 0,
            #[cfg(feature = "bc3hash")]
            bc3: None,
        })
    }

    /// Load the BC3Hash kernel for `window` and upload its tables. The JIT
    /// of a 2.8 MB PTX takes a while the first time on a card; the driver
    /// caches the result.
    #[cfg(feature = "bc3hash")]
    fn load_bc3(&mut self, window: u32) -> Result<Bc3Kernel, String> {
        let ptx = BC3HASH_PTX
            .iter()
            .find(|(w, _)| *w == window)
            .map(|(_, ptx)| *ptx)
            .ok_or_else(|| format!("no BC3Hash kernel for a window of {window} bits"))?;
        let module = self
            .ctx
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|e| format!("could not load the BC3Hash kernel (window {window}): {e}"))?;
        let scan = module
            .load_function("bc3hash_scan")
            .map_err(|e| format!("load_function bc3hash_scan: {e}"))?;
        let batch = module
            .load_function("bc3hash_batch")
            .map_err(|e| format!("load_function bc3hash_batch: {e}"))?;
        let tables = super::bc3::host_tables(window)?;
        let d_tables = self
            .stream
            .clone_htod(&tables[..])
            .map_err(|e| format!("upload BC3Hash tables ({} MiB): {e}", (tables.len() * 8) >> 20))?;
        let sms = self
            .ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap_or(1)
            .max(1) as u32;
        let per_sm = scan
            .occupancy_max_active_blocks_per_multiprocessor(BC3_BLOCK_DIM, 0, None)
            .unwrap_or(1)
            .max(1);
        let wave = sms * per_sm * BC3_BLOCK_DIM;
        Ok(Bc3Kernel { window, scan, batch, d_tables, wave })
    }

    /// BC3Hash of 80-byte headers on the GPU, through the kernel's batch
    /// entry point - the self-test's view of the kernel.
    #[cfg(feature = "bc3hash")]
    fn bc3_batch(&mut self, headers: &[[u8; 80]]) -> Result<Vec<[u8; 32]>, String> {
        let k = self.bc3.as_ref().ok_or("BC3Hash kernel not loaded")?;
        let flat: Vec<u8> = headers.iter().flatten().copied().collect();
        let d_in = self.stream.clone_htod(&flat[..]).map_err(|e| format!("htod headers: {e}"))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<u8>(headers.len() * 32)
            .map_err(|e| format!("alloc digests: {e}"))?;
        let count = headers.len() as u32;
        let mut launch = self.stream.launch_builder(&k.batch);
        launch.arg(&k.d_tables).arg(&d_in).arg(&count).arg(&mut d_out);
        unsafe {
            launch
                .launch(launch_config(count, BC3_BLOCK_DIM))
                .map_err(|e| format!("bc3hash_batch launch: {e}"))?;
        }
        let out = self.stream.clone_dtoh(&d_out).map_err(|e| format!("dtoh digests: {e}"))?;
        self.stream.synchronize().map_err(|e| format!("synchronize: {e}"))?;
        Ok(out.as_chunks::<32>().0.to_vec())
    }

    /// Queue the copies and the kernel of one launch on slot `i`.
    fn queue_kernel(&mut self, i: usize, l: &Launch) -> Result<(), String> {
        match l.algo {
            PowAlgo::Bc3Hash => self.queue_bc3(i, l),
            algo if kernel_supports(algo) => self.queue_sha3t(i, l),
            algo => Err(format!("no kernel for {}", algo.name())),
        }
    }

    fn queue_sha3t(&mut self, i: usize, l: &Launch) -> Result<(), String> {
        let stream = &self.stream;
        let slot = &mut self.slots[i];
        let lanes = kernel_lanes(l.header76);
        if slot.lanes_of != Some(lanes) {
            slot.lanes_of = None;
            stream.memcpy_htod(&lanes[..], &mut slot.lanes).map_err(|e| format!("htod lanes: {e}"))?;
            slot.lanes_of = Some(lanes);
        }
        let t3 = kernel_target(l.target);
        let max_hits = MAX_HITS as u32;
        stream
            .memset_zeros(&mut slot.hits.slice_mut(0..1))
            .map_err(|e| format!("reset hits: {e}"))?;
        slot.start.record(stream).map_err(|e| format!("record: {e}"))?;
        let mut launch = stream.launch_builder(&self.func);
        launch
            .arg(&slot.lanes)
            .arg(&l.start)
            .arg(&l.count)
            .arg(&t3)
            .arg(&mut slot.hits)
            .arg(&max_hits);
        unsafe { launch.launch(launch_config(l.count, BLOCK_DIM)) }
            .map_err(|e| format!("kernel-launch: {e}"))?;
        Ok(())
    }

    #[cfg(feature = "bc3hash")]
    fn queue_bc3(&mut self, i: usize, l: &Launch) -> Result<(), String> {
        let k = self.bc3.as_ref().ok_or("BC3Hash kernel not loaded")?;
        let stream = &self.stream;
        let slot = &mut self.slots[i];
        if slot.header_of.as_ref() != Some(l.header76) {
            slot.header_of = None;
            stream
                .memcpy_htod(&l.header76[..], &mut slot.header)
                .map_err(|e| format!("htod header: {e}"))?;
            slot.header_of = Some(*l.header76);
        }
        let limbs = super::bc3::target_limbs(l.target);
        if slot.target_of != Some(limbs) {
            slot.target_of = None;
            stream.memcpy_htod(&limbs[..], &mut slot.target).map_err(|e| format!("htod target: {e}"))?;
            slot.target_of = Some(limbs);
        }
        let max_hits = MAX_HITS as u32;
        stream
            .memset_zeros(&mut slot.hits.slice_mut(0..1))
            .map_err(|e| format!("reset hits: {e}"))?;
        slot.start.record(stream).map_err(|e| format!("record: {e}"))?;
        let mut launch = stream.launch_builder(&k.scan);
        launch
            .arg(&k.d_tables)
            .arg(&slot.header)
            .arg(&l.start)
            .arg(&l.count)
            .arg(&slot.target)
            .arg(&mut slot.hits)
            .arg(&max_hits);
        unsafe { launch.launch(launch_config(l.count, BC3_BLOCK_DIM)) }
            .map_err(|e| format!("bc3hash_scan launch: {e}"))?;
        Ok(())
    }

    #[cfg(not(feature = "bc3hash"))]
    fn queue_bc3(&mut self, _i: usize, _l: &Launch) -> Result<(), String> {
        Err("this build has no BC3Hash kernel".into())
    }
}

/// One thread per nonce, in blocks of `block`; the kernel ignores the threads
/// past `count` in the last block.
fn launch_config(count: u32, block: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (count.div_ceil(block), 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }
}

#[cfg(feature = "bc3hash")]
impl super::bc3::Bc3Ops for CudaBackend {
    fn bc3_batch(&mut self, headers: &[[u8; 80]]) -> Result<Vec<[u8; 32]>, String> {
        CudaBackend::bc3_batch(self, headers)
    }

    fn bc3_scan(
        &mut self,
        header76: &[u8; 76],
        start: u32,
        count: u32,
        target: &Target,
    ) -> Result<Vec<u32>, String> {
        self.scan_range(PowAlgo::Bc3Hash, header76, start, count, target)
    }
}

impl MiningBackend for CudaBackend {
    fn name(&self) -> String {
        format!("CUDA: {}", self.name)
    }

    fn has_kernel(&self, algo: PowAlgo) -> bool {
        match algo {
            PowAlgo::Bc3Hash => cfg!(feature = "bc3hash"),
            PowAlgo::Sha256d | PowAlgo::Sha3t => kernel_supports(algo),
        }
    }

    fn supports(&self, algo: PowAlgo) -> bool {
        match algo {
            #[cfg(feature = "bc3hash")]
            PowAlgo::Bc3Hash => self.bc3.is_some(),
            #[cfg(not(feature = "bc3hash"))]
            PowAlgo::Bc3Hash => false,
            PowAlgo::Sha256d | PowAlgo::Sha3t => kernel_supports(algo),
        }
    }

    #[cfg(feature = "bc3hash")]
    fn prepare_bc3hash(&mut self, window: u32) -> Result<std::time::Duration, String> {
        let t0 = std::time::Instant::now();
        // Nothing of the old kernel may still be running when its tables go.
        self.abandon();
        // The old window's tables leave the card before the new ones arrive:
        // at 16 bits they are 256 MiB.
        self.bc3 = None;
        let kernel = self.load_bc3(window)?;
        self.bc3 = Some(kernel);
        if let Err(e) = super::bc3::self_test(self) {
            self.bc3 = None;
            return Err(format!("self-test failed: {e}"));
        }
        Ok(t0.elapsed())
    }

    #[cfg(feature = "bc3hash")]
    fn bc3hash_window(&self) -> Option<u32> {
        self.bc3.as_ref().map(|k| k.window)
    }

    #[cfg(feature = "bc3hash")]
    fn compile_bc3hash_ahead(&self, window: u32) {
        let (index, name) = (self.index, self.name());
        let spawned = std::thread::Builder::new().name(format!("jit-{index}")).spawn(move || {
            match compile_bc3hash_ahead(index, window) {
                // Anything quicker came from the driver's cache: not news.
                Ok(took) if took > Duration::from_secs(3) => crate::human!(
                    "[gpu] {name}: BC3Hash kernel (window {window}) compiled ahead in {:.0} s",
                    took.as_secs_f64()
                ),
                Ok(_) => {}
                Err(e) => eprintln!("[gpu] {name}: compiling the BC3Hash kernel ahead failed: {e}"),
            }
        });
        if let Err(e) = spawned {
            eprintln!("[gpu] could not start the BC3Hash compile thread: {e}");
        }
    }

    fn launch_granule(&self, algo: PowAlgo) -> u32 {
        match algo {
            #[cfg(feature = "bc3hash")]
            PowAlgo::Bc3Hash => self.bc3.as_ref().map_or(1, |k| k.wave),
            _ => 1,
        }
    }

    fn enqueue(&mut self, l: &Launch) -> Result<(), String> {
        if l.count == 0 {
            return Err("a launch of 0 nonces".into());
        }
        if self.queued.len() >= QUEUE_DEPTH {
            return Err("too many launches in flight".into());
        }
        let i = self.next_slot;
        if let Err(e) = self.queue_kernel(i, l) {
            self.slots[i].forget();
            return Err(e);
        }
        let stream = &self.stream;
        let slot = &mut self.slots[i];
        stream
            .memcpy_dtoh(&slot.hits, &mut slot.hits_host)
            .map_err(|e| format!("dtoh hits: {e}"))?;
        slot.done.record(stream).map_err(|e| format!("record: {e}"))?;
        // WDDM (Windows) batches queued work in the driver until something
        // makes it submit; without this the launch could sit there until the
        // host waits for the one before it, which is exactly the gap the
        // queue exists to close. A query flushes, and costs nothing here.
        let _ = slot.done.is_complete();
        self.queued.push_back(i);
        self.next_slot = (i + 1) % QUEUE_DEPTH;
        Ok(())
    }

    fn collect(&mut self) -> Result<Scan, String> {
        let i = self.queued.pop_front().ok_or("no launch in flight")?;
        let slot = &self.slots[i];
        slot.done.synchronize().map_err(|e| format!("synchronize: {e}"))?;
        let busy = slot
            .start
            .elapsed_ms(&slot.done)
            .ok()
            .filter(|ms| ms.is_finite() && *ms >= 0.0)
            .map(|ms| Duration::from_secs_f64(f64::from(ms) / 1e3));
        // The pinned slice's own event is never recorded with a single
        // stream (cudarc only tracks it in multi-stream mode), so as_slice
        // does not wait: the synchronize above is what makes this safe.
        let hits = slot.hits_host.as_slice().map_err(|e| format!("read hits: {e}"))?;
        let reported = hits[0] as usize;
        let n = reported.min(MAX_HITS);
        Ok(Scan { hits: hits[1..1 + n].to_vec(), reported, busy })
    }

    fn in_flight(&self) -> usize {
        self.queued.len()
    }

    fn abandon(&mut self) {
        let _ = self.stream.synchronize();
        self.queued.clear();
        for slot in &mut self.slots {
            slot.forget();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::{
        compact_to_target, hash_meets_target, sha3t, BlockHeader,
    };

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// CPU ground truth: every nonce in the range whose sha3t hash <= target.
    fn cpu_hits(header76: &[u8; 76], start: u32, count: u32, target: &Target) -> Vec<u32> {
        let mut header80 = [0u8; 80];
        header80[..76].copy_from_slice(header76);
        (0..count)
            .filter_map(|i| {
                let nonce = start.wrapping_add(i);
                header80[76..80].copy_from_slice(&nonce.to_le_bytes());
                hash_meets_target(&sha3t(&header80), target).then_some(nonce)
            })
            .collect()
    }

    // Run in a CUDA container: cargo test -- --ignored --nocapture
    #[test]
    #[ignore = "needs an NVIDIA GPU (run inside a --gpus all container)"]
    fn cuda_finds_genesis_nonce() {
        let mut merkle: [u8; 32] =
            hex::decode("8e1df52fddd25c460304ff8ea7bcb570850bf0b0c027eecf8ebf8ab17d3e93b1")
                .unwrap()
                .try_into()
                .unwrap();
        merkle.reverse();
        let genesis = BlockHeader {
            version: 1,
            prev_hash: [0u8; 32],
            merkle_root: merkle,
            time: 1_777_245_555,
            bits: 0x1d00ffff,
            nonce: 2_442_659_435,
        };
        let ser = genesis.serialize();
        let header76: [u8; 76] = ser[..76].try_into().unwrap();
        // Note: the genesis block is SHA256d-mined; here we only verify that
        // the GPU's sha3t over the header matches the CPU's sha3t bit for
        // bit, with target = exactly the genesis header's sha3t hash (a
        // guaranteed hit). That hash is a "large" number, so nearly every
        // nonce hits too - the range is kept <= MAX_HITS so the whole hit
        // set can be compared.
        let hash = sha3t(&ser);
        let mut target = [0u8; 32];
        for (i, b) in hash.iter().rev().enumerate() {
            target[i] = *b; // target is big-endian of the little-endian value
        }

        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        println!("backend: {}", backend.name());
        let count = MAX_HITS as u32;
        let start = genesis.nonce - count / 2;
        let mut gpu = backend.scan_range(PowAlgo::Sha3t, &header76, start, count, &target).unwrap();
        let mut cpu = cpu_hits(&header76, start, count, &target);
        gpu.sort_unstable();
        cpu.sort_unstable();
        assert!(gpu.contains(&genesis.nonce), "genesis nonce missing from {gpu:?}");
        assert_eq!(gpu, cpu);
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU (run inside a --gpus all container)"]
    fn cuda_matches_cpu_on_random_headers() {
        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        let mut seed = 0xbc3_cafe_u64;
        for round in 0..10 {
            let mut header76 = [0u8; 76];
            for b in header76.iter_mut() {
                *b = xorshift(&mut seed) as u8;
            }
            let start = xorshift(&mut seed) as u32;
            let count = 4096u32;

            // Target = the SMALLEST hash in the range (computed on the CPU)
            // -> exactly one guaranteed hit; any keccak bit error anywhere in
            // the range that produced a smaller hash would show up as an
            // extra hit.
            let mut header80 = [0u8; 80];
            header80[..76].copy_from_slice(&header76);
            let (pick, min_hash) = (0..count)
                .map(|i| {
                    let nonce = start.wrapping_add(i);
                    header80[76..80].copy_from_slice(&nonce.to_le_bytes());
                    let mut key = sha3t(&header80);
                    key.reverse(); // big-endian -> compares lexicographically
                    (nonce, key)
                })
                .min_by(|a, b| a.1.cmp(&b.1))
                .unwrap();
            let target: Target = min_hash; // targets are big-endian bytes

            let mut gpu = backend.scan_range(PowAlgo::Sha3t, &header76, start, count, &target).unwrap();
            let mut cpu = cpu_hits(&header76, start, count, &target);
            gpu.sort_unstable();
            cpu.sort_unstable();
            assert!(gpu.contains(&pick), "round {round}: the chosen nonce is missing");
            assert_eq!(gpu, cpu, "round {round}: the GPU and CPU sets differ");
        }
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU (run inside a --gpus all container)"]
    fn cuda_low_diff_share_targets() {
        // Realistic pool scenario: a low share target, check the exact hit
        // set over a larger range.
        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        let target = crate::consensus::target_for_difficulty(0.00001);
        let mut seed = 0x51ab_e77a_u64;
        let mut header76 = [0u8; 76];
        for b in header76.iter_mut() {
            *b = xorshift(&mut seed) as u8;
        }
        let count = 1u32 << 20;
        let mut gpu = backend.scan_range(PowAlgo::Sha3t, &header76, 0, count, &target).unwrap();
        let mut cpu = cpu_hits(&header76, 0, count, &target);
        gpu.sort_unstable();
        cpu.sort_unstable();
        assert_eq!(gpu, cpu);
        assert!(!cpu.is_empty(), "the test should yield at least one hit (diff 1e-5, 2^20 nonces)");
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU - hashrate measurement, run with --nocapture"]
    fn cuda_hashrate() {
        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        let target = compact_to_target(0x1d00ffff).unwrap(); // few hits
        let header76 = [0x42u8; 76];
        // Warm-up.
        backend.scan_range(PowAlgo::Sha3t, &header76, 0, 1 << 20, &target).unwrap();
        let batch = 1u32 << 24;
        let rounds = 16u32;
        let t0 = std::time::Instant::now();
        for r in 0..rounds {
            backend
                .scan_range(PowAlgo::Sha3t, &header76, r.wrapping_mul(batch), batch, &target)
                .unwrap();
        }
        let dt = t0.elapsed().as_secs_f64();
        let rate = (batch as f64 * rounds as f64) / dt;
        println!(
            "{}: {:.2} MH/s ({} hashes in {dt:.2}s)",
            backend.name(),
            rate / 1e6,
            batch as u64 * rounds as u64
        );
    }

    /// The hashrate with launches queued two deep, as the GPU worker runs
    /// them, next to the same launches one at a time - the gap between the
    /// two is what the host round trip cost.
    #[test]
    #[ignore = "needs an NVIDIA GPU - hashrate measurement, run with --nocapture"]
    fn cuda_hashrate_queued() {
        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        let target = compact_to_target(0x1d00ffff).unwrap();
        let header76 = [0x42u8; 76];
        backend.scan_range(PowAlgo::Sha3t, &header76, 0, 1 << 20, &target).unwrap();
        let batch = 1u32 << 24;
        let rounds = 16u32;
        let launch = |r: u32| Launch {
            algo: PowAlgo::Sha3t,
            header76: &header76,
            start: r.wrapping_mul(batch),
            count: batch,
            target: &target,
        };
        let t0 = std::time::Instant::now();
        let mut busy = Duration::ZERO;
        for r in 0..rounds {
            backend.enqueue(&launch(r)).unwrap();
            if backend.in_flight() == QUEUE_DEPTH {
                busy += backend.collect().unwrap().busy.unwrap();
            }
        }
        while backend.in_flight() > 0 {
            busy += backend.collect().unwrap().busy.unwrap();
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "{}: {:.2} MH/s queued, device busy {:.1} % of {dt:.2} s",
            backend.name(),
            (batch as f64 * rounds as f64) / dt / 1e6,
            100.0 * busy.as_secs_f64() / dt
        );
    }

    /// Launches queued behind each other must each come back with their own
    /// hits - their own header, range and target - in order, with the slot
    /// buffers reused round-robin and their copies skipped when unchanged.
    #[test]
    #[ignore = "needs an NVIDIA GPU (run inside a --gpus all container)"]
    fn cuda_queued_launches_keep_their_own_inputs() {
        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        let mut seed = 0x51ab_0bc3_u64;
        let mut header = || {
            let mut h = [0u8; 76];
            for b in h.iter_mut() {
                *b = xorshift(&mut seed) as u8;
            }
            h
        };
        let (a, b) = (header(), header());
        // About 32 and 8 hits per 2^16 nonces.
        let easy = crate::consensus::target_for_difficulty(1.0 / (1u64 << 21) as f64);
        let hard = crate::consensus::target_for_difficulty(1.0 / (1u64 << 19) as f64);
        // Two slots in turn: slot 0 gets a, b, a and slot 1 b, a, a. Copies
        // over another header, a skipped copy (slot 1's second a) and a
        // change of target alone.
        let plan = [
            (a, easy, 0u32),
            (b, easy, 5000),
            (b, easy, 10_000),
            (a, easy, 15_000),
            (a, hard, 20_000),
            (a, hard, 25_000),
        ];
        let mut results = Vec::new();
        for (h, t, start) in &plan {
            backend
                .enqueue(&Launch { algo: PowAlgo::Sha3t, header76: h, start: *start, count: 1 << 16, target: t })
                .unwrap();
            if backend.in_flight() == QUEUE_DEPTH {
                results.push(backend.collect().unwrap());
            }
        }
        while backend.in_flight() > 0 {
            results.push(backend.collect().unwrap());
        }
        assert!(backend.collect().is_err(), "nothing left to collect");
        for ((h, t, start), scan) in plan.iter().zip(&results) {
            let mut gpu = scan.hits.clone();
            gpu.sort_unstable();
            let cpu = cpu_hits(h, *start, 1 << 16, t);
            assert_eq!(gpu, cpu, "launch at {start}");
            assert!(!cpu.is_empty() && cpu.len() <= MAX_HITS, "{} hits", cpu.len());
            assert_eq!(scan.reported, cpu.len());
            assert!(scan.busy.is_some_and(|d| d > Duration::ZERO), "no device time");
        }
    }

    /// Device time per BC3Hash at different launch sizes, interleaved in one
    /// process so the card's clock drifts the same for all of them. A launch
    /// ends when its slowest block does; if that tail were costly, short
    /// launches would come out slower per hash.
    ///     BC3_WINDOW=16 cargo test --release -- --ignored cuda_bc3hash_launch_sizes --nocapture
    #[cfg(feature = "bc3hash")]
    #[test]
    #[ignore = "needs an NVIDIA GPU - measurement, run with --nocapture"]
    fn cuda_bc3hash_launch_sizes() {
        let window = std::env::var("BC3_WINDOW").ok().and_then(|w| w.parse().ok()).unwrap_or(16);
        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        backend.prepare_bc3hash(window).expect("BC3Hash passes the self-test");
        let wave = backend.launch_granule(PowAlgo::Bc3Hash);
        let header = [7u8; 76];
        let target = [0u8; 32];
        let sizes = [1u32, 2, 4, 8, 16];
        let mut busy = vec![Duration::ZERO; sizes.len()];
        let mut launches = vec![0u64; sizes.len()];
        let mut start = 0u32;
        for _round in 0..6 {
            for (i, &waves) in sizes.iter().enumerate() {
                // Two in flight, as the worker runs them.
                for _ in 0..3 {
                    let l = Launch { algo: PowAlgo::Bc3Hash, header76: &header, start, count: waves * wave, target: &target };
                    start = start.wrapping_add(waves * wave);
                    backend.enqueue(&l).unwrap();
                    if backend.in_flight() == QUEUE_DEPTH {
                        busy[i] += backend.collect().unwrap().busy.unwrap();
                        launches[i] += 1;
                    }
                }
                while backend.in_flight() > 0 {
                    busy[i] += backend.collect().unwrap().busy.unwrap();
                    launches[i] += 1;
                }
            }
        }
        println!("{} window {window}, wave {wave} nonces:", backend.name());
        for (i, &waves) in sizes.iter().enumerate() {
            let total = f64::from(waves * wave) * launches[i] as f64;
            println!(
                "  {waves:>2} waves/launch: {:>8.1} ms a launch, {:.3} us a hash ({:.1} kH/s)",
                busy[i].as_secs_f64() * 1e3 / launches[i] as f64,
                busy[i].as_secs_f64() * 1e6 / total,
                total / busy[i].as_secs_f64() / 1e3
            );
        }
    }

    /// Compiling the BC3Hash kernel ahead holds up no launch on the mining
    /// context, and leaves it in the driver's cache: the load that follows
    /// takes a second, not the JIT. Run with an empty cache, alone:
    ///     CUDA_CACHE_PATH=/tmp/jit-test cargo test --release -- --ignored cuda_bc3hash_compiled_ahead --nocapture
    #[cfg(feature = "bc3hash")]
    #[test]
    #[ignore = "needs an NVIDIA GPU and an empty JIT cache - run alone, with --nocapture"]
    fn cuda_bc3hash_compiled_ahead() {
        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        let target = compact_to_target(0x1d00ffff).unwrap();
        let header76 = [0x42u8; 76];
        let launch = |start| Launch { algo: PowAlgo::Sha3t, header76: &header76, start, count: 1 << 21, target: &target };
        backend.scan_range(PowAlgo::Sha3t, &header76, 0, 1 << 21, &target).unwrap();
        let jit = std::thread::spawn(|| compile_bc3hash_ahead(0, 16));
        let t0 = std::time::Instant::now();
        let mut worst = Duration::ZERO;
        let mut n = 0u32;
        while !jit.is_finished() || n < 20 {
            let t = std::time::Instant::now();
            backend.enqueue(&launch(n.wrapping_mul(1 << 21))).unwrap();
            backend.collect().unwrap();
            worst = worst.max(t.elapsed());
            n += 1;
        }
        let took = jit.join().unwrap().expect("compiled");
        let t1 = std::time::Instant::now();
        backend.ctx.load_module(cudarc::nvrtc::Ptx::from_src(BC3HASH_PTX[3].1)).expect("loaded");
        let load = t1.elapsed();
        println!(
            "compiled ahead in {:.1} s; meanwhile {n} SHA3-256t launches of 2^21 in {:.1} s, the \
             slowest {:.1} ms; then loaded on the mining context in {:.2} s",
            took.as_secs_f64(),
            t0.elapsed().as_secs_f64(),
            worst.as_secs_f64() * 1e3,
            load.as_secs_f64()
        );
        assert!(worst < Duration::from_millis(500), "a launch waited for the JIT");
        assert!(load < took / 2 || load < Duration::from_secs(2), "the load compiled again");
    }

    /// The same for BC3Hash, through the window-10 kernel - its header and
    /// target live in the slots too.
    #[cfg(feature = "bc3hash")]
    #[test]
    #[ignore = "needs an NVIDIA GPU (run inside a --gpus all container)"]
    fn cuda_bc3hash_queued_launches_match_the_cpu() {
        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        backend.prepare_bc3hash(10).expect("BC3Hash window 10 passes the self-test");
        let mut header80 = [0u8; 80];
        header80[..4].copy_from_slice(&0x2000_1800u32.to_le_bytes());
        let mut a = header80;
        a[4] = 0xa1;
        let mut b = header80;
        b[4] = 0xb2;
        let mut eighth: Target = [0xff; 32];
        eighth[0] = 0x1f;
        let mut half: Target = [0xff; 32];
        half[0] = 0x7f;
        // Two slots in turn: slot 1 gets b twice, the second time with a new
        // target - the slot's own target buffer, for BC3Hash.
        let plan = [(a, eighth, 100u32), (b, eighth, 300), (a, eighth, 500), (b, half, 700), (a, half, 100)];
        let mut results = Vec::new();
        for (h, t, start) in &plan {
            let h76: [u8; 76] = h[..76].try_into().unwrap();
            backend
                .enqueue(&Launch { algo: PowAlgo::Bc3Hash, header76: &h76, start: *start, count: 96, target: t })
                .unwrap();
            if backend.in_flight() == QUEUE_DEPTH {
                results.push(backend.collect().unwrap());
            }
        }
        while backend.in_flight() > 0 {
            results.push(backend.collect().unwrap());
        }
        for ((h, t, start), scan) in plan.iter().zip(&results) {
            let mut h = *h;
            let cpu: Vec<u32> = (*start..*start + 96)
                .filter(|&n| {
                    h[76..80].copy_from_slice(&n.to_le_bytes());
                    hash_meets_target(&bc3hash::bc3hash(&h), t)
                })
                .collect();
            let mut gpu = scan.hits.clone();
            gpu.sort_unstable();
            assert_eq!(gpu, cpu, "launch at {start}");
            assert!(!cpu.is_empty());
        }
    }
}
