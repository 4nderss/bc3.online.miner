//! CUDA backend via `cudarc` (driver API).
//!
//! The BC3Hash kernel is PRECOMPILED to PTX at build time (see build.rs), once
//! per table window, and embedded in the binary. At runtime we therefore only
//! need `nvcuda.dll`/`libcuda.so` - that is, the graphics driver - which JITs
//! the PTX into the card's machine code. NVRTC is NOT used: it ships with the
//! CUDA Toolkit, which end users do not have installed. libcuda is loaded
//! dynamically, so the binary starts even with no NVIDIA driver at all
//! (detection then returns an empty list).
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

use super::{kernel_supports, Launch, MiningBackend, Scan, MAX_HITS, QUEUE_DEPTH};
use crate::consensus::PowAlgo;
use cudarc::driver::{
    sys, CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, LaunchConfig,
    PinnedHostSlice, PushKernelArg,
};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

/// The BC3Hash kernel, one PTX per secp256k1 table window. Every card starts
/// at 16; a card with the memory for them is moved to a bigger window while it
/// mines, if it hashes faster there (src/tuning.rs).
const BC3HASH_PTX: [(u32, &str); 6] = [
    (16, include_str!(env!("BC3_BC3HASH_W16_PTX"))),
    (18, include_str!(env!("BC3_BC3HASH_W18_PTX"))),
    (19, include_str!(env!("BC3_BC3HASH_W19_PTX"))),
    (20, include_str!(env!("BC3_BC3HASH_W20_PTX"))),
    (22, include_str!(env!("BC3_BC3HASH_W22_PTX"))),
    (24, include_str!(env!("BC3_BC3HASH_W24_PTX"))),
];

/// The kernel that builds the curve tables on the card, one row at a time,
/// for every window (`bc3hash_build_row`; the source built with
/// `-DBC3_BUILD_ONLY`).
const BUILD_PTX: &str = include_str!(env!("BC3_BC3HASH_BUILD_PTX"));

/// Threads per block for BC3Hash. Must stay 64: the kernel declares
/// `__launch_bounds__(64, 6)` - its register budget - and a larger block does
/// not launch. A small block also keeps the tail of a launch short.
const BC3_BLOCK_DIM: u32 = 64;

/// The table builder's launches: blocks of 64, eight an SM - 96-106
/// registers a thread let ten fit. Each thread makes a stretch of a row's
/// points, so any number of threads builds all of them, and the fewer the
/// threads the longer the stretches over which a thread's first point - 24
/// doublings - is paid for.
const BUILD_BLOCK_DIM: u32 = 64;
const BUILD_BLOCKS_PER_SM: u32 = 8;

/// Card memory a window's tables must leave free: the launch buffers, and
/// the hash kernel's local memory before its first launch has taken it. The
/// tuning keeps a GiB free (`tuning::largest_fitting`); this is the floor.
const BUILD_SPARE: u64 = 256 << 20;

/// The oldest driver that can load the kernel, as `cuDriverGetVersion` reports
/// it (1000 x major + 10 x minor of the CUDA version it supports): the PTX
/// declares ISA 8.0 (`MAX_PTX_ISA` in build.rs), which came with CUDA 12.0 -
/// driver R525.
const MIN_DRIVER_CUDA: i32 = 12_000;

/// One loaded BC3Hash kernel: its window, the two entry points, and the
/// tables they read (uploaded once).
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
    /// The header and the target limbs.
    header: CudaSlice<u8>,
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
    /// What `header` and `target` hold, so the copies that would change
    /// nothing - nearly all of them - are skipped.
    header_of: Option<[u8; 76]>,
    target_of: Option<[u64; 4]>,
}

impl Slot {
    fn new(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Result<Self, String> {
        let alloc = |what: &str, e| format!("alloc {what}: {e}");
        Ok(Self {
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
            header_of: None,
            target_of: None,
        })
    }

    /// Forget what the device buffers hold - after a failed or abandoned
    /// launch nobody knows.
    fn forget(&mut self) {
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

/// Refuse a driver too old to load the kernel, when the card is opened.
///
/// The PTX is loaded only when the first BC3Hash job comes, and a load that
/// fails there just leaves the card idle. Failing the open instead sends the
/// card to OpenCL (`open_backend_with_fallback`), whose runtime ships with
/// the same driver and compiles the kernel from source. Until the SHA3-256t
/// kernel was removed its PTX did that check, loaded at every open.
fn check_driver() -> Result<(), String> {
    let mut version: std::ffi::c_int = 0;
    // SAFETY: a plain query into a local; the library is loaded (the context
    // has been created by now).
    unsafe { sys::cuDriverGetVersion(&mut version) }
        .result()
        .map_err(|e| format!("cuDriverGetVersion: {e}"))?;
    if version < MIN_DRIVER_CUDA {
        return Err(format!(
            "the NVIDIA driver supports CUDA {}.{}, and the kernel needs 12.0 (driver R525) or \
             newer - update the driver",
            version / 1000,
            version % 1000 / 10
        ));
    }
    Ok(())
}

/// The kernels' machine code, by PTX and architecture (build.rs).
mod embedded {
    include!(concat!(env!("OUT_DIR"), "/sass.rs"));
}

/// The machine code of the PTX named `stem` for a card of architecture `sm`
/// (major * 10 + minor), and the architecture it was made for: the card's
/// own, or the newest earlier minor of its major - machine code runs on the
/// later minors of its own major (sm_86 code on sm_87), never on an earlier
/// one or another major. None: the card JITs the PTX.
fn pick_sass(
    sass: &'static [(&'static str, u32, &'static [u8])],
    stem: &str,
    sm: u32,
) -> Option<(u32, &'static [u8])> {
    sass.iter()
        .filter(|(name, s, _)| *name == stem && s / 10 == sm / 10 && *s <= sm)
        .max_by_key(|(_, s, _)| *s)
        .map(|(_, s, code)| (*s, *code))
}

/// The PTX of the kernel for `window`.
fn bc3hash_ptx(window: u32) -> Result<&'static str, String> {
    BC3HASH_PTX
        .iter()
        .find(|(w, _)| *w == window)
        .map(|(_, ptx)| *ptx)
        .ok_or_else(|| format!("no BC3Hash kernel for a window of {window} bits"))
}

/// The name of the PTX for `window`, as build.rs and tools/ptx.sh have it.
fn bc3hash_stem(window: u32) -> String {
    format!("bc3hash_w{window}")
}

/// Compile the BC3Hash kernel for `window` into the driver's JIT cache, so
/// that loading it later - when the window tuning switches the card - comes
/// from the cache in about a second instead of 10-60 s of JIT. Only for a
/// card this build has no machine code for (`pick_sass`).
///
/// On a context of its own: a JIT on the mining context blocks every launch
/// on it until it is done (window 16 on an i9-12900HK: 16.7 s, and one launch
/// waited all of it), while on a second context nothing waits (1234 launches
/// meanwhile, the slowest 28 ms). The context goes when this returns, and the
/// memory it held with it.
pub fn compile_bc3hash_ahead(index: usize, window: u32) -> Result<Duration, String> {
    let t0 = std::time::Instant::now();
    let ptx = bc3hash_ptx(window)?;
    let ctx = CudaContext::new_non_primary(index, 0).map_err(|e| format!("context: {e}"))?;
    ctx.load_module(cudarc::nvrtc::Ptx::from_src(ptx))
        .map_err(|e| format!("could not load the BC3Hash kernel (window {window}): {e}"))?;
    Ok(t0.elapsed())
}

pub struct CudaBackend {
    /// The device's CUDA ordinal, for a context of its own to compile on.
    index: usize,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    name: String,
    /// One per launch that may be in flight, taken in turn.
    slots: Vec<Slot>,
    /// Slots of the launches in flight, oldest first.
    queued: VecDeque<usize>,
    /// The slot the next launch takes.
    next_slot: usize,
    /// The BC3Hash kernel in use - loaded on demand, for one window at a time.
    bc3: Option<Bc3Kernel>,
    /// The table builder, loaded with the first tables and kept.
    builder: Option<CudaFunction>,
    /// The card's architecture, major * 10 + minor: which machine code it
    /// takes (`pick_sass`).
    sm: u32,
    /// Its multiprocessors.
    sms: u32,
    /// Whether the card has been told that it JITs the PTX - said once.
    told_jit: bool,
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
        check_driver()?;
        let name = ctx.name().unwrap_or_else(|_| format!("CUDA device {index}"));
        let stream = ctx.default_stream();
        let slots = (0..QUEUE_DEPTH).map(|_| Slot::new(&ctx, &stream)).collect::<Result<Vec<_>, _>>()?;
        use sys::CUdevice_attribute::{
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as MAJOR,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as MINOR,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as SMS,
        };
        let sm = (ctx.attribute(MAJOR).unwrap_or(0) * 10 + ctx.attribute(MINOR).unwrap_or(0)).max(0) as u32;
        let sms = ctx.attribute(SMS).unwrap_or(1).max(1) as u32;
        Ok(Self {
            index,
            ctx,
            stream,
            name,
            slots,
            queued: VecDeque::new(),
            next_slot: 0,
            bc3: None,
            builder: None,
            sm,
            sms,
            told_jit: false,
        })
    }

    /// The module of the PTX named `stem`: the machine code built for this
    /// card's architecture where this build has it, else the PTX, which the
    /// driver JITs - 20-60 s for a window's kernel, unless its cache has the
    /// result.
    fn load_module(&mut self, stem: &str, ptx: &'static str) -> Result<Arc<cudarc::driver::CudaModule>, String> {
        match pick_sass(embedded::SASS, stem, self.sm) {
            Some((sm, packed)) => match miniz_oxide::inflate::decompress_to_vec(packed) {
                Ok(code) => match self.ctx.load_module(cudarc::nvrtc::Ptx::from_binary(code)) {
                    Ok(module) => return Ok(module),
                    Err(e) => eprintln!(
                        "[gpu] {}: the machine code for sm_{sm} would not load ({e}) - the driver \
                         compiles the kernel instead",
                        self.name
                    ),
                },
                Err(e) => eprintln!("[gpu] {}: the machine code for sm_{sm} is damaged ({e:?})", self.name),
            },
            None if !self.told_jit => crate::human!(
                "[gpu] {}: this build has no machine code for sm_{} - the driver compiles the \
                 kernel, which takes up to a minute where it has not cached it",
                self.name,
                self.sm
            ),
            None => {}
        }
        self.told_jit = true;
        self.ctx
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|e| format!("could not load the BC3Hash kernel {stem}: {e}"))
    }

    /// Load the BC3Hash kernel for `window` and build its tables.
    fn load_bc3(&mut self, window: u32) -> Result<Bc3Kernel, String> {
        let module = self.load_module(&bc3hash_stem(window), bc3hash_ptx(window)?)?;
        let scan = module
            .load_function("bc3hash_scan")
            .map_err(|e| format!("load_function bc3hash_scan: {e}"))?;
        let batch = module
            .load_function("bc3hash_batch")
            .map_err(|e| format!("load_function bc3hash_batch: {e}"))?;
        let d_tables = self.build_tables(window)?;
        let per_sm = scan
            .occupancy_max_active_blocks_per_multiprocessor(BC3_BLOCK_DIM, 0, None)
            .unwrap_or(1)
            .max(1);
        let wave = self.sms * per_sm * BC3_BLOCK_DIM;
        Ok(Bc3Kernel { window, scan, batch, d_tables, wave })
    }

    /// The table builder, loaded with the first tables and kept.
    fn builder(&mut self) -> Result<CudaFunction, String> {
        if let Some(f) = &self.builder {
            return Ok(f.clone());
        }
        let f = self
            .load_module("bc3hash_build", BUILD_PTX)?
            .load_function("bc3hash_build_row")
            .map_err(|e| format!("load_function bc3hash_build_row: {e}"))?;
        self.builder = Some(f.clone());
        Ok(f)
    }

    /// The tables of `window`, built on the card: the power tables from the
    /// host, then every row of the four curves by `bc3hash_build_row`, from
    /// the row's base 2^pos * P, which the host works out. Until 2.1 the host
    /// built all of it, on every core, and uploaded it: 18.7 s for window 20
    /// on 20 threads and 3.3 GiB of host memory, a laptop's GPU down from 40
    /// to 27 W meanwhile - and window 24's 28 GiB out of reach. On an RTX 3050
    /// Ti window 20 takes 0.4 s, and the host's work is a few dozen scalar
    /// multiplications. Some entries of every row are checked against the
    /// host's own (`check_tables`).
    fn build_tables(&mut self, window: u32) -> Result<CudaSlice<u64>, String> {
        let rows = crate::tuning::rows(window).ok_or_else(|| format!("no tables for a window of {window} bits"))?;
        let build = self.builder()?;
        let power = bc3hash::gpu_power_tables()?;
        let curve_u64: usize = rows.iter().map(|r| r.places as usize * 8).sum();
        let total = power.len() + 4 * curve_u64;
        // The card's own memory, not what the driver can lend: on Windows a
        // CUDA allocation the card has no room for falls back to system memory
        // (NVIDIA's "CUDA - Sysmem Fallback Policy", on by default since R536),
        // and window 24's 28 GiB on an RTX 4090 (24 GB) did not fail - the
        // card sat for minutes over tables half in host memory instead. The
        // old tables went before this; the synchronize gives their memory back.
        self.stream.synchronize().map_err(|e| format!("synchronize: {e}"))?;
        let need = (total * 8) as u64;
        if let Ok((free, _)) = self.ctx.mem_get_info() {
            if need + BUILD_SPARE > free as u64 {
                return Err(format!(
                    "window {window} needs {} MiB of tables, and the card has {} MiB free",
                    need >> 20,
                    free as u64 >> 20
                ));
            }
        }
        // Zeroed: the places past the last point of a uniform window's last
        // row hold zeros, as the host's tables have them.
        let mut tables = self
            .stream
            .alloc_zeros::<u64>(total)
            .map_err(|e| format!("could not allocate {} MiB for the BC3Hash tables: {e}", (total * 8) >> 20))?;
        self.stream
            .memcpy_htod(&power[..], &mut tables.slice_mut(..power.len()))
            .map_err(|e| format!("htod power tables: {e}"))?;
        let mut bases = Vec::with_capacity(4 * rows.len() * 8);
        for curve in 0..4 {
            for row in &rows {
                bases.extend_from_slice(&bc3hash::gpu_point(curve, row.pos, 1)?);
            }
        }
        let bases = self.stream.clone_htod(&bases[..]).map_err(|e| format!("htod row bases: {e}"))?;
        let most = self.sms * BUILD_BLOCKS_PER_SM;
        let mut at = power.len();
        for curve in 0..4 {
            for (i, row) in rows.iter().enumerate() {
                let places = row.places as usize * 8;
                let count = row.points as u32;
                let base_at = (curve * rows.len() + i) * 8;
                let mut out = tables.slice_mut(at..at + places);
                let base = bases.slice(base_at..base_at + 8);
                let blocks = count.div_ceil(BUILD_BLOCK_DIM).clamp(1, most);
                let mut launch = self.stream.launch_builder(&build);
                launch.arg(&mut out).arg(&base).arg(&count);
                let config = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (BUILD_BLOCK_DIM, 1, 1), shared_mem_bytes: 0 };
                // SAFETY: the kernel writes `count` points of 8 limbs into
                // `out`, which has room for `places >= count` of them, and
                // reads the 8 limbs of `base`.
                unsafe { launch.launch(config) }.map_err(|e| format!("bc3hash_build_row launch: {e}"))?;
                at += places;
            }
        }
        self.stream.synchronize().map_err(|e| format!("building the BC3Hash tables: {e}"))?;
        self.check_tables(&tables, power.len(), &rows)?;
        Ok(tables)
    }

    /// The first, a middle and the last point of every row the card built,
    /// against the host's b * 2^pos * P: a row built from the wrong base, at
    /// the wrong place or not to its end shows here by name, where the
    /// self-test after it could only say that hashes come out wrong.
    fn check_tables(&self, tables: &CudaSlice<u64>, power: usize, rows: &[crate::tuning::Row]) -> Result<(), String> {
        let curve_u64: usize = rows.iter().map(|r| r.places as usize * 8).sum();
        let mut samples = Vec::new();
        for curve in 0..4 {
            let mut at = power + curve * curve_u64;
            for (i, row) in rows.iter().enumerate() {
                for b in [1, row.points / 2 + 1, row.points] {
                    let e = at + (b as usize - 1) * 8;
                    let got = self.stream.clone_dtoh(&tables.slice(e..e + 8)).map_err(|e| format!("dtoh: {e}"))?;
                    samples.push((curve, i, row.pos, b, got));
                }
                at += row.places as usize * 8;
            }
        }
        self.stream.synchronize().map_err(|e| format!("synchronize: {e}"))?;
        for (curve, i, pos, b, got) in samples {
            if got[..] != bc3hash::gpu_point(curve, pos, b)? {
                return Err(format!("the tables built on the card are wrong: curve {curve}, row {i}, point {b}"));
            }
        }
        Ok(())
    }

    /// BC3Hash of 80-byte headers on the GPU, through the kernel's batch
    /// entry point - the self-test's view of the kernel.
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
            algo @ (PowAlgo::Sha256d | PowAlgo::Sha3t) => Err(format!("no kernel for {}", algo.name())),
        }
    }

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
}

/// One thread per nonce, in blocks of `block`; the kernel ignores the threads
/// past `count` in the last block.
fn launch_config(count: u32, block: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (count.div_ceil(block), 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }
}

impl super::bc3::Bc3Ops for CudaBackend {
    fn bc3_batch(&mut self, headers: &[[u8; 80]]) -> Result<Vec<[u8; 32]>, String> {
        CudaBackend::bc3_batch(self, headers)
    }
}

impl MiningBackend for CudaBackend {
    fn name(&self) -> String {
        format!("CUDA: {}", self.name)
    }

    fn has_kernel(&self, algo: PowAlgo) -> bool {
        kernel_supports(algo)
    }

    fn supports(&self, algo: PowAlgo) -> bool {
        match algo {
            PowAlgo::Bc3Hash => self.bc3.is_some(),
            PowAlgo::Sha256d | PowAlgo::Sha3t => false,
        }
    }

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

    fn bc3hash_window(&self) -> Option<u32> {
        self.bc3.as_ref().map(|k| k.window)
    }

    fn memory(&self) -> Option<(u64, u64)> {
        self.ctx.mem_get_info().ok().map(|(free, total)| (free as u64, total as u64))
    }

    fn compile_bc3hash_ahead(&self, window: u32) -> Option<std::sync::mpsc::Receiver<()>> {
        // Machine code for the card: the switch loads it as it is.
        if pick_sass(embedded::SASS, &bc3hash_stem(window), self.sm).is_some() {
            return None;
        }
        let (index, name) = (self.index, self.name());
        let (tx, rx) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new().name(format!("jit-{index}")).spawn(move || {
            // Whatever happens, say it is over: the tuning waits for it.
            struct Over(std::sync::mpsc::Sender<()>);
            impl Drop for Over {
                fn drop(&mut self) {
                    let _ = self.0.send(());
                }
            }
            let _over = Over(tx);
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
        match spawned {
            Ok(_) => Some(rx),
            Err(e) => {
                eprintln!("[gpu] could not start the BC3Hash compile thread: {e}");
                None
            }
        }
    }

    fn launch_granule(&self, algo: PowAlgo) -> u32 {
        match algo {
            PowAlgo::Bc3Hash => self.bc3.as_ref().map_or(1, |k| k.wave),
            PowAlgo::Sha256d | PowAlgo::Sha3t => 1,
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
    use crate::consensus::{hash_meets_target, Target};

    /// Device time per BC3Hash at different launch sizes, interleaved in one
    /// process so the card's clock drifts the same for all of them. A launch
    /// ends when its slowest block does; if that tail were costly, short
    /// launches would come out slower per hash.
    ///     BC3_WINDOW=16 cargo test --release -- --ignored cuda_bc3hash_launch_sizes --nocapture
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

    #[test]
    fn machine_code_is_the_cards_own_or_its_majors_earlier_minor() {
        static HAVE: [(&str, u32, &[u8]); 5] = [
            ("bc3hash_w16", 61, b"61"),
            ("bc3hash_w16", 75, b"75"),
            ("bc3hash_w16", 86, b"86"),
            ("bc3hash_w16", 89, b"89"),
            ("bc3hash_w16", 120, b"120"),
        ];
        let pick = |sm| pick_sass(&HAVE, "bc3hash_w16", sm).map(|(s, _)| s);
        assert_eq!(pick(86), Some(86));
        assert_eq!(pick(89), Some(89));
        assert_eq!(pick(87), Some(86), "Jetson Orin runs Ampere's code");
        assert_eq!(pick(121), Some(120));
        assert_eq!(pick(62), Some(61));
        assert_eq!(pick(80), None, "an A100 cannot run sm_86 code");
        assert_eq!(pick(90), None);
        assert_eq!(pick(70), None, "nor a V100 Turing's");
        assert_eq!(pick(60), None);
        assert_eq!(pick_sass(&HAVE, "bc3hash_w18", 86), None, "nor one kernel another's");
    }

    /// What build.rs embedded unpacks to cubins, for kernels there is a PTX
    /// for. Empty in a build without tools/sass.sh's output.
    #[test]
    fn the_embedded_machine_code_unpacks_to_cubins() {
        for (stem, sm, packed) in embedded::SASS {
            let known = *stem == "bc3hash_build" || BC3HASH_PTX.iter().any(|(w, _)| bc3hash_stem(*w) == *stem);
            assert!(known, "{stem}");
            let code = miniz_oxide::inflate::decompress_to_vec(packed).expect("inflate");
            assert_eq!(&code[..4], b"\x7fELF", "{stem}, sm_{sm}");
        }
    }

    /// The tables a card builds itself are bit for bit the host's, the last
    /// row of a window that does not divide 256 included (its zeros too),
    /// on a window small enough to compare whole and on 16.
    #[test]
    #[ignore = "needs an NVIDIA GPU (run inside a --gpus all container)"]
    fn cuda_tables_built_on_the_card_are_the_hosts() {
        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        for window in [5, 9, 16] {
            let t0 = std::time::Instant::now();
            let built = backend.build_tables(window).unwrap_or_else(|e| panic!("window {window}: {e}"));
            let took = t0.elapsed();
            let card = backend.stream.clone_dtoh(&built).expect("dtoh");
            backend.stream.synchronize().expect("synchronize");
            let host = bc3hash::gpu_tables(window).unwrap();
            assert_eq!(card.len(), host.len(), "window {window}");
            assert!(card == host, "window {window}: the card's tables differ from the host's");
            println!("window {window}: {} MiB, built on the card in {:.3} s", (card.len() * 8) >> 20, took.as_secs_f64());
        }
    }

    /// Every window the card has the memory for passes the self-test - the
    /// node's header vectors, random headers against the CPU reference, a
    /// scan against the CPU's hit set - and the card switches between them
    /// and back to 16 on one context, as the tuning does.
    #[test]
    #[ignore = "needs an NVIDIA GPU (run inside a --gpus all container)"]
    fn cuda_bc3hash_every_window_passes_the_self_test() {
        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        let windows: Vec<u32> = BC3HASH_PTX.iter().map(|(w, _)| *w).chain([16]).collect();
        let mut tested = 0;
        for window in windows {
            backend.prepare_bc3hash(16).expect("window 16 first");
            let (free, _) = backend.memory().expect("memory");
            // The window's tables replace 16's, which are held now.
            let free = free + crate::tuning::table_bytes(16);
            if crate::tuning::table_bytes(window) + (512 << 20) > free {
                println!("window {window}: skipped, {} MiB free", free >> 20);
                continue;
            }
            let took = backend.prepare_bc3hash(window).unwrap_or_else(|e| panic!("window {window}: {e}"));
            assert_eq!(backend.bc3hash_window(), Some(window));
            println!("window {window}: self-test passed ({:.1} s)", took.as_secs_f64());
            tested += 1;
        }
        assert!(tested >= 2, "window 16, tested first and last, must always fit");
    }

    /// Launches queued behind each other must each come back with their own
    /// hits - their own header, range and target - in order, through the
    /// window-16 kernel, with the slot buffers reused round-robin and their
    /// copies skipped when unchanged.
    #[test]
    #[ignore = "needs an NVIDIA GPU (run inside a --gpus all container)"]
    fn cuda_bc3hash_queued_launches_match_the_cpu() {
        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        backend.prepare_bc3hash(16).expect("BC3Hash window 16 passes the self-test");
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
        // target - the slot's own target buffer.
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
        assert!(backend.collect().is_err(), "nothing left to collect");
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
            assert_eq!(scan.reported, cpu.len());
            assert!(scan.busy.is_some_and(|d| d > Duration::ZERO), "no device time");
        }
    }
}

