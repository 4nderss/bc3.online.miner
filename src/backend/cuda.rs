//! CUDA backend via `cudarc` (driver API).
//!
//! The kernel is PRECOMPILED to PTX at build time (see build.rs) and embedded
//! in the binary. At runtime we therefore only need `nvcuda.dll`/`libcuda.so`
//! - that is, the graphics driver - which JITs the PTX into the card's
//! machine code. NVRTC is NOT used: it ships with the CUDA Toolkit, which end
//! users do not have installed. libcuda is loaded dynamically, so the binary
//! starts even with no NVIDIA driver at all (detection then returns an empty
//! list).

use super::{kernel_lanes, kernel_supports, kernel_target, MiningBackend, MAX_HITS};
use crate::consensus::{PowAlgo, Target};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

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
/// device buffers it reads - the tables (uploaded once) and the per-launch
/// header and target.
#[cfg(feature = "bc3hash")]
struct Bc3Kernel {
    window: u32,
    scan: CudaFunction,
    batch: CudaFunction,
    d_tables: CudaSlice<u64>,
    d_header: CudaSlice<u8>,
    d_target: CudaSlice<u64>,
    /// Nonces the card runs at once: SMs x resident blocks x block size.
    /// At 255 registers a thread only 4 blocks of 64 fit on an SM, so a
    /// launch that is not a whole number of these waves leaves most of the
    /// card idle for its last one - 16384 nonces on an RTX 3050 Ti is 3.2
    /// waves, which cost 14 %.
    wave: u32,
}

/// The host side of the GPU tables, built once per window for every card in
/// the process. Only the latest window is kept: at 16 bits that is 256 MiB.
#[cfg(feature = "bc3hash")]
fn host_tables(window: u32) -> Result<Arc<Vec<u64>>, String> {
    static CACHE: std::sync::Mutex<Option<(u32, Arc<Vec<u64>>)>> = std::sync::Mutex::new(None);
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((w, tables)) = cache.as_ref() {
        if *w == window {
            return Ok(tables.clone());
        }
    }
    *cache = None; // the old window's tables go first, not after
    let tables = Arc::new(bc3hash::gpu_tables(window)?);
    *cache = Some((window, tables.clone()));
    Ok(tables)
}

/// The share target as the kernel takes it: four u64 limbs, least
/// significant first. Our `Target` is 32 big-endian bytes.
#[cfg(feature = "bc3hash")]
fn target_limbs(target: &Target) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let at = 32 - 8 * (i + 1);
        *limb = u64::from_be_bytes(target[at..at + 8].try_into().unwrap());
    }
    limbs
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

pub struct CudaBackend {
    #[cfg(feature = "bc3hash")]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    func: CudaFunction,
    name: String,
    /// Reused GPU buffers (hits: [counter, nonce0, nonce1, ...]).
    d_hits: CudaSlice<u32>,
    d_lanes: CudaSlice<u64>,
    hits_reset: Vec<u32>,
    /// The BC3Hash kernel in use - loaded on demand, for one window at a time.
    #[cfg(feature = "bc3hash")]
    bc3: Option<Bc3Kernel>,
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

        let d_hits = stream
            .alloc_zeros::<u32>(1 + MAX_HITS)
            .map_err(|e| format!("alloc hits: {e}"))?;
        let d_lanes = stream
            .alloc_zeros::<u64>(10)
            .map_err(|e| format!("alloc lanes: {e}"))?;

        Ok(Self {
            #[cfg(feature = "bc3hash")]
            ctx,
            stream,
            func,
            name,
            d_hits,
            d_lanes,
            hits_reset: vec![0u32; 1 + MAX_HITS],
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
        let tables = host_tables(window)?;
        let d_tables = self
            .stream
            .clone_htod(&tables[..])
            .map_err(|e| format!("upload BC3Hash tables ({} MiB): {e}", (tables.len() * 8) >> 20))?;
        let d_header = self.stream.alloc_zeros::<u8>(76).map_err(|e| format!("alloc header: {e}"))?;
        let d_target = self.stream.alloc_zeros::<u64>(4).map_err(|e| format!("alloc target: {e}"))?;
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
        Ok(Bc3Kernel { window, scan, batch, d_tables, d_header, d_target, wave })
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
        let cfg = LaunchConfig {
            grid_dim: (count.div_ceil(BC3_BLOCK_DIM), 1, 1),
            block_dim: (BC3_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&k.batch);
        launch.arg(&k.d_tables).arg(&d_in).arg(&count).arg(&mut d_out);
        unsafe {
            launch.launch(cfg).map_err(|e| format!("bc3hash_batch launch: {e}"))?;
        }
        let out = self.stream.clone_dtoh(&d_out).map_err(|e| format!("dtoh digests: {e}"))?;
        self.stream.synchronize().map_err(|e| format!("synchronize: {e}"))?;
        Ok(out.as_chunks::<32>().0.to_vec())
    }

    /// Check the loaded BC3Hash kernel against the node: the node's header
    /// vectors and random headers against the CPU reference through the
    /// batch entry point, then a range through the scan entry point - the
    /// exact mining path, nonce placement and target compare included -
    /// against the CPU's own hit set. Every architecture and driver JITs the
    /// PTX for itself, so this runs on every card, for every window.
    #[cfg(feature = "bc3hash")]
    fn self_test_bc3(&mut self) -> Result<(), String> {
        let mut seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x0bc3)
            | 1;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut headers = Vec::new();
        let mut expected = Vec::new();
        for (header, digest) in bc3hash::header_vectors() {
            headers.push(header);
            expected.push(digest);
        }
        for _ in 0..14 {
            let mut header = [0u8; 80];
            for b in header.iter_mut() {
                *b = next() as u8;
            }
            header[0..4].copy_from_slice(&0x2000_1800u32.to_le_bytes());
            expected.push(bc3hash::bc3hash(&header));
            headers.push(header);
        }
        let got = self.bc3_batch(&headers)?;
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            if g != e {
                return Err(format!(
                    "header {i} ({}): GPU {} but CPU {}",
                    if i < 2 { "the node's vector" } else { "random" },
                    hex::encode(g),
                    hex::encode(e)
                ));
            }
        }

        // Target 2^253 - 1: about one hash in eight is a hit, so 256 nonces
        // give the scan's compare and hit list something to do both ways.
        let mut header76 = [0u8; 76];
        for b in header76.iter_mut() {
            *b = next() as u8;
        }
        header76[0..4].copy_from_slice(&0x2000_1800u32.to_le_bytes());
        let mut target: Target = [0xff; 32];
        target[0] = 0x1f;
        let start = next() as u32;
        let mut gpu = self.scan_bc3(&header76, start, 256, &target)?;
        let mut header80 = [0u8; 80];
        header80[..76].copy_from_slice(&header76);
        let mut cpu: Vec<u32> = (0..256u32)
            .map(|i| start.wrapping_add(i))
            .filter(|&nonce| {
                header80[76..80].copy_from_slice(&nonce.to_le_bytes());
                crate::consensus::hash_meets_target(&bc3hash::bc3hash(&header80), &target)
            })
            .collect();
        gpu.sort_unstable();
        cpu.sort_unstable();
        if gpu != cpu {
            return Err(format!(
                "scan over 256 nonces: GPU found {} hits, CPU {} (first GPU {:?}, first CPU {:?})",
                gpu.len(),
                cpu.len(),
                gpu.first(),
                cpu.first()
            ));
        }
        Ok(())
    }

    #[cfg(feature = "bc3hash")]
    fn scan_bc3(
        &mut self,
        header76: &[u8; 76],
        start_nonce: u32,
        count: u32,
        target: &Target,
    ) -> Result<Vec<u32>, String> {
        let k = self.bc3.as_mut().ok_or("BC3Hash kernel not loaded")?;
        self.stream
            .memcpy_htod(&header76[..], &mut k.d_header)
            .map_err(|e| format!("htod header: {e}"))?;
        self.stream
            .memcpy_htod(&target_limbs(target)[..], &mut k.d_target)
            .map_err(|e| format!("htod target: {e}"))?;
        self.stream
            .memcpy_htod(&self.hits_reset, &mut self.d_hits)
            .map_err(|e| format!("htod hits: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: (count.div_ceil(BC3_BLOCK_DIM), 1, 1),
            block_dim: (BC3_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let max_hits = MAX_HITS as u32;
        let mut launch = self.stream.launch_builder(&k.scan);
        launch
            .arg(&k.d_tables)
            .arg(&k.d_header)
            .arg(&start_nonce)
            .arg(&count)
            .arg(&k.d_target)
            .arg(&mut self.d_hits)
            .arg(&max_hits);
        unsafe {
            launch.launch(cfg).map_err(|e| format!("bc3hash_scan launch: {e}"))?;
        }
        let hits = self.stream.clone_dtoh(&self.d_hits).map_err(|e| format!("dtoh hits: {e}"))?;
        self.stream.synchronize().map_err(|e| format!("synchronize: {e}"))?;
        let n = (hits[0] as usize).min(MAX_HITS);
        Ok(hits[1..1 + n].to_vec())
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
        // The old window's tables leave the card before the new ones arrive:
        // at 16 bits they are 256 MiB.
        self.bc3 = None;
        let kernel = self.load_bc3(window)?;
        self.bc3 = Some(kernel);
        if let Err(e) = self.self_test_bc3() {
            self.bc3 = None;
            return Err(format!("self-test failed: {e}"));
        }
        Ok(t0.elapsed())
    }

    #[cfg(feature = "bc3hash")]
    fn bc3hash_window(&self) -> Option<u32> {
        self.bc3.as_ref().map(|k| k.window)
    }

    fn launch_granule(&self, algo: PowAlgo) -> u32 {
        match algo {
            #[cfg(feature = "bc3hash")]
            PowAlgo::Bc3Hash => self.bc3.as_ref().map_or(1, |k| k.wave),
            _ => 1,
        }
    }

    fn scan_range(
        &mut self,
        algo: PowAlgo,
        header76: &[u8; 76],
        start_nonce: u32,
        count: u32,
        target: &Target,
    ) -> Result<Vec<u32>, String> {
        #[cfg(feature = "bc3hash")]
        if algo == PowAlgo::Bc3Hash {
            if count == 0 {
                return Ok(vec![]);
            }
            return self.scan_bc3(header76, start_nonce, count, target);
        }
        if !kernel_supports(algo) {
            return Err(format!("no kernel for {}", algo.name()));
        }
        if count == 0 {
            return Ok(vec![]);
        }
        let lanes = kernel_lanes(header76);
        let t3 = kernel_target(target);

        self.stream
            .memcpy_htod(&lanes, &mut self.d_lanes)
            .map_err(|e| format!("htod lanes: {e}"))?;
        self.stream
            .memcpy_htod(&self.hits_reset, &mut self.d_hits)
            .map_err(|e| format!("htod hits: {e}"))?;

        let cfg = LaunchConfig {
            grid_dim: (count.div_ceil(BLOCK_DIM), 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let max_hits = MAX_HITS as u32;
        let mut launch = self.stream.launch_builder(&self.func);
        launch
            .arg(&self.d_lanes)
            .arg(&start_nonce)
            .arg(&count)
            .arg(&t3)
            .arg(&mut self.d_hits)
            .arg(&max_hits);
        unsafe {
            launch.launch(cfg).map_err(|e| format!("kernel-launch: {e}"))?;
        }

        let hits = self
            .stream
            .clone_dtoh(&self.d_hits)
            .map_err(|e| format!("dtoh hits: {e}"))?;
        self.stream
            .synchronize()
            .map_err(|e| format!("synchronize: {e}"))?;
        let n = (hits[0] as usize).min(MAX_HITS);
        Ok(hits[1..1 + n].to_vec())
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
}
