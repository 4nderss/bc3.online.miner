//! CUDA-backend via `cudarc` (driver-API).
//!
//! Kerneln är FÖRKOMPILERAD till PTX vid bygget (se build.rs) och inbäddad i
//! binären. Runtime behöver därför bara `nvcuda.dll`/`libcuda.so` — alltså
//! grafikdrivrutinen — som JIT:ar PTX:en till kortets maskinkod. NVRTC
//! används INTE: den ligger i CUDA Toolkit som slutanvändare inte har
//! installerat. libcuda laddas dynamiskt, så binären startar även helt utan
//! NVIDIA-drivrutin (detekteringen returnerar då en tom lista).

use super::{header_lanes, target_limbs, MiningBackend, MAX_HITS};
use crate::consensus::Target;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

const BLOCK_DIM: u32 = 256;

/// Förkompilerad kernel-PTX (byggd av build.rs).
const SHA3T_PTX: &str = include_str!(env!("BC3_SHA3T_PTX"));

/// Lista CUDA-enheter; tom lista om drivrutin/enheter saknas.
///
/// cudarc:s dynamiska laddare PANIKAR om libcuda/nvcuda inte finns på
/// systemet, så första kontakten görs bakom catch_unwind (med tystad
/// panic-hook) — utan drivrutin blir resultatet en tom lista i stället
/// för en krasch. Kräver att panic = "abort" INTE används (se Cargo.toml).
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
            let name = ctx.name().unwrap_or_else(|_| format!("CUDA-enhet {i}"));
            Some(super::GpuDevice::Cuda { index: i, name })
        })
        .collect()
}

pub struct CudaBackend {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    name: String,
    /// Återanvända GPU-buffertar (hits: [räknare, nonce0, nonce1, ...]).
    d_hits: CudaSlice<u32>,
    d_lanes: CudaSlice<u64>,
    hits_reset: Vec<u32>,
}

impl CudaBackend {
    pub fn new(index: usize) -> Result<Self, String> {
        let ctx = CudaContext::new(index).map_err(|e| format!("CudaContext::new: {e}"))?;
        let name = ctx.name().unwrap_or_else(|_| format!("CUDA-enhet {index}"));
        let stream = ctx.default_stream();

        // Ladda den inbäddade PTX:en; drivrutinen JIT:ar den för kortet.
        let ptx = cudarc::nvrtc::Ptx::from_src(SHA3T_PTX);
        let module = ctx.load_module(ptx).map_err(|e| {
            format!("kunde inte ladda kernel-PTX (drivrutin för gammal?): {e}")
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
            stream,
            func,
            name,
            d_hits,
            d_lanes,
            hits_reset: vec![0u32; 1 + MAX_HITS],
        })
    }
}

impl MiningBackend for CudaBackend {
    fn name(&self) -> String {
        format!("CUDA: {}", self.name)
    }

    fn scan_range(
        &mut self,
        header76: &[u8; 76],
        start_nonce: u32,
        count: u32,
        target: &Target,
    ) -> Result<Vec<u32>, String> {
        if count == 0 {
            return Ok(vec![]);
        }
        let lanes = header_lanes(header76);
        let t = target_limbs(target);

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
            .arg(&t[0])
            .arg(&t[1])
            .arg(&t[2])
            .arg(&t[3])
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

    /// CPU-facit: alla noncer i intervallet vars sha3t-hash ≤ target.
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

    // Körs i CUDA-container: cargo test -- --ignored --nocapture
    #[test]
    #[ignore = "kräver NVIDIA-GPU (kör i --gpus all-container)"]
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
        // Obs: genesis-blocket är SHA256d-minat; här verifierar vi bara att
        // GPU:ns sha3t över headern matchar CPU:ns sha3t bit för bit, med
        // target = exakt genesis-headerns sha3t-hash (garanterad träff).
        // Hashen är ett "stort" tal, så nästan alla noncer träffar också —
        // intervallet hålls ≤ MAX_HITS så hela träffmängden kan jämföras.
        let hash = sha3t(&ser);
        let mut target = [0u8; 32];
        for (i, b) in hash.iter().rev().enumerate() {
            target[i] = *b; // target är big-endian av det lilla-endian-talet
        }

        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        println!("backend: {}", backend.name());
        let count = MAX_HITS as u32;
        let start = genesis.nonce - count / 2;
        let mut gpu = backend.scan_range(&header76, start, count, &target).unwrap();
        let mut cpu = cpu_hits(&header76, start, count, &target);
        gpu.sort_unstable();
        cpu.sort_unstable();
        assert!(gpu.contains(&genesis.nonce), "genesis-noncen saknas i {gpu:?}");
        assert_eq!(gpu, cpu);
    }

    #[test]
    #[ignore = "kräver NVIDIA-GPU (kör i --gpus all-container)"]
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

            // Target = MINSTA hashen i intervallet (CPU-beräknad) ⇒ exakt en
            // garanterad träff; varje keccak-bitfel någonstans i intervallet
            // som ger en mindre hash skulle synas som extra träff.
            let mut header80 = [0u8; 80];
            header80[..76].copy_from_slice(&header76);
            let (pick, min_hash) = (0..count)
                .map(|i| {
                    let nonce = start.wrapping_add(i);
                    header80[76..80].copy_from_slice(&nonce.to_le_bytes());
                    let mut key = sha3t(&header80);
                    key.reverse(); // big-endian ⇒ jämförbar lexikografiskt
                    (nonce, key)
                })
                .min_by(|a, b| a.1.cmp(&b.1))
                .unwrap();
            let target: Target = min_hash; // targets är big-endian-bytes

            let mut gpu = backend.scan_range(&header76, start, count, &target).unwrap();
            let mut cpu = cpu_hits(&header76, start, count, &target);
            gpu.sort_unstable();
            cpu.sort_unstable();
            assert!(gpu.contains(&pick), "runda {round}: vald nonce saknas");
            assert_eq!(gpu, cpu, "runda {round}: GPU- och CPU-mängderna skiljer");
        }
    }

    #[test]
    #[ignore = "kräver NVIDIA-GPU (kör i --gpus all-container)"]
    fn cuda_low_diff_share_targets() {
        // Realistiskt pool-scenario: lågt share-target, kontrollera exakt
        // träffmängd över ett större intervall.
        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        let target = crate::consensus::target_for_difficulty(0.00001);
        let mut seed = 0x51ab_e77a_u64;
        let mut header76 = [0u8; 76];
        for b in header76.iter_mut() {
            *b = xorshift(&mut seed) as u8;
        }
        let count = 1u32 << 20;
        let mut gpu = backend.scan_range(&header76, 0, count, &target).unwrap();
        let mut cpu = cpu_hits(&header76, 0, count, &target);
        gpu.sort_unstable();
        cpu.sort_unstable();
        assert_eq!(gpu, cpu);
        assert!(!cpu.is_empty(), "testet bör ge minst någon träff (diff 1e-5, 2^20 noncer)");
    }

    #[test]
    #[ignore = "kräver NVIDIA-GPU — hashrate-mätning, kör med --nocapture"]
    fn cuda_hashrate() {
        let mut backend = CudaBackend::new(0).expect("CUDA-backend");
        let target = compact_to_target(0x1d00ffff).unwrap(); // få träffar
        let header76 = [0x42u8; 76];
        // Uppvärmning.
        backend.scan_range(&header76, 0, 1 << 20, &target).unwrap();
        let batch = 1u32 << 24;
        let rounds = 16u32;
        let t0 = std::time::Instant::now();
        for r in 0..rounds {
            backend
                .scan_range(&header76, r.wrapping_mul(batch), batch, &target)
                .unwrap();
        }
        let dt = t0.elapsed().as_secs_f64();
        let rate = (batch as f64 * rounds as f64) / dt;
        println!(
            "{}: {:.2} MH/s ({} hashar på {dt:.2}s)",
            backend.name(),
            rate / 1e6,
            batch as u64 * rounds as u64
        );
    }
}
