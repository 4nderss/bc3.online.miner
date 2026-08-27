//! OpenCL-backend via `opencl3` (NVIDIA/AMD/Intel). Delar kernelkälla med
//! CUDA-backenden (src/kernels/sha3t.cl) — bitexaktheten följer därmed av
//! CUDA-testerna; runtime-test av just OpenCL-vägen kräver native Windows
//! (ingen OpenCL-GPU-runtime i WSL/Docker), se test nederst.

use super::{header_lanes, target_limbs, MiningBackend, KERNEL_SOURCE, MAX_HITS};
use crate::consensus::Target;
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::CL_BLOCKING;
use std::ptr;

const LOCAL_SIZE: usize = 256;

/// Lista alla GPU-enheter över alla OpenCL-plattformar; tom vid fel.
pub fn list_devices() -> Vec<super::GpuDevice> {
    let Ok(platforms) = get_platforms() else {
        return vec![];
    };
    let mut out = Vec::new();
    for (pi, platform) in platforms.iter().enumerate() {
        let Ok(ids) = platform.get_devices(CL_DEVICE_TYPE_GPU) else {
            continue;
        };
        for (di, id) in ids.into_iter().enumerate() {
            let name = Device::new(id)
                .name()
                .unwrap_or_else(|_| format!("OpenCL-enhet {pi}.{di}"));
            out.push(super::GpuDevice::Opencl { platform: pi, device: di, name });
        }
    }
    out
}

pub struct OpenClBackend {
    queue: CommandQueue,
    kernel: Kernel,
    name: String,
    d_lanes: Buffer<u64>,
    d_hits: Buffer<u32>,
    hits_reset: Vec<u32>,
    // Context måste överleva buffertarna.
    _context: Context,
}

impl OpenClBackend {
    pub fn new(platform_idx: usize, device_idx: usize) -> Result<Self, String> {
        let platforms = get_platforms().map_err(|e| format!("get_platforms: {e}"))?;
        let platform = platforms
            .get(platform_idx)
            .ok_or_else(|| format!("OpenCL-plattform {platform_idx} saknas"))?;
        let ids = platform
            .get_devices(CL_DEVICE_TYPE_GPU)
            .map_err(|e| format!("get_devices: {e}"))?;
        let id = *ids
            .get(device_idx)
            .ok_or_else(|| format!("OpenCL-enhet {platform_idx}.{device_idx} saknas"))?;
        let device = Device::new(id);
        let name = device
            .name()
            .unwrap_or_else(|_| format!("OpenCL-enhet {platform_idx}.{device_idx}"));

        let context = Context::from_device(&device).map_err(|e| format!("Context: {e}"))?;
        let program = Program::create_and_build_from_source(&context, KERNEL_SOURCE, "")
            .map_err(|e| format!("OpenCL-bygge av sha3t.cl misslyckades:\n{e}"))?;
        let kernel = Kernel::create(&program, "sha3t_scan").map_err(|e| format!("Kernel: {e}"))?;
        // Legacy-kön (clCreateCommandQueue) fungerar på alla OpenCL-versioner
        // — den "moderna" varianten kräver OpenCL 2.0-runtime.
        #[allow(deprecated)]
        let queue =
            CommandQueue::create_default(&context, 0).map_err(|e| format!("CommandQueue: {e}"))?;

        let d_lanes = unsafe {
            Buffer::<u64>::create(&context, CL_MEM_READ_ONLY, 10, ptr::null_mut())
                .map_err(|e| format!("alloc lanes: {e}"))?
        };
        let d_hits = unsafe {
            Buffer::<u32>::create(&context, CL_MEM_READ_WRITE, 1 + MAX_HITS, ptr::null_mut())
                .map_err(|e| format!("alloc hits: {e}"))?
        };

        Ok(Self {
            queue,
            kernel,
            name,
            d_lanes,
            d_hits,
            hits_reset: vec![0u32; 1 + MAX_HITS],
            _context: context,
        })
    }
}

impl MiningBackend for OpenClBackend {
    fn name(&self) -> String {
        format!("OpenCL: {}", self.name)
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

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.d_lanes, CL_BLOCKING, 0, &lanes, &[])
                .map_err(|e| format!("write lanes: {e}"))?;
            self.queue
                .enqueue_write_buffer(&mut self.d_hits, CL_BLOCKING, 0, &self.hits_reset, &[])
                .map_err(|e| format!("write hits: {e}"))?;

            // Global rundas upp till multipel av local; kerneln vaktar count.
            let global = (count as usize).div_ceil(LOCAL_SIZE) * LOCAL_SIZE;
            ExecuteKernel::new(&self.kernel)
                .set_arg(&self.d_lanes)
                .set_arg(&start_nonce)
                .set_arg(&count)
                .set_arg(&t[0])
                .set_arg(&t[1])
                .set_arg(&t[2])
                .set_arg(&t[3])
                .set_arg(&self.d_hits)
                .set_arg(&(MAX_HITS as u32))
                .set_global_work_size(global)
                .set_local_work_size(LOCAL_SIZE)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| format!("enqueue: {e}"))?;

            let mut hits = vec![0u32; 1 + MAX_HITS];
            self.queue
                .enqueue_read_buffer(&self.d_hits, CL_BLOCKING, 0, &mut hits, &[])
                .map_err(|e| format!("read hits: {e}"))?;
            let n = (hits[0] as usize).min(MAX_HITS);
            Ok(hits[1..1 + n].to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::{hash_meets_target, sha3t};

    // Kan bara köras på en maskin med OpenCL-GPU-runtime (t.ex. native
    // Windows med NVIDIA-drivrutin) — inte i WSL/Docker. Samma facit-metod
    // som CUDA-testerna: exakt träffmängdsjämförelse mot CPU-referensen.
    #[test]
    #[ignore = "kräver OpenCL-GPU-runtime (native Windows) — ej WSL/Docker"]
    fn opencl_matches_cpu_on_random_headers() {
        let mut backend = OpenClBackend::new(0, 0).expect("OpenCL-backend");
        println!("backend: {}", backend.name());
        let mut seed = 0xbc3_0c1_u64;
        let mut xorshift = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..10 {
            let mut header76 = [0u8; 76];
            for b in header76.iter_mut() {
                *b = xorshift() as u8;
            }
            let start = xorshift() as u32;
            let count = 4096u32;
            // Target = minsta hashen i intervallet ⇒ exakt en garanterad träff
            // (samma metod som CUDA-testet).
            let mut header80 = [0u8; 80];
            header80[..76].copy_from_slice(&header76);
            let (pick, target) = (0..count)
                .map(|i| {
                    let nonce = start.wrapping_add(i);
                    header80[76..80].copy_from_slice(&nonce.to_le_bytes());
                    let mut key = sha3t(&header80);
                    key.reverse();
                    (nonce, key)
                })
                .min_by(|a, b| a.1.cmp(&b.1))
                .unwrap();

            let mut gpu = backend.scan_range(&header76, start, count, &target).unwrap();
            gpu.sort_unstable();
            let mut cpu: Vec<u32> = (0..count)
                .filter_map(|i| {
                    let nonce = start.wrapping_add(i);
                    header80[76..80].copy_from_slice(&nonce.to_le_bytes());
                    hash_meets_target(&sha3t(&header80), &target).then_some(nonce)
                })
                .collect();
            cpu.sort_unstable();
            assert!(gpu.contains(&pick), "runda {round}: vald nonce saknas");
            assert_eq!(gpu, cpu, "runda {round}: GPU- och CPU-mängderna skiljer");
        }
    }
}
