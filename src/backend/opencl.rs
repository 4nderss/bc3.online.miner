//! OpenCL backend (AMD/Intel/NVIDIA). Shares the kernel source with the CUDA
//! backend (src/kernels/sha3t.cl), so bit-exactness follows from it being the
//! same kernel.
//!
//! The runtime is loaded dynamically (see `cl_sys`) - the binary starts and
//! falls back to the CPU even on machines with no OpenCL at all.

use super::cl_sys::{
    check, cl, cl_handle, cl_int, cl_uint, cstr, err_str, info_string, CL_DEVICE_NAME,
    CL_DEVICE_TYPE_GPU, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_PROGRAM_BUILD_LOG, CL_SUCCESS,
    CL_TRUE,
};
use super::{header_lanes, target_limbs, MiningBackend, KERNEL_SOURCE, MAX_HITS};
use crate::consensus::Target;
use std::ffi::c_void;
use std::ptr;

const LOCAL_SIZE: usize = 256;

/// Devices of a platform, in the same order `list_devices` numbers them.
fn devices_of(platform: cl_handle, dtype: u64) -> Vec<cl_handle> {
    let Ok(cl) = cl() else { return vec![] };
    let mut n: cl_uint = 0;
    // SAFETY: OpenCL calls with valid pointers; n is written only on success.
    unsafe {
        if (cl.get_device_ids)(platform, dtype, 0, ptr::null_mut(), &mut n)
            != CL_SUCCESS
            || n == 0
        {
            return vec![];
        }
        let mut ids = vec![ptr::null_mut(); n as usize];
        if (cl.get_device_ids)(platform, dtype, n, ids.as_mut_ptr(), ptr::null_mut())
            != CL_SUCCESS
        {
            return vec![];
        }
        ids
    }
}

fn platforms() -> Vec<cl_handle> {
    let Ok(cl) = cl() else { return vec![] };
    let mut n: cl_uint = 0;
    unsafe {
        if (cl.get_platform_ids)(0, ptr::null_mut(), &mut n) != CL_SUCCESS || n == 0 {
            return vec![];
        }
        let mut ids = vec![ptr::null_mut(); n as usize];
        if (cl.get_platform_ids)(n, ids.as_mut_ptr(), ptr::null_mut()) != CL_SUCCESS {
            return vec![];
        }
        ids
    }
}

fn device_name(device: cl_handle) -> Option<String> {
    let cl = cl().ok()?;
    info_string(|size, buf, len| unsafe {
        (cl.get_device_info)(device, CL_DEVICE_NAME, size, buf, len)
    })
    .ok()
    .filter(|s| !s.is_empty())
}

/// List all GPU devices across all platforms; empty list on error or if there
/// is no OpenCL runtime.
pub fn list_devices() -> Vec<super::GpuDevice> {
    let mut out = Vec::new();
    for (pi, platform) in platforms().into_iter().enumerate() {
        for (di, device) in devices_of(platform, CL_DEVICE_TYPE_GPU).into_iter().enumerate() {
            let name = device_name(device).unwrap_or_else(|| format!("OpenCL device {pi}.{di}"));
            out.push(super::GpuDevice::Opencl { platform: pi, device: di, name });
        }
    }
    out
}

pub struct OpenClBackend {
    context: cl_handle,
    queue: cl_handle,
    program: cl_handle,
    kernel: cl_handle,
    d_lanes: cl_handle,
    d_hits: cl_handle,
    name: String,
    hits_reset: Vec<u32>,
}

impl OpenClBackend {
    pub fn new(platform_idx: usize, device_idx: usize) -> Result<Self, String> {
        Self::new_typed(platform_idx, device_idx, CL_DEVICE_TYPE_GPU)
    }

    /// The device type is a parameter so the tests can run against pocl,
    /// which exposes a CPU device. The production path takes GPUs only - an
    /// OpenCL CPU would be slower than our own CPU backend.
    fn new_typed(platform_idx: usize, device_idx: usize, dtype: u64) -> Result<Self, String> {
        let cl = cl()?;
        let platform = *platforms()
            .get(platform_idx)
            .ok_or_else(|| format!("OpenCL platform {platform_idx} not found"))?;
        let device = *devices_of(platform, dtype)
            .get(device_idx)
            .ok_or_else(|| format!("OpenCL device {platform_idx}.{device_idx} not found"))?;
        let name =
            device_name(device).unwrap_or_else(|| format!("OpenCL device {platform_idx}.{device_idx}"));

        unsafe {
            let mut err: cl_int = CL_SUCCESS;
            let context = (cl.create_context)(
                ptr::null(),
                1,
                &device,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut err,
            );
            check("clCreateContext", err)?;
            if context.is_null() {
                return Err("clCreateContext returned null".into());
            }
            // Clean up whatever got created if a later step fails.
            let guard = |c: cl_handle| {
                let _ = (cl.release_context)(c);
            };

            // The queue: the old variant works everywhere, the new one is
            // required on runtimes that dropped it (OpenCL 2.0+ without
            // backwards compatibility).
            let queue = if let Some(f) = cl.create_command_queue {
                f(context, device, 0, &mut err)
            } else {
                let props = [0u64; 1]; // empty, nul-terminated property list
                (cl.create_command_queue_with_properties.unwrap())(
                    context,
                    device,
                    props.as_ptr(),
                    &mut err,
                )
            };
            if err != CL_SUCCESS || queue.is_null() {
                guard(context);
                return Err(format!("clCreateCommandQueue: {}", err_str(err)));
            }

            let src = cstr(KERNEL_SOURCE);
            let src_ptr = src.as_ptr();
            let src_len = KERNEL_SOURCE.len();
            let program =
                (cl.create_program_with_source)(context, 1, &src_ptr, &src_len, &mut err);
            if err != CL_SUCCESS || program.is_null() {
                let _ = (cl.release_command_queue)(queue);
                guard(context);
                return Err(format!("clCreateProgramWithSource: {}", err_str(err)));
            }

            let code = (cl.build_program)(
                program,
                1,
                &device,
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
            );
            if code != CL_SUCCESS {
                // The build log is the only thing that says WHAT went wrong.
                let log = info_string(|size, buf, len| {
                    (cl.get_program_build_info)(
                        program,
                        device,
                        CL_PROGRAM_BUILD_LOG,
                        size,
                        buf,
                        len,
                    )
                })
                .unwrap_or_else(|e| format!("(build log unavailable: {e})"));
                let _ = (cl.release_program)(program);
                let _ = (cl.release_command_queue)(queue);
                guard(context);
                return Err(format!(
                    "building sha3t.cl failed: {}\n{log}",
                    err_str(code)
                ));
            }

            let kname = cstr("sha3t_scan");
            let kernel = (cl.create_kernel)(program, kname.as_ptr(), &mut err);
            if err != CL_SUCCESS || kernel.is_null() {
                let _ = (cl.release_program)(program);
                let _ = (cl.release_command_queue)(queue);
                guard(context);
                return Err(format!("clCreateKernel: {}", err_str(err)));
            }

            let d_lanes = (cl.create_buffer)(
                context,
                CL_MEM_READ_ONLY,
                10 * std::mem::size_of::<u64>(),
                ptr::null_mut(),
                &mut err,
            );
            check("clCreateBuffer(lanes)", err)?;
            let d_hits = (cl.create_buffer)(
                context,
                CL_MEM_READ_WRITE,
                (1 + MAX_HITS) * std::mem::size_of::<u32>(),
                ptr::null_mut(),
                &mut err,
            );
            check("clCreateBuffer(hits)", err)?;

            Ok(Self {
                context,
                queue,
                program,
                kernel,
                d_lanes,
                d_hits,
                name,
                hits_reset: vec![0u32; 1 + MAX_HITS],
            })
        }
    }
}

impl Drop for OpenClBackend {
    fn drop(&mut self) {
        let Ok(cl) = cl() else { return };
        // Reverse of creation order; error codes are of no interest here.
        unsafe {
            let _ = (cl.release_mem_object)(self.d_hits);
            let _ = (cl.release_mem_object)(self.d_lanes);
            let _ = (cl.release_kernel)(self.kernel);
            let _ = (cl.release_program)(self.program);
            let _ = (cl.release_command_queue)(self.queue);
            let _ = (cl.release_context)(self.context);
        }
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
        let cl = cl()?;
        let lanes = header_lanes(header76);
        let t = target_limbs(target);
        let max_hits = MAX_HITS as u32;

        unsafe {
            check(
                "write lanes",
                (cl.enqueue_write_buffer)(
                    self.queue,
                    self.d_lanes,
                    CL_TRUE,
                    0,
                    std::mem::size_of_val(&lanes),
                    lanes.as_ptr() as *mut c_void,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                ),
            )?;
            check(
                "reset hits",
                (cl.enqueue_write_buffer)(
                    self.queue,
                    self.d_hits,
                    CL_TRUE,
                    0,
                    std::mem::size_of_val(&self.hits_reset[..]),
                    self.hits_reset.as_ptr() as *mut c_void,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                ),
            )?;

            // The argument order must match sha3t_scan in sha3t.cl exactly.
            let set = |i: cl_uint, size: usize, p: *const c_void| -> Result<(), String> {
                check(&format!("set arg {i}"), (cl.set_kernel_arg)(self.kernel, i, size, p))
            };
            let hsz = std::mem::size_of::<cl_handle>();
            set(0, hsz, &self.d_lanes as *const _ as *const c_void)?;
            set(1, 4, &start_nonce as *const _ as *const c_void)?;
            set(2, 4, &count as *const _ as *const c_void)?;
            for (k, limb) in t.iter().enumerate() {
                set(3 + k as cl_uint, 8, limb as *const _ as *const c_void)?;
            }
            set(7, hsz, &self.d_hits as *const _ as *const c_void)?;
            set(8, 4, &max_hits as *const _ as *const c_void)?;

            // Global rounds up to a multiple of local; the kernel guards count.
            let global = (count as usize).div_ceil(LOCAL_SIZE) * LOCAL_SIZE;
            let local = LOCAL_SIZE;
            check(
                "clEnqueueNDRangeKernel",
                (cl.enqueue_nd_range_kernel)(
                    self.queue,
                    self.kernel,
                    1,
                    ptr::null(),
                    &global,
                    &local,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                ),
            )?;

            let mut hits = vec![0u32; 1 + MAX_HITS];
            check(
                "read hits",
                (cl.enqueue_read_buffer)(
                    self.queue,
                    self.d_hits,
                    CL_TRUE,
                    0,
                    std::mem::size_of_val(&hits[..]),
                    hits.as_mut_ptr() as *mut c_void,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                ),
            )?;
            check("clFinish", (cl.finish)(self.queue))?;

            let n = (hits[0] as usize).min(MAX_HITS);
            Ok(hits[1..1 + n].to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cl_sys::CL_DEVICE_TYPE_ALL;
    use crate::consensus::{hash_meets_target, sha3t};

    /// Requires an OpenCL runtime. In CI/Docker pocl (a CPU implementation)
    /// is enough:
    ///   apt-get install -y pocl-opencl-icd
    ///   cargo test --features opencl -- --ignored opencl
    /// On a real machine the GPU driver's runtime gets tested instead.
    ///
    /// Same ground-truth method as the CUDA tests: exact comparison of the
    /// hit set against the CPU reference, plus one nonce that MUST be found.
    #[test]
    #[ignore = "requires an OpenCL runtime (pocl in Docker, or a GPU driver)"]
    fn opencl_matches_cpu_on_random_headers() {
        // ALL instead of GPU: pocl exposes a CPU device, and the point of
        // the test is the kernel path - not what kind of device runs it.
        let mut backend = OpenClBackend::new_typed(0, 0, CL_DEVICE_TYPE_ALL)
            .expect("OpenCL backend");
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
            // Target = smallest hash in the range -> exactly one guaranteed
            // hit.
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
            assert!(gpu.contains(&pick), "round {round}: the chosen nonce is missing");
            assert_eq!(gpu, cpu, "round {round}: GPU and CPU hit sets differ");
        }
    }

    /// An empty range must never make it as far as a kernel launch.
    #[test]
    #[ignore = "requires an OpenCL runtime"]
    fn zero_count_is_a_no_op() {
        let mut backend = OpenClBackend::new_typed(0, 0, CL_DEVICE_TYPE_ALL)
            .expect("OpenCL backend");
        let hits = backend.scan_range(&[0u8; 76], 0, 0, &[0xff; 32]).unwrap();
        assert!(hits.is_empty());
    }

    /// Without a runtime, list_devices must give an empty list, not a panic.
    /// Can be run anywhere.
    #[test]
    fn list_devices_never_panics() {
        let _ = list_devices();
    }
}
