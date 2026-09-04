//! OpenCL backend (AMD/Intel/NVIDIA). Shares the kernel source with the CUDA
//! backend (src/kernels/sha3t.cl), so bit-exactness follows from it being the
//! same kernel.
//!
//! The runtime is loaded dynamically (see `cl_sys`) - the binary starts and
//! falls back to the CPU even on machines with no OpenCL at all.

use super::cl_sys::{
    check, cl, cl_handle, cl_int, cl_uint, cstr, err_str, info_string, CL_DEVICE_NAME,
    CL_DEVICE_TYPE_GPU, CL_DEVICE_VENDOR, CL_KERNEL_WORK_GROUP_SIZE, CL_MEM_READ_ONLY,
    CL_MEM_READ_WRITE, CL_PROGRAM_BUILD_LOG, CL_SUCCESS, CL_TRUE,
};
use super::{header_lanes, target_limbs, HitBudget, MiningBackend, KERNEL_SOURCE, MAX_HITS};
use crate::consensus::Target;
use std::ffi::c_void;
use std::ptr;

/// Preferred work-group size. Only an upper bound now: the real one comes from
/// CL_KERNEL_WORK_GROUP_SIZE for the built kernel. See `clamp_local_size`.
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

fn device_string(device: cl_handle, param: cl_uint) -> Option<String> {
    let cl = cl().ok()?;
    info_string(|size, buf, len| unsafe {
        (cl.get_device_info)(device, param, size, buf, len)
    })
    .ok()
    .filter(|s| !s.is_empty())
}

fn device_name(device: cl_handle) -> Option<String> {
    device_string(device, CL_DEVICE_NAME)
}

/// List all GPU devices across all platforms; empty list on error or if there
/// is no OpenCL runtime.
///
/// The same physical card can appear on more than one platform - see
/// `super::dedup_opencl`, which is what turns this list into one entry per
/// card.
pub fn list_devices() -> Vec<super::GpuDevice> {
    let mut out = Vec::new();
    for (pi, platform) in platforms().into_iter().enumerate() {
        for (di, device) in devices_of(platform, CL_DEVICE_TYPE_GPU).into_iter().enumerate() {
            let name = device_name(device).unwrap_or_else(|| format!("OpenCL device {pi}.{di}"));
            let vendor = device_string(device, CL_DEVICE_VENDOR).unwrap_or_default();
            out.push(super::GpuDevice::Opencl { platform: pi, device: di, name, vendor });
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
    /// Work-group size this device accepts for this kernel, <= LOCAL_SIZE.
    local_size: usize,
    name: String,
    hits_reset: Vec<u32>,
    budget: HitBudget,
}

/// Largest usable work-group size, given what the device reported.
///
/// `LOCAL_SIZE` used to be passed unconditionally. On a device whose
/// CL_KERNEL_WORK_GROUP_SIZE for this kernel is smaller - keccak is
/// register-heavy, and a small GPU runs out of registers long before 256
/// work-items - EVERY launch failed with CL_INVALID_WORK_GROUP_SIZE, and the
/// worker then gave up on the card as if it had disappeared.
///
/// Rounded down to a power of two: it keeps the launch geometry the kernel was
/// measured with, and stays under CL_DEVICE_MAX_WORK_ITEM_SIZES[0], which is a
/// power of two on every runtime we have seen. A device reporting 0 is broken
/// enough that 1 is the only size left to try.
fn clamp_local_size(device_max: usize) -> usize {
    let n = device_max.min(LOCAL_SIZE);
    if n <= 1 {
        return 1;
    }
    // Highest power of two <= n.
    1usize << (usize::BITS - 1 - n.leading_zeros())
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

        // Every handle below is owned by `me` the instant it exists, so any
        // `return` from here on releases what has been created, through Drop.
        // The two clCreateBuffer paths used to return without releasing the
        // context, queue, program and kernel above them - and the second one
        // leaked the first buffer as well. That is one leaked OpenCL context
        // per failed open, on a worker that retries.
        let mut me = Self {
            context: ptr::null_mut(),
            queue: ptr::null_mut(),
            program: ptr::null_mut(),
            kernel: ptr::null_mut(),
            d_lanes: ptr::null_mut(),
            d_hits: ptr::null_mut(),
            local_size: LOCAL_SIZE,
            name,
            hits_reset: vec![0u32; 1 + MAX_HITS],
            budget: HitBudget::default(),
        };

        unsafe {
            let mut err: cl_int = CL_SUCCESS;
            me.context = (cl.create_context)(
                ptr::null(),
                1,
                &device,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut err,
            );
            check("clCreateContext", err)?;
            if me.context.is_null() {
                return Err("clCreateContext returned null".into());
            }

            // The queue: the old variant works everywhere, the new one is
            // required on runtimes that dropped it (OpenCL 2.0+ without
            // backwards compatibility).
            me.queue = if let Some(f) = cl.create_command_queue {
                f(me.context, device, 0, &mut err)
            } else {
                let props = [0u64; 1]; // empty, nul-terminated property list
                (cl.create_command_queue_with_properties.unwrap())(
                    me.context,
                    device,
                    props.as_ptr(),
                    &mut err,
                )
            };
            if err != CL_SUCCESS || me.queue.is_null() {
                return Err(format!("clCreateCommandQueue: {}", err_str(err)));
            }

            let src = cstr(KERNEL_SOURCE);
            let src_ptr = src.as_ptr();
            let src_len = KERNEL_SOURCE.len();
            me.program =
                (cl.create_program_with_source)(me.context, 1, &src_ptr, &src_len, &mut err);
            if err != CL_SUCCESS || me.program.is_null() {
                return Err(format!("clCreateProgramWithSource: {}", err_str(err)));
            }

            let code = (cl.build_program)(
                me.program,
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
                        me.program,
                        device,
                        CL_PROGRAM_BUILD_LOG,
                        size,
                        buf,
                        len,
                    )
                })
                .unwrap_or_else(|e| format!("(build log unavailable: {e})"));
                return Err(format!(
                    "building sha3t.cl failed: {}\n{log}",
                    err_str(code)
                ));
            }

            let kname = cstr("sha3t_scan");
            me.kernel = (cl.create_kernel)(me.program, kname.as_ptr(), &mut err);
            if err != CL_SUCCESS || me.kernel.is_null() {
                return Err(format!("clCreateKernel: {}", err_str(err)));
            }

            // Must be asked after the build: this is what the compiled kernel
            // fits on this device, not what the device could do in general.
            let mut device_max: usize = 0;
            let code = (cl.get_kernel_work_group_info)(
                me.kernel,
                device,
                CL_KERNEL_WORK_GROUP_SIZE,
                std::mem::size_of::<usize>(),
                &mut device_max as *mut usize as *mut c_void,
                ptr::null_mut(),
            );
            // A runtime that will not answer keeps the size the miner has
            // always used - no better guess is available, and dropping to 1
            // over a failed query would cost far more than it saves.
            me.local_size = if code == CL_SUCCESS {
                clamp_local_size(device_max)
            } else {
                LOCAL_SIZE
            };

            // Worth a line only when it is NOT the size we always used - that
            // is the case that used to fail every launch, and the one someone
            // reading a bug report needs to see.
            if me.local_size != LOCAL_SIZE {
                eprintln!(
                    "[gpu] {}: work-group size {} (device reports {device_max} for this kernel)",
                    me.name, me.local_size
                );
            }

            me.d_lanes = (cl.create_buffer)(
                me.context,
                CL_MEM_READ_ONLY,
                10 * std::mem::size_of::<u64>(),
                ptr::null_mut(),
                &mut err,
            );
            check("clCreateBuffer(lanes)", err)?;
            if me.d_lanes.is_null() {
                return Err("clCreateBuffer(lanes) returned null".into());
            }
            me.d_hits = (cl.create_buffer)(
                me.context,
                CL_MEM_READ_WRITE,
                (1 + MAX_HITS) * std::mem::size_of::<u32>(),
                ptr::null_mut(),
                &mut err,
            );
            check("clCreateBuffer(hits)", err)?;
            if me.d_hits.is_null() {
                return Err("clCreateBuffer(hits) returned null".into());
            }
        }

        Ok(me)
    }
}

impl Drop for OpenClBackend {
    fn drop(&mut self) {
        let Ok(cl) = cl() else { return };
        // Reverse of creation order; error codes are of no interest here.
        //
        // The null checks are not decoration: `new_typed` builds Self with
        // null handles and fills them in, so a constructor that fails part of
        // the way through drops a half-built backend, and this is what makes
        // that safe.
        unsafe {
            for (release, handle) in [
                (cl.release_mem_object, self.d_hits),
                (cl.release_mem_object, self.d_lanes),
                (cl.release_kernel, self.kernel),
                (cl.release_program, self.program),
                (cl.release_command_queue, self.queue),
                (cl.release_context, self.context),
            ] {
                if !handle.is_null() {
                    let _ = release(handle);
                }
            }
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

        // The header does not change across the chunks below, so it is written
        // once rather than per launch.
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
        }

        // Normally one pass over the whole range: the budget only starts
        // splitting after the kernel has overflowed its hit buffer once.
        let mut out = Vec::new();
        let mut done = 0u32;
        while done < count {
            let chunk = self.budget.chunk(count - done);
            let (mut hits, reported) =
                self.scan_chunk(&t, start_nonce.wrapping_add(done), chunk)?;
            out.append(&mut hits);
            done += chunk;
            if let Some(warning) = self.budget.overflowed(chunk, reported) {
                eprintln!("{warning}");
            }
        }
        Ok(out)
    }
}

impl OpenClBackend {
    /// One kernel launch. Returns the hits it could read back, and the count
    /// the kernel reported - which can exceed MAX_HITS, and is what the caller
    /// needs in order to notice that hits were lost.
    fn scan_chunk(
        &self,
        t: &[u64; 4],
        start_nonce: u32,
        count: u32,
    ) -> Result<(Vec<u32>, usize), String> {
        let cl = cl()?;
        let max_hits = MAX_HITS as u32;

        unsafe {
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
            let local = self.local_size;
            let global = (count as usize).div_ceil(local) * local;
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

            // hits[0] is what the kernel COUNTED, which can be more than it
            // had room to store. The caller needs the raw number to see that.
            let reported = hits[0] as usize;
            let n = reported.min(MAX_HITS);
            Ok((hits[1..1 + n].to_vec(), reported))
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

    /// A device that cannot run 256 work-items on this kernel must get a size
    /// it can run. Passing 256 unconditionally failed EVERY launch on such a
    /// card with CL_INVALID_WORK_GROUP_SIZE.
    #[test]
    fn local_size_is_clamped_to_what_the_device_reports() {
        // The common case: the device allows at least our preferred size.
        assert_eq!(clamp_local_size(1024), LOCAL_SIZE);
        assert_eq!(clamp_local_size(256), 256);
        // Smaller devices get the largest power of two that fits.
        assert_eq!(clamp_local_size(255), 128);
        assert_eq!(clamp_local_size(64), 64);
        assert_eq!(clamp_local_size(100), 64);
        assert_eq!(clamp_local_size(3), 2);
        assert_eq!(clamp_local_size(1), 1);
        // A device reporting 0 is broken; 1 is the only legal size left.
        assert_eq!(clamp_local_size(0), 1);
    }

    /// The launch geometry must stay legal for every clamped size: global has
    /// to be a whole multiple of local, or the enqueue is rejected.
    #[test]
    fn global_size_is_a_multiple_of_every_clamped_local_size() {
        for device_max in [0usize, 1, 2, 3, 31, 32, 64, 100, 255, 256, 1024] {
            let local = clamp_local_size(device_max);
            assert!(local >= 1 && local <= LOCAL_SIZE);
            for count in [1u32, 2, 63, 64, 4096, 1_048_576] {
                let global = (count as usize).div_ceil(local) * local;
                assert_eq!(global % local, 0, "device_max {device_max}, count {count}");
                assert!(global >= count as usize);
            }
        }
    }

    /// Once the hit budget has split a scan, the launches must together cover
    /// exactly the range that was asked for - no gap between chunks, no
    /// overlap. Nothing reaches that path on its own (it needs the kernel to
    /// overflow its hit buffer first), so the overflow is forced here. A wrong
    /// per-chunk offset would show up as missing or duplicated nonces.
    #[test]
    #[ignore = "requires an OpenCL runtime"]
    fn a_split_scan_covers_the_same_range_as_one_launch() {
        let mut backend =
            OpenClBackend::new_typed(0, 0, CL_DEVICE_TYPE_ALL).expect("OpenCL backend");

        let mut header76 = [0u8; 76];
        for (i, b) in header76.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        // Deliberately near the top of the space, so a chunk offset that
        // wraps differently from the kernel's would be caught.
        let start = 0xffff_f000u32;
        let count = 4096u32;

        let mut header80 = [0u8; 80];
        header80[..76].copy_from_slice(&header76);
        let mut keys: Vec<(u32, crate::consensus::Target)> = (0..count)
            .map(|i| {
                let nonce = start.wrapping_add(i);
                header80[76..80].copy_from_slice(&nonce.to_le_bytes());
                let mut key = sha3t(&header80);
                key.reverse();
                (nonce, key)
            })
            .collect();
        keys.sort_by(|a, b| a.1.cmp(&b.1));
        // The 5th smallest: several guaranteed hits, spread over the range,
        // so a lost chunk cannot pass unnoticed.
        let target = keys[4].1;

        let mut cpu: Vec<u32> = (0..count)
            .filter_map(|i| {
                let nonce = start.wrapping_add(i);
                header80[76..80].copy_from_slice(&nonce.to_le_bytes());
                hash_meets_target(&sha3t(&header80), &target).then_some(nonce)
            })
            .collect();
        cpu.sort_unstable();
        assert!(cpu.len() >= 5, "the test target must give several hits, got {}", cpu.len());

        // Baseline: one launch over the whole range.
        let mut whole = backend.scan_range(&header76, start, count, &target).unwrap();
        whole.sort_unstable();
        assert_eq!(whole, cpu, "the unsplit scan already disagrees with the CPU");

        // Force the budget to split, then scan the identical range again.
        backend.budget.overflowed(count, MAX_HITS + 1);
        assert!(backend.budget.chunk(count) < count, "the budget did not split");

        let mut split = backend.scan_range(&header76, start, count, &target).unwrap();
        split.sort_unstable();
        assert_eq!(split, cpu, "the split scan lost or duplicated part of the range");
    }
}
