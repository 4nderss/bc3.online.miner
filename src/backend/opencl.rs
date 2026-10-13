//! OpenCL backend (AMD/Intel/NVIDIA). Builds the BC3Hash kernel from the
//! source the CUDA backend's PTX is made from (src/kernels/bc3hash.cl), at run
//! time and once per table window. The two builds do not share all of their
//! arithmetic (AGENTS.md), which is one more reason every card self-tests its
//! kernel before it mines (bc3.rs).
//!
//! The runtime is loaded dynamically (see `cl_sys`) - the binary starts and
//! falls back to the CPU even on machines with no OpenCL at all.
//!
//! Launches are queued like CUDA's (see `QUEUE_DEPTH`): non-blocking copies
//! and an event per kernel and per readback, on a queue with profiling on so
//! the kernel's own start and end can be read back.

use super::cl_sys::{
    check, cl, cl_handle, cl_int, cl_uint, cstr, err_str, info_string, Cl, CL_DEVICE_NAME,
    CL_DEVICE_TYPE_GPU, CL_DEVICE_VENDOR, CL_FALSE, CL_KERNEL_WORK_GROUP_SIZE, CL_MEM_READ_ONLY,
    CL_MEM_READ_WRITE, CL_PROFILING_COMMAND_END, CL_PROFILING_COMMAND_START, CL_PROGRAM_BUILD_LOG,
    CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_PROPERTIES, CL_SUCCESS, CL_TRUE,
};
use super::{kernel_supports, Launch, MiningBackend, Scan, MAX_HITS, QUEUE_DEPTH};
use crate::consensus::PowAlgo;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr;
use std::time::Duration;

/// Work-group size for BC3Hash: the kernel needs ~255 registers a work-item,
/// and 64 is what it was measured with (32-256 the same on CUDA). Only an
/// upper bound: the real one comes from CL_KERNEL_WORK_GROUP_SIZE for the
/// built kernel. See `clamp_local_size`.
const LOCAL_SIZE: usize = 64;

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
    name: String,
    /// One per launch that may be in flight.
    slots: Vec<ClSlot>,
    /// Slots of the launches in flight, oldest first.
    queued: VecDeque<usize>,
    /// The slot the next launch takes.
    next_slot: usize,
    /// The device itself, for building the BC3Hash program later.
    device: cl_handle,
    /// The BC3Hash program in use - built on demand, for one window at a time.
    bc3: Option<Bc3Program>,
}

/// What one launch in flight owns - the counterpart of the CUDA backend's
/// `Slot`: its inputs, its hit buffer, and the events of its kernel (the
/// device time) and of its readback (done).
struct ClSlot {
    /// The header and the target limbs.
    header: cl_handle,
    target: cl_handle,
    hits: cl_handle,
    /// Host memory the non-blocking copies read from and write to after the
    /// enqueue has returned - boxed, so it stays put while they are pending.
    host: Box<ClHost>,
    kernel_done: cl_handle,
    read_done: cl_handle,
    /// What `header` and `target` hold, so unchanged ones are not written
    /// again.
    header_of: Option<[u8; 76]>,
    target_of: Option<[u64; 4]>,
}

struct ClHost {
    header: [u8; 76],
    target: [u64; 4],
    zero: u32,
    hits: [u32; 1 + MAX_HITS],
}

impl ClSlot {
    fn new(context: cl_handle) -> Result<Self, String> {
        let mut slot = Self {
            header: ptr::null_mut(),
            target: ptr::null_mut(),
            hits: ptr::null_mut(),
            host: Box::new(ClHost { header: [0; 76], target: [0; 4], zero: 0, hits: [0; 1 + MAX_HITS] }),
            kernel_done: ptr::null_mut(),
            read_done: ptr::null_mut(),
            header_of: None,
            target_of: None,
        };
        // Owned from here on: a failure below releases what exists (Drop).
        unsafe {
            slot.header = create_buffer(context, CL_MEM_READ_ONLY, 76, "header")?;
            slot.target = create_buffer(context, CL_MEM_READ_ONLY, 32, "target")?;
            slot.hits = create_buffer(context, CL_MEM_READ_WRITE, (1 + MAX_HITS) * 4, "hits")?;
        }
        Ok(slot)
    }

    fn release_events(&mut self) {
        let Ok(cl) = cl() else { return };
        for event in [&mut self.kernel_done, &mut self.read_done] {
            if !event.is_null() {
                unsafe {
                    let _ = (cl.release_event)(*event);
                }
                *event = ptr::null_mut();
            }
        }
    }

    /// After a failed or abandoned launch nobody knows what the buffers hold.
    fn forget(&mut self) {
        self.header_of = None;
        self.target_of = None;
    }
}

impl Drop for ClSlot {
    fn drop(&mut self) {
        self.release_events();
        let Ok(cl) = cl() else { return };
        for buffer in [self.header, self.target, self.hits] {
            if !buffer.is_null() {
                unsafe {
                    let _ = (cl.release_mem_object)(buffer);
                }
            }
        }
    }
}

/// One built BC3Hash program: its window, both kernels, and the tables they
/// read (written once). Every handle is released on drop, however far the
/// build got.
struct Bc3Program {
    window: u32,
    program: cl_handle,
    scan: cl_handle,
    batch: cl_handle,
    d_tables: cl_handle,
    local: usize,
}

impl Drop for Bc3Program {
    fn drop(&mut self) {
        let Ok(cl) = cl() else { return };
        unsafe {
            for (release, handle) in [
                (cl.release_mem_object, self.d_tables),
                (cl.release_kernel, self.batch),
                (cl.release_kernel, self.scan),
                (cl.release_program, self.program),
            ] {
                if !handle.is_null() {
                    let _ = release(handle);
                }
            }
        }
    }
}

/// Largest usable work-group size, given what the device reported.
///
/// A fixed size used to be passed unconditionally. On a device whose
/// CL_KERNEL_WORK_GROUP_SIZE for the kernel is smaller - a register-heavy
/// kernel on a small GPU runs out of registers long before the size it was
/// measured with - EVERY launch failed with CL_INVALID_WORK_GROUP_SIZE, and
/// the worker then gave up on the card as if it had disappeared.
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

/// A buffer on `context`; never a null handle on Ok.
unsafe fn create_buffer(context: cl_handle, flags: u64, bytes: usize, what: &str) -> Result<cl_handle, String> {
    let cl = cl()?;
    let mut err: cl_int = CL_SUCCESS;
    let buffer = (cl.create_buffer)(context, flags, bytes, ptr::null_mut(), &mut err);
    check(&format!("clCreateBuffer({what})"), err)?;
    if buffer.is_null() {
        return Err(format!("clCreateBuffer({what}) returned null"));
    }
    Ok(buffer)
}

/// A command queue, with event profiling if `profiling`: the old entry point
/// works everywhere, the new one is required on runtimes that dropped it
/// (OpenCL 2.0+ without backwards compatibility).
unsafe fn create_queue(cl: &Cl, context: cl_handle, device: cl_handle, profiling: bool) -> Result<cl_handle, String> {
    let mut err: cl_int = CL_SUCCESS;
    let props = if profiling { CL_QUEUE_PROFILING_ENABLE } else { 0 };
    let queue = if let Some(f) = cl.create_command_queue {
        f(context, device, props, &mut err)
    } else {
        let list = [CL_QUEUE_PROPERTIES, props, 0];
        (cl.create_command_queue_with_properties.unwrap())(context, device, list.as_ptr(), &mut err)
    };
    if err != CL_SUCCESS || queue.is_null() {
        if !queue.is_null() {
            let _ = (cl.release_command_queue)(queue);
        }
        return Err(format!("clCreateCommandQueue: {}", err_str(err)));
    }
    Ok(queue)
}

/// A non-blocking write: `data` must stay valid and unchanged until the queue
/// has executed it.
unsafe fn write_async(queue: cl_handle, buffer: cl_handle, data: *const c_void, bytes: usize, what: &str) -> Result<(), String> {
    let cl = cl()?;
    check(
        &format!("write {what}"),
        (cl.enqueue_write_buffer)(queue, buffer, CL_FALSE, 0, bytes, data as *mut c_void, 0, ptr::null(), ptr::null_mut()),
    )
}

/// How long the command behind `event` ran on the device; None without
/// profiling.
fn device_time(cl: &Cl, event: cl_handle) -> Option<Duration> {
    let read = |param: cl_uint| {
        let mut ns: u64 = 0;
        let code = unsafe {
            (cl.get_event_profiling_info)(event, param, 8, &mut ns as *mut u64 as *mut c_void, ptr::null_mut())
        };
        (code == CL_SUCCESS).then_some(ns)
    };
    let (start, end) = (read(CL_PROFILING_COMMAND_START)?, read(CL_PROFILING_COMMAND_END)?);
    (start > 0 && end >= start).then(|| Duration::from_nanos(end - start))
}

/// A kernel argument for `set_args`.
enum Arg {
    Mem(cl_handle),
    U32(u32),
}

/// Set a kernel's arguments, in order: buffers by handle, numbers by value.
/// The runtime copies them; the kernel is launched with what they are at the
/// time of the enqueue.
unsafe fn set_args(kernel: cl_handle, args: &[Arg]) -> Result<(), String> {
    let cl = cl()?;
    for (i, arg) in args.iter().enumerate() {
        let (size, p) = match arg {
            Arg::Mem(h) => (std::mem::size_of::<cl_handle>(), h as *const cl_handle as *const c_void),
            Arg::U32(v) => (4, v as *const u32 as *const c_void),
        };
        check(&format!("set arg {i}"), (cl.set_kernel_arg)(kernel, i as cl_uint, size, p))?;
    }
    Ok(())
}

/// One work-item per nonce; global rounds up to a multiple of local, and the
/// kernel ignores the items past its count.
unsafe fn launch(queue: cl_handle, kernel: cl_handle, items: u32, local: usize, event: *mut cl_handle) -> Result<(), String> {
    let cl = cl()?;
    let global = (items as usize).div_ceil(local) * local;
    check(
        "clEnqueueNDRangeKernel",
        (cl.enqueue_nd_range_kernel)(queue, kernel, 1, ptr::null(), &global, &local, 0, ptr::null(), event),
    )
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
        // Early returns used to skip that: one leaked OpenCL context per
        // failed open, on a worker that retries.
        let mut me = Self {
            context: ptr::null_mut(),
            queue: ptr::null_mut(),
            name,
            slots: Vec::new(),
            queued: VecDeque::new(),
            next_slot: 0,
            device,
            bc3: None,
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

            // Profiling gives each launch's device time (the benchmark's
            // "busy" figure, and the autotune's measure). Every device must
            // support it, but a runtime that refuses still gets a queue.
            me.queue = match create_queue(cl, me.context, device, true) {
                Ok(q) => q,
                Err(_) => create_queue(cl, me.context, device, false)?,
            };
        }
        for _ in 0..QUEUE_DEPTH {
            let slot = ClSlot::new(me.context)?;
            me.slots.push(slot);
        }

        Ok(me)
    }

    /// Queue the writes, the kernel and the readback of one launch on slot
    /// `i`, and submit them.
    fn queue_launch(&mut self, i: usize, l: &Launch) -> Result<(), String> {
        let (kernel, local, args) = match l.algo {
            PowAlgo::Bc3Hash => self.bc3_inputs(i, l)?,
            algo @ (PowAlgo::Sha256d | PowAlgo::Sha3t) => {
                return Err(format!("no kernel for {}", algo.name()))
            }
        };
        let cl = cl()?;
        let queue = self.queue;
        let slot = &mut self.slots[i];
        unsafe {
            slot.host.zero = 0;
            write_async(queue, slot.hits, &slot.host.zero as *const u32 as *const c_void, 4, "hit count")?;
            set_args(kernel, &args)?;
            launch(queue, kernel, l.count, local, &mut slot.kernel_done)?;
            check(
                "read hits",
                (cl.enqueue_read_buffer)(
                    queue,
                    slot.hits,
                    CL_FALSE,
                    0,
                    std::mem::size_of_val(&slot.host.hits),
                    slot.host.hits.as_mut_ptr() as *mut c_void,
                    0,
                    ptr::null(),
                    &mut slot.read_done,
                ),
            )?;
            // Submit now: a runtime may otherwise hold the commands back until
            // the host waits, which is the gap the queue is there to close.
            check("clFlush", (cl.flush)(queue))?;
        }
        Ok(())
    }

    /// BC3Hash: the header and target into slot `i`, and the kernel arguments.
    fn bc3_inputs(&mut self, i: usize, l: &Launch) -> Result<(cl_handle, usize, Vec<Arg>), String> {
        let k = self.bc3.as_ref().ok_or("BC3Hash program not built")?;
        let queue = self.queue;
        let slot = &mut self.slots[i];
        if slot.header_of.as_ref() != Some(l.header76) {
            slot.header_of = None;
            slot.host.header = *l.header76;
            unsafe { write_async(queue, slot.header, slot.host.header.as_ptr() as *const c_void, 76, "header")? };
            slot.header_of = Some(*l.header76);
        }
        let limbs = super::bc3::target_limbs(l.target);
        if slot.target_of != Some(limbs) {
            slot.target_of = None;
            slot.host.target = limbs;
            unsafe { write_async(queue, slot.target, slot.host.target.as_ptr() as *const c_void, 32, "target")? };
            slot.target_of = Some(limbs);
        }
        // The argument order must match bc3hash_scan in bc3hash.cl.
        let args = vec![
            Arg::Mem(k.d_tables),
            Arg::Mem(slot.header),
            Arg::U32(l.start),
            Arg::U32(l.count),
            Arg::Mem(slot.target),
            Arg::Mem(slot.hits),
            Arg::U32(MAX_HITS as u32),
        ];
        Ok((k.scan, k.local, args))
    }
}

impl Drop for OpenClBackend {
    fn drop(&mut self) {
        let Ok(cl) = cl() else { return };
        // Pending copies read and write the slots' host memory: let the queue
        // finish before anything is freed.
        if !self.queue.is_null() {
            unsafe {
                let _ = (cl.finish)(self.queue);
            }
        }
        self.slots.clear();
        // The BC3Hash program and buffers belong to the context: first.
        drop(self.bc3.take());
        // Reverse of creation order; error codes are of no interest here.
        //
        // The null checks are not decoration: `new_typed` builds Self with
        // null handles and fills them in, so a constructor that fails part of
        // the way through drops a half-built backend, and this is what makes
        // that safe.
        unsafe {
            for (release, handle) in [
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
        // Nothing of the old program may still run when its buffers go.
        self.abandon();
        // The old window's buffers leave the device before the new ones come.
        self.bc3 = None;
        let program = self.build_bc3(window)?;
        self.bc3 = Some(program);
        if let Err(e) = super::bc3::self_test(self) {
            self.bc3 = None;
            return Err(format!("self-test failed: {e}"));
        }
        Ok(t0.elapsed())
    }

    fn bc3hash_window(&self) -> Option<u32> {
        self.bc3.as_ref().map(|k| k.window)
    }

    fn enqueue(&mut self, l: &Launch) -> Result<(), String> {
        if l.count == 0 {
            return Err("a launch of 0 nonces".into());
        }
        if self.queued.len() >= QUEUE_DEPTH {
            return Err("too many launches in flight".into());
        }
        let i = self.next_slot;
        if let Err(e) = self.queue_launch(i, l) {
            // Whatever made it into the queue may still run; abandon() is
            // what waits for it.
            let slot = &mut self.slots[i];
            slot.release_events();
            slot.forget();
            return Err(e);
        }
        self.queued.push_back(i);
        self.next_slot = (i + 1) % QUEUE_DEPTH;
        Ok(())
    }

    fn collect(&mut self) -> Result<Scan, String> {
        let cl = cl()?;
        let i = self.queued.pop_front().ok_or("no launch in flight")?;
        let slot = &mut self.slots[i];
        let waited = unsafe { (cl.wait_for_events)(1, &slot.read_done) };
        let busy = if waited == CL_SUCCESS { device_time(cl, slot.kernel_done) } else { None };
        slot.release_events();
        check("clWaitForEvents", waited)?;
        let reported = slot.host.hits[0] as usize;
        let n = reported.min(MAX_HITS);
        Ok(Scan { hits: slot.host.hits[1..1 + n].to_vec(), reported, busy })
    }

    fn in_flight(&self) -> usize {
        self.queued.len()
    }

    fn abandon(&mut self) {
        if let Ok(cl) = cl() {
            unsafe {
                let _ = (cl.finish)(self.queue);
            }
        }
        for slot in &mut self.slots {
            slot.release_events();
            slot.forget();
        }
        self.queued.clear();
    }
}

impl OpenClBackend {
    /// Build the BC3Hash program for `window` (`-DBC3_WINDOW=N`) and write its
    /// tables to the device. The runtime compiles a kernel this size for a
    /// while - once per window and run.
    fn build_bc3(&self, window: u32) -> Result<Bc3Program, String> {
        // The host builds an OpenCL card's tables, and those of the uniform
        // windows only; 22 and 24 - mixed schedules, 8 and 28 GiB - only a
        // CUDA card builds, on itself.
        if !(2..=20).contains(&window) {
            return Err(format!("window {window} is for CUDA cards only"));
        }
        let cl = cl()?;
        // Owned from the start, so every early return releases what exists.
        let mut k = Bc3Program {
            window,
            program: ptr::null_mut(),
            scan: ptr::null_mut(),
            batch: ptr::null_mut(),
            d_tables: ptr::null_mut(),
            local: LOCAL_SIZE,
        };
        unsafe {
            let mut err: cl_int = CL_SUCCESS;
            let source = super::bc3::SOURCE;
            let src = cstr(source);
            let src_ptr = src.as_ptr();
            let src_len = source.len();
            k.program =
                (cl.create_program_with_source)(self.context, 1, &src_ptr, &src_len, &mut err);
            if err != CL_SUCCESS || k.program.is_null() {
                return Err(format!("clCreateProgramWithSource (BC3Hash): {}", err_str(err)));
            }
            let options = cstr(&format!("-DBC3_WINDOW={window}"));
            let code = (cl.build_program)(
                k.program,
                1,
                &self.device,
                options.as_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
            );
            if code != CL_SUCCESS {
                // The build log is the only thing that says WHAT went wrong.
                let log = info_string(|size, buf, len| {
                    (cl.get_program_build_info)(
                        k.program,
                        self.device,
                        CL_PROGRAM_BUILD_LOG,
                        size,
                        buf,
                        len,
                    )
                })
                .unwrap_or_else(|e| format!("(build log unavailable: {e})"));
                return Err(format!(
                    "building bc3hash.cl (window {window}) failed: {}\n{log}",
                    err_str(code)
                ));
            }
            for (name, slot) in [("bc3hash_scan", &mut k.scan), ("bc3hash_batch", &mut k.batch)] {
                let kname = cstr(name);
                *slot = (cl.create_kernel)(k.program, kname.as_ptr(), &mut err);
                if err != CL_SUCCESS || slot.is_null() {
                    return Err(format!("clCreateKernel {name}: {}", err_str(err)));
                }
            }
            // Must be asked after the build: this is what the compiled kernel
            // fits on this device, not what the device could do in general.
            // A runtime that will not answer keeps the size the kernel was
            // measured with - no better guess is available, and dropping to
            // 1 over a failed query would cost far more than it saves.
            let mut device_max: usize = 0;
            let code = (cl.get_kernel_work_group_info)(
                k.scan,
                self.device,
                CL_KERNEL_WORK_GROUP_SIZE,
                std::mem::size_of::<usize>(),
                &mut device_max as *mut usize as *mut c_void,
                ptr::null_mut(),
            );
            if code == CL_SUCCESS {
                k.local = clamp_local_size(device_max);
            }
            let tables = super::bc3::host_tables(window)?;
            let bytes = tables.len() * std::mem::size_of::<u64>();
            k.d_tables = create_buffer(self.context, CL_MEM_READ_ONLY, bytes, "tables")?;
            self.write_blocking(k.d_tables, tables.as_ptr() as *const c_void, bytes, "tables")?;
        }
        Ok(k)
    }

    unsafe fn write_blocking(&self, buffer: cl_handle, data: *const c_void, bytes: usize, what: &str) -> Result<(), String> {
        let cl = cl()?;
        check(
            &format!("write {what}"),
            (cl.enqueue_write_buffer)(self.queue, buffer, CL_TRUE, 0, bytes, data as *mut c_void, 0, ptr::null(), ptr::null_mut()),
        )
    }
}

impl super::bc3::Bc3Ops for OpenClBackend {
    fn bc3_batch(&mut self, headers: &[[u8; 80]]) -> Result<Vec<[u8; 32]>, String> {
        let k = self.bc3.as_ref().ok_or("BC3Hash program not built")?;
        let api = cl()?;
        let flat: Vec<u8> = headers.iter().flatten().copied().collect();
        let count = headers.len() as u32;
        let mut out = vec![0u8; headers.len() * 32];
        // Released on every path out, like the rest of this backend's handles.
        struct Temp(cl_handle);
        impl Drop for Temp {
            fn drop(&mut self) {
                if let Ok(cl) = cl() {
                    if !self.0.is_null() {
                        unsafe {
                            let _ = (cl.release_mem_object)(self.0);
                        }
                    }
                }
            }
        }
        unsafe {
            let d_in = Temp(create_buffer(self.context, CL_MEM_READ_ONLY, flat.len(), "headers")?);
            let d_out = Temp(create_buffer(self.context, CL_MEM_READ_WRITE, out.len(), "digests")?);
            self.write_blocking(d_in.0, flat.as_ptr() as *const c_void, flat.len(), "headers")?;
            set_args(k.batch, &[Arg::Mem(k.d_tables), Arg::Mem(d_in.0), Arg::U32(count), Arg::Mem(d_out.0)])?;
            launch(self.queue, k.batch, count, k.local, ptr::null_mut())?;
            check(
                "read digests",
                (api.enqueue_read_buffer)(
                    self.queue,
                    d_out.0,
                    CL_TRUE,
                    0,
                    out.len(),
                    out.as_mut_ptr() as *mut c_void,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                ),
            )?;
            check("clFinish", (api.finish)(self.queue))?;
        }
        Ok(out.as_chunks::<32>().0.to_vec())
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cl_sys::CL_DEVICE_TYPE_ALL;
    use crate::consensus::{hash_meets_target, Target};

    /// The CPU's hit set over `count` nonces from `start`: the nonces whose
    /// BC3Hash, by the node's own C++, meets `target`.
    fn cpu_hits(header76: &[u8; 76], start: u32, count: u32, target: &Target) -> Vec<u32> {
        let mut header80 = [0u8; 80];
        header80[..76].copy_from_slice(header76);
        (0..count)
            .map(|i| start.wrapping_add(i))
            .filter(|&nonce| {
                header80[76..80].copy_from_slice(&nonce.to_le_bytes());
                hash_meets_target(&bc3hash::bc3hash(&header80), target)
            })
            .collect()
    }

    /// The BC3Hash program builds on this runtime and passes the self-test
    /// every card runs: the node's header vectors, random headers and a scan
    /// range against the CPU reference. Then a range through `scan_range`,
    /// the mining path, against the CPU's hit set.
    ///
    /// Requires an OpenCL runtime. In CI/Docker pocl (a CPU implementation)
    /// is enough - see the README:
    ///   apt-get install -y pocl-opencl-icd
    ///   cargo test --release opencl -- --ignored --test-threads=1
    /// On a real machine the GPU driver's runtime gets tested instead.
    #[test]
    #[ignore = "requires an OpenCL runtime; building the kernel takes pocl minutes"]
    fn opencl_bc3hash_passes_the_self_test() {
        // ALL instead of GPU: pocl exposes a CPU device, and the point of
        // the test is the kernel path - not what kind of device runs it.
        let mut backend =
            OpenClBackend::new_typed(0, 0, CL_DEVICE_TYPE_ALL).expect("OpenCL backend");
        let took = backend.prepare_bc3hash(10).expect("BC3Hash on OpenCL");
        println!("{}: BC3Hash window 10 ready, self-test passed, {:.1} s", backend.name(), took.as_secs_f64());
        assert!(backend.supports(PowAlgo::Bc3Hash));
        assert_eq!(backend.bc3hash_window(), Some(10));

        let mut header80 = [0u8; 80];
        header80[..4].copy_from_slice(&0x2000_1800u32.to_le_bytes());
        header80[4] = 0x5a;
        let header76: [u8; 76] = header80[..76].try_into().unwrap();
        let mut target: Target = [0xff; 32];
        target[0] = 0x0f; // about one hash in sixteen
        let mut gpu = backend.scan_range(PowAlgo::Bc3Hash, &header76, 1000, 128, &target).unwrap();
        gpu.sort_unstable();
        let cpu = cpu_hits(&header76, 1000, 128, &target);
        assert_eq!(gpu, cpu);
        assert!(!cpu.is_empty());
    }

    /// An empty range must never make it as far as a kernel launch.
    #[test]
    #[ignore = "requires an OpenCL runtime"]
    fn zero_count_is_a_no_op() {
        let mut backend = OpenClBackend::new_typed(0, 0, CL_DEVICE_TYPE_ALL)
            .expect("OpenCL backend");
        let hits = backend.scan_range(PowAlgo::Bc3Hash, &[0u8; 76], 0, 0, &[0xff; 32]).unwrap();
        assert!(hits.is_empty());
    }

    /// Without a runtime, list_devices must give an empty list, not a panic.
    /// Can be run anywhere.
    #[test]
    fn list_devices_never_panics() {
        let _ = list_devices();
    }

    /// A device that cannot run the kernel's work-group size must get a size
    /// it can run. Passing a fixed size unconditionally failed EVERY launch on
    /// such a card with CL_INVALID_WORK_GROUP_SIZE.
    #[test]
    fn local_size_is_clamped_to_what_the_device_reports() {
        // The common case: the device allows at least our preferred size.
        assert_eq!(clamp_local_size(1024), LOCAL_SIZE);
        assert_eq!(clamp_local_size(64), 64);
        // Smaller devices get the largest power of two that fits.
        assert_eq!(clamp_local_size(63), 32);
        assert_eq!(clamp_local_size(48), 32);
        assert_eq!(clamp_local_size(3), 2);
        assert_eq!(clamp_local_size(1), 1);
        // A device reporting 0 is broken; 1 is the only legal size left.
        assert_eq!(clamp_local_size(0), 1);
    }

    /// The launch geometry must stay legal for every clamped size: global has
    /// to be a whole multiple of local, or the enqueue is rejected.
    #[test]
    fn global_size_is_a_multiple_of_every_clamped_local_size() {
        for device_max in [0usize, 1, 2, 3, 31, 32, 63, 64, 100, 256, 1024] {
            let local = clamp_local_size(device_max);
            assert!(local >= 1 && local <= LOCAL_SIZE);
            for count in [1u32, 2, 63, 64, 4096, 1_048_576] {
                let global = (count as usize).div_ceil(local) * local;
                assert_eq!(global % local, 0, "device_max {device_max}, count {count}");
                assert!(global >= count as usize);
            }
        }
    }

    /// Launches queued behind each other must each come back with their own
    /// hits - their own header, range and target - in order, and with the
    /// time the device spent on them. Slot 0 gets a, b, a and slot 1 gets
    /// b, b, a: copies over another header, a skipped copy, a new target.
    #[test]
    #[ignore = "requires an OpenCL runtime; building the kernel takes pocl minutes"]
    fn opencl_queued_launches_keep_their_own_inputs() {
        let mut backend =
            OpenClBackend::new_typed(0, 0, CL_DEVICE_TYPE_ALL).expect("OpenCL backend");
        backend.prepare_bc3hash(10).expect("BC3Hash window 10 passes the self-test");
        let mut a = [0u8; 76];
        a[..4].copy_from_slice(&0x2000_1800u32.to_le_bytes());
        let mut b = a;
        a[4] = 0xa1;
        b[4] = 0xb2;
        // About 12 and 48 hits in 96 nonces.
        let mut eighth: Target = [0xff; 32];
        eighth[0] = 0x1f;
        let mut half: Target = [0xff; 32];
        half[0] = 0x7f;
        let count = 96u32;
        let plan = [
            (a, eighth, 100u32),
            (b, eighth, 300),
            (b, eighth, 500),
            (b, eighth, 700),
            (a, half, 900),
            (a, half, 0xffff_ff00),
        ];
        let mut results = Vec::new();
        for (h, t, start) in &plan {
            backend
                .enqueue(&Launch { algo: PowAlgo::Bc3Hash, header76: h, start: *start, count, target: t })
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
            let cpu = cpu_hits(h, *start, count, t);
            assert_eq!(gpu, cpu, "launch at {start}");
            assert!(!cpu.is_empty() && cpu.len() <= MAX_HITS, "{} hits", cpu.len());
            assert!(scan.busy.is_some_and(|d| d > Duration::ZERO), "no device time from profiling");
        }
    }

    /// A launch the caller gave up on leaves nothing behind: the next one
    /// starts on a clean queue and reports only its own hits.
    #[test]
    #[ignore = "requires an OpenCL runtime; building the kernel takes pocl minutes"]
    fn opencl_abandon_clears_the_queue() {
        let mut backend =
            OpenClBackend::new_typed(0, 0, CL_DEVICE_TYPE_ALL).expect("OpenCL backend");
        backend.prepare_bc3hash(10).expect("BC3Hash window 10 passes the self-test");
        let target: Target = [0xff; 32];
        let mut header = [3u8; 76];
        header[..4].copy_from_slice(&0x2000_1800u32.to_le_bytes());
        let launch = Launch { algo: PowAlgo::Bc3Hash, header76: &header, start: 0, count: 32, target: &target };
        backend.enqueue(&launch).unwrap();
        backend.enqueue(&launch).unwrap();
        assert!(backend.enqueue(&launch).is_err(), "a third launch must not fit");
        backend.abandon();
        assert_eq!(backend.in_flight(), 0);
        let hits = backend.scan_range(PowAlgo::Bc3Hash, &header, 0, 32, &target).unwrap();
        assert_eq!(hits.len(), 32, "every nonce meets an all-ones target");
    }
}
