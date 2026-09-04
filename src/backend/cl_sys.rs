//! Minimal OpenCL binding, loaded DYNAMICALLY at runtime.
//!
//! Why not `opencl3`/`ocl`: they link against `OpenCL.lib`/`libOpenCL.so` at
//! build time. Such a binary refuses to start on a machine without an
//! OpenCL runtime - so an NVIDIA user without OpenCL, or anyone without a
//! GPU at all, could not run the miner at all. Same reason CUDA is loaded
//! dynamically (cudarc `fallback-dynamic-loading`).
//!
//! With this loader it stays ONE binary: CUDA on NVIDIA, OpenCL on
//! AMD/Intel, CPU if there is neither. If the library is missing we get an
//! Err instead of a process that will not start.
//!
//! Only the ~20 functions the miner actually uses are bound.

#![allow(non_camel_case_types)]

use libloading::{Library, Symbol};
use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::sync::OnceLock;

pub type cl_int = i32;
pub type cl_uint = u32;
pub type cl_bitfield = u64;
pub type cl_bool = cl_uint;
/// All OpenCL objects are opaque pointers.
pub type cl_handle = *mut c_void;

pub const CL_SUCCESS: cl_int = 0;
pub const CL_TRUE: cl_bool = 1;
pub const CL_DEVICE_TYPE_GPU: cl_bitfield = 1 << 2;
/// All device types - for tests only (pocl exposes a CPU device).
pub const CL_DEVICE_TYPE_ALL: cl_bitfield = 0xFFFF_FFFF;
pub const CL_DEVICE_NAME: cl_uint = 0x102B;
pub const CL_DEVICE_VENDOR: cl_uint = 0x102C;
pub const CL_MEM_READ_WRITE: cl_bitfield = 1 << 0;
pub const CL_MEM_READ_ONLY: cl_bitfield = 1 << 2;
pub const CL_PROGRAM_BUILD_LOG: cl_uint = 0x1183;
/// Largest work-group size this device can run THIS kernel with. It is a
/// per-kernel limit, not a device one: register pressure decides it, and
/// keccak is register-heavy.
pub const CL_KERNEL_WORK_GROUP_SIZE: cl_uint = 0x11B0;

type FnGetPlatformIDs = unsafe extern "C" fn(cl_uint, *mut cl_handle, *mut cl_uint) -> cl_int;
type FnGetDeviceIDs =
    unsafe extern "C" fn(cl_handle, cl_bitfield, cl_uint, *mut cl_handle, *mut cl_uint) -> cl_int;
type FnGetDeviceInfo =
    unsafe extern "C" fn(cl_handle, cl_uint, usize, *mut c_void, *mut usize) -> cl_int;
type FnCreateContext = unsafe extern "C" fn(
    *const isize,
    cl_uint,
    *const cl_handle,
    *mut c_void,
    *mut c_void,
    *mut cl_int,
) -> cl_handle;
type FnCreateCommandQueue =
    unsafe extern "C" fn(cl_handle, cl_handle, cl_bitfield, *mut cl_int) -> cl_handle;
type FnCreateCommandQueueWithProperties =
    unsafe extern "C" fn(cl_handle, cl_handle, *const cl_bitfield, *mut cl_int) -> cl_handle;
type FnCreateProgramWithSource = unsafe extern "C" fn(
    cl_handle,
    cl_uint,
    *const *const c_char,
    *const usize,
    *mut cl_int,
) -> cl_handle;
type FnBuildProgram = unsafe extern "C" fn(
    cl_handle,
    cl_uint,
    *const cl_handle,
    *const c_char,
    *mut c_void,
    *mut c_void,
) -> cl_int;
type FnGetProgramBuildInfo =
    unsafe extern "C" fn(cl_handle, cl_handle, cl_uint, usize, *mut c_void, *mut usize) -> cl_int;
type FnCreateKernel = unsafe extern "C" fn(cl_handle, *const c_char, *mut cl_int) -> cl_handle;
type FnGetKernelWorkGroupInfo =
    unsafe extern "C" fn(cl_handle, cl_handle, cl_uint, usize, *mut c_void, *mut usize) -> cl_int;
type FnCreateBuffer =
    unsafe extern "C" fn(cl_handle, cl_bitfield, usize, *mut c_void, *mut cl_int) -> cl_handle;
type FnSetKernelArg = unsafe extern "C" fn(cl_handle, cl_uint, usize, *const c_void) -> cl_int;
type FnEnqueueBuffer = unsafe extern "C" fn(
    cl_handle,
    cl_handle,
    cl_bool,
    usize,
    usize,
    *mut c_void,
    cl_uint,
    *const cl_handle,
    *mut cl_handle,
) -> cl_int;
type FnEnqueueNDRange = unsafe extern "C" fn(
    cl_handle,
    cl_handle,
    cl_uint,
    *const usize,
    *const usize,
    *const usize,
    cl_uint,
    *const cl_handle,
    *mut cl_handle,
) -> cl_int;
type FnFinish = unsafe extern "C" fn(cl_handle) -> cl_int;
type FnRelease = unsafe extern "C" fn(cl_handle) -> cl_int;

pub struct Cl {
    pub get_platform_ids: FnGetPlatformIDs,
    pub get_device_ids: FnGetDeviceIDs,
    pub get_device_info: FnGetDeviceInfo,
    pub create_context: FnCreateContext,
    /// Removed from the OpenCL 2.0 headers but still exported by every
    /// ICD. `None` only if a runtime really does not have it.
    pub create_command_queue: Option<FnCreateCommandQueue>,
    pub create_command_queue_with_properties: Option<FnCreateCommandQueueWithProperties>,
    pub create_program_with_source: FnCreateProgramWithSource,
    pub build_program: FnBuildProgram,
    pub get_program_build_info: FnGetProgramBuildInfo,
    pub create_kernel: FnCreateKernel,
    pub get_kernel_work_group_info: FnGetKernelWorkGroupInfo,
    pub create_buffer: FnCreateBuffer,
    pub set_kernel_arg: FnSetKernelArg,
    pub enqueue_write_buffer: FnEnqueueBuffer,
    pub enqueue_read_buffer: FnEnqueueBuffer,
    pub enqueue_nd_range_kernel: FnEnqueueNDRange,
    pub finish: FnFinish,
    pub release_mem_object: FnRelease,
    pub release_kernel: FnRelease,
    pub release_program: FnRelease,
    pub release_command_queue: FnRelease,
    pub release_context: FnRelease,
    /// Must stay alive for as long as the function pointers are used.
    _lib: Library,
}

/// Library names per platform. The ICD loader goes by different names and is
/// not always on the search path under the same one.
#[cfg(target_os = "windows")]
const CANDIDATES: &[&str] = &["OpenCL.dll"];
#[cfg(target_os = "macos")]
const CANDIDATES: &[&str] = &["/System/Library/Frameworks/OpenCL.framework/OpenCL"];
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const CANDIDATES: &[&str] = &["libOpenCL.so.1", "libOpenCL.so", "libOpenCL.so.1.0.0"];

static CL: OnceLock<Result<Cl, String>> = OnceLock::new();

/// Get the binding. Loaded once; subsequent calls are free.
pub fn cl() -> Result<&'static Cl, String> {
    CL.get_or_init(load).as_ref().map_err(|e| e.clone())
}

fn load() -> Result<Cl, String> {
    let mut last = String::new();
    for name in CANDIDATES {
        // SAFETY: we load a named system library and only look up symbols
        // whose signatures come straight from the OpenCL spec.
        match unsafe { Library::new(name) } {
            Ok(lib) => return build(lib),
            Err(e) => last = format!("{name}: {e}"),
        }
    }
    Err(format!(
        "no OpenCL runtime found (tried {}) - last error: {last}",
        CANDIDATES.join(", ")
    ))
}

/// Look up a mandatory symbol.
unsafe fn sym<T: Copy>(lib: &Library, name: &[u8]) -> Result<T, String> {
    let s: Symbol<T> = lib
        .get(name)
        .map_err(|e| format!("OpenCL runtime is missing {}: {e}", String::from_utf8_lossy(name)))?;
    Ok(*s)
}

/// Optional symbol - if it is missing we use the other queue variant.
unsafe fn sym_opt<T: Copy>(lib: &Library, name: &[u8]) -> Option<T> {
    lib.get::<T>(name).ok().map(|s| *s)
}

fn build(lib: Library) -> Result<Cl, String> {
    unsafe {
        let cl = Cl {
            get_platform_ids: sym(&lib, b"clGetPlatformIDs\0")?,
            get_device_ids: sym(&lib, b"clGetDeviceIDs\0")?,
            get_device_info: sym(&lib, b"clGetDeviceInfo\0")?,
            create_context: sym(&lib, b"clCreateContext\0")?,
            create_command_queue: sym_opt(&lib, b"clCreateCommandQueue\0"),
            create_command_queue_with_properties: sym_opt(
                &lib,
                b"clCreateCommandQueueWithProperties\0",
            ),
            create_program_with_source: sym(&lib, b"clCreateProgramWithSource\0")?,
            build_program: sym(&lib, b"clBuildProgram\0")?,
            get_program_build_info: sym(&lib, b"clGetProgramBuildInfo\0")?,
            create_kernel: sym(&lib, b"clCreateKernel\0")?,
            get_kernel_work_group_info: sym(&lib, b"clGetKernelWorkGroupInfo\0")?,
            create_buffer: sym(&lib, b"clCreateBuffer\0")?,
            set_kernel_arg: sym(&lib, b"clSetKernelArg\0")?,
            enqueue_write_buffer: sym(&lib, b"clEnqueueWriteBuffer\0")?,
            enqueue_read_buffer: sym(&lib, b"clEnqueueReadBuffer\0")?,
            enqueue_nd_range_kernel: sym(&lib, b"clEnqueueNDRangeKernel\0")?,
            finish: sym(&lib, b"clFinish\0")?,
            release_mem_object: sym(&lib, b"clReleaseMemObject\0")?,
            release_kernel: sym(&lib, b"clReleaseKernel\0")?,
            release_program: sym(&lib, b"clReleaseProgram\0")?,
            release_command_queue: sym(&lib, b"clReleaseCommandQueue\0")?,
            release_context: sym(&lib, b"clReleaseContext\0")?,
            _lib: lib,
        };
        if cl.create_command_queue.is_none() && cl.create_command_queue_with_properties.is_none() {
            return Err("OpenCL runtime exports no way to create a command queue".into());
        }
        Ok(cl)
    }
}

/// Translate an error code into something readable. Only the codes we can
/// plausibly get.
pub fn err_str(code: cl_int) -> String {
    let name = match code {
        0 => "CL_SUCCESS",
        -1 => "CL_DEVICE_NOT_FOUND",
        -2 => "CL_DEVICE_NOT_AVAILABLE",
        -3 => "CL_COMPILER_NOT_AVAILABLE",
        -4 => "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        -5 => "CL_OUT_OF_RESOURCES",
        -6 => "CL_OUT_OF_HOST_MEMORY",
        -11 => "CL_BUILD_PROGRAM_FAILURE",
        -30 => "CL_INVALID_VALUE",
        -33 => "CL_INVALID_DEVICE",
        -34 => "CL_INVALID_CONTEXT",
        -36 => "CL_INVALID_COMMAND_QUEUE",
        -38 => "CL_INVALID_MEM_OBJECT",
        -45 => "CL_INVALID_PROGRAM_EXECUTABLE",
        -48 => "CL_INVALID_KERNEL",
        -49 => "CL_INVALID_ARG_INDEX",
        -50 => "CL_INVALID_ARG_VALUE",
        -51 => "CL_INVALID_ARG_SIZE",
        -52 => "CL_INVALID_KERNEL_ARGS",
        -54 => "CL_INVALID_WORK_GROUP_SIZE",
        -55 => "CL_INVALID_WORK_ITEM_SIZE",
        -61 => "CL_INVALID_BUFFER_SIZE",
        _ => "CL_ERROR",
    };
    format!("{name} ({code})")
}

/// `Err` if the code is not CL_SUCCESS.
pub fn check(what: &str, code: cl_int) -> Result<(), String> {
    if code == CL_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what}: {}", err_str(code)))
    }
}

/// Read a string property (device name, build logs) with a two-step call.
pub fn info_string(
    mut query: impl FnMut(usize, *mut c_void, *mut usize) -> cl_int,
) -> Result<String, String> {
    let mut len: usize = 0;
    let code = query(0, std::ptr::null_mut(), &mut len);
    check("query size", code)?;
    if len == 0 {
        return Ok(String::new());
    }
    let mut buf = vec![0u8; len];
    let code = query(len, buf.as_mut_ptr() as *mut c_void, std::ptr::null_mut());
    check("query value", code)?;
    // The strings are nul-terminated; drop the terminator and trailing junk.
    if let Some(pos) = buf.iter().position(|&b| b == 0) {
        buf.truncate(pos);
    }
    Ok(String::from_utf8_lossy(&buf).trim().to_string())
}

/// Helper: make a nul-terminated C string that cannot panic on interior nuls.
pub fn cstr(s: &str) -> CString {
    CString::new(s.replace('\0', " ")).expect("no interior nul after replace")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The constants must match the OpenCL spec exactly - a wrong value here
    /// turns into baffling runtime errors much later.
    #[test]
    fn constants_match_the_spec() {
        assert_eq!(CL_DEVICE_TYPE_GPU, 4);
        assert_eq!(CL_DEVICE_NAME, 0x102B);
        assert_eq!(CL_DEVICE_VENDOR, 0x102C);
        assert_eq!(CL_MEM_READ_WRITE, 1);
        assert_eq!(CL_MEM_READ_ONLY, 4);
        assert_eq!(CL_PROGRAM_BUILD_LOG, 0x1183);
        assert_eq!(CL_KERNEL_WORK_GROUP_SIZE, 0x11B0);
        assert_eq!(CL_SUCCESS, 0);
    }

    /// A missing runtime must give a sensible error, not a crash.
    #[test]
    fn missing_runtime_is_an_error_not_a_panic() {
        // We cannot uninstall OpenCL inside the test, so we only check that
        // the call returns (Ok or Err) without panicking.
        let r = cl();
        match r {
            Ok(_) => {}
            Err(e) => assert!(!e.is_empty(), "the error message must say something"),
        }
    }

    #[test]
    fn error_strings_name_the_code() {
        assert_eq!(err_str(-11), "CL_BUILD_PROGRAM_FAILURE (-11)");
        assert_eq!(err_str(-9999), "CL_ERROR (-9999)");
        assert!(check("x", 0).is_ok());
        assert!(check("x", -5).unwrap_err().contains("CL_OUT_OF_RESOURCES"));
    }

    /// Interior nul bytes must not break the CString conversion.
    #[test]
    fn cstr_survives_interior_nul() {
        assert_eq!(cstr("a\0b").to_str().unwrap(), "a b");
    }
}
