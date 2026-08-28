//! Minimal OpenCL-bindning som laddas DYNAMISKT i runtime.
//!
//! Varför inte `opencl3`/`ocl`: de länkar mot `OpenCL.lib`/`libOpenCL.so` vid
//! byggtillfället. En sådan binär vägrar starta på en maskin utan
//! OpenCL-runtime — alltså skulle en NVIDIA-användare utan OpenCL, eller vem
//! som helst utan GPU, inte kunna köra minern alls. Samma skäl som gör att
//! CUDA laddas dynamiskt (cudarc `fallback-dynamic-loading`).
//!
//! Med den här laddaren blir det EN binär: CUDA på NVIDIA, OpenCL på
//! AMD/Intel, CPU om inget finns. Saknas biblioteket får vi ett Err i stället
//! för en process som inte startar.
//!
//! Bara de ~20 funktioner minern faktiskt använder bindas.

#![allow(non_camel_case_types)]

use libloading::{Library, Symbol};
use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::sync::OnceLock;

pub type cl_int = i32;
pub type cl_uint = u32;
pub type cl_bitfield = u64;
pub type cl_bool = cl_uint;
/// Alla OpenCL-objekt är ogenomskinliga pekare.
pub type cl_handle = *mut c_void;

pub const CL_SUCCESS: cl_int = 0;
pub const CL_TRUE: cl_bool = 1;
pub const CL_DEVICE_TYPE_GPU: cl_bitfield = 1 << 2;
/// Alla enhetstyper — bara för tester (pocl exponerar en CPU-enhet).
pub const CL_DEVICE_TYPE_ALL: cl_bitfield = 0xFFFF_FFFF;
pub const CL_DEVICE_NAME: cl_uint = 0x102B;
pub const CL_MEM_READ_WRITE: cl_bitfield = 1 << 0;
pub const CL_MEM_READ_ONLY: cl_bitfield = 1 << 2;
pub const CL_PROGRAM_BUILD_LOG: cl_uint = 0x1183;

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
    /// Borttagen ur OpenCL 2.0-headers men fortfarande exporterad av alla
    /// ICD:er. `None` bara om en runtime verkligen saknar den.
    pub create_command_queue: Option<FnCreateCommandQueue>,
    pub create_command_queue_with_properties: Option<FnCreateCommandQueueWithProperties>,
    pub create_program_with_source: FnCreateProgramWithSource,
    pub build_program: FnBuildProgram,
    pub get_program_build_info: FnGetProgramBuildInfo,
    pub create_kernel: FnCreateKernel,
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
    /// Måste leva så länge funktionspekarna används.
    _lib: Library,
}

/// Biblioteksnamn per plattform. ICD-loadern heter olika saker och ligger
/// inte alltid i sökvägen med samma namn.
#[cfg(target_os = "windows")]
const CANDIDATES: &[&str] = &["OpenCL.dll"];
#[cfg(target_os = "macos")]
const CANDIDATES: &[&str] = &["/System/Library/Frameworks/OpenCL.framework/OpenCL"];
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const CANDIDATES: &[&str] = &["libOpenCL.so.1", "libOpenCL.so", "libOpenCL.so.1.0.0"];

static CL: OnceLock<Result<Cl, String>> = OnceLock::new();

/// Hämta bindningen. Laddas en gång; efterföljande anrop är gratis.
pub fn cl() -> Result<&'static Cl, String> {
    CL.get_or_init(load).as_ref().map_err(|e| e.clone())
}

fn load() -> Result<Cl, String> {
    let mut last = String::new();
    for name in CANDIDATES {
        // SAFETY: vi laddar ett namngivet systembibliotek och slår bara upp
        // symboler med signaturer ur OpenCL-specen.
        match unsafe { Library::new(name) } {
            Ok(lib) => return build(lib),
            Err(e) => last = format!("{name}: {e}"),
        }
    }
    Err(format!(
        "no OpenCL runtime found (tried {}) — last error: {last}",
        CANDIDATES.join(", ")
    ))
}

/// Slå upp en obligatorisk symbol.
unsafe fn sym<T: Copy>(lib: &Library, name: &[u8]) -> Result<T, String> {
    let s: Symbol<T> = lib
        .get(name)
        .map_err(|e| format!("OpenCL runtime is missing {}: {e}", String::from_utf8_lossy(name)))?;
    Ok(*s)
}

/// Valfri symbol — saknas den använder vi den andra kövarianten.
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

/// Översätt en felkod till något läsbart. Bara koderna vi rimligen kan få.
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

/// `Err` om koden inte är CL_SUCCESS.
pub fn check(what: &str, code: cl_int) -> Result<(), String> {
    if code == CL_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what}: {}", err_str(code)))
    }
}

/// Läs en strängegenskap (enhetsnamn, byggloggar) med tvåstegsanrop.
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
    // Strängarna är nullterminerade; ta bort terminatorn och skräp efter den.
    if let Some(pos) = buf.iter().position(|&b| b == 0) {
        buf.truncate(pos);
    }
    Ok(String::from_utf8_lossy(&buf).trim().to_string())
}

/// Hjälpare: gör en nullterminerad C-sträng utan att kunna panika på inre nollor.
pub fn cstr(s: &str) -> CString {
    CString::new(s.replace('\0', " ")).expect("no interior nul after replace")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Konstanterna måste matcha OpenCL-specen exakt — fel värde här ger
    /// obegripliga runtime-fel långt senare.
    #[test]
    fn constants_match_the_spec() {
        assert_eq!(CL_DEVICE_TYPE_GPU, 4);
        assert_eq!(CL_DEVICE_NAME, 0x102B);
        assert_eq!(CL_MEM_READ_WRITE, 1);
        assert_eq!(CL_MEM_READ_ONLY, 4);
        assert_eq!(CL_PROGRAM_BUILD_LOG, 0x1183);
        assert_eq!(CL_SUCCESS, 0);
    }

    /// Saknad runtime ska ge ett begripligt fel, inte en krasch.
    #[test]
    fn missing_runtime_is_an_error_not_a_panic() {
        // Vi kan inte avinstallera OpenCL i testet, så vi kontrollerar bara
        // att anropet returnerar (Ok eller Err) utan att panika.
        let r = cl();
        match r {
            Ok(_) => {}
            Err(e) => assert!(!e.is_empty(), "felmeddelandet måste säga något"),
        }
    }

    #[test]
    fn error_strings_name_the_code() {
        assert_eq!(err_str(-11), "CL_BUILD_PROGRAM_FAILURE (-11)");
        assert_eq!(err_str(-9999), "CL_ERROR (-9999)");
        assert!(check("x", 0).is_ok());
        assert!(check("x", -5).unwrap_err().contains("CL_OUT_OF_RESOURCES"));
    }

    /// Inre nollbytes får inte fälla CString-konverteringen.
    #[test]
    fn cstr_survives_interior_nul() {
        assert_eq!(cstr("a\0b").to_str().unwrap(), "a b");
    }
}
