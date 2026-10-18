//! A line on stderr when the process dies of a hardware exception on
//! Windows: which exception, at what address it read or wrote, and in which
//! module at what offset - "bc3-miner: crashed - access violation
//! (0xC0000005) reading 0x0000000000000010 in nvml.dll+0x1a2b3".
//!
//! Without it, the GUI could only say "miner exited (code -1073741819)", and
//! a crash on someone else's machine could not be placed: in our code, in the
//! node's C++, or in a driver library we load (nvml.dll, nvcuda.dll, an
//! OpenCL ICD). The GUI shows the core's stderr in its activity log, so the
//! line reaches whoever reports the crash.
//!
//! The filter only reports. It returns EXCEPTION_CONTINUE_SEARCH, so Windows
//! ends the process exactly as before: same exit code, same Windows Error
//! Reporting entry. Rust's own stack-overflow message comes from a vectored
//! handler that runs first and is not affected.

/// Install the filter. Call first thing in main, before any thread starts.
#[cfg(windows)]
pub fn install() {
    // SAFETY: registers a process-wide callback; `filter` is a plain
    // extern "system" function with the signature Windows expects.
    unsafe {
        imp::SetUnhandledExceptionFilter(Some(imp::filter));
    }
}

#[cfg(not(windows))]
pub fn install() {}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;

    #[repr(C)]
    pub struct ExceptionRecord {
        pub code: u32,
        pub flags: u32,
        pub record: *mut ExceptionRecord,
        pub address: *mut c_void,
        pub parameters: u32,
        pub information: [usize; 15],
    }

    #[repr(C)]
    pub struct ExceptionPointers {
        pub record: *mut ExceptionRecord,
        pub context: *mut c_void,
    }

    type Filter = unsafe extern "system" fn(*mut ExceptionPointers) -> i32;

    extern "system" {
        pub fn SetUnhandledExceptionFilter(filter: Option<Filter>) -> Option<Filter>;
        fn GetModuleHandleExW(flags: u32, address: *const u16, module: *mut *mut c_void) -> i32;
        fn GetModuleFileNameW(module: *mut c_void, name: *mut u16, size: u32) -> u32;
        fn GetStdHandle(which: u32) -> *mut c_void;
        fn WriteFile(file: *mut c_void, data: *const u8, len: u32, written: *mut u32, overlapped: *mut c_void) -> i32;
    }

    const FROM_ADDRESS: u32 = 0x4;
    const UNCHANGED_REFCOUNT: u32 = 0x2;
    const STD_ERROR_HANDLE: u32 = -12i32 as u32;
    const CONTINUE_SEARCH: i32 = 0;

    /// A fixed buffer: the filter must not allocate, since a damaged heap may
    /// be what crashed.
    pub struct Line {
        buf: [u8; 512],
        len: usize,
    }

    impl Line {
        pub fn new() -> Self {
            Self { buf: [0; 512], len: 0 }
        }
        pub fn push(&mut self, bytes: &[u8]) {
            for &b in bytes {
                if self.len < self.buf.len() {
                    self.buf[self.len] = b;
                    self.len += 1;
                }
            }
        }
        pub fn hex(&mut self, value: u64, digits: usize) {
            const HEX: &[u8; 16] = b"0123456789ABCDEF";
            for i in (0..digits).rev() {
                self.push(&[HEX[((value >> (i * 4)) & 0xF) as usize]]);
            }
        }
        pub fn bytes(&self) -> &[u8] {
            &self.buf[..self.len]
        }
    }

    pub fn name(code: u32) -> &'static [u8] {
        match code {
            0xC000_0005 => b"access violation",
            0xC000_001D => b"illegal instruction",
            0xC000_0096 => b"privileged instruction",
            0xC000_0094 => b"integer division by zero",
            0xC000_00FD => b"stack overflow",
            0xC000_0006 => b"in-page error",
            0x8000_0002 => b"datatype misalignment",
            0xC000_0409 => b"stack buffer overrun",
            0xC000_0374 => b"heap corruption",
            _ => b"exception",
        }
    }

    /// The report for one exception, without the module part.
    pub fn describe(line: &mut Line, code: u32, parameters: u32, information: &[usize; 15]) {
        line.push(b"bc3-miner: crashed - ");
        line.push(name(code));
        line.push(b" (0x");
        line.hex(code as u64, 8);
        line.push(b")");
        // An access violation says what it did and where: 0 read, 1 write,
        // 8 execute (DEP).
        if (code == 0xC000_0005 || code == 0xC000_0006) && parameters >= 2 {
            line.push(match information[0] {
                0 => b" reading 0x" as &[u8],
                1 => b" writing 0x",
                8 => b" executing 0x",
                _ => b" at 0x",
            });
            line.hex(information[1] as u64, 16);
        }
    }

    pub unsafe extern "system" fn filter(info: *mut ExceptionPointers) -> i32 {
        let mut line = Line::new();
        let record = if info.is_null() { std::ptr::null_mut() } else { (*info).record };
        if record.is_null() {
            line.push(b"bc3-miner: crashed - unknown exception");
        } else {
            let r = &*record;
            describe(&mut line, r.code, r.parameters, &r.information);
            // The module the faulting instruction belongs to, and the offset
            // into it: enough to find the function in that module's build.
            let mut module: *mut c_void = std::ptr::null_mut();
            if GetModuleHandleExW(FROM_ADDRESS | UNCHANGED_REFCOUNT, r.address as *const u16, &mut module) != 0
                && !module.is_null()
            {
                let mut name = [0u16; 260];
                let n = GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32) as usize;
                let path = &name[..n.min(name.len())];
                let base = path.iter().rposition(|&c| c == b'\\' as u16).map_or(0, |i| i + 1);
                line.push(b" in ");
                for &c in &path[base..] {
                    line.push(&[if c < 0x80 { c as u8 } else { b'?' }]);
                }
                line.push(b"+0x");
                let offset = (r.address as usize).wrapping_sub(module as usize) as u64;
                line.hex(offset, if offset > 0xFFFF_FFFF { 16 } else { 8 });
            } else {
                line.push(b" at 0x");
                line.hex(r.address as u64, 16);
            }
        }
        line.push(b"\r\n");
        let out = GetStdHandle(STD_ERROR_HANDLE);
        if !out.is_null() && out as isize != -1 {
            let mut written = 0u32;
            WriteFile(out, line.bytes().as_ptr(), line.bytes().len() as u32, &mut written, std::ptr::null_mut());
        }
        CONTINUE_SEARCH
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::imp::*;

    #[test]
    fn an_access_violation_says_what_and_where() {
        let mut info = [0usize; 15];
        info[0] = 0;
        info[1] = 0x10;
        let mut line = Line::new();
        describe(&mut line, 0xC000_0005, 2, &info);
        assert_eq!(
            std::str::from_utf8(line.bytes()).unwrap(),
            "bc3-miner: crashed - access violation (0xC0000005) reading 0x0000000000000010"
        );
    }

    /// The real thing: a child process that dereferences a bad pointer with
    /// the filter installed. It must still die with the same exit code as
    /// without the filter, and say where on stderr.
    #[test]
    fn a_crash_is_reported_on_stderr() {
        if std::env::var_os("BC3_CRASH_CHILD").is_some() {
            super::install();
            // SAFETY: deliberately not - the point is the hardware exception.
            unsafe {
                std::ptr::read_volatile(std::ptr::without_provenance::<u32>(0x10));
            }
            unreachable!();
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "crash::tests::a_crash_is_reported_on_stderr", "--nocapture", "--test-threads=1"])
            .env("BC3_CRASH_CHILD", "1")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(0xC000_0005u32 as i32), "stderr: {stderr}");
        assert!(
            stderr.contains("crashed - access violation (0xC0000005) reading 0x0000000000000010 in ")
                && stderr.contains(".exe+0x"),
            "stderr: {stderr}"
        );
    }
}
