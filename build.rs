//! Build script: precompile the SHA3-256t kernel to PTX.
//!
//! Why: NVRTC (`nvrtc64_*.dll` / `libnvrtc.so`) ships with the CUDA
//! **Toolkit**, not with the graphics driver. An end user with only the
//! driver installed therefore has no NVRTC - runtime compilation crashes
//! then. By compiling the kernel to PTX at BUILD time and embedding the
//! result in the binary, the client only needs `nvcuda.dll`/`libcuda.so`
//! (the driver), which JITs the PTX to machine code for the exact card
//! installed.
//!
//! Order:
//!   1. `BC3_PTX_PATH` - explicit path (used by CI).
//!   2. `nvcc` in PATH - compile src/kernels/sha3t.cl -> OUT_DIR/sha3t.ptx.
//!   3. Checked-in `src/kernels/sha3t.ptx` - fallback when nvcc is missing.
//!
//! PTX is forward compatible: `compute_52` is JITed by the driver to
//! everything from Maxwell and upwards.

use std::path::{Path, PathBuf};
use std::process::Command;

const KERNEL: &str = "src/kernels/sha3t.cl";
const CHECKED_IN_PTX: &str = "src/kernels/sha3t.ptx";
/// Lowest architecture the PTX must be JITable for (Maxwell and upwards).
const ARCH: &str = "compute_52";

/// Highest PTX ISA version the embedded kernel may declare.
///
/// A driver can only JIT the ISA versions it knows: build the PTX with a
/// newer toolkit than the user's driver and `cuModuleLoadData` fails with
/// CUDA_ERROR_UNSUPPORTED_PTX_VERSION - the miner then finds the card, opens
/// no backend for it and hashes nothing. This has already shipped once, with
/// ISA 8.8 (CUDA 12.9), which needs driver R575.
///
/// 8.0 comes from CUDA 12.0 and is JITable by driver R525 and newer, which
/// covers everything in use. The kernel needs nothing newer - it is LOP3,
/// funnel shifts and 64-bit logic, all of it available since Maxwell.
const MAX_PTX_ISA: (u32, u32) = (8, 0);

fn main() {
    println!("cargo:rerun-if-changed={KERNEL}");
    println!("cargo:rerun-if-changed={CHECKED_IN_PTX}");
    println!("cargo:rerun-if-env-changed=BC3_PTX_PATH");

    // The CUDA backend is the only one that needs PTX.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is not set"));
    let out_ptx = out_dir.join("sha3t.ptx");

    if let Some(path) = std::env::var_os("BC3_PTX_PATH") {
        let path = PathBuf::from(path);
        std::fs::copy(&path, &out_ptx)
            .unwrap_or_else(|e| panic!("cannot read BC3_PTX_PATH {}: {e}", path.display()));
        emit(&out_ptx);
        return;
    }

    // nvcc found on PATH is a CONVENIENCE, not a promise. Any toolkit newer
    // than 12.0 emits an ISA above what we ship, and failing the build there
    // would mean nobody with a current CUDA install could compile the miner at
    // all - over a file that is checked in and already correct. Warn and use
    // the committed PTX instead. The panic is kept for the two paths where a
    // human chose the file: BC3_PTX_PATH and the checked-in PTX itself.
    if compile_with_nvcc(&out_ptx) {
        match ptx_isa(&out_ptx) {
            Ok(v) if v <= MAX_PTX_ISA => {
                emit(&out_ptx);
                return;
            }
            Ok((major, minor)) => println!(
                "cargo:warning=nvcc produced PTX ISA {major}.{minor}, above the {}.{} we ship - \
                 using the checked-in {CHECKED_IN_PTX} instead. To regenerate it, use the CUDA \
                 12.0 toolkit (see AGENTS.md).",
                MAX_PTX_ISA.0, MAX_PTX_ISA.1
            ),
            Err(e) => println!("cargo:warning=could not read the PTX nvcc produced ({e}) - using {CHECKED_IN_PTX}"),
        }
    }

    let checked_in = Path::new(CHECKED_IN_PTX);
    if checked_in.exists() {
        println!("cargo:warning=nvcc not found - using the checked-in PTX ({CHECKED_IN_PTX})");
        std::fs::copy(checked_in, &out_ptx).expect("cannot copy the checked-in PTX");
        emit(&out_ptx);
        return;
    }

    panic!(
        "kan inte producera sha3t.ptx: varken nvcc, BC3_PTX_PATH eller {CHECKED_IN_PTX} finns.\n\
         Bygg med CUDA-toolkit i PATH, eller checka in en PTX (se docker/Dockerfile.cuda-dev)."
    );
}

/// Compile the kernel source with nvcc. `false` if nvcc is missing.
fn compile_with_nvcc(out_ptx: &Path) -> bool {
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let mut cmd = Command::new(&nvcc);
    cmd.arg("--ptx")
        .arg("-arch")
        .arg(ARCH)
        .arg("-O3")
        // The kernel source is .cl (shared with OpenCL) - force the CUDA C++
        // frontend.
        .arg("-x")
        .arg("cu")
        .arg(KERNEL)
        .arg("-o")
        .arg(out_ptx);

    match cmd.output() {
        Ok(out) if out.status.success() => true,
        Ok(out) => panic!(
            "nvcc failed:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        // nvcc is not in PATH - fall back on the checked-in PTX.
        Err(_) => false,
    }
}

fn emit(ptx: &Path) {
    check_ptx_isa(ptx);
    println!("cargo:rustc-env=BC3_SHA3T_PTX={}", ptx.display());
}

/// The ISA version a PTX file declares.
fn ptx_isa(ptx: &Path) -> Result<(u32, u32), String> {
    let text = std::fs::read_to_string(ptx).map_err(|e| format!("cannot read {}: {e}", ptx.display()))?;
    let version = text
        .lines()
        .find_map(|l| l.trim().strip_prefix(".version "))
        .map(str::trim)
        .ok_or_else(|| format!("{} has no .version directive", ptx.display()))?;
    version
        .split_once('.')
        .and_then(|(a, b)| Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?)))
        .ok_or_else(|| format!("cannot read the ISA version from '{version}'"))
}

/// Refuse to embed a PTX that the target drivers cannot load. See MAX_PTX_ISA.
fn check_ptx_isa(ptx: &Path) {
    let (major, minor) = ptx_isa(ptx).unwrap_or_else(|e| panic!("{e}"));
    if (major, minor) > MAX_PTX_ISA {
        panic!(
            "{} declares PTX ISA {major}.{minor}, but the highest we ship is {}.{}.\n\
             A driver older than that version's cannot JIT the kernel, and the miner then\n\
             finds the GPU, opens no backend for it and hashes nothing.\n\
             Rebuild the kernel with the CUDA 12.0 toolkit (ISA 8.0):\n  \
             nvcc --ptx -arch {ARCH} -O3 -x cu {KERNEL} -o {CHECKED_IN_PTX}",
            ptx.display(),
            MAX_PTX_ISA.0,
            MAX_PTX_ISA.1,
        );
    }
}
