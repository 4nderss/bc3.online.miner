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

    if compile_with_nvcc(&out_ptx) {
        emit(&out_ptx);
        return;
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
    println!("cargo:rustc-env=BC3_SHA3T_PTX={}", ptx.display());
}
