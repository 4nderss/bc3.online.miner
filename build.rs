//! Build script: precompile the GPU kernels to PTX - SHA3-256t, and BC3Hash.
//!
//! Why: NVRTC (`nvrtc64_*.dll` / `libnvrtc.so`) ships with the CUDA
//! **Toolkit**, not with the graphics driver. An end user with only the
//! driver installed therefore has no NVRTC - runtime compilation crashes
//! then. By compiling the kernels to PTX at BUILD time and embedding the
//! result in the binary, the client only needs `nvcuda.dll`/`libcuda.so`
//! (the driver), which JITs the PTX to machine code for the exact card
//! installed.
//!
//! BC3Hash is built four times, once per secp256k1 table window (10, 12, 14
//! and 16 bits, `-DBC3_WINDOW=N`): which one is fastest depends on the card's
//! L2. A release mines with 16; an internal-tools build picks one at run time
//! (`--bc3-window`), and its benchmark compares them.
//!
//! Order, per PTX:
//!   1. An explicit path in the environment (`BC3_PTX_PATH` for SHA3-256t,
//!      `BC3_BC3HASH_W<N>_PTX_PATH` for BC3Hash) - used by CI.
//!   2. `nvcc` in PATH - compile src/kernels/<source>.cl -> OUT_DIR.
//!   3. The checked-in `src/kernels/<name>.ptx` - fallback when nvcc is
//!      missing, which is the normal case.
//!
//! PTX is forward compatible: `compute_52` is JITed by the driver to
//! everything from Maxwell and upwards.

use std::path::{Path, PathBuf};
use std::process::Command;

/// One PTX: its file stem, the kernel source it comes from, the window it is
/// built for (BC3Hash only), the environment variable that can name a PTX
/// for it, and the env var the embedded path is handed to the crate in.
struct Kernel {
    name: &'static str,
    source: &'static str,
    window: Option<u32>,
    path_env: &'static str,
    rustc_env: &'static str,
}

const KERNELS: [Kernel; 5] = [
    Kernel {
        name: "sha3t",
        source: "sha3t",
        window: None,
        path_env: "BC3_PTX_PATH",
        rustc_env: "BC3_SHA3T_PTX",
    },
    Kernel {
        name: "bc3hash_w10",
        source: "bc3hash",
        window: Some(10),
        path_env: "BC3_BC3HASH_W10_PTX_PATH",
        rustc_env: "BC3_BC3HASH_W10_PTX",
    },
    Kernel {
        name: "bc3hash_w12",
        source: "bc3hash",
        window: Some(12),
        path_env: "BC3_BC3HASH_W12_PTX_PATH",
        rustc_env: "BC3_BC3HASH_W12_PTX",
    },
    Kernel {
        name: "bc3hash_w14",
        source: "bc3hash",
        window: Some(14),
        path_env: "BC3_BC3HASH_W14_PTX_PATH",
        rustc_env: "BC3_BC3HASH_W14_PTX",
    },
    Kernel {
        name: "bc3hash_w16",
        source: "bc3hash",
        window: Some(16),
        path_env: "BC3_BC3HASH_W16_PTX_PATH",
        rustc_env: "BC3_BC3HASH_W16_PTX",
    },
];

/// Lowest architecture the PTX must be JITable for (Maxwell and upwards).
const ARCH: &str = "compute_52";

/// Highest PTX ISA version the embedded kernels may declare.
///
/// A driver can only JIT the ISA versions it knows: build the PTX with a
/// newer toolkit than the user's driver and `cuModuleLoadData` fails with
/// CUDA_ERROR_UNSUPPORTED_PTX_VERSION - the miner then finds the card, opens
/// no backend for it and hashes nothing. This has already shipped once, with
/// ISA 8.8 (CUDA 12.9), which needs driver R575.
///
/// 8.0 comes from CUDA 12.0 and is JITable by driver R525 and newer, which
/// covers everything in use. Neither kernel needs anything newer - LOP3,
/// funnel shifts and 64-bit integer arithmetic, all of it there since Maxwell.
const MAX_PTX_ISA: (u32, u32) = (8, 0);

fn main() {
    // The CUDA backend is the only one that needs PTX.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is not set"));
    for kernel in &KERNELS {
        build_kernel(kernel, &out_dir);
    }
}

fn build_kernel(k: &Kernel, out_dir: &Path) {
    let source = format!("src/kernels/{}.cl", k.source);
    let checked_in = format!("src/kernels/{}.ptx", k.name);
    println!("cargo:rerun-if-changed={source}");
    println!("cargo:rerun-if-changed={checked_in}");
    println!("cargo:rerun-if-env-changed={}", k.path_env);
    let out_ptx = out_dir.join(format!("{}.ptx", k.name));

    if let Some(path) = std::env::var_os(k.path_env) {
        let path = PathBuf::from(path);
        std::fs::copy(&path, &out_ptx).unwrap_or_else(|e| {
            panic!("cannot read {} {}: {e}", k.path_env, path.display())
        });
        emit(k, &out_ptx);
        return;
    }

    // nvcc found on PATH is a CONVENIENCE, not a promise. Any toolkit newer
    // than 12.0 emits an ISA above what we ship, and failing the build there
    // would mean nobody with a current CUDA install could compile the miner at
    // all - over a file that is checked in and already correct. Warn and use
    // the committed PTX instead. The panic is kept for the two paths where a
    // human chose the file: the env path and the checked-in PTX itself.
    if compile_with_nvcc(&source, k.window, &out_ptx) {
        match ptx_isa(&out_ptx) {
            Ok(v) if v <= MAX_PTX_ISA => {
                emit(k, &out_ptx);
                return;
            }
            Ok((major, minor)) => println!(
                "cargo:warning=nvcc produced PTX ISA {major}.{minor}, above the {}.{} we ship - \
                 using the checked-in {checked_in} instead. To regenerate it, use the CUDA \
                 12.0 toolkit (see AGENTS.md).",
                MAX_PTX_ISA.0, MAX_PTX_ISA.1
            ),
            Err(e) => println!(
                "cargo:warning=could not read the PTX nvcc produced ({e}) - using {checked_in}"
            ),
        }
    }

    let checked_in_path = Path::new(&checked_in);
    if checked_in_path.exists() {
        println!("cargo:warning=nvcc not found - using the checked-in PTX ({checked_in})");
        std::fs::copy(checked_in_path, &out_ptx).expect("cannot copy the checked-in PTX");
        emit(k, &out_ptx);
        return;
    }

    panic!(
        "cannot produce {}.ptx: neither nvcc, {} nor {checked_in} is there.\n\
         Build with the CUDA toolkit on PATH, or check in a PTX (see AGENTS.md).",
        k.name, k.path_env
    );
}

/// Compile a kernel source with nvcc. `false` if nvcc is missing.
fn compile_with_nvcc(source: &str, window: Option<u32>, out_ptx: &Path) -> bool {
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let mut cmd = Command::new(&nvcc);
    cmd.arg("--ptx").arg("-arch").arg(ARCH).arg("-O3");
    if let Some(w) = window {
        cmd.arg(format!("-DBC3_WINDOW={w}"));
    }
    // The kernel sources are .cl (shared with OpenCL) - force the CUDA C++
    // frontend.
    cmd.arg("-x").arg("cu").arg(source).arg("-o").arg(out_ptx);

    match cmd.output() {
        Ok(out) if out.status.success() => true,
        Ok(out) => panic!(
            "nvcc failed on {source}:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        // nvcc is not in PATH - fall back on the checked-in PTX.
        Err(_) => false,
    }
}

fn emit(k: &Kernel, ptx: &Path) {
    check_ptx_isa(k, ptx);
    println!("cargo:rustc-env={}={}", k.rustc_env, ptx.display());
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
fn check_ptx_isa(k: &Kernel, ptx: &Path) {
    let (major, minor) = ptx_isa(ptx).unwrap_or_else(|e| panic!("{e}"));
    if (major, minor) > MAX_PTX_ISA {
        panic!(
            "{} declares PTX ISA {major}.{minor}, but the highest we ship is {}.{}.\n\
             A driver older than that version's cannot JIT the kernel, and the miner then\n\
             finds the GPU, opens no backend for it and hashes nothing.\n\
             Rebuild the kernel with the CUDA 12.0 toolkit (ISA 8.0):\n  \
             nvcc --ptx -arch {ARCH} -O3 {define}-x cu src/kernels/{source}.cl -o src/kernels/{name}.ptx",
            ptx.display(),
            MAX_PTX_ISA.0,
            MAX_PTX_ISA.1,
            define = k.window.map(|w| format!("-DBC3_WINDOW={w} ")).unwrap_or_default(),
            source = k.source,
            name = k.name,
        );
    }
}
