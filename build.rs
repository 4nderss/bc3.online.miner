//! Bygg-script: förkompilera SHA3-256t-kerneln till PTX.
//!
//! Varför: NVRTC (`nvrtc64_*.dll` / `libnvrtc.so`) ingår i CUDA **Toolkit**,
//! inte i grafikdrivrutinen. En slutanvändare med bara drivrutin installerad
//! har alltså ingen NVRTC — runtime-kompilering kraschar då. Genom att
//! kompilera kerneln till PTX vid BYGGET och bädda in resultatet i binären
//! behöver klienten bara `nvcuda.dll`/`libcuda.so` (drivrutinen), som JIT:ar
//! PTX:en till maskinkod för just det installerade kortet.
//!
//! Ordning:
//!   1. `BC3_PTX_PATH` — explicit sökväg (används av CI).
//!   2. `nvcc` i PATH — kompilera src/kernels/sha3t.cl → OUT_DIR/sha3t.ptx.
//!   3. Incheckad `src/kernels/sha3t.ptx` — fallback när nvcc saknas.
//!
//! PTX är framåtkompatibel: `compute_52` JIT:as av drivrutinen till allt från
//! Maxwell och uppåt.

use std::path::{Path, PathBuf};
use std::process::Command;

const KERNEL: &str = "src/kernels/sha3t.cl";
const CHECKED_IN_PTX: &str = "src/kernels/sha3t.ptx";
/// Lägsta arkitektur PTX:en ska kunna JIT:as för (Maxwell och uppåt).
const ARCH: &str = "compute_52";

fn main() {
    println!("cargo:rerun-if-changed={KERNEL}");
    println!("cargo:rerun-if-changed={CHECKED_IN_PTX}");
    println!("cargo:rerun-if-env-changed=BC3_PTX_PATH");

    // CUDA-backenden är den enda som behöver PTX.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR saknas"));
    let out_ptx = out_dir.join("sha3t.ptx");

    if let Some(path) = std::env::var_os("BC3_PTX_PATH") {
        let path = PathBuf::from(path);
        std::fs::copy(&path, &out_ptx)
            .unwrap_or_else(|e| panic!("kan inte läsa BC3_PTX_PATH {}: {e}", path.display()));
        emit(&out_ptx);
        return;
    }

    if compile_with_nvcc(&out_ptx) {
        emit(&out_ptx);
        return;
    }

    let checked_in = Path::new(CHECKED_IN_PTX);
    if checked_in.exists() {
        println!("cargo:warning=nvcc saknas — använder incheckad PTX ({CHECKED_IN_PTX})");
        std::fs::copy(checked_in, &out_ptx).expect("kan inte kopiera incheckad PTX");
        emit(&out_ptx);
        return;
    }

    panic!(
        "kan inte producera sha3t.ptx: varken nvcc, BC3_PTX_PATH eller {CHECKED_IN_PTX} finns.\n\
         Bygg med CUDA-toolkit i PATH, eller checka in en PTX (se docker/Dockerfile.cuda-dev)."
    );
}

/// Kompilera kernelkällan med nvcc. `false` om nvcc saknas.
fn compile_with_nvcc(out_ptx: &Path) -> bool {
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let mut cmd = Command::new(&nvcc);
    cmd.arg("--ptx")
        .arg("-arch")
        .arg(ARCH)
        .arg("-O3")
        // Kernelkällan är .cl (delad med OpenCL) — tvinga CUDA C++-frontend.
        .arg("-x")
        .arg("cu")
        .arg(KERNEL)
        .arg("-o")
        .arg(out_ptx);

    match cmd.output() {
        Ok(out) if out.status.success() => true,
        Ok(out) => panic!(
            "nvcc misslyckades:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        // nvcc finns inte i PATH — fall tillbaka på incheckad PTX.
        Err(_) => false,
    }
}

fn emit(ptx: &Path) {
    println!("cargo:rustc-env=BC3_SHA3T_PTX={}", ptx.display());
}
