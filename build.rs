//! Build script: precompile the BC3Hash GPU kernel to PTX.
//!
//! Why: NVRTC (`nvrtc64_*.dll` / `libnvrtc.so`) ships with the CUDA
//! **Toolkit**, not with the graphics driver. An end user with only the
//! driver installed therefore has no NVRTC - runtime compilation crashes
//! then. By compiling the kernel to PTX at BUILD time and embedding the
//! result in the binary, the client only needs `nvcuda.dll`/`libcuda.so`
//! (the driver), which JITs the PTX to machine code for the exact card
//! installed.
//!
//! The kernel is built once per secp256k1 table window (16, 18, 19, 20, 22
//! and 24 bits): every card starts at 16, and the window tuning
//! (src/tuning.rs) moves a card that hashes faster with a bigger one. Once
//! more it is built as the table builder alone (`-DBC3_BUILD_ONLY`), which a
//! CUDA card runs to make a window's tables on itself. `KERNELS` has the
//! defines of each, as tools/ptx.sh does - the list CI checks the committed
//! PTX against.
//!
//! Beside the PTX goes the machine code tools/sass.sh made from it for the
//! common architectures (`embed_sass`): a card that has its own loads it in
//! a second, where the driver's JIT of the PTX takes 20-60 s - at every
//! start where its cache does not keep the result, and at every switch of
//! the window tuning.
//!
//! Order, per PTX:
//!   1. An explicit path in the environment (`BC3_<STEM>_PTX_PATH`, as
//!      `BC3_BC3HASH_W16_PTX_PATH`).
//!   2. `nvcc` in PATH - compile src/kernels/bc3hash.cl -> OUT_DIR.
//!   3. The checked-in `src/kernels/<stem>.ptx` - fallback when nvcc is
//!      missing, which is the normal case.
//!
//! PTX is forward compatible: `compute_52` is JITed by the driver to
//! everything from Maxwell and upwards.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The kernel source every PTX is built from.
const SOURCE: &str = "src/kernels/bc3hash.cl";

/// Every PTX the CUDA backend embeds, and the defines it is built with - the
/// same as tools/ptx.sh's. 16 is where every card starts; the bigger windows
/// are those the tuning (src/tuning.rs) may move a card with the memory to.
/// 22 and 24 are mixed schedules: 12 digits of 21 and 22 bits, 11 of 23 and
/// 24 (`BC3_WIDE` in the kernel). 10-14 never beat 16 on any card measured,
/// and cost 2.4 MB each.
const KERNELS: [Kernel; 7] = [
    Kernel { stem: "bc3hash_w16", defines: &["BC3_WINDOW=16"] },
    Kernel { stem: "bc3hash_w18", defines: &["BC3_WINDOW=18"] },
    Kernel { stem: "bc3hash_w19", defines: &["BC3_WINDOW=19"] },
    Kernel { stem: "bc3hash_w20", defines: &["BC3_WINDOW=20"] },
    Kernel { stem: "bc3hash_w22", defines: &["BC3_WINDOW=21", "BC3_WIDE=4"] },
    Kernel { stem: "bc3hash_w24", defines: &["BC3_WINDOW=23", "BC3_WIDE=3"] },
    Kernel { stem: "bc3hash_build", defines: &["BC3_BUILD_ONLY"] },
];

/// Lowest architecture the PTX must be JITable for (Maxwell and upwards).
const ARCH: &str = "compute_52";

/// Highest PTX ISA version the embedded kernels may declare.
///
/// A driver can only JIT the ISA versions it knows: build the PTX with a
/// newer toolkit than the user's driver and `cuModuleLoadData` fails with
/// CUDA_ERROR_UNSUPPORTED_PTX_VERSION - the miner then finds the card and
/// hashes nothing on it. This has already shipped once, with ISA 8.8 (CUDA
/// 12.9), which needs driver R575.
///
/// 8.0 comes from CUDA 12.0 and is JITable by driver R525 and newer, which
/// covers everything in use; the CUDA backend refuses an older driver when it
/// opens a card (`MIN_DRIVER_CUDA` in src/backend/cuda.rs), so that the card
/// goes to OpenCL instead. The kernel needs nothing newer: integer and
/// carry-chain arithmetic, all of it there since Maxwell.
const MAX_PTX_ISA: (u32, u32) = (8, 0);

/// One PTX: its file stem - in src/kernels/ and in OUT_DIR - and the defines
/// it is built with.
struct Kernel {
    stem: &'static str,
    defines: &'static [&'static str],
}

impl Kernel {
    /// The environment variable that can name a PTX for it.
    fn path_env(&self) -> String {
        format!("BC3_{}_PTX_PATH", self.stem.to_uppercase())
    }

    /// The variable the embedded path is handed to the crate in.
    fn rustc_env(&self) -> String {
        format!("BC3_{}_PTX", self.stem.to_uppercase())
    }
}

fn main() {
    // The CUDA backend is the only one that needs PTX.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is not set"));
    println!("cargo:rerun-if-changed={SOURCE}");
    for k in &KERNELS {
        build_kernel(k, &out_dir);
    }
    embed_sass(&out_dir);
}

/// The architectures there is machine code for (tools/sass.sh): Pascal,
/// Turing, Ampere, Ada and Blackwell - sm_61 runs on sm_62, sm_86 on sm_87,
/// sm_120 on sm_121. Anything else (sm_70, sm_80, sm_90, sm_100) JITs the PTX.
const SASS_ARCHS: [u32; 5] = [61, 75, 86, 89, 120];

/// Embed the kernels' machine code from tools/sass.sh - BC3_SASS_DIR, else
/// src/kernels/sass - deflated, and write `sass.rs` to OUT_DIR: the table the
/// CUDA backend loads a card's code from. A kernel's code goes in only if the
/// manifest says it was made from the very PTX this build embeds; stale code
/// would at best fail the cards' self-test. With BC3_REQUIRE_SASS set, as in
/// the release workflow, anything missing fails the build. Without it the
/// build goes on with what there is, and the other cards JIT the PTX.
fn embed_sass(out_dir: &Path) {
    println!("cargo:rerun-if-env-changed=BC3_SASS_DIR");
    println!("cargo:rerun-if-env-changed=BC3_REQUIRE_SASS");
    let dir = std::env::var_os("BC3_SASS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("src/kernels/sass"));
    println!("cargo:rerun-if-changed={}", dir.display());
    let require = std::env::var("BC3_REQUIRE_SASS").is_ok_and(|v| !v.is_empty() && v != "0");
    let manifest = std::fs::read_to_string(dir.join("manifest.txt")).unwrap_or_default();

    let mut entries = Vec::new();
    let mut missing = Vec::new();
    for k in &KERNELS {
        let name = k.stem;
        // The PTX embedded beside it, its line endings as tools/sass.sh hashed
        // them: a Windows checkout has CRLF.
        let ptx = std::fs::read(out_dir.join(format!("{name}.ptx"))).expect("the embedded PTX");
        let lf: Vec<u8> = ptx.into_iter().filter(|&b| b != b'\r').collect();
        let digest = sha256_hex(&lf);
        let made_from = manifest.lines().find_map(|l| {
            let (hash, file) = l.split_once("  ")?;
            (file.trim() == format!("{name}.ptx")).then(|| hash.trim().to_string())
        });
        if made_from.as_deref() != Some(digest.as_str()) {
            if !manifest.is_empty() {
                println!(
                    "cargo:warning=no machine code for {name}: {} was made from another PTX - \
                     run tools/sass.sh again",
                    dir.display()
                );
            }
            missing.extend(SASS_ARCHS.iter().map(|sm| format!("{name}_sm{sm}")));
            continue;
        }
        for sm in SASS_ARCHS {
            let Ok(code) = std::fs::read(dir.join(format!("{name}_sm{sm}.cubin"))) else {
                missing.push(format!("{name}_sm{sm}"));
                continue;
            };
            let file = format!("{name}_sm{sm}.cubin.z");
            std::fs::write(out_dir.join(&file), miniz_oxide::deflate::compress_to_vec(&code, 10))
                .expect("write the packed machine code");
            entries.push((name, sm, file));
        }
    }
    if require && !missing.is_empty() {
        panic!(
            "BC3_REQUIRE_SASS is set, and machine code is missing or stale: {}\n\
             Run tools/sass.sh (see AGENTS.md).",
            missing.join(", ")
        );
    }
    let mut src = String::from(
        "/// The BC3Hash kernels' machine code, deflated: (PTX stem, sm, bytes). See build.rs.\n\
         pub static SASS: &[(&str, u32, &[u8])] = &[\n",
    );
    for (name, sm, file) in &entries {
        src.push_str(&format!(
            "    (\"{name}\", {sm}, include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{file}\"))),\n"
        ));
    }
    src.push_str("];\n");
    std::fs::write(out_dir.join("sass.rs"), src).expect("write sass.rs");
}

fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

fn build_kernel(k: &Kernel, out_dir: &Path) {
    let name = k.stem;
    let path_env = k.path_env();
    let checked_in = format!("src/kernels/{name}.ptx");
    println!("cargo:rerun-if-changed={checked_in}");
    println!("cargo:rerun-if-env-changed={path_env}");
    let out_ptx = out_dir.join(format!("{name}.ptx"));

    if let Some(path) = std::env::var_os(&path_env) {
        let path = PathBuf::from(path);
        std::fs::copy(&path, &out_ptx).unwrap_or_else(|e| {
            panic!("cannot read {path_env} {}: {e}", path.display())
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
    if compile_with_nvcc(k, &out_ptx) {
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
        "cannot produce {name}.ptx: neither nvcc, {path_env} nor {checked_in} is there.\n\
         Build with the CUDA toolkit on PATH, or check in a PTX (see AGENTS.md)."
    );
}

/// Compile one kernel with nvcc. `false` if nvcc is missing.
fn compile_with_nvcc(k: &Kernel, out_ptx: &Path) -> bool {
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let mut cmd = Command::new(&nvcc);
    cmd.arg("--ptx").arg("-arch").arg(ARCH).arg("-O3");
    for define in k.defines {
        cmd.arg(format!("-D{define}"));
    }
    // The kernel source is .cl (shared with OpenCL) - force the CUDA C++
    // frontend.
    cmd.arg("-x").arg("cu").arg(SOURCE).arg("-o").arg(out_ptx);

    match cmd.output() {
        Ok(out) if out.status.success() => true,
        Ok(out) => panic!(
            "nvcc failed on {SOURCE} ({}):\n{}\n{}",
            k.stem,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        // nvcc is not in PATH - fall back on the checked-in PTX.
        Err(_) => false,
    }
}

fn emit(k: &Kernel, ptx: &Path) {
    check_ptx_isa(k, ptx);
    println!("cargo:rustc-env={}={}", k.rustc_env(), ptx.display());
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
             finds the GPU and hashes nothing on it.\n\
             Rebuild {} with the CUDA 12.0 toolkit (ISA 8.0): tools/ptx.sh, see AGENTS.md.",
            ptx.display(),
            MAX_PTX_ISA.0,
            MAX_PTX_ISA.1,
            k.stem,
        );
    }
}
