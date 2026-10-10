//! Builds the node's BC3Hash reference (C++) and the secp256k1 it uses, from
//! the node's own sources: `vendor/bc3` is `src/crypto/bc3` and
//! `vendor/secp256k1` is the `src/secp256k1` subtree of bitcoiniii-core, at the
//! commit in `vendor/SOURCE_COMMIT`. Nothing in vendor/ is edited here; the
//! only code of ours is `src/shim.cpp`.
//!
//! `BC3HASH_VENDOR` points the build at the same layout somewhere else.
//!
//! secp256k1 is built with the choices Core makes in cmake/secp256k1.cmake,
//! minus the modules - the mix only parses, combines and serializes public
//! keys.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=BC3HASH_VENDOR");
    println!("cargo:rerun-if-changed=src/shim.cpp");
    let vendor = std::env::var_os("BC3HASH_VENDOR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("vendor"));
    println!("cargo:rerun-if-changed={}", vendor.display());
    if !vendor.join("bc3/mix.cpp").exists() || !vendor.join("secp256k1/src/secp256k1.c").exists()
    {
        panic!(
            "the BC3Hash sources are missing from {}: copy src/crypto/bc3 and src/secp256k1 \
             of bitcoiniii-core (commit caae93e8ae) to vendor/bc3 and vendor/secp256k1, or set \
             BC3HASH_VENDOR to a directory laid out that way",
            vendor.display()
        );
    }
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    let native = std::env::var("TARGET").ok() == std::env::var("HOST").ok();

    let mut secp = cc::Build::new();
    secp.include(vendor.join("secp256k1/include"))
        .include(vendor.join("secp256k1/src"))
        .file(vendor.join("secp256k1/src/secp256k1.c"))
        .file(vendor.join("secp256k1/src/precomputed_ecmult.c"))
        .file(vendor.join("secp256k1/src/precomputed_ecmult_gen.c"))
        .define("ECMULT_WINDOW_SIZE", "15")
        .define("COMB_BLOCKS", "43")
        .define("COMB_TEETH", "6")
        // Linked statically. On Windows the header otherwise declares every
        // function __declspec(dllimport) and the link finds none of them.
        .define("SECP256K1_STATIC", None)
        .opt_level(2)
        .warnings(false);
    if msvc {
        use_clang_cl(&mut secp);
    } else if native {
        prefer_clang(&mut secp, "CC", "clang");
    }
    secp.compile("bc3secp256k1");

    let mut bc3 = cc::Build::new();
    bc3.cpp(true)
        .std("c++20")
        .include(vendor.join("bc3/include"))
        .include(vendor.join("bc3"))
        .include(vendor.join("secp256k1/include"))
        .define("SECP256K1_STATIC", None)
        .file(vendor.join("bc3/mix.cpp"))
        .file(vendor.join("bc3/hash.cpp"))
        .file(vendor.join("bc3/standard_hashes.cpp"))
        .file("src/shim.cpp")
        // As the node's Release build: -O3, nothing host-specific, so one
        // binary runs on every x86-64.
        .opt_level(3)
        .warnings(false);
    if msvc {
        use_clang_cl(&mut bc3);
    } else if native {
        prefer_clang(&mut bc3, "CXX", "clang++");
    }
    // GCC 13 (Ubuntu 24.04's g++ 13.3) miscompiles mix.cpp at -O3: it runs
    // and gets every hash wrong. Bisected: right at -O0 to -O2, and at -O3
    // with either -fno-tree-loop-vectorize or -fno-strict-aliasing; GCC 12
    // and 14 get it right. Both go on for every compiler, in its own spelling,
    // because nobody has shown whether the code or the compiler breaks the
    // rule - and the next compiler should not get the chance either. The
    // self-test in lib.rs is the backstop.
    let tool = bc3.get_compiler();
    let guards: &[&str] = if tool.is_like_clang_cl() {
        // clang-cl relaxes aliasing by default already; spelled out anyway.
        &["/clang:-fno-strict-aliasing", "/clang:-fno-vectorize"]
    } else if tool.is_like_clang() {
        &["-fno-strict-aliasing", "-fno-vectorize"]
    } else if tool.is_like_gnu() {
        &["-fno-strict-aliasing", "-fno-tree-loop-vectorize"]
    } else {
        panic!("BC3Hash needs GCC, Clang or clang-cl, not {}", tool.path().display());
    };
    for flag in guards {
        bc3.flag(flag);
    }
    bc3.compile("bc3hashref");
}

/// Clang where the build has not named a compiler of its own: the reference
/// code is ~10 % faster built with Clang 19 than with GCC 14 - 1065 against
/// 970 H/s per thread with the fast tables and the guard flags on, alternated
/// runs on an i9-12900HK - and it sidesteps GCC 13 altogether. Same code,
/// same hashes; the vector tests and the self-test check every build. Native
/// builds only - a cross build has its own toolchain to answer to.
fn prefer_clang(build: &mut cc::Build, var: &str, clang: &str) {
    let target = std::env::var("TARGET").unwrap_or_default();
    let named = [
        var.to_string(),
        format!("TARGET_{var}"),
        format!("{var}_{target}"),
        format!("{var}_{}", target.replace('-', "_")),
    ];
    for name in &named {
        println!("cargo:rerun-if-env-changed={name}");
    }
    if named.iter().any(|name| std::env::var_os(name).is_some()) {
        return;
    }
    let found = std::process::Command::new(clang)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if found {
        build.compiler(clang);
    } else {
        println!(
            "cargo:warning=BC3Hash built with the default compiler: install clang for ~10% more \
             hashrate (see bc3hash/build.rs)"
        );
    }
}

/// The reference code needs a GCC/Clang-compatible compiler: it uses
/// `unsigned __int128` and `__builtin_clzll`/`__builtin_prefetch`, none of
/// which MSVC's cl.exe has. clang-cl has all of them and takes the same flags,
/// links against the same CRT and produces the same ABI, so on an MSVC target
/// it replaces cl.exe. cargo-xwin already selects it; on a Windows host it
/// comes with the LLVM install (GitHub's Windows images have one).
fn use_clang_cl(build: &mut cc::Build) {
    let tool = build.get_compiler();
    if tool.is_like_clang_cl() || !tool.is_like_msvc() {
        return;
    }
    let installed = Path::new(r"C:\Program Files\LLVM\bin\clang-cl.exe");
    if installed.exists() {
        build.compiler(installed);
    } else {
        build.compiler("clang-cl");
    }
}
