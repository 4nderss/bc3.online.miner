//! BC3Hash: the proof of work and block hash of BC3's second fork (version
//! bit 11).
//!
//! Not an implementation of our own. This is the node's reference code
//! (`src/crypto/bc3` in bitcoiniii-core, commit in `vendor/SOURCE_COMMIT`)
//! behind a C interface. A port could differ from the node in a single bit and
//! produce shares that look valid but never become blocks; with the node's
//! own code that cannot happen - provided the compiler builds it correctly,
//! which is why `initialize` checks the build against the node's vectors
//! before anything is hashed (see `SELF_TEST`).
//!
//! The hash is about 800 times as expensive as SHA3-256t, the proof of work
//! before it - elliptic-curve multiplications and big modular
//! exponentiations. This is the CPU's BC3Hash, and the reference the miner's
//! GPU kernel (src/kernels/bc3hash.cl) is checked against: its self-test,
//! and every hit before it is submitted. The kernel's tables come from here
//! too: whole for an OpenCL card (`gpu_tables`), as the points a CUDA card
//! builds its own from and is checked against (`gpu_point`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, RwLock};

#[cfg(target_arch = "wasm32")]
mod wasm_malloc;

extern "C" {
    fn bc3hash_initialize(fast: i32) -> i32;
    fn bc3hash_hash(data: *const u8, len: usize, out32: *mut u8) -> i32;
    fn bc3hash_mix(data: *const u8, len: usize, out32: *mut u8) -> i32;
    fn bc3hash_table_bytes() -> usize;
    fn bc3hash_gpu_part_u64(window: u32, part: i32) -> usize;
    fn bc3hash_gpu_table_part(window: u32, part: i32, out: *mut u64, len: usize) -> i32;
    fn bc3hash_gpu_rows(window: u32) -> u32;
    fn bc3hash_gpu_row_u64(window: u32) -> usize;
    fn bc3hash_gpu_curve_row(window: u32, curve: i32, row: u32, out: *mut u64, len: usize) -> i32;
    fn bc3hash_gpu_point(curve: i32, pos: u32, b: u64, out: *mut u64) -> i32;
}

/// The lookup tables the GPU kernel (src/kernels/bc3hash.cl in the miner)
/// reads, for secp256k1 windows of `window` bits: the FixedPower tables of
/// the node's g64, g128 and g256, then the four curve tables b * 2^(w*i) * P
/// as affine x, y limbs - the kernel's BC3_TABLE_* layout. Built from the
/// node's own constants with the same secp256k1 calls as its CPU tables, and
/// independent of `configure`.
///
/// Every row of every curve is built on its own (its base straight from the
/// curve's point by one scalar multiplication), on as many threads as the
/// machine has. On an i9-12900HK (20 threads): window 16 (256 MiB) in 1.3 s,
/// ~3 s with a thread per curve before; 18 in 5.2 s, 19 in 9.6 s, 20 (3.3 GiB)
/// in 18.7 s.
pub fn gpu_tables(window: u32) -> Result<Vec<u64>, String> {
    if !(2..=20).contains(&window) {
        return Err(format!("no GPU tables for a window of {window} bits"));
    }
    // SAFETY: plain integer arguments, cannot throw.
    let (pow_len, rows, row_len) =
        unsafe { (bc3hash_gpu_part_u64(window, 0), bc3hash_gpu_rows(window) as usize, bc3hash_gpu_row_u64(window)) };
    let mut out = vec![0u64; pow_len + 4 * rows * row_len];
    // Work items: the power tables, then (curve, row) for each curve's rows,
    // in the buffer's order - BC3_TABLE_* of the kernel.
    let (pow, curves) = out.split_at_mut(pow_len);
    let mut items: Vec<(usize, &mut [u64])> = vec![(0, pow)];
    items.extend(curves.chunks_mut(row_len).enumerate().map(|(i, slice)| (i + 1, slice)));
    items.reverse();
    let queue = std::sync::Mutex::new(items);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 64);
    let failed = AtomicBool::new(false);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let Some((item, slice)) = queue.lock().unwrap_or_else(|e| e.into_inner()).pop() else { break };
                // SAFETY: each slice is exactly its item's length, and no
                // other thread touches it.
                let rc = unsafe {
                    if item == 0 {
                        bc3hash_gpu_table_part(window, 0, slice.as_mut_ptr(), slice.len())
                    } else {
                        let (curve, row) = ((item - 1) / rows, (item - 1) % rows);
                        bc3hash_gpu_curve_row(window, curve as i32, row as u32, slice.as_mut_ptr(), slice.len())
                    }
                };
                if rc != 0 {
                    failed.store(true, Ordering::Relaxed);
                }
            });
        }
    });
    if failed.load(Ordering::Relaxed) {
        Err(format!("building the BC3Hash GPU tables (window {window}) failed"))
    } else {
        Ok(out)
    }
}

/// The power tables at the start of the GPU tables, the same for every
/// window - for a card that builds the curve tables after them itself.
pub fn gpu_power_tables() -> Result<Vec<u64>, String> {
    // SAFETY: plain integer arguments; the slice is exactly the part's
    // length, as the call checks.
    let len = unsafe { bc3hash_gpu_part_u64(16, 0) };
    let mut out = vec![0u64; len];
    match unsafe { bc3hash_gpu_table_part(16, 0, out.as_mut_ptr(), len) } {
        0 => Ok(out),
        rc => Err(format!("building the BC3Hash power tables failed ({rc})")),
    }
}

/// b * 2^pos * P for the kernel's curve point P number `curve` (0-3), as the
/// GPU tables hold a point: affine x and y, four limbs each, least
/// significant first. A CUDA card builds its curve tables on itself from the
/// rows' bases (b = 1), and its entries are checked against this.
pub fn gpu_point(curve: usize, pos: u32, b: u64) -> Result<[u64; 8], String> {
    let mut out = [0u64; 8];
    let curve_arg = i32::try_from(curve).unwrap_or(-1);
    // SAFETY: `out` has the 8 limbs the call writes.
    match unsafe { bc3hash_gpu_point(curve_arg, pos, b, out.as_mut_ptr()) } {
        0 => Ok(out),
        rc => Err(format!("no point {b} * 2^{pos} on curve {curve} ({rc})")),
    }
}

/// The curve tables the way 2.0.0 built them - one curve after the other,
/// each row's base from the row before - for the test that the row-parallel
/// build gives the same table.
#[cfg(test)]
fn gpu_tables_by_curve(window: u32) -> Vec<u64> {
    let sizes: Vec<usize> = (0..5).map(|part| unsafe { bc3hash_gpu_part_u64(window, part) }).collect();
    let mut out = vec![0u64; sizes.iter().sum()];
    let mut at = 0;
    for (part, &size) in sizes.iter().enumerate() {
        let slice = &mut out[at..at + size];
        assert_eq!(unsafe { bc3hash_gpu_table_part(window, part as i32, slice.as_mut_ptr(), size) }, 0);
        at += size;
    }
    out
}

/// The node's header vectors among the self-test vectors: 80-byte BC3Hash
/// headers and their digests, for checking other implementations (the GPU
/// kernel) the same way `initialize` checks this one.
pub fn header_vectors() -> Vec<([u8; 80], [u8; 32])> {
    SELF_TEST
        .iter()
        .filter(|(input, _)| input.len() == 160)
        .map(|(input, digest)| {
            let header: [u8; 80] = from_hex(input).try_into().expect("80-byte header");
            let digest: [u8; 32] = from_hex(digest).try_into().expect("32-byte digest");
            (header, digest)
        })
        .collect()
}

/// How big the shared precomputed curve tables are. Both modes give identical
/// hashes; they differ in memory and speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// About 7 MiB.
    Compact,
    /// About 257 MiB.
    Fast,
}

static FAST: AtomicBool = AtomicBool::new(false);
/// The outcome of building the tables and checking the build - once, by
/// whoever gets there first.
static READY: OnceLock<Result<(), InitError>> = OnceLock::new();

/// Held for reading by every hash; taken for writing, for good, at exit.
///
/// The node's mix.cpp keeps its secp256k1 context and its tables in function
/// statics, which the C++ runtime destroys when the process exits - with a
/// worker thread perhaps still inside a hash, reading them. The miner never
/// joins its workers, and neither do the tests that start them: the 2.1.0
/// release workflow's test binary died of SIGSEGV after every test had
/// passed (on a 4-core runner; 0 in 25 runs on 20 cores). `stop_hashing`,
/// registered to run at exit before those destructors, lets the hashes in
/// flight finish and starts no new one. A hash costs about a millisecond,
/// the lock two atomic operations.
static HASHING: RwLock<()> = RwLock::new(());

/// Register `stop_hashing` to run at exit, ahead of the C++ destructors.
/// Exit handlers run in the reverse order of their registration, and each
/// static registered its destructor when it was built - which `initialize`
/// and its self-test hashes have done by the time this is called.
#[cfg(not(target_arch = "wasm32"))]
fn stop_hashing_at_exit() {
    extern "C" {
        fn atexit(f: extern "C" fn()) -> std::os::raw::c_int;
    }
    extern "C" fn stop_hashing() {
        // Never given back: the process is ending, and whatever asks for a
        // hash from here on waits for that.
        std::mem::forget(HASHING.write().unwrap_or_else(|e| e.into_inner()));
    }
    // SAFETY: registers a function that takes nothing and returns nothing,
    // as atexit requires; called once, from `build_and_check`.
    unsafe { atexit(stop_hashing) };
}

/// No exit, and no threads, in the browser.
#[cfg(target_arch = "wasm32")]
fn stop_hashing_at_exit() {}

/// Choose the table mode. Only the first `initialize` (or hash) uses it, so
/// call this before anything hashes.
pub fn configure(mode: Mode) {
    FAST.store(mode == Mode::Fast, Ordering::Relaxed);
}

/// Why BC3Hash cannot be used in this process. Either way it stays so: the
/// node's code locks the table mode at the first attempt, and a wrong build
/// does not get right by trying again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InitError {
    /// The tables could not be built - in practice, not enough memory for the
    /// fast ones.
    Tables { fast: bool },
    /// The build computes BC3Hash wrongly: one of the node's vectors came out
    /// different. Mining with it would only produce rejected shares.
    WrongResult { input: String, got: String, expected: String },
}

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitError::Tables { fast } => {
                let (mode, size) = if *fast { ("fast", "~256 MiB") } else { ("compact", "~7 MiB") };
                write!(f, "could not build the BC3Hash {mode} tables ({size}) - out of memory?")
            }
            InitError::WrongResult { input, got, expected } => {
                let input = match input.get(..24) {
                    Some(start) if input.len() > 24 => format!("{start}..."),
                    _ => input.clone(),
                };
                write!(
                    f,
                    "this build computes BC3Hash incorrectly - report it. The node's test \
                     vector for input {input} gave {got}, expected {expected}"
                )
            }
        }
    }
}

impl std::error::Error for InitError {}

/// The node's own vectors, checked by `initialize` before anything is hashed:
/// two 80-byte headers from doc/bc3/vectors/header-vectors.json (bit 11
/// alone, and bits 11 and 12 together - what the pool sends) and two inputs
/// from reference-vectors.tsv. Input hex ("" for empty), BC3Hash hex.
///
/// GCC 13 builds mix.cpp at -O3 into code that runs happily and gets every
/// hash wrong; nothing downstream could tell until the pool had rejected every
/// share. build.rs works around that compiler. This catches the next one, for
/// the price of four hashes at startup.
const SELF_TEST: [(&str, &str); 4] = [
    (
        "00080020000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f\
         202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f00d2496bffff7f202a000000",
        "eef9a6934aa1719f19bdea11824927d4cd58b0995033b99edb4bf2370d77160e",
    ),
    (
        "00180020000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f\
         202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f00d2496bffff7f202a000000",
        "102b13f7ba825a57fdd460fc6a7caaa473ea2cd7e312f5988d8d5c41438d3975",
    ),
    ("", "069fcba272f2d061b70b0ab121e40d289c0fe8e277ac453450bf6090ad1795a4"),
    ("616263", "72e6793f4631eabb8755e35cd41706c245f56ed64d4a17d4ecc2769a93e419a1"),
];

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn from_hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("vector hex"))
        .collect()
}

/// Build the tables, then check the build against `SELF_TEST`.
fn build_and_check() -> Result<(), InitError> {
    let fast = FAST.load(Ordering::Relaxed);
    // SAFETY: plain integer argument; the C++ side is thread-safe.
    if unsafe { bc3hash_initialize(i32::from(fast)) } != 0 {
        return Err(InitError::Tables { fast });
    }
    for (input, expected) in SELF_TEST {
        let got = to_hex(&raw_hash(&from_hex(input)));
        if got != expected {
            return Err(InitError::WrongResult {
                input: if input.is_empty() { "(empty)".into() } else { input.into() },
                got,
                expected: expected.into(),
            });
        }
    }
    stop_hashing_at_exit();
    Ok(())
}

/// Build the tables and check the build against the node's vectors. Takes a
/// noticeable time - the fast tables about ten seconds - which belongs at
/// startup, not inside the first hash. Idempotent; a concurrent caller waits
/// for the first one to finish.
pub fn initialize() -> Result<(), InitError> {
    READY.get_or_init(build_and_check).clone()
}

/// What `initialize` came to, without waiting for it: None until it has
/// finished (or before it started).
pub fn status() -> Option<Result<(), InitError>> {
    READY.get().cloned()
}

/// Bytes of precomputed tables in the configured mode.
pub fn table_bytes() -> usize {
    // SAFETY: no arguments, cannot throw.
    unsafe { bc3hash_table_bytes() }
}

fn raw_hash(data: &[u8]) -> [u8; 32] {
    let _hashing = HASHING.read().unwrap_or_else(|e| e.into_inner());
    let mut out = [0u8; 32];
    // SAFETY: data/len describe a valid slice and out holds 32 bytes.
    let rc = unsafe { bc3hash_hash(data.as_ptr(), data.len(), out.as_mut_ptr()) };
    // Throws only for inputs above 2^509 bytes or on allocation failure, and
    // the hash allocates nothing once the tables exist.
    assert_eq!(rc, 0, "BC3Hash threw an exception");
    out
}

/// BC3Hash of `data` (for a block header: its 80 serialized bytes). The digest
/// comes back in the order the node stores it in a uint256 - internal order,
/// the same as `prev_hash`.
///
/// The `initialize` check costs one atomic load once it has run. Panics if it
/// failed: without tables there is no hash, and a build that fails the
/// vectors must not produce one.
#[inline]
pub fn bc3hash(data: &[u8]) -> [u8; 32] {
    if let Err(e) = READY.get_or_init(build_and_check) {
        panic!("{e}");
    }
    raw_hash(data)
}

/// BC3Mix, the expensive inner step. Exposed for the reference vectors.
pub fn bc3mix(data: &[u8]) -> [u8; 32] {
    if let Err(e) = READY.get_or_init(build_and_check) {
        panic!("{e}");
    }
    let _hashing = HASHING.read().unwrap_or_else(|e| e.into_inner());
    let mut out = [0u8; 32];
    // SAFETY: as above.
    let rc = unsafe { bc3hash_mix(data.as_ptr(), data.len(), out.as_mut_ptr()) };
    assert_eq!(rc, 0, "BC3Mix threw an exception");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The node's vectors: `vectors/` next to the vendored sources, or
    /// `BC3HASH_VECTORS` when they are elsewhere.
    fn vectors(name: &str) -> String {
        let dir = std::env::var_os("BC3HASH_VECTORS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("vectors"));
        let path = dir.join(name);
        std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("{}: {e} - copy doc/bc3/vectors from the node", path.display())
        })
    }

    /// doc/bc3/vectors/reference-vectors.tsv: input, bc3mix, bc3hash, made
    /// with the node's frozen Python reference (bc3hash.py). Covers the empty
    /// input, the 64-byte path without folding, and folded longer inputs.
    #[test]
    fn reference_vectors() {
        let tsv = vectors("reference-vectors.tsv");
        let mut n = 0;
        for line in tsv.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let cols: Vec<&str> = line.split('\t').collect();
            assert_eq!(cols.len(), 3, "line: {line}");
            let input = if cols[0] == "-" { vec![] } else { hex::decode(cols[0]).unwrap() };
            assert_eq!(hex::encode(bc3mix(&input)), cols[1], "bc3mix({})", cols[0]);
            assert_eq!(hex::encode(bc3hash(&input)), cols[2], "bc3hash({})", cols[0]);
            n += 1;
        }
        assert!(n >= 4, "too few vectors: {n}");
    }

    /// The vectors built into the self-test are the node's, digit for digit,
    /// and this build passes them.
    #[test]
    fn self_test_vectors_are_the_nodes() {
        let headers = vectors("header-vectors.json");
        let tsv = vectors("reference-vectors.tsv");
        for (input, expected) in SELF_TEST {
            let in_headers = input.len() == 160
                && headers.split('{').any(|v| v.contains(input) && v.contains(expected));
            let in_tsv = tsv.lines().any(|l| {
                let c: Vec<&str> = l.split('\t').collect();
                c.len() == 3 && (c[0] == input || (input.is_empty() && c[0] == "-")) && c[2] == expected
            });
            assert!(in_headers || in_tsv, "{input} -> {expected} is not among the node's vectors");
        }
        assert_eq!(initialize(), Ok(()));
        assert_eq!(status(), Some(Ok(())));
    }

    /// The GPU tables in the kernel's layout: the right size, and anchored to
    /// values that are known independently - g^0 and g^1 at the start of each
    /// power table, and the curve's own base point first in its table.
    #[test]
    fn gpu_tables_have_the_kernel_layout() {
        let window = 10u32;
        let points = 26 * 1023;
        let t = gpu_tables(window).unwrap();
        let (p64, p128, p256, curve) = (0, 2048, 2048 + 8192, 2048 + 8192 + 32768);
        assert_eq!(t.len(), curve + 4 * points * 8, "BC3_TABLE_U64 at window {window}");
        // Row 0: g^0 = 1, g^1 = g (the node's g64 / g128 low limb / g256 low limb).
        assert_eq!(&t[p64..p64 + 2], &[1, 0x7b48_4f6c_1cf4_97f2]);
        assert_eq!(&t[p128..p128 + 2], &[1, 0]);
        assert_eq!(&t[p128 + 2..p128 + 4], &[0xb223_833f_64cd_ffc9, 0x77c6_0c1d_221a_0804]);
        assert_eq!(t[p256 + 4], 0x43d7_82e9_b25a_ae2b);
        // Curve 0 starts at H itself: x = 01f2a775...27dd1f51.
        assert_eq!(t[curve], 0x9e37_6c27_27dd_1f51);
        assert_eq!(t[curve + 3], 0x01f2_a775_8a89_4831);
        // The last row of every curve is padded with zeros past 2^(256-250)-1.
        let last_row = curve + (25 * 1023 + 63) * 8;
        assert!(t[last_row..curve + points * 8].iter().all(|&v| v == 0));
        assert!(t[last_row - 8..last_row].iter().any(|&v| v != 0));
        assert!(gpu_tables(1).is_err());
        assert_eq!(header_vectors().len(), 2);
    }

    /// What a card that builds its own curve tables starts from and is
    /// checked against: the power tables, and b * 2^pos * P - the entries of
    /// the host's tables, the last row of a window that does not divide 256
    /// included.
    #[test]
    fn gpu_points_are_the_tables_entries() {
        let window = 5u32;
        let t = gpu_tables(window).unwrap();
        let pow = gpu_power_tables().unwrap();
        assert_eq!(pow[..], t[..pow.len()]);
        let (rows, row_u64) = (52usize, 31 * 8);
        for curve in 0..4 {
            for row in [0, 1, 26, 50, 51] {
                let pos = window * row as u32;
                let bits = window.min(256 - pos);
                for b in [1u64, 2, 7, 31].into_iter().filter(|&b| b < 1 << bits) {
                    let at = pow.len() + (curve * rows + row) * row_u64 + (b as usize - 1) * 8;
                    assert_eq!(gpu_point(curve, pos, b).unwrap(), t[at..at + 8], "curve {curve}, row {row}, b {b}");
                }
            }
        }
        // b * 2^pos itself past n: two scalars below it.
        assert!(gpu_point(3, 255, (1 << 24) - 1).is_ok());
        assert!(gpu_point(4, 0, 1).is_err());
        assert!(gpu_point(0, 256, 1).is_err());
        assert!(gpu_point(0, 0, 0).is_err());
    }

    /// The row-parallel build gives exactly the table the curve-by-curve one
    /// did, every window the kernel is built for.
    #[test]
    fn gpu_tables_rows_match_the_curve_build() {
        for window in [10u32, 12] {
            assert!(gpu_tables(window).unwrap() == gpu_tables_by_curve(window), "window {window}");
        }
    }

    /// How long the GPU tables take to build per window, and how big they are:
    ///     cargo test --release -- --ignored gpu_table_build_times --nocapture
    #[test]
    #[ignore]
    fn gpu_table_build_times() {
        for window in [10u32, 12, 14, 16, 18, 19, 20] {
            let t0 = std::time::Instant::now();
            let t = gpu_tables(window).unwrap();
            println!(
                "window {window}: {:.1} MiB in {:.2} s",
                t.len() as f64 * 8.0 / (1u64 << 20) as f64,
                t0.elapsed().as_secs_f64()
            );
        }
    }

    #[test]
    fn a_wrong_build_says_so() {
        let e = InitError::WrongResult {
            input: "(empty)".into(),
            got: "e75220283bdd".into(),
            expected: "069fcba272f2".into(),
        };
        let text = e.to_string();
        assert!(text.contains("incorrectly") && text.contains("report it"), "{text}");
    }

    /// Hashing from many threads at once gives the same answers as one
    /// thread - the tables are shared and must be read-only after the build.
    #[test]
    fn concurrent_hashing_agrees() {
        let header = |n: u32| {
            let mut h = [7u8; 80];
            h[76..80].copy_from_slice(&n.to_le_bytes());
            h
        };
        let expected: Vec<[u8; 32]> = (0..16).map(|n| bc3hash(&header(n))).collect();
        let threads: Vec<_> = (0..8)
            .map(|_| std::thread::spawn(move || (0..16).map(|n| bc3hash(&header(n))).collect::<Vec<_>>()))
            .collect();
        for t in threads {
            assert_eq!(t.join().unwrap(), expected);
        }
    }
}
