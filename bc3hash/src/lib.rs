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
//! The hash is about 800 times as expensive as SHA3-256t - elliptic-curve
//! multiplications and big modular exponentiations - and has no GPU kernel.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

extern "C" {
    fn bc3hash_initialize(fast: i32) -> i32;
    fn bc3hash_hash(data: *const u8, len: usize, out32: *mut u8) -> i32;
    fn bc3hash_mix(data: *const u8, len: usize, out32: *mut u8) -> i32;
    fn bc3hash_table_bytes() -> usize;
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
