//! The search kernel for the web miner, compiled to WebAssembly: BC3Hash, the
//! PoW since the second fork, with the `bc3hash` feature (the default) - the
//! node's own C++ via the `bc3hash` crate, as in the CLI. A job without
//! BC3Hash's version bit is refused, as the CLI refuses it.
//!
//! Deliberately WITHOUT wasm-bindgen: the interface is four numbers and two
//! byte buffers, so bindgen would only add glue and a build dependency. JS
//! writes directly into the wasm memory and calls the exports.
//!
//! The consensus code is shared with the CLI miner via `#[path]` instead of
//! being copied - a web share must be hashed exactly like a GPU share, or
//! else the pool rejects it.

// `consensus` pulls in more functions than the search loop uses (merkle,
// bip34, addresses). They are tested and cost nothing in the compiled module.
#[allow(dead_code)]
#[path = "../../src/consensus.rs"]
mod consensus;

use consensus::{hash_meets_target, PowAlgo, Target};

/// PoW algorithms as the numbers JS passes across the wasm boundary. Part of
/// the ABI: a number, once given out, keeps its meaning.
fn algo_id(algo: PowAlgo) -> i32 {
    match algo {
        PowAlgo::Sha256d => 0,
        PowAlgo::Sha3t => 1,
        PowAlgo::Bc3Hash => 2,
    }
}

/// The algorithm behind an id - only if this module can hash it. BC3Hash can
/// only with the `bc3hash` feature, and only once its tables are built and
/// the build has passed the node's vectors (`bc3hash_init`, which this runs
/// on first use): a build that fails them must not hash at all.
fn algo_from_id(id: u32) -> Option<PowAlgo> {
    [PowAlgo::Sha256d, PowAlgo::Sha3t, PowAlgo::Bc3Hash]
        .into_iter()
        .find(|a| algo_id(*a) == id as i32 && a.hash_fn().is_some())
        .filter(|a| *a != PowAlgo::Bc3Hash || bc3hash_init() == 1)
}

/// The PoW to mine a job with, from its version - the node's rule, the same
/// decision the CLI makes at notify time.
///
/// Returns the algorithm id for `search_algo`, or -1 for a PoW this module
/// does not hash - every one but BC3Hash, and BC3Hash too in a build without
/// the `bc3hash` feature: the page must stop hashing and tell the user, not
/// grind the job into rejected shares.
#[no_mangle]
pub extern "C" fn job_pow_algo(version: u32) -> i32 {
    let algo = PowAlgo::for_version(version);
    match algo.hash_fn() {
        Some(_) => algo_id(algo),
        None => -1,
    }
}

/// Build this instance's BC3Hash tables (compact mode, ~7 MiB) and check the
/// build against the node's vectors - the CLI's `bc3hash::initialize`. Call
/// it before the first BC3Hash search: the first hash would do it anyway,
/// but a build that fails the vectors would then trap mid-search instead of
/// saying so here.
///
/// 1 = ready; 0 = the tables could not be built (out of memory); -1 = this
/// build computes BC3Hash wrongly - mine no BC3Hash with it; -2 = built
/// without the `bc3hash` feature. Idempotent and cheap after the first call.
#[no_mangle]
pub extern "C" fn bc3hash_init() -> i32 {
    #[cfg(feature = "bc3hash")]
    {
        match bc3hash::initialize() {
            Ok(()) => 1,
            Err(bc3hash::InitError::Tables { .. }) => 0,
            Err(bc3hash::InitError::WrongResult { .. }) => -1,
        }
    }
    #[cfg(not(feature = "bc3hash"))]
    {
        -2
    }
}

/// The search loop over one nonce range, with the job's hash. Next to a
/// BC3Hash a call through a function pointer per nonce is noise.
fn search_with(
    pow: impl Fn(&[u8; 80]) -> [u8; 32],
    mut header: [u8; 80],
    target: &Target,
    nonce_start: u32,
    nonce_count: u32,
) -> f64 {
    for i in 0..nonce_count {
        // wrapping: the nonce space is circular, and a worker that gets a
        // range close to u32::MAX must not panic.
        let nonce = nonce_start.wrapping_add(i);
        header[76..80].copy_from_slice(&nonce.to_le_bytes());
        if hash_meets_target(&pow(&header), target) {
            return nonce as f64;
        }
    }
    -1.0
}

/// Copy the 80-byte header and 32-byte target out of wasm memory.
unsafe fn read_header_target(header_ptr: *const u8, target_ptr: *const u8) -> ([u8; 80], Target) {
    let mut header = [0u8; 80];
    core::ptr::copy_nonoverlapping(header_ptr, header.as_mut_ptr(), 80);
    let mut target: Target = [0u8; 32];
    core::ptr::copy_nonoverlapping(target_ptr, target.as_mut_ptr(), 32);
    (header, target)
}

/// Wasm has no allocator that JS can reach directly. These two exports let
/// JS reserve memory for the header and the target.
///
/// # Safety
/// The caller owns the block until `dealloc` is called with the same length.
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    core::mem::forget(buf);
    ptr
}

/// # Safety
/// `ptr` must come from `alloc` with the same `len`.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// The original export: search a nonce range for a SHA3-256t hash, which
/// this module no longer computes. It returns `-2.0` ("cannot hash this"),
/// as `search_algo` does for SHA3-256t's id. Kept with its signature for a
/// page that still looks for it; the page calls `search_algo` whenever the
/// module has it, and every module that has no SHA3-256t does.
///
/// # Safety
/// The buffers must be 80 and 32 bytes respectively and come from `alloc`.
#[no_mangle]
pub unsafe extern "C" fn search(
    header_ptr: *const u8,
    target_ptr: *const u8,
    nonce_start: u32,
    nonce_count: u32,
) -> f64 {
    search_algo(algo_id(PowAlgo::Sha3t) as u32, header_ptr, target_ptr, nonce_start, nonce_count)
}

/// Search a nonce range for a hash of the algorithm `algo` (an id from
/// `job_pow_algo`) that meets the target.
///
/// `header_ptr` points at 80 bytes of serialized block header; the nonce at
/// offset 76 is overwritten for every attempt. `target_ptr` points at 32
/// bytes of big-endian target (the pool's share target, not the network's).
///
/// Returns the nonce as an f64 - `-1.0` means "found nothing in the range".
/// f64 instead of i64 to avoid BigInt in JS; every u32 fits exactly in an
/// f64. Returns `-2.0` for an id this module does not hash, so a page newer
/// than its wasm cannot mistake "cannot hash this" for "found nothing".
///
/// # Safety
/// The buffers must be 80 and 32 bytes respectively and come from `alloc`.
#[no_mangle]
pub unsafe extern "C" fn search_algo(
    algo: u32,
    header_ptr: *const u8,
    target_ptr: *const u8,
    nonce_start: u32,
    nonce_count: u32,
) -> f64 {
    if header_ptr.is_null() || target_ptr.is_null() {
        return -1.0;
    }
    let Some(pow) = algo_from_id(algo).and_then(|a| a.hash_fn()) else {
        return -2.0;
    };
    // Copy the header onto the stack: writing the nonce through a raw
    // pointer in the loop blocks optimizations and forces one read per
    // iteration.
    let (header, target) = read_header_target(header_ptr, target_ptr);
    search_with(pow, header, &target, nonce_start, nonce_count)
}

/// The original export: SHA3-256t of a header, which this module no longer
/// computes - it writes nothing. Kept with its signature for a page that
/// still looks for it; the page calls `hash_header_algo` whenever the module
/// has it.
///
/// # Safety
/// `header_ptr` must point at 80 bytes, `out_ptr` at at least 32.
#[no_mangle]
pub unsafe extern "C" fn hash_header(header_ptr: *const u8, out_ptr: *mut u8) {
    hash_header_algo(algo_id(PowAlgo::Sha3t) as u32, header_ptr, out_ptr);
}

/// Hash a header once with the algorithm `algo` (an id from `job_pow_algo`)
/// and write the result to `out_ptr` (32 bytes). Used by the main thread to
/// verify a hit before it is submitted. Returns 1 when it hashed, 0 for an id
/// this module does not hash or a null pointer - then nothing is written.
///
/// # Safety
/// `header_ptr` must point at 80 bytes, `out_ptr` at at least 32.
#[no_mangle]
pub unsafe extern "C" fn hash_header_algo(algo: u32, header_ptr: *const u8, out_ptr: *mut u8) -> u32 {
    if header_ptr.is_null() || out_ptr.is_null() {
        return 0;
    }
    let Some(pow) = algo_from_id(algo).and_then(|a| a.hash_fn()) else {
        return 0;
    };
    let mut header = [0u8; 80];
    core::ptr::copy_nonoverlapping(header_ptr, header.as_mut_ptr(), 80);
    let hash = pow(&header);
    core::ptr::copy_nonoverlapping(hash.as_ptr(), out_ptr, 32);
    1
}

/// BC3Hash (`which` = 0) or BC3Mix (`which` = 1) of an arbitrary buffer,
/// for running the node's reference vectors (any input length) against the
/// built module: wasm/test/vectors.mjs. 1 = written, 0 = not available
/// (feature off, or `bc3hash_init` did not return 1) or a null pointer.
///
/// # Safety
/// `ptr` must point at `len` readable bytes (or be anything when `len` is 0),
/// `out_ptr` at at least 32.
#[no_mangle]
pub unsafe extern "C" fn bc3hash_bytes_into(which: u32, ptr: *const u8, len: usize, out_ptr: *mut u8) -> u32 {
    if out_ptr.is_null() || (ptr.is_null() && len > 0) || bc3hash_init() != 1 {
        return 0;
    }
    #[cfg(feature = "bc3hash")]
    {
        let data: &[u8] = if len == 0 { &[] } else { core::slice::from_raw_parts(ptr, len) };
        let digest = match which {
            0 => bc3hash::bc3hash(data),
            1 => bc3hash::bc3mix(data),
            _ => return 0,
        };
        core::ptr::copy_nonoverlapping(digest.as_ptr(), out_ptr, 32);
        1
    }
    #[cfg(not(feature = "bc3hash"))]
    {
        let _ = which;
        0
    }
}

/// SHA256d over an arbitrary buffer (coinbase txid, merkle steps).
///
/// # Safety
/// `ptr` must point at `len` readable bytes, `out_ptr` at at least 32.
#[no_mangle]
pub unsafe extern "C" fn sha256d_into(ptr: *const u8, len: usize, out_ptr: *mut u8) {
    if ptr.is_null() || out_ptr.is_null() {
        return;
    }
    let data = core::slice::from_raw_parts(ptr, len);
    let h = consensus::sha256d(data);
    core::ptr::copy_nonoverlapping(h.as_ptr(), out_ptr, 32);
}

/// Stratum's prevhash ordering -> internal ordering (4-byte word reversal).
///
/// # Safety
/// Both pointers must point at at least 32 bytes.
#[no_mangle]
pub unsafe extern "C" fn swab32_into(in_ptr: *const u8, out_ptr: *mut u8) {
    if in_ptr.is_null() || out_ptr.is_null() {
        return;
    }
    let mut input = [0u8; 32];
    core::ptr::copy_nonoverlapping(in_ptr, input.as_mut_ptr(), 32);
    let out = consensus::swab32(&input);
    core::ptr::copy_nonoverlapping(out.as_ptr(), out_ptr, 32);
}

/// Merkle root from the coinbase txid and stratum's branches
/// (`steps_count` x 32 bytes).
///
/// # Safety
/// `txid_ptr` 32 bytes, `steps_ptr` `steps_count * 32` bytes, `out_ptr` 32.
#[no_mangle]
pub unsafe extern "C" fn merkle_root_into(
    txid_ptr: *const u8,
    steps_ptr: *const u8,
    steps_count: usize,
    out_ptr: *mut u8,
) {
    if txid_ptr.is_null() || out_ptr.is_null() {
        return;
    }
    let mut txid = [0u8; 32];
    core::ptr::copy_nonoverlapping(txid_ptr, txid.as_mut_ptr(), 32);
    let mut steps = Vec::with_capacity(steps_count);
    for i in 0..steps_count {
        let mut s = [0u8; 32];
        core::ptr::copy_nonoverlapping(steps_ptr.add(i * 32), s.as_mut_ptr(), 32);
        steps.push(s);
    }
    let root = consensus::root_from_steps(&txid, &steps);
    core::ptr::copy_nonoverlapping(root.as_ptr(), out_ptr, 32);
}

/// Share target for a stratum difficulty.
///
/// # Safety
/// `out_ptr` must point at at least 32 bytes.
#[no_mangle]
pub unsafe extern "C" fn target_for_difficulty_into(difficulty: f64, out_ptr: *mut u8) {
    if out_ptr.is_null() {
        return;
    }
    let t = consensus::target_for_difficulty(difficulty);
    core::ptr::copy_nonoverlapping(t.as_ptr(), out_ptr, 32);
}

/// Serialize an 80-byte header. The field ordering lives in Rust so that JS
/// can never put a field at the wrong offset.
///
/// # Safety
/// `prev_ptr` and `merkle_ptr` 32 bytes each, `out_ptr` at least 80.
#[no_mangle]
pub unsafe extern "C" fn build_header(
    version: u32,
    prev_ptr: *const u8,
    merkle_ptr: *const u8,
    time: u32,
    bits: u32,
    out_ptr: *mut u8,
) {
    if prev_ptr.is_null() || merkle_ptr.is_null() || out_ptr.is_null() {
        return;
    }
    let mut prev_hash = [0u8; 32];
    let mut merkle_root = [0u8; 32];
    core::ptr::copy_nonoverlapping(prev_ptr, prev_hash.as_mut_ptr(), 32);
    core::ptr::copy_nonoverlapping(merkle_ptr, merkle_root.as_mut_ptr(), 32);
    let header = consensus::BlockHeader {
        version,
        prev_hash,
        merkle_root,
        time,
        bits,
        nonce: 0,
    };
    let ser = header.serialize();
    core::ptr::copy_nonoverlapping(ser.as_ptr(), out_ptr, 80);
}

/// The difficulty a hash corresponds to - for showing "best share" in the UI.
///
/// # Safety
/// `hash_ptr` must point at 32 bytes.
#[no_mangle]
pub unsafe extern "C" fn difficulty_of_hash(hash_ptr: *const u8) -> f64 {
    if hash_ptr.is_null() {
        return 0.0;
    }
    let mut h = [0u8; 32];
    core::ptr::copy_nonoverlapping(hash_ptr, h.as_mut_ptr(), 32);
    consensus::difficulty_of_hash(&h)
}

/// Self test against a chain-verified vector: BC3's genesis block, whose hash
/// is SHA256d over its header - the header serialization and the SHA256d the
/// page builds every header, txid and merkle root with. Returns 1 on the
/// right answer.
///
/// It exists so that the web miner can refuse to start if the compiled
/// module hashes incorrectly - a broken kernel would otherwise just produce
/// rejected shares without anyone understanding why. BC3Hash has its own
/// check against the node's vectors, `bc3hash_init`. (Until SHA3-256t was
/// removed this checked a SHA3-256t digest; the page only asks for 1.)
#[no_mangle]
pub extern "C" fn self_test() -> u32 {
    let Ok(Ok(mut merkle_root)) = hex::decode(GENESIS_MERKLE_ROOT).map(<[u8; 32]>::try_from) else {
        return 0;
    };
    merkle_root.reverse(); // display order -> internal
    let genesis = consensus::BlockHeader {
        version: 1,
        prev_hash: [0u8; 32],
        merkle_root,
        time: 1_777_245_555,
        bits: 0x1d00_ffff,
        nonce: 2_442_659_435,
    };
    u32::from(consensus::hash_to_display(&consensus::sha256d(&genesis.serialize())) == GENESIS_HASH)
}

/// BC3's genesis block as an explorer shows it: its merkle root and its hash
/// (byte-reversed hex). The same chain-verified vector as the CLI's
/// `consensus::tests::genesis_vectors`.
const GENESIS_MERKLE_ROOT: &str = "8e1df52fddd25c460304ff8ea7bcb570850bf0b0c027eecf8ebf8ab17d3e93b1";
const GENESIS_HASH: &str = "000000000c226a41e70717f6d4fbdcb6bfb4fdc40831ccc87fa9cfdd2c57bff6";

#[cfg(test)]
mod tests {
    use super::*;

    /// The search loop finds exactly the first nonce whose hash meets the
    /// target, with the nonce at offset 76, little-endian - the header the
    /// pool validates - and returns -1 when the range holds none. SHA256d
    /// stands in for the hash: the loop is the same for any.
    #[test]
    fn search_finds_the_first_matching_nonce() {
        let pow = |h: &[u8; 80]| consensus::sha256d(h);
        let mut header = [0u8; 80];
        header[0..4].copy_from_slice(&0x2000_1800u32.to_le_bytes());
        // A loose target: one hash in sixteen.
        let mut target = [0xffu8; 32];
        target[0] = 0x0f;
        let expected = (0..2000u32)
            .find(|&n| {
                let mut h = header;
                h[76..80].copy_from_slice(&n.to_le_bytes());
                hash_meets_target(&pow(&h), &target)
            })
            .expect("a loose target must produce a hit");
        assert_eq!(search_with(pow, header, &target, 0, 2000), expected as f64);
        // A range that ends before the hit -> no hit.
        assert_eq!(search_with(pow, header, &target, 0, expected), -1.0, "a hit outside the range");
    }

    /// A nonce close to u32::MAX must not panic - the range wraps.
    #[test]
    fn nonce_wraps_instead_of_panicking() {
        let pow = |h: &[u8; 80]| consensus::sha256d(h);
        let target = [0u8; 32]; // never hit
        assert_eq!(search_with(pow, [0u8; 80], &target, u32::MAX - 2, 10), -1.0);
    }

    /// The PoWs before the second fork are not hashed: their ids, and the
    /// original SHA3-256t exports, refuse instead - nothing written, -2 from
    /// a search - so a page cannot grind such a job into rejected shares.
    #[test]
    fn the_older_pows_are_refused() {
        assert_eq!(job_pow_algo(0x2000_1000), -1);
        assert_eq!(job_pow_algo(0x2000_0000), -1);
        assert_eq!(algo_id(PowAlgo::Sha256d), 0);
        assert_eq!(algo_id(PowAlgo::Sha3t), 1);
        let header = [0u8; 80];
        let target = [0xffu8; 32]; // any hash would meet it
        for id in [0, 1, 99] {
            assert_eq!(algo_from_id(id), None, "id {id}");
            assert_eq!(unsafe { search_algo(id, header.as_ptr(), target.as_ptr(), 0, 10) }, -2.0, "id {id}");
            let mut out = [7u8; 32];
            assert_eq!(unsafe { hash_header_algo(id, header.as_ptr(), out.as_mut_ptr()) }, 0, "id {id}");
            assert_eq!(out, [7u8; 32], "id {id} wrote a hash");
        }
        assert_eq!(unsafe { search(header.as_ptr(), target.as_ptr(), 0, 10) }, -2.0);
        let mut out = [7u8; 32];
        unsafe { hash_header(header.as_ptr(), out.as_mut_ptr()) };
        assert_eq!(out, [7u8; 32], "hash_header wrote a hash");
    }

    /// The page asks the module which PoW a job uses, by the same version-bit
    /// rule as the CLI. Without the `bc3hash` feature BC3Hash is -1 and
    /// refused everywhere rather than hashed with something else.
    #[test]
    #[cfg(not(feature = "bc3hash"))]
    fn job_pow_algo_follows_the_version_bits() {
        assert_eq!(job_pow_algo(0x2000_1800), -1);
        assert_eq!(job_pow_algo(0x2000_0800), -1);
        assert_eq!(algo_id(PowAlgo::Bc3Hash), 2);
        assert_eq!(algo_from_id(2), None);
        assert_eq!(bc3hash_init(), -2);
        let header = [0u8; 80];
        let target = [0xffu8; 32];
        assert_eq!(unsafe { search_algo(2, header.as_ptr(), target.as_ptr(), 0, 10) }, -2.0);
        let mut out = [0u8; 32];
        assert_eq!(unsafe { hash_header_algo(2, header.as_ptr(), out.as_mut_ptr()) }, 0);
    }

    /// With the `bc3hash` feature, bit 11 is BC3Hash (id 2) and wins over
    /// bit 12, and the exports hash it exactly as the CLI's reference does.
    #[test]
    #[cfg(feature = "bc3hash")]
    fn job_pow_algo_follows_the_version_bits() {
        assert_eq!(job_pow_algo(0x2000_1800), 2);
        assert_eq!(job_pow_algo(0x2000_0800), 2);
        assert_eq!(bc3hash_init(), 1);
        assert_eq!(algo_from_id(2), Some(PowAlgo::Bc3Hash));

        let mut header = [0u8; 80];
        header[0..4].copy_from_slice(&0x2000_1800u32.to_le_bytes());
        let mut out = [0u8; 32];
        assert_eq!(unsafe { hash_header_algo(2, header.as_ptr(), out.as_mut_ptr()) }, 1);
        assert_eq!(out, bc3hash::bc3hash(&header));

        // A share-difficulty target (1/64 of the space) finds the first
        // nonce that a straightforward loop over the reference finds.
        let mut target = [0xffu8; 32];
        target[0] = 0x03;
        let expected = (0..2000u32)
            .find(|&n| {
                let mut h = header;
                h[76..80].copy_from_slice(&n.to_le_bytes());
                hash_meets_target(&bc3hash::bc3hash(&h), &target)
            })
            .expect("a hit within 2000 nonces at 1/64");
        let got = unsafe { search_algo(2, header.as_ptr(), target.as_ptr(), 0, 2000) };
        assert_eq!(got, expected as f64);
        let miss = unsafe { search_algo(2, header.as_ptr(), target.as_ptr(), 0, expected) };
        assert_eq!(miss, -1.0);
    }

    /// The native half of the cross-check in wasm/test/vectors.mjs: SHA-256
    /// over the BC3Hash digests of 64 fixed pseudo-random headers, hashed by
    /// the host build of the node's C++. The harness computes the same with
    /// the wasm32 build; the two builds agree when both match this constant.
    #[test]
    #[cfg(feature = "bc3hash")]
    fn cross_check_matches_the_wasm_harness() {
        use sha2::{Digest, Sha256};
        let mut agg = Sha256::new();
        for i in 0..64u32 {
            let mut h = [0u8; 80];
            for (j, b) in h.iter_mut().enumerate() {
                *b = ((j as u32 * 31 + i * 7 + 0x5a) & 0xff) as u8;
            }
            h[0..4].copy_from_slice(&0x2000_1800u32.to_le_bytes());
            h[76..80].copy_from_slice(&i.to_le_bytes());
            agg.update(bc3hash::bc3hash(&h));
        }
        let got = hex::encode(agg.finalize());
        assert_eq!(got, CROSS_CHECK, "the harness's CROSS_CHECK must be {got}");
        let harness = include_str!("../test/vectors.mjs");
        assert!(harness.contains(&format!("\"{CROSS_CHECK}\"")), "vectors.mjs has another CROSS_CHECK");
    }

    #[cfg(feature = "bc3hash")]
    const CROSS_CHECK: &str = "0b625bcc9887a866a75689b8b1a91f21a5c0fee4b3ce7c319aef84e146a3ebeb";

    /// The self test passes on a correct build - and only on the genesis
    /// block: the same header with another nonce does not hash to it.
    #[test]
    fn the_self_test_passes_on_the_genesis_block() {
        assert_eq!(self_test(), 1);
        let mut merkle_root: [u8; 32] = hex::decode(GENESIS_MERKLE_ROOT).unwrap().try_into().unwrap();
        merkle_root.reverse();
        let other = consensus::BlockHeader {
            version: 1,
            prev_hash: [0u8; 32],
            merkle_root,
            time: 1_777_245_555,
            bits: 0x1d00_ffff,
            nonce: 2_442_659_436,
        };
        assert_ne!(consensus::hash_to_display(&consensus::sha256d(&other.serialize())), GENESIS_HASH);
    }
}
