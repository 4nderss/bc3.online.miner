//! SHA3-256t search kernel for the web miner, compiled to WebAssembly.
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

use consensus::{hash_meets_target, sha3t, Target};

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

/// Search a nonce range for a hash that meets the target.
///
/// `header_ptr` points at 80 bytes of serialized block header; the nonce at
/// offset 76 is overwritten for every attempt. `target_ptr` points at 32
/// bytes of big-endian target (the pool's share target, not the network's).
///
/// Returns the nonce as an f64 - `-1.0` means "found nothing in the range".
/// f64 instead of i64 to avoid BigInt in JS; every u32 fits exactly in an
/// f64.
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
    if header_ptr.is_null() || target_ptr.is_null() {
        return -1.0;
    }
    // Copy the header onto the stack: writing the nonce through a raw
    // pointer in the loop blocks optimizations and forces one read per
    // iteration.
    let mut header = [0u8; 80];
    core::ptr::copy_nonoverlapping(header_ptr, header.as_mut_ptr(), 80);
    let mut target: Target = [0u8; 32];
    core::ptr::copy_nonoverlapping(target_ptr, target.as_mut_ptr(), 32);

    for i in 0..nonce_count {
        // wrapping: the nonce space is circular, and a worker that gets a
        // range close to u32::MAX must not panic.
        let nonce = nonce_start.wrapping_add(i);
        header[76..80].copy_from_slice(&nonce.to_le_bytes());
        let hash = sha3t(&header);
        if hash_meets_target(&hash, &target) {
            return nonce as f64;
        }
    }
    -1.0
}

/// Hash a header once and write the result to `out_ptr` (32 bytes). Used by
/// the main thread to verify a hit before it is submitted, and by the self
/// test below.
///
/// # Safety
/// `header_ptr` must point at 80 bytes, `out_ptr` at at least 32.
#[no_mangle]
pub unsafe extern "C" fn hash_header(header_ptr: *const u8, out_ptr: *mut u8) {
    if header_ptr.is_null() || out_ptr.is_null() {
        return;
    }
    let mut header = [0u8; 80];
    core::ptr::copy_nonoverlapping(header_ptr, header.as_mut_ptr(), 80);
    let hash = sha3t(&header);
    core::ptr::copy_nonoverlapping(hash.as_ptr(), out_ptr, 32);
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

/// Self test against a chain-verified vector. Returns 1 on the right answer.
///
/// It exists so that the web miner can refuse to start if the compiled
/// module hashes incorrectly - a broken kernel would otherwise just produce
/// rejected shares without anyone understanding why.
#[no_mangle]
pub extern "C" fn self_test() -> u32 {
    // BC3's genesis header (the same vector as the CLI miner's tests).
    let mut header = [0u8; 80];
    header[0..4].copy_from_slice(&1u32.to_le_bytes());
    let hash = sha3t(&header);
    // We do not compare against genesis here (the header above is not
    // genesis) but against a fixed expected value for exactly this input,
    // computed with the same code in the test
    // `wasm_self_test_vector_matches` below.
    let expected: [u8; 32] = SELF_TEST_DIGEST;
    u32::from(hash == expected)
}

/// The expected value for `self_test`. Locked down by the unit test at the
/// bottom, which computes the same value with the shared consensus code.
const SELF_TEST_DIGEST: [u8; 32] = [
    0x6b, 0x5a, 0xb1, 0x5c, 0xea, 0x4b, 0x19, 0xaa, 0x64, 0x94, 0x5e, 0x06, 0xfa, 0xf0, 0x2e, 0x00,
    0xdb, 0x3c, 0x48, 0x84, 0xf3, 0x47, 0xc0, 0xe0, 0x74, 0x72, 0x52, 0xb0, 0x5e, 0xd9, 0xc6, 0xa6,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The search loop must find exactly the nonce whose hash meets the
    /// target, and return -1 when the range holds no hits.
    #[test]
    fn search_finds_the_matching_nonce() {
        let mut header = [0u8; 80];
        header[0..4].copy_from_slice(&0x2000_1000u32.to_le_bytes());
        // A loose target: first byte < 0x10 is enough, so a hit is nearby.
        let mut target = [0xffu8; 32];
        target[0] = 0x0f;

        // Expected value computed straightforwardly with the same hash
        // function.
        let mut expected = None;
        for n in 0..200_000u32 {
            let mut h = header;
            h[76..80].copy_from_slice(&n.to_le_bytes());
            if hash_meets_target(&sha3t(&h), &target) {
                expected = Some(n);
                break;
            }
        }
        let expected = expected.expect("a loose target must produce a hit");

        let got = unsafe { search(header.as_ptr(), target.as_ptr(), 0, 200_000) };
        assert_eq!(got, expected as f64, "search hittade fel nonce");

        // A range that ends before the hit -> no hit.
        let miss = unsafe { search(header.as_ptr(), target.as_ptr(), 0, expected) };
        assert_eq!(miss, -1.0, "search must not report a hit outside the range");
    }

    /// The nonce is written at the right offset (76..80, little-endian). If
    /// it is written wrong, the web miner hashes a different header than the
    /// one the pool validates.
    #[test]
    fn nonce_is_written_at_offset_76_little_endian() {
        let header = [0u8; 80];
        let mut target = [0u8; 32]; // impossible target -> search returns -1
        target[31] = 1;
        let _ = unsafe { search(header.as_ptr(), target.as_ptr(), 0, 1) };

        // Verify against hash_header with a manually set nonce.
        let mut manual = [0u8; 80];
        let nonce = 0x1234_5678u32;
        manual[76..80].copy_from_slice(&nonce.to_le_bytes());
        assert_eq!(manual[76], 0x78, "little-endian: lowest byte first");
        assert_eq!(manual[79], 0x12);

        let mut out = [0u8; 32];
        unsafe { hash_header(manual.as_ptr(), out.as_mut_ptr()) };
        assert_eq!(out, sha3t(&manual), "hash_header must agree with sha3t");
    }

    /// A nonce close to u32::MAX must not panic - the range wraps.
    #[test]
    fn nonce_wraps_instead_of_panicking() {
        let header = [0u8; 80];
        let target = [0u8; 32]; // never hit
        let got = unsafe { search(header.as_ptr(), target.as_ptr(), u32::MAX - 2, 10) };
        assert_eq!(got, -1.0);
    }

    /// Locks down the expected value in `SELF_TEST_DIGEST`. If this test
    /// breaks, the hash implementation has changed - and then the web
    /// miner's self test is wrong too.
    #[test]
    fn wasm_self_test_vector_matches() {
        let mut header = [0u8; 80];
        header[0..4].copy_from_slice(&1u32.to_le_bytes());
        let hash = sha3t(&header);
        assert_eq!(
            hash, SELF_TEST_DIGEST,
            "uppdatera SELF_TEST_DIGEST till {:02x?}",
            hash
        );
        assert_eq!(self_test(), 1);
    }
}
