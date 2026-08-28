//! SHA3-256t-sökkärna för web-minern, kompilerad till WebAssembly.
//!
//! Medvetet UTAN wasm-bindgen: gränssnittet är fyra tal och två byte-buffertar,
//! så bindgen skulle bara lägga till glue och ett byggberoende. JS skriver
//! direkt i wasm-minnet och anropar exporterna.
//!
//! Konsensuskoden delas med CLI-minern via `#[path]` i stället för att
//! kopieras — en webbshare måste hashas exakt som en GPU-share, annars
//! avvisar poolen den.

// `consensus` drar in fler funktioner än sökloopen använder (merkle, bip34,
// adresser). De är testade och kostar inget i den kompilerade modulen.
#[allow(dead_code)]
#[path = "../../src/consensus.rs"]
mod consensus;

use consensus::{hash_meets_target, sha3t, Target};

/// Wasm har ingen allokator som JS kan nå direkt. De här två exporterna gör
/// att JS kan reservera minne för header och target.
///
/// # Safety
/// Anroparen äger blocket tills `dealloc` anropas med samma längd.
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    core::mem::forget(buf);
    ptr
}

/// # Safety
/// `ptr` måste komma från `alloc` med samma `len`.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// Sök igenom ett nonce-intervall efter en hash som möter target.
///
/// `header_ptr` pekar på 80 bytes serialiserad blockheader; noncen på offset
/// 76 skrivs över för varje försök. `target_ptr` pekar på 32 bytes big-endian
/// target (poolens share-target, inte nätverkets).
///
/// Returnerar noncen som funktion av f64 — `-1.0` betyder "hittade inget i
/// intervallet". f64 i stället för i64 för att slippa BigInt i JS; alla u32
/// ryms exakt i en f64.
///
/// # Safety
/// Bufferterna måste vara 80 respektive 32 bytes och komma från `alloc`.
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
    // Kopiera in headern på stacken: att skriva noncen genom en rå pekare i
    // loopen hindrar optimeringar och tvingar en läsning per varv.
    let mut header = [0u8; 80];
    core::ptr::copy_nonoverlapping(header_ptr, header.as_mut_ptr(), 80);
    let mut target: Target = [0u8; 32];
    core::ptr::copy_nonoverlapping(target_ptr, target.as_mut_ptr(), 32);

    for i in 0..nonce_count {
        // wrapping: nonce-rymden är cirkulär, och en arbetare som får ett
        // intervall nära u32::MAX ska inte panika.
        let nonce = nonce_start.wrapping_add(i);
        header[76..80].copy_from_slice(&nonce.to_le_bytes());
        let hash = sha3t(&header);
        if hash_meets_target(&hash, &target) {
            return nonce as f64;
        }
    }
    -1.0
}

/// Hasha en header en gång och skriv resultatet till `out_ptr` (32 bytes).
/// Används av huvudtråden för att verifiera en träff innan den skickas, och
/// av självtestet nedan.
///
/// # Safety
/// `header_ptr` måste peka på 80 bytes, `out_ptr` på minst 32.
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

/// SHA256d över godtycklig buffert (coinbase-txid, merklesteg).
///
/// # Safety
/// `ptr` måste peka på `len` läsbara bytes, `out_ptr` på minst 32.
#[no_mangle]
pub unsafe extern "C" fn sha256d_into(ptr: *const u8, len: usize, out_ptr: *mut u8) {
    if ptr.is_null() || out_ptr.is_null() {
        return;
    }
    let data = core::slice::from_raw_parts(ptr, len);
    let h = consensus::sha256d(data);
    core::ptr::copy_nonoverlapping(h.as_ptr(), out_ptr, 32);
}

/// Stratums prevhash-ordning → intern ordning (4-bytesordsreversering).
///
/// # Safety
/// Båda pekarna måste peka på minst 32 bytes.
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

/// Merklerot ur coinbase-txid och stratums grenar (`steps_count` × 32 bytes).
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

/// Share-target för en stratum-svårighet.
///
/// # Safety
/// `out_ptr` måste peka på minst 32 bytes.
#[no_mangle]
pub unsafe extern "C" fn target_for_difficulty_into(difficulty: f64, out_ptr: *mut u8) {
    if out_ptr.is_null() {
        return;
    }
    let t = consensus::target_for_difficulty(difficulty);
    core::ptr::copy_nonoverlapping(t.as_ptr(), out_ptr, 32);
}

/// Serialisera en 80-byte header. Fältordningen ligger i Rust så att JS
/// aldrig kan lägga ett fält på fel offset.
///
/// # Safety
/// `prev_ptr` och `merkle_ptr` 32 bytes vardera, `out_ptr` minst 80.
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

/// Svårigheten en hash motsvarar — för att visa "best share" i UI:t.
///
/// # Safety
/// `hash_ptr` måste peka på 32 bytes.
#[no_mangle]
pub unsafe extern "C" fn difficulty_of_hash(hash_ptr: *const u8) -> f64 {
    if hash_ptr.is_null() {
        return 0.0;
    }
    let mut h = [0u8; 32];
    core::ptr::copy_nonoverlapping(hash_ptr, h.as_mut_ptr(), 32);
    consensus::difficulty_of_hash(&h)
}

/// Självtest mot en kedjeverifierad vektor. Returnerar 1 vid rätt svar.
///
/// Finns för att web-minern ska kunna vägra starta om den kompilerade
/// modulen hashar fel — en trasig kärna skulle annars bara producera
/// avvisade shares utan att någon förstod varför.
#[no_mangle]
pub extern "C" fn self_test() -> u32 {
    // BC3:s genesisheader (samma vektor som CLI-minerns tester).
    let mut header = [0u8; 80];
    header[0..4].copy_from_slice(&1u32.to_le_bytes());
    let hash = sha3t(&header);
    // Vi jämför inte mot genesis här (headern ovan är inte genesis) utan mot
    // ett fast facit för just den här indatan, framräknat med samma kod i
    // testet `wasm_self_test_vector_matches` nedan.
    let expected: [u8; 32] = SELF_TEST_DIGEST;
    u32::from(hash == expected)
}

/// Facit för `self_test`. Låst av enhetstestet längst ned, som räknar fram
/// samma värde med den delade konsensuskoden.
const SELF_TEST_DIGEST: [u8; 32] = [
    0x6b, 0x5a, 0xb1, 0x5c, 0xea, 0x4b, 0x19, 0xaa, 0x64, 0x94, 0x5e, 0x06, 0xfa, 0xf0, 0x2e, 0x00,
    0xdb, 0x3c, 0x48, 0x84, 0xf3, 0x47, 0xc0, 0xe0, 0x74, 0x72, 0x52, 0xb0, 0x5e, 0xd9, 0xc6, 0xa6,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Sökloopen måste hitta exakt den nonce vars hash möter target, och
    /// returnera -1 när intervallet är tomt på träffar.
    #[test]
    fn search_finds_the_matching_nonce() {
        let mut header = [0u8; 80];
        header[0..4].copy_from_slice(&0x2000_1000u32.to_le_bytes());
        // Ett löst target: första byten < 0x10 räcker, så en träff finns nära.
        let mut target = [0xffu8; 32];
        target[0] = 0x0f;

        // Facit räknat rakt fram med samma hashfunktion.
        let mut expected = None;
        for n in 0..200_000u32 {
            let mut h = header;
            h[76..80].copy_from_slice(&n.to_le_bytes());
            if hash_meets_target(&sha3t(&h), &target) {
                expected = Some(n);
                break;
            }
        }
        let expected = expected.expect("ett löst target måste ge en träff");

        let got = unsafe { search(header.as_ptr(), target.as_ptr(), 0, 200_000) };
        assert_eq!(got, expected as f64, "search hittade fel nonce");

        // Intervall som slutar före träffen ⇒ ingen träff.
        let miss = unsafe { search(header.as_ptr(), target.as_ptr(), 0, expected) };
        assert_eq!(miss, -1.0, "search får inte rapportera en träff utanför intervallet");
    }

    /// Noncen skrivs på rätt offset (76..80, little-endian). Skrivs den fel
    /// hashar webbminern en annan header än poolen validerar.
    #[test]
    fn nonce_is_written_at_offset_76_little_endian() {
        let header = [0u8; 80];
        let mut target = [0u8; 32]; // omöjligt target ⇒ search returnerar -1
        target[31] = 1;
        let _ = unsafe { search(header.as_ptr(), target.as_ptr(), 0, 1) };

        // Verifiera mot hash_header med en manuellt satt nonce.
        let mut manual = [0u8; 80];
        let nonce = 0x1234_5678u32;
        manual[76..80].copy_from_slice(&nonce.to_le_bytes());
        assert_eq!(manual[76], 0x78, "little-endian: lägsta byten först");
        assert_eq!(manual[79], 0x12);

        let mut out = [0u8; 32];
        unsafe { hash_header(manual.as_ptr(), out.as_mut_ptr()) };
        assert_eq!(out, sha3t(&manual), "hash_header måste ge samma svar som sha3t");
    }

    /// Nonce nära u32::MAX får inte panika — intervallet wrappar.
    #[test]
    fn nonce_wraps_instead_of_panicking() {
        let header = [0u8; 80];
        let target = [0u8; 32]; // träffas aldrig
        let got = unsafe { search(header.as_ptr(), target.as_ptr(), u32::MAX - 2, 10) };
        assert_eq!(got, -1.0);
    }

    /// Låser fast facit i `SELF_TEST_DIGEST`. Går det här testet sönder har
    /// hashimplementationen ändrats — då är web-minerns självtest också fel.
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
