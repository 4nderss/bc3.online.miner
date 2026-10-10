//! BC3 consensus primitives the miner needs: SHA3-256t, header
//! serialization, targets and merkle folding.
//!
//! Deliberately duplicated from the pool's private `bc3-core` (this client is
//! open source). The test vectors at the bottom are chain-verified - the same
//! expected values as the pool's - and lock the implementations to each other.

use sha3::{Digest, Sha3_256};

/// Version bit marking SHA3-256t hashing (set in every post-fork block).
pub const SHA3_VBIT: u32 = 0x0000_1000;

/// SHA3-256t: triple NIST SHA3-256 - BC3's PoW function.
#[inline]
pub fn sha3t(data: &[u8]) -> [u8; 32] {
    let h1 = Sha3_256::digest(data);
    let h2 = Sha3_256::digest(h1);
    let h3 = Sha3_256::digest(h2);
    h3.into()
}

/// SHA256d - for txids/merkle (unchanged from Bitcoin).
pub fn sha256d(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest as _, Sha256};
    let first = Sha256::digest(data);
    Sha256::digest(first).into()
}

/// A proof-of-work function, chosen per job.
///
/// BC3 has changed its PoW once already (SHA256d -> SHA3-256t at the version
/// bit) and a second hard fork will replace it again. Everything that hashes a
/// header goes through this enum, and every `match` on it is exhaustive on
/// purpose: adding the next variant must fail to compile at each place that
/// has to learn about it, rather than silently fall into a default arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowAlgo {
    /// Pre-fork blocks (no version bit). Never sent by the pool today; kept
    /// so the chain-verified genesis vector still runs through this path.
    Sha256d,
    /// SHA3-256t, selected by `SHA3_VBIT`.
    Sha3t,
    // NEW POW: add the variant for the next hard fork here, then fix every
    // compile error - `name`, `hash`, `PowSchedule::algo_for`, the backends'
    // `supports` and the wasm algo ids.
}

impl PowAlgo {
    /// The name used in logs and shared with the pool.
    pub fn name(&self) -> &'static str {
        match self {
            PowAlgo::Sha256d => "SHA256d",
            PowAlgo::Sha3t => "SHA3-256t",
        }
    }

    /// The version-bit rule alone, without the height schedule.
    pub fn for_version(version: u32) -> PowAlgo {
        if version & SHA3_VBIT != 0 {
            PowAlgo::Sha3t
        } else {
            PowAlgo::Sha256d
        }
    }

    /// Hash a serialized 80-byte header. This is the CPU reference for every
    /// algorithm - and the oracle every GPU hit is re-checked against.
    pub fn hash(&self, header80: &[u8; 80]) -> [u8; 32] {
        match self {
            PowAlgo::Sha256d => sha256d(header80),
            PowAlgo::Sha3t => sha3t(header80),
        }
    }

    /// The hash as a plain function, so a hot loop can resolve the algorithm
    /// once per job instead of matching on every nonce.
    pub fn hash_fn(&self) -> fn(&[u8; 80]) -> [u8; 32] {
        match self {
            PowAlgo::Sha256d => |h: &[u8; 80]| sha256d(h),
            PowAlgo::Sha3t => |h: &[u8; 80]| sha3t(h),
        }
    }
}

/// Height of the next PoW hard fork on mainnet. `None` until the spec and the
/// height are known; set it in the same release that adds the `PowAlgo`
/// variant, or this build stops mining at that height (which is the point:
/// grinding the old PoW past the fork only produces rejected shares).
pub const MAINNET_NEXT_POW_HEIGHT: Option<u32> = None;

/// Which PoW a block at a given height uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowSchedule {
    /// First height of the next PoW. From that height on this build has no
    /// implementation and `algo_for` answers `None`.
    pub next_pow_height: Option<u32>,
}

impl PowSchedule {
    pub const MAINNET: PowSchedule = PowSchedule {
        next_pow_height: MAINNET_NEXT_POW_HEIGHT,
    };

    /// The PoW for a block at `height` with header `version`. `None` means a
    /// PoW this build does not implement - do not hash at all.
    pub fn algo_for(&self, height: u32, version: u32) -> Option<PowAlgo> {
        match self.next_pow_height {
            // NEW POW: once the variant exists, return it here instead.
            Some(h) if height >= h => None,
            _ => Some(PowAlgo::for_version(version)),
        }
    }

    /// `algo_for` for a job whose height may not be readable. Without a fork
    /// scheduled the height does not matter; with one, an unknown height could
    /// be on either side of it, and guessing wrong grinds a dead PoW.
    pub fn algo_for_job(&self, height: Option<u32>, version: u32) -> Option<PowAlgo> {
        match (height, self.next_pow_height) {
            (Some(h), _) => self.algo_for(h, version),
            (None, None) => Some(PowAlgo::for_version(version)),
            (None, Some(_)) => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BlockHeader {
    pub version: u32,
    pub prev_hash: [u8; 32],
    pub merkle_root: [u8; 32],
    pub time: u32,
    pub bits: u32,
    pub nonce: u32,
}

impl BlockHeader {
    pub fn serialize(&self) -> [u8; 80] {
        let mut out = [0u8; 80];
        out[0..4].copy_from_slice(&self.version.to_le_bytes());
        out[4..36].copy_from_slice(&self.prev_hash);
        out[36..68].copy_from_slice(&self.merkle_root);
        out[68..72].copy_from_slice(&self.time.to_le_bytes());
        out[72..76].copy_from_slice(&self.bits.to_le_bytes());
        out[76..80].copy_from_slice(&self.nonce.to_le_bytes());
        out
    }

    /// The PoW hash by the version-bit rule alone. Mining goes through the
    /// job's `PowAlgo` instead, which also knows the fork height; this stays
    /// for the chain-verified vectors - test-only, so nothing mines with it.
    #[cfg(test)]
    pub fn hash(&self) -> [u8; 32] {
        PowAlgo::for_version(self.version).hash(&self.serialize())
    }

    #[cfg(test)]
    pub fn hash_display(&self) -> String {
        hash_to_display(&self.hash())
    }
}

/// A hash in display order (byte-reversed hex), as explorers show it.
pub fn hash_to_display(hash: &[u8; 32]) -> String {
    let mut h = *hash;
    h.reverse();
    hex::encode(h)
}

/// 256-bit target as big-endian bytes; compared lexicographically.
pub type Target = [u8; 32];

/// Expand compact nBits into a target (None for invalid/negative ones).
pub fn compact_to_target(bits: u32) -> Option<Target> {
    let exponent = (bits >> 24) as usize;
    let mantissa = bits & 0x007f_ffff;
    if bits & 0x0080_0000 != 0 || exponent > 34 {
        return None;
    }
    let mut out = [0u8; 32];
    let bytes = [(mantissa >> 16) as u8, (mantissa >> 8) as u8, mantissa as u8];
    for (i, b) in bytes.iter().enumerate() {
        // The most significant mantissa byte lands at position 32 - exponent.
        let pos = 32usize.checked_sub(exponent)?.checked_add(i)?;
        if pos < 32 {
            out[pos] = *b;
        } else if *b != 0 {
            return None; // would be truncated
        }
    }
    Some(out)
}

/// Does the hash (internal order) meet the target? hash <= target.
#[inline]
pub fn hash_meets_target(hash: &[u8; 32], target: &Target) -> bool {
    // The hash is little-endian as a number - compare big-endian in reverse.
    for i in (0..32).rev() {
        let h = hash[i];
        let t = target[31 - i];
        if h < t {
            return true;
        }
        if h > t {
            return false;
        }
    }
    true
}

/// Target for a stratum difficulty: diff1 target / difficulty.
/// Fractional difficulties are supported via 2^24 scaling (as the pool does).
pub fn target_for_difficulty(difficulty: f64) -> Target {
    // diff1 = 0x00000000ffff0000...0000. Division is done in u128 limbs.
    const SCALE_BITS: u32 = 24;
    let divisor = ((difficulty * (1u64 << SCALE_BITS) as f64).max(1.0)) as u128;
    // diff1 << 24 as a 256-bit number in four 64-bit big-endian limbs:
    // diff1 = 0xffff << 208 -> shifted = 0xffff << 232 ... as a byte array.
    // Simple long division over 32 bytes.
    let mut dividend = [0u8; 33]; // extra byte for the <<24 overhang
    // diff1 (32B BE) has 0xff at bytes 4,5. In the 33-byte array that becomes
    // idx 5,6; <<24 moves it up three bytes -> idx 2,3.
    dividend[2] = 0xff;
    dividend[3] = 0xff;
    let mut out = [0u8; 32];
    let mut rem: u128 = 0;
    for i in 0..33 {
        rem = (rem << 8) | dividend[i] as u128;
        let q = rem / divisor;
        rem %= divisor;
        if i > 0 {
            out[i - 1] = q as u8; // q < 256 guaranteed by the long-division invariant
        } else {
            debug_assert_eq!(q, 0);
        }
    }
    out
}

/// Read the block height from the coinbase scriptSig (BIP34: the height is
/// the first push).
///
/// `coinb1` is stratum's first coinbase part: version(4) | in-count(1) |
/// prevout(36) | scriptSig-length(varint) | scriptSig prefix... The height
/// comes first in the scriptSig as `<len> <len bytes little-endian>`.
///
/// This height now also picks the job's PoW (see `PowSchedule`), so a wrong
/// answer is worse than none: anything that is not a well-formed,
/// non-negative push gives `None`.
pub fn bip34_height(coinb1: &[u8]) -> Option<u32> {
    let mut pos = 4 + 1 + 36; // version + input count + null prevout
    // Skip past the scriptSig length (varint).
    let first = *coinb1.get(pos)?;
    pos += match first {
        0xfd => 3,
        0xfe => 5,
        0xff => 9,
        _ => 1,
    };
    let op = *coinb1.get(pos)?;
    // Bitcoin Core writes the height with `CScript() << nHeight`, which uses
    // the small-integer opcodes for 0..=16 - only a regtest chain ever sees
    // them, and regtest is exactly where a fork height gets tested.
    match op {
        0x00 => return Some(0), // OP_0
        0x51..=0x60 => return Some((op - 0x50) as u32), // OP_1..OP_16
        _ => {}
    }
    let push_len = op as usize;
    // A minimal CScriptNum: 1-4 bytes cover every realistic height (>4 would
    // be invalid here).
    if push_len == 0 || push_len > 4 {
        return None;
    }
    pos += 1;
    let bytes = coinb1.get(pos..pos + push_len)?;
    // The top bit of the last byte is the sign. A negative height is not a
    // height.
    if bytes[push_len - 1] & 0x80 != 0 {
        return None;
    }
    let mut height = 0u32;
    for (i, b) in bytes.iter().enumerate() {
        height |= (*b as u32) << (8 * i);
    }
    Some(height)
}

/// The difficulty a hash actually reached ("best share") = diff1 target / hash.
/// The same definition the pool uses for its best-share statistics.
pub fn difficulty_of_hash(hash: &[u8; 32]) -> f64 {
    // The hash is a little-endian number; convert to f64 in a big-endian loop.
    let mut h = 0.0f64;
    for b in hash.iter().rev() {
        h = h * 256.0 + *b as f64;
    }
    if h == 0.0 {
        return f64::INFINITY;
    }
    let mut d1 = 0.0f64;
    for b in compact_to_target(0x1d00ffff).unwrap() {
        d1 = d1 * 256.0 + b as f64;
    }
    d1 / h
}

/// Network difficulty from nBits (for the ETA calculation).
pub fn difficulty_of_bits(bits: u32) -> f64 {
    let Some(t) = compact_to_target(bits) else {
        return 0.0;
    };
    let mut target = 0.0f64;
    for b in t {
        target = target * 256.0 + b as f64;
    }
    if target == 0.0 {
        return 0.0;
    }
    // The diff1 target as an f64.
    let mut d1 = 0.0f64;
    for b in compact_to_target(0x1d00ffff).unwrap() {
        d1 = d1 * 256.0 + b as f64;
    }
    d1 / target
}

/// Merkle root from the coinbase txid + the stratum branches.
pub fn root_from_steps(coinbase_txid: &[u8; 32], steps: &[[u8; 32]]) -> [u8; 32] {
    let mut acc = *coinbase_txid;
    let mut buf = [0u8; 64];
    for step in steps {
        buf[..32].copy_from_slice(&acc);
        buf[32..].copy_from_slice(step);
        acc = sha256d(&buf);
    }
    acc
}

/// Stratum's prevhash field <-> internal order: 4-byte word swap (involution).
pub fn swab32(bytes: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, chunk) in bytes.chunks(4).enumerate() {
        for (j, b) in chunk.iter().rev().enumerate() {
            out[i * 4 + j] = *b;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_from_display(s: &str) -> [u8; 32] {
        let mut h: [u8; 32] = hex::decode(s).unwrap().try_into().unwrap();
        h.reverse();
        h
    }

    // Same chain-verified expected values as the pool's bc3-core - this is
    // what locks the two implementations together.
    #[test]
    fn genesis_vectors() {
        let genesis = BlockHeader {
            version: 1,
            prev_hash: [0u8; 32],
            merkle_root: hash_from_display(
                "8e1df52fddd25c460304ff8ea7bcb570850bf0b0c027eecf8ebf8ab17d3e93b1",
            ),
            time: 1_777_245_555,
            bits: 0x1d00ffff,
            nonce: 2_442_659_435,
        };
        assert_eq!(
            genesis.hash_display(),
            "000000000c226a41e70717f6d4fbdcb6bfb4fdc40831ccc87fa9cfdd2c57bff6"
        );
        let mut h = sha3t(&genesis.serialize());
        h.reverse();
        assert_eq!(
            hex::encode(h),
            "e60e0c32fbfed8d800c0d179a1843c0537d20ab2c8b7c2859bca4bf142cac9b5"
        );
    }

    #[test]
    fn compact_diff1() {
        let t = compact_to_target(0x1d00ffff).unwrap();
        assert_eq!(
            hex::encode(t),
            "00000000ffff0000000000000000000000000000000000000000000000000000"
        );
        assert!((difficulty_of_bits(0x1d00ffff) - 1.0).abs() < 1e-9);
        assert_eq!(compact_to_target(0x0180_0000), None);
    }

    #[test]
    fn target_difficulty_scaling() {
        assert_eq!(target_for_difficulty(1.0), compact_to_target(0x1d00ffff).unwrap());
        let t16 = target_for_difficulty(16.0);
        // diff 16 = diff1 >> 4.
        assert_eq!(
            hex::encode(t16),
            "000000000ffff000000000000000000000000000000000000000000000000000"
        );
    }

    #[test]
    fn genesis_meets_diff1() {
        let h = hash_from_display(
            "000000000c226a41e70717f6d4fbdcb6bfb4fdc40831ccc87fa9cfdd2c57bff6",
        );
        assert!(hash_meets_target(&h, &compact_to_target(0x1d00ffff).unwrap()));
        assert!(!hash_meets_target(&h, &target_for_difficulty(100.0)));
    }

    #[test]
    fn bip34_height_reads_the_coinbase() {
        // Build a coinb1 the way the pool does: version | 1 input |
        // null prevout | scriptlen | <3><ce e7 00> (=59342) | tag...
        let mut cb = Vec::new();
        cb.extend_from_slice(&2u32.to_le_bytes());
        cb.push(0x01);
        cb.extend_from_slice(&[0u8; 32]);
        cb.extend_from_slice(&0xffff_ffffu32.to_le_bytes());
        cb.push(24); // scriptSig length (varint < 0xfd)
        cb.extend_from_slice(&[0x03, 0xce, 0xe7, 0x00]);
        cb.extend_from_slice(b"/bc3.online/solo/");
        assert_eq!(bip34_height(&cb), Some(59_342));

        // Two-byte height (the fork height 30240 = 0x7620).
        let mut cb2 = cb[..41].to_vec();
        cb2.push(20);
        cb2.extend_from_slice(&[0x02, 0x20, 0x76]);
        assert_eq!(bip34_height(&cb2), Some(30_240));

        // Broken input must give None, not a panic.
        assert_eq!(bip34_height(&[]), None);
        assert_eq!(bip34_height(&cb[..30]), None);
    }

    /// The parser now decides the PoW, so the edges matter: regtest heights
    /// written as small-integer opcodes, a push cut off by the end of coinb1,
    /// oversized and negative pushes.
    #[test]
    fn bip34_height_edge_cases() {
        let prefix = |script: &[u8]| {
            let mut cb = Vec::new();
            cb.extend_from_slice(&1u32.to_le_bytes());
            cb.push(0x01);
            cb.extend_from_slice(&[0u8; 32]);
            cb.extend_from_slice(&0xffff_ffffu32.to_le_bytes());
            cb.push(script.len() as u8 + 8);
            cb.extend_from_slice(script);
            cb
        };
        assert_eq!(bip34_height(&prefix(&[0x00])), Some(0)); // OP_0
        assert_eq!(bip34_height(&prefix(&[0x51])), Some(1)); // OP_1
        assert_eq!(bip34_height(&prefix(&[0x60])), Some(16)); // OP_16
        assert_eq!(bip34_height(&prefix(&[0x01, 0x11])), Some(17));
        // 128 needs a second byte, or the sign bit would make it negative.
        assert_eq!(bip34_height(&prefix(&[0x02, 0x80, 0x00])), Some(128));
        assert_eq!(bip34_height(&prefix(&[0x01, 0x80])), None);
        assert_eq!(bip34_height(&prefix(&[0x04, 0xff, 0xff, 0xff, 0x7f])), Some(0x7fff_ffff));
        // Too long for a height, and a push that runs past the data.
        assert_eq!(bip34_height(&prefix(&[0x05, 1, 2, 3, 4, 5])), None);
        assert_eq!(bip34_height(&prefix(&[0x03, 0x01])), None);
        // A three-byte varint for the scriptSig length is skipped correctly.
        let mut cb = prefix(&[]);
        cb.pop();
        cb.extend_from_slice(&[0xfd, 0x00, 0x01, 0x03, 0xce, 0xe7, 0x00]);
        assert_eq!(bip34_height(&cb), Some(59_342));
    }

    #[test]
    fn pow_schedule_without_a_fork_follows_the_version_bit() {
        let s = PowSchedule { next_pow_height: None };
        assert_eq!(s.algo_for(10, 1), Some(PowAlgo::Sha256d));
        assert_eq!(s.algo_for(30_239, 0x2000_0000), Some(PowAlgo::Sha256d));
        assert_eq!(s.algo_for(59_342, 0x2000_1000), Some(PowAlgo::Sha3t));
        assert_eq!(s.algo_for(u32::MAX, SHA3_VBIT), Some(PowAlgo::Sha3t));
        // Mainnet has no next fork until a release sets one.
        assert_eq!(PowSchedule::MAINNET.next_pow_height, MAINNET_NEXT_POW_HEIGHT);
    }

    #[test]
    fn pow_schedule_refuses_heights_at_and_after_the_next_fork() {
        let s = PowSchedule { next_pow_height: Some(100_000) };
        assert_eq!(s.algo_for(99_999, 0x2000_1000), Some(PowAlgo::Sha3t));
        assert_eq!(s.algo_for(99_999, 0x2000_0000), Some(PowAlgo::Sha256d));
        assert_eq!(s.algo_for(100_000, 0x2000_1000), None);
        assert_eq!(s.algo_for(100_001, 0x2000_1000), None);
        assert_eq!(s.algo_for(u32::MAX, 0), None);
    }

    /// An unreadable height is harmless without a fork scheduled, and a
    /// reason to stop with one: it could be on either side.
    #[test]
    fn pow_schedule_with_an_unknown_height() {
        let none = PowSchedule { next_pow_height: None };
        assert_eq!(none.algo_for_job(None, 0x2000_1000), Some(PowAlgo::Sha3t));
        let fork = PowSchedule { next_pow_height: Some(5) };
        assert_eq!(fork.algo_for_job(None, 0x2000_1000), None);
        assert_eq!(fork.algo_for_job(Some(4), 0x2000_1000), Some(PowAlgo::Sha3t));
        assert_eq!(fork.algo_for_job(Some(5), 0x2000_1000), None);
    }

    /// `hash`, `hash_fn` and the version-bit `BlockHeader::hash` must agree.
    #[test]
    fn pow_algo_dispatch_matches_the_reference() {
        let mut h = BlockHeader {
            version: 1,
            prev_hash: [3u8; 32],
            merkle_root: [5u8; 32],
            time: 1_777_245_555,
            bits: 0x1d00ffff,
            nonce: 42,
        };
        for algo in [PowAlgo::Sha256d, PowAlgo::Sha3t] {
            h.version = match algo {
                PowAlgo::Sha256d => 0x2000_0000,
                PowAlgo::Sha3t => 0x2000_1000,
            };
            let ser = h.serialize();
            assert_eq!(PowAlgo::for_version(h.version), algo);
            assert_eq!(algo.hash(&ser), h.hash(), "{}", algo.name());
            assert_eq!((algo.hash_fn())(&ser), h.hash(), "{}", algo.name());
        }
        assert_eq!(PowAlgo::Sha3t.hash(&h.serialize()), sha3t(&h.serialize()));
        assert_eq!(PowAlgo::Sha256d.name(), "SHA256d");
        assert_eq!(PowAlgo::Sha3t.name(), "SHA3-256t");
    }

    #[test]
    fn difficulty_of_hash_matches_known_values() {
        // The genesis hash has ~36 zero bits; diff1 matches 32 -> about 2^4.
        let h = hash_from_display(
            "000000000c226a41e70717f6d4fbdcb6bfb4fdc40831ccc87fa9cfdd2c57bff6",
        );
        let d = difficulty_of_hash(&h);
        assert!(d > 1.0 && d < 100.0, "d = {d}");
        // A target equal to diff1 is by definition difficulty 1.
        let diff1 = compact_to_target(0x1d00ffff).unwrap();
        let mut le = diff1;
        le.reverse(); // target is big-endian; a hash is read little-endian
        assert!((difficulty_of_hash(&le) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn swab_is_involution() {
        let mut b = [0u8; 32];
        for (i, x) in b.iter_mut().enumerate() {
            *x = i as u8;
        }
        assert_eq!(swab32(&swab32(&b)), b);
        assert_eq!(&swab32(&b)[..4], &[3, 2, 1, 0]);
    }
}
