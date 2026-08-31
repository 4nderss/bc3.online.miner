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

    /// The PoW hash (SHA3-256t when the version bit is set - always so for
    /// jobs from the pool; the SHA256d branch exists only for the test
    /// vectors).
    pub fn hash(&self) -> [u8; 32] {
        let ser = self.serialize();
        if self.version & SHA3_VBIT != 0 {
            sha3t(&ser)
        } else {
            sha256d(&ser)
        }
    }

    pub fn hash_display(&self) -> String {
        let mut h = self.hash();
        h.reverse();
        hex::encode(h)
    }
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
    let push_len = *coinb1.get(pos)? as usize;
    // The BIP34 height is a minimal CScriptNum: 1-4 bytes cover every
    // realistic height (>4 would be invalid here).
    if push_len == 0 || push_len > 4 {
        return None;
    }
    pos += 1;
    let bytes = coinb1.get(pos..pos + push_len)?;
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
