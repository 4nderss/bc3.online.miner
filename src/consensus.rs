//! BC3 consensus primitives the miner needs: the PoW functions, header
//! serialization, targets and merkle folding.
//!
//! Deliberately duplicated from the pool's private `bc3-core` (this client is
//! open source). The test vectors at the bottom are chain-verified - the same
//! expected values as the pool's - and lock the implementations to each other.
//! BC3Hash is the exception: it is the node's own C++ (the `bc3hash` crate),
//! because a port could differ from the node in one bit and nobody would know
//! until no block was ever accepted.

use sha3::{Digest, Sha3_256};

/// Version bit marking SHA3-256t hashing (set in every block since the first
/// fork, and still set after the second).
pub const SHA3_VBIT: u32 = 0x0000_1000;

/// Version bit 11, marking BC3Hash (set from the second fork on). It takes
/// precedence over `SHA3_VBIT`; both stay in the hashed header.
pub const BC3HASH_VBIT: u32 = 0x0000_0800;

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
/// BC3 has changed its PoW twice (SHA256d -> SHA3-256t -> BC3Hash), each time
/// at a version bit. Everything that hashes a header goes through this enum,
/// and every `match` on it is exhaustive on purpose: a new variant must fail
/// to compile at each place that has to learn about it, rather than silently
/// fall into a default arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowAlgo {
    /// No version bit: blocks before the first fork. Never sent by the pool;
    /// kept so the chain-verified genesis vector still runs through this path.
    Sha256d,
    /// `SHA3_VBIT` without `BC3HASH_VBIT`.
    Sha3t,
    /// `BC3HASH_VBIT`. CPU only: there is no GPU kernel for it.
    Bc3Hash,
}

/// A PoW over one serialized 80-byte header.
pub type PowFn = fn(&[u8; 80]) -> [u8; 32];

/// BC3Hash where this build has it. The browser build (wasm/) shares this file
/// but not the C++ behind it, and leaves the `bc3hash` feature off.
#[cfg(feature = "bc3hash")]
const BC3HASH: Option<PowFn> = Some(|h: &[u8; 80]| ::bc3hash::bc3hash(h));
#[cfg(not(feature = "bc3hash"))]
const BC3HASH: Option<PowFn> = None;

impl PowAlgo {
    /// Every variant, in `index` order.
    pub const ALL: [PowAlgo; 3] = [PowAlgo::Sha256d, PowAlgo::Sha3t, PowAlgo::Bc3Hash];

    /// Position in `ALL`, for per-algorithm tables.
    pub fn index(&self) -> usize {
        match self {
            PowAlgo::Sha256d => 0,
            PowAlgo::Sha3t => 1,
            PowAlgo::Bc3Hash => 2,
        }
    }

    /// The name used in logs and shared with the pool.
    pub fn name(&self) -> &'static str {
        match self {
            PowAlgo::Sha256d => "SHA256d",
            PowAlgo::Sha3t => "SHA3-256t",
            PowAlgo::Bc3Hash => "BC3Hash",
        }
    }

    /// The node's rule, `CBlockHeader::GetHash`: bit 11 before bit 12, and
    /// neither means SHA256d. The version alone decides - not the height - so
    /// the job's version is all a miner needs.
    pub fn for_version(version: u32) -> PowAlgo {
        if version & BC3HASH_VBIT != 0 {
            PowAlgo::Bc3Hash
        } else if version & SHA3_VBIT != 0 {
            PowAlgo::Sha3t
        } else {
            PowAlgo::Sha256d
        }
    }

    /// The CPU implementation of this PoW, as a plain function so a hot loop
    /// resolves it once per job instead of matching on every nonce. It is the
    /// reference every GPU hit is re-checked against. None where this build
    /// has no implementation - BC3Hash in the browser build.
    pub fn hash_fn(&self) -> Option<PowFn> {
        match self {
            PowAlgo::Sha256d => Some(|h: &[u8; 80]| sha256d(h)),
            PowAlgo::Sha3t => Some(|h: &[u8; 80]| sha3t(h)),
            PowAlgo::Bc3Hash => BC3HASH,
        }
    }

    /// Hash one serialized 80-byte header; None as for `hash_fn`.
    #[cfg(test)]
    pub fn hash(&self, header80: &[u8; 80]) -> Option<[u8; 32]> {
        self.hash_fn().map(|f| f(header80))
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

    /// The block hash, as the node's `GetHash` computes it. Test-only: mining
    /// resolves the job's `PowAlgo` once per job instead of once per header.
    #[cfg(test)]
    pub fn hash(&self) -> [u8; 32] {
        let algo = PowAlgo::for_version(self.version);
        algo.hash(&self.serialize())
            .unwrap_or_else(|| panic!("this build has no {}", algo.name()))
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

/// Target for a stratum difficulty: diff1 target / difficulty, computed
/// exactly as the pool's bc3-core does - floor((diff1 << 31) /
/// floor(difficulty * 2^31)) - so the two agree on every share.
///
/// 31 bits, not the 24 this used to scale by: BC3Hash share difficulties sit
/// around 1e-6, and the pool's floor is 1e-7. At 24 bits the difficulty is
/// rounded to whole steps of 6e-8, so at 1e-7 the target came out 1.7 times
/// too easy and ~40 % of the shares sent were below the pool's target - about
/// 5 % at 1e-6. diff1 is below 2^224, so the shift still fits in 256 bits.
pub fn target_for_difficulty(difficulty: f64) -> Target {
    const SCALE_BITS: u32 = 31;
    // The remainder below is shifted by a byte per step inside a u128, so the
    // divisor has to stay under 2^120 - a share difficulty of 2^89, which no
    // pool comes near.
    let divisor = ((difficulty * (1u64 << SCALE_BITS) as f64).max(1.0) as u128)
        .min((1u128 << 120) - 1);
    let diff1 = compact_to_target(0x1d00ffff).expect("diff1 is a valid target");
    let dividend = shl(&diff1, SCALE_BITS);
    // Long division over the 32 big-endian bytes.
    let mut out = [0u8; 32];
    let mut rem: u128 = 0;
    for (q, d) in out.iter_mut().zip(dividend) {
        rem = (rem << 8) | d as u128;
        *q = (rem / divisor) as u8; // < 256: rem < divisor before the shift
        rem %= divisor;
    }
    out
}

/// `x << n` for a 256-bit big-endian number, n < 256; bits shifted past the
/// top are lost.
fn shl(x: &[u8; 32], n: u32) -> [u8; 32] {
    let (bytes, bits) = ((n / 8) as usize, n % 8);
    let byte = |i: usize| x.get(i).copied().unwrap_or(0);
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = if bits == 0 {
            byte(i + bytes)
        } else {
            (byte(i + bytes) << bits) | (byte(i + bytes + 1) >> (8 - bits))
        };
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
        // The smallest step: diff1 << 31 over 1, i.e. 0xffff << 239.
        assert_eq!(
            hex::encode(target_for_difficulty(1.0 / (1u64 << 31) as f64)),
            "7fff800000000000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(target_for_difficulty(1e-12), target_for_difficulty(1.0 / (1u64 << 31) as f64));
    }

    /// The pool's formula at the difficulties BC3Hash uses. For 1e-7 the
    /// pool divides by floor(1e-7 * 2^31) = 214, and the target must be the
    /// exact quotient: target * 214 <= diff1 << 31 < (target + 1) * 214.
    /// With the old 2^24 scaling the divisor was 1 and the target 1.7 times
    /// too large.
    #[test]
    fn small_difficulties_match_the_pool() {
        fn times(x: &[u8; 32], m: u64) -> [u8; 33] {
            let mut out = [0u8; 33];
            let mut carry = 0u128;
            for i in (0..32).rev() {
                let v = x[i] as u128 * m as u128 + carry;
                out[i + 1] = v as u8;
                carry = v >> 8;
            }
            out[0] = carry as u8;
            out
        }
        let mut dividend = [0u8; 33];
        dividend[1..].copy_from_slice(&shl(&compact_to_target(0x1d00ffff).unwrap(), 31));
        for (difficulty, divisor) in [(1e-7, 214u64), (1e-6, 2147), (2.5e-5, 53_687), (0.1, 214_748_364)] {
            let target = target_for_difficulty(difficulty);
            assert!(times(&target, divisor) <= dividend, "{difficulty}: target too large");
            let mut next = target;
            for b in next.iter_mut().rev() {
                let (v, overflow) = b.overflowing_add(1);
                *b = v;
                if !overflow {
                    break;
                }
            }
            assert!(times(&next, divisor) > dividend, "{difficulty}: target too small");
        }
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

    /// The edges: regtest heights written as small-integer opcodes, a push
    /// cut off by the end of coinb1, oversized and negative pushes.
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

    /// The node's `GetHash` rule: bit 11 wins whatever else is set, then bit
    /// 12, then SHA256d. Every combination, plus neighbouring bits that must
    /// not matter.
    #[test]
    fn the_version_bits_pick_the_pow() {
        for (version, algo) in [
            (0x0000_0001, PowAlgo::Sha256d),
            (0x2000_0000, PowAlgo::Sha256d),
            (0x2000_1000, PowAlgo::Sha3t),
            (0x2000_0800, PowAlgo::Bc3Hash),
            (0x2000_1800, PowAlgo::Bc3Hash),
            (0x2000_0400, PowAlgo::Sha256d),
            (0x2000_2000, PowAlgo::Sha256d),
            (0xffff_ffff, PowAlgo::Bc3Hash),
            (0xffff_f7ff, PowAlgo::Sha3t),
            (0xffff_e7ff, PowAlgo::Sha256d),
        ] {
            assert_eq!(PowAlgo::for_version(version), algo, "version {version:08x}");
        }
        assert_eq!(PowAlgo::Sha256d.name(), "SHA256d");
        assert_eq!(PowAlgo::Sha3t.name(), "SHA3-256t");
        assert_eq!(PowAlgo::Bc3Hash.name(), "BC3Hash");
        for (i, algo) in PowAlgo::ALL.iter().enumerate() {
            assert_eq!(algo.index(), i, "{}", algo.name());
        }
    }

    /// `hash`, `hash_fn` and the version-bit `BlockHeader::hash` must agree.
    #[test]
    fn pow_algo_dispatch_matches_the_reference() {
        let h = BlockHeader {
            version: 0x2000_1000,
            prev_hash: [3u8; 32],
            merkle_root: [5u8; 32],
            time: 1_777_245_555,
            bits: 0x1d00ffff,
            nonce: 42,
        };
        let ser = h.serialize();
        assert_eq!(PowAlgo::Sha3t.hash(&ser), Some(sha3t(&ser)));
        assert_eq!(PowAlgo::Sha3t.hash(&ser), Some(h.hash()));
        assert_eq!(PowAlgo::Sha256d.hash(&ser), Some(sha256d(&ser)));
        for algo in [PowAlgo::Sha256d, PowAlgo::Sha3t, PowAlgo::Bc3Hash] {
            if let Some(f) = algo.hash_fn() {
                assert_eq!(Some(f(&ser)), algo.hash(&ser), "{}", algo.name());
            }
        }
        // Only the browser build may lack BC3Hash.
        assert_eq!(PowAlgo::Bc3Hash.hash_fn().is_some(), cfg!(feature = "bc3hash"));
    }

    /// The node's header vectors (doc/bc3/vectors/header-vectors.json): one
    /// 80-byte header under all four combinations of bits 11 and 12, with the
    /// digest `GetHash` gives and the block id RPC shows. Run through the
    /// miner's own path - version -> PowAlgo -> hash - not the crate's.
    #[cfg(feature = "bc3hash")]
    #[test]
    fn node_header_vectors_through_the_miner_path() {
        // bc3hash/vectors from the root crate; the wasm crate compiles this
        // file too, one directory down, and CI runs its tests without
        // BC3HASH_VECTORS.
        let dir = std::env::var_os("BC3HASH_VECTORS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
                [here.join("bc3hash/vectors"), here.join("../bc3hash/vectors")]
                    .into_iter()
                    .find(|d| d.is_dir())
                    .unwrap_or_else(|| here.join("bc3hash/vectors"))
            });
        let path = dir.join("header-vectors.json");
        let json = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e} - copy doc/bc3/vectors from the node", path.display()));
        let vectors: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        let mut seen = std::collections::HashSet::new();
        for v in &vectors {
            let header: [u8; 80] = hex::decode(v["header"].as_str().unwrap())
                .unwrap()
                .try_into()
                .unwrap();
            let version = u32::from_le_bytes(header[0..4].try_into().unwrap());
            let algo = PowAlgo::for_version(version);
            seen.insert(version & (BC3HASH_VBIT | SHA3_VBIT));
            let digest = algo.hash(&header).unwrap();
            assert_eq!(hex::encode(digest), v["digest"], "version {version:08x}, {}", algo.name());
            assert_eq!(hash_to_display(&digest), v["block_id"], "version {version:08x}");
        }
        assert_eq!(seen.len(), 4, "the vectors must cover all four bit combinations");
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
