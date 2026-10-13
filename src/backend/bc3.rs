//! What the CUDA and OpenCL backends share for BC3Hash: the kernel source,
//! the host side of OpenCL's tables, the target in the kernel's limb order,
//! and the self-test every card runs before it mines.

use super::{Launch, MiningBackend, QUEUE_DEPTH};
use crate::consensus::{PowAlgo, Target};
#[cfg(feature = "opencl")]
use std::sync::{Arc, Mutex, Weak};

/// The kernel source, for the OpenCL backend, which builds it at run time
/// (`-DBC3_WINDOW=N`); CUDA embeds the PTX that build.rs made from it.
#[cfg(feature = "opencl")]
pub const SOURCE: &str = include_str!("../kernels/bc3hash.cl");

/// The host side of an OpenCL card's tables for `window`, shared by every
/// card that loads them at the same time - at 16 bits they are 256 MiB. A
/// CUDA card builds its own, on the card (cuda.rs).
///
/// Held only as long as a card is uploading them. Kept for the life of the
/// process, they were 256 MiB of RAM nobody read again after the start; a
/// card that loads them later (a reopen, another window) builds them again,
/// about 1.3 s at window 16 on 20 threads.
#[cfg(feature = "opencl")]
pub fn host_tables(window: u32) -> Result<Arc<Vec<u64>>, String> {
    static CACHE: Mutex<Option<(u32, Weak<Vec<u64>>)>> = Mutex::new(None);
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((w, tables)) = cache.as_ref() {
        if *w == window {
            if let Some(tables) = tables.upgrade() {
                return Ok(tables);
            }
        }
    }
    *cache = None;
    let tables = Arc::new(bc3hash::gpu_tables(window)?);
    *cache = Some((window, Arc::downgrade(&tables)));
    Ok(tables)
}

/// The share target as the kernel takes it: four u64 limbs, least
/// significant first. Our `Target` is 32 big-endian bytes.
pub fn target_limbs(target: &Target) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let at = 32 - 8 * (i + 1);
        *limb = u64::from_be_bytes(target[at..at + 8].try_into().unwrap());
    }
    limbs
}

/// A backend with a loaded BC3Hash kernel, as the self-test needs it: the
/// batch entry point next to the mining path.
pub trait Bc3Ops: MiningBackend {
    /// BC3Hash of whole 80-byte headers, through the batch entry point.
    fn bc3_batch(&mut self, headers: &[[u8; 80]]) -> Result<Vec<[u8; 32]>, String>;
}

/// Check a loaded BC3Hash kernel against the node: the node's header vectors
/// and random headers against the CPU reference through the batch entry
/// point, then three launches through the launch queue - the exact mining
/// path, every slot, the nonce placement up to the top of the nonce space and
/// the target compare - each against the CPU's own hit set. Every
/// architecture and driver compiles the kernel for itself, so this runs on
/// every card, for every window. Leaves nothing in flight.
pub fn self_test(k: &mut impl Bc3Ops) -> Result<(), String> {
    let mut seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x0bc3)
        | 1;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut headers = Vec::new();
    let mut expected = Vec::new();
    for (header, digest) in bc3hash::header_vectors() {
        headers.push(header);
        expected.push(digest);
    }
    for _ in 0..14 {
        let mut header = [0u8; 80];
        for b in header.iter_mut() {
            *b = next() as u8;
        }
        header[0..4].copy_from_slice(&0x2000_1800u32.to_le_bytes());
        expected.push(bc3hash::bc3hash(&header));
        headers.push(header);
    }
    let got = k.bc3_batch(&headers)?;
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        if g != e {
            return Err(format!(
                "header {i} ({}): GPU {} but CPU {}",
                if i < 2 { "the node's vector" } else { "random" },
                hex::encode(g),
                hex::encode(e)
            ));
        }
    }

    // Target 2^253 - 1: about one hash in eight is a hit, so 128 nonces a
    // launch give the compare and the hit list something to do both ways.
    // The last launch ends at nonce 2^32 - 1.
    let mut target: Target = [0xff; 32];
    target[0] = 0x1f;
    const COUNT: u32 = 128;
    let launches: Vec<([u8; 76], u32)> = (0..QUEUE_DEPTH + 1)
        .map(|i| {
            let mut header76 = [0u8; 76];
            for b in header76.iter_mut() {
                *b = next() as u8;
            }
            header76[0..4].copy_from_slice(&0x2000_1800u32.to_le_bytes());
            // Never across 2^32: the worker does not launch that way either.
            let start = if i == QUEUE_DEPTH { 0u32.wrapping_sub(COUNT) } else { next() as u32 % (0u32.wrapping_sub(COUNT)) };
            (header76, start)
        })
        .collect();
    let scans = queued_scans(k, &launches, COUNT, &target).inspect_err(|_| k.abandon())?;
    for (i, ((header76, start), gpu)) in launches.iter().zip(scans).enumerate() {
        let mut header80 = [0u8; 80];
        header80[..76].copy_from_slice(header76);
        let cpu: Vec<u32> = (0..COUNT)
            .map(|n| start.wrapping_add(n))
            .filter(|&nonce| {
                header80[76..80].copy_from_slice(&nonce.to_le_bytes());
                crate::consensus::hash_meets_target(&bc3hash::bc3hash(&header80), &target)
            })
            .collect();
        if gpu != cpu {
            return Err(format!(
                "launch {i} over {COUNT} nonces from {start:08x}: GPU found {} hits, CPU {} \
                 (first GPU {:?}, first CPU {:?})",
                gpu.len(),
                cpu.len(),
                gpu.first(),
                cpu.first()
            ));
        }
    }
    Ok(())
}

/// The hits of each launch, sorted, the launches queued as deep as the
/// worker queues them.
fn queued_scans(
    k: &mut impl Bc3Ops,
    launches: &[([u8; 76], u32)],
    count: u32,
    target: &Target,
) -> Result<Vec<Vec<u32>>, String> {
    let mut scans = Vec::new();
    let mut collect = |k: &mut dyn MiningBackend| -> Result<(), String> {
        let mut hits = k.collect()?.hits;
        hits.sort_unstable();
        scans.push(hits);
        Ok(())
    };
    for (header76, start) in launches {
        k.enqueue(&Launch { algo: PowAlgo::Bc3Hash, header76, start: *start, count, target })?;
        if k.in_flight() == QUEUE_DEPTH {
            collect(k)?;
        }
    }
    while k.in_flight() > 0 {
        collect(k)?;
    }
    Ok(scans)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Limb 0 is the least significant 64 bits of the big-endian target.
    #[test]
    fn target_limbs_are_least_significant_first() {
        let mut t: Target = [0; 32];
        t[0] = 0x12; // most significant byte
        t[31] = 0x34; // least significant byte
        let limbs = target_limbs(&t);
        assert_eq!(limbs[0], 0x34);
        assert_eq!(limbs[3], 0x12 << 56);
        assert_eq!((limbs[1], limbs[2]), (0, 0));
    }

    /// The layout a CUDA card builds its tables in (`tuning::rows`, from
    /// `bc3hash::gpu_point`) is the host's: the size, and the first and last
    /// point of every row, the short last row of a window that does not
    /// divide 256 included.
    #[test]
    fn cards_build_their_tables_in_the_hosts_layout() {
        for window in [5u32, 9] {
            let t = bc3hash::gpu_tables(window).unwrap();
            assert_eq!(t.len() as u64 * 8, crate::tuning::table_bytes(window), "window {window}");
            let rows = crate::tuning::rows(window).unwrap();
            let curve_u64: usize = rows.iter().map(|r| r.places as usize * 8).sum();
            for curve in 0..4 {
                let mut at = crate::tuning::POWER_TABLES_U64 as usize + curve * curve_u64;
                for r in &rows {
                    for b in [1, r.points] {
                        let e = at + (b as usize - 1) * 8;
                        let point = bc3hash::gpu_point(curve, r.pos, b).unwrap();
                        assert_eq!(point, t[e..e + 8], "window {window}, curve {curve}, bit {}, point {b}", r.pos);
                    }
                    at += r.places as usize * 8;
                }
            }
        }
    }

    /// The host tables are shared while a card holds them, and freed after.
    #[test]
    #[cfg(feature = "opencl")]
    fn host_tables_live_as_long_as_a_card_holds_them() {
        let a = host_tables(2).unwrap();
        let b = host_tables(2).unwrap();
        assert!(Arc::ptr_eq(&a, &b), "two cards loading at once share one copy");
        let weak = Arc::downgrade(&a);
        drop((a, b));
        assert!(weak.upgrade().is_none(), "nobody holds them: they must be gone");
        // And they come back, the same, when a card asks again.
        assert_eq!(host_tables(2).unwrap().len(), bc3hash::gpu_tables(2).unwrap().len());
    }
}
