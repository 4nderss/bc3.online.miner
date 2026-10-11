//! What the CUDA and OpenCL backends share for BC3Hash: the kernel source,
//! the host side of its tables, the target in the kernel's limb order, and
//! the self-test every card runs before it mines.

use crate::consensus::Target;
use std::sync::Arc;

/// The kernel source, for the OpenCL backend, which builds it at run time
/// (`-DBC3_WINDOW=N`); CUDA embeds the PTX that build.rs made from it.
#[cfg(feature = "opencl")]
pub const SOURCE: &str = include_str!("../kernels/bc3hash.cl");

/// The host side of the GPU tables, built once per window for every card in
/// the process. Only the latest window is kept: at 16 bits that is 256 MiB.
pub fn host_tables(window: u32) -> Result<Arc<Vec<u64>>, String> {
    static CACHE: std::sync::Mutex<Option<(u32, Arc<Vec<u64>>)>> = std::sync::Mutex::new(None);
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((w, tables)) = cache.as_ref() {
        if *w == window {
            return Ok(tables.clone());
        }
    }
    *cache = None; // the old window's tables go first, not after
    let tables = Arc::new(bc3hash::gpu_tables(window)?);
    *cache = Some((window, tables.clone()));
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

/// The two ways into a loaded BC3Hash kernel, as the self-test needs them.
pub trait Bc3Ops {
    /// BC3Hash of whole 80-byte headers, through the batch entry point.
    fn bc3_batch(&mut self, headers: &[[u8; 80]]) -> Result<Vec<[u8; 32]>, String>;
    /// The nonces in [start, start+count) whose hash meets `target`, through
    /// the scan entry point - the mining path.
    fn bc3_scan(
        &mut self,
        header76: &[u8; 76],
        start: u32,
        count: u32,
        target: &Target,
    ) -> Result<Vec<u32>, String>;
}

/// Check a loaded BC3Hash kernel against the node: the node's header vectors
/// and random headers against the CPU reference through the batch entry
/// point, then a range through the scan entry point - the exact mining path,
/// nonce placement and target compare included - against the CPU's own hit
/// set. Every architecture and driver compiles the kernel for itself, so this
/// runs on every card, for every window.
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

    // Target 2^253 - 1: about one hash in eight is a hit, so 256 nonces give
    // the scan's compare and hit list something to do both ways.
    let mut header76 = [0u8; 76];
    for b in header76.iter_mut() {
        *b = next() as u8;
    }
    header76[0..4].copy_from_slice(&0x2000_1800u32.to_le_bytes());
    let mut target: Target = [0xff; 32];
    target[0] = 0x1f;
    let start = next() as u32;
    let mut gpu = k.bc3_scan(&header76, start, 256, &target)?;
    let mut header80 = [0u8; 80];
    header80[..76].copy_from_slice(&header76);
    let mut cpu: Vec<u32> = (0..256u32)
        .map(|i| start.wrapping_add(i))
        .filter(|&nonce| {
            header80[76..80].copy_from_slice(&nonce.to_le_bytes());
            crate::consensus::hash_meets_target(&bc3hash::bc3hash(&header80), &target)
        })
        .collect();
    gpu.sort_unstable();
    cpu.sort_unstable();
    if gpu != cpu {
        return Err(format!(
            "scan over 256 nonces: GPU found {} hits, CPU {} (first GPU {:?}, first CPU {:?})",
            gpu.len(),
            cpu.len(),
            gpu.first(),
            cpu.first()
        ));
    }
    Ok(())
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
}
