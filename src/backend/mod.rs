//! Mining-backends: abstraktion över CUDA/OpenCL-GPU:er (och CPU-fallback).
//!
//! En backend grindar ett nonce-intervall för en fast 76-bytes headerprefix
//! och returnerar de noncer vars SHA3-256t-hash ≤ share-target. Alla träffar
//! CPU-verifieras i gpu_worker innan submit — kernelbuggar kan aldrig ge
//! felaktiga shares, bara missade.

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "opencl")]
pub mod cl_sys;
#[cfg(feature = "opencl")]
pub mod opencl;

use crate::consensus::Target;

/// Delad kernelkälla (CUDA via NVRTC + OpenCL). Se filen för schema.
#[allow(dead_code)]
pub const KERNEL_SOURCE: &str = include_str!("../kernels/sha3t.cl");

/// Max antal träffar per launch som kerneln rapporterar (fler än så i en
/// batch händer i praktiken aldrig med rimliga share-difficulties).
#[allow(dead_code)]
pub const MAX_HITS: usize = 64;

pub trait MiningBackend {
    /// Människoläsbart namn ("CUDA: NVIDIA GeForce RTX 3050 Ti ...").
    fn name(&self) -> String;

    /// Grinda [start_nonce, start_nonce+count) för headern (nonce-fältet
    /// exkluderat) och returnera noncer med hash ≤ target.
    fn scan_range(
        &mut self,
        header76: &[u8; 76],
        start_nonce: u32,
        count: u32,
        target: &Target,
    ) -> Result<Vec<u32>, String>;
}

/// En upptäckt GPU — bara data (Send), själva backenden öppnas i arbetstråden.
#[derive(Clone, Debug)]
pub enum GpuDevice {
    #[cfg(feature = "cuda")]
    Cuda { index: usize, name: String },
    #[cfg(feature = "opencl")]
    Opencl { platform: usize, device: usize, name: String },
}

impl GpuDevice {
    pub fn describe(&self) -> String {
        match self {
            #[cfg(feature = "cuda")]
            GpuDevice::Cuda { index, name } => format!("CUDA #{index}: {name}"),
            #[cfg(feature = "opencl")]
            GpuDevice::Opencl { platform, device, name } => {
                format!("OpenCL {platform}.{device}: {name}")
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

/// Vilka backends användaren bett om.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum BackendKind {
    /// CUDA om möjligt, annars OpenCL, annars CPU.
    Auto,
    Cuda,
    Opencl,
    Cpu,
}

/// Lista GPU:er enligt önskad backend. `gpu_id` filtrerar till en enhet
/// (CUDA-index resp. löpnummer i OpenCL-listan).
pub fn detect_gpus(kind: BackendKind, gpu_id: Option<usize>) -> Vec<GpuDevice> {
    let mut found: Vec<GpuDevice> = Vec::new();

    #[cfg(feature = "cuda")]
    if matches!(kind, BackendKind::Auto | BackendKind::Cuda) {
        found.extend(cuda::list_devices());
    }
    #[cfg(feature = "opencl")]
    if matches!(kind, BackendKind::Opencl)
        || (matches!(kind, BackendKind::Auto) && found.is_empty())
    {
        found.extend(opencl::list_devices());
    }
    let _ = kind; // (om inga GPU-features är aktiva)

    if let Some(id) = gpu_id {
        found = found.into_iter().skip(id).take(1).collect();
    }
    found
}

/// Öppna en backend för en upptäckt enhet (körs i GPU-arbetstråden).
pub fn open_backend(dev: &GpuDevice) -> Result<Box<dyn MiningBackend>, String> {
    match dev {
        #[cfg(feature = "cuda")]
        GpuDevice::Cuda { index, .. } => Ok(Box::new(cuda::CudaBackend::new(*index)?)),
        #[cfg(feature = "opencl")]
        GpuDevice::Opencl { platform, device, .. } => {
            Ok(Box::new(opencl::OpenClBackend::new(*platform, *device)?))
        }
        #[allow(unreachable_patterns)]
        _ => unreachable!(),
    }
}

/// Packa 80-bytesheadern (med nonce = 0) som tio LE-u64-lanes åt kerneln.
#[allow(dead_code)]
pub fn header_lanes(header76: &[u8; 76]) -> [u64; 10] {
    let mut buf = [0u8; 80];
    buf[..76].copy_from_slice(header76);
    let mut lanes = [0u64; 10];
    for (i, lane) in lanes.iter_mut().enumerate() {
        *lane = u64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
    }
    lanes
}

/// Target ([u8;32] big-endian) → fyra u64-limbar [t0..t3], t3 mest signifikant.
/// Matchar kernelns tolkning: hash-limb k = LE-u64 ur hashbytes 8k..8k+7.
#[allow(dead_code)]
pub fn target_limbs(target: &Target) -> [u64; 4] {
    let mut t = [0u64; 4];
    for (k, limb) in t.iter_mut().enumerate() {
        let off = (3 - k) * 8;
        *limb = u64::from_be_bytes(target[off..off + 8].try_into().unwrap());
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::{
        compact_to_target, hash_meets_target, sha3t, target_for_difficulty, BlockHeader, SHA3_VBIT,
    };

    // ------------------------------------------------------------------
    // Rust-spegel av kernelalgoritmen (samma lane-schema, padding och
    // jämförelse som src/kernels/sha3t.cl). Verifierar kernelns matematik
    // mot CPU-referensen utan GPU — bitexaktheten på riktig GPU låses sedan
    // av de #[ignore]-markerade testen i cuda-backenden.
    // ------------------------------------------------------------------

    const RC: [u64; 24] = [
        0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
        0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
        0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
    ];
    const ROTC: [u32; 24] = [
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
    ];
    const PILN: [usize; 24] = [
        10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
    ];

    fn keccakf(st: &mut [u64; 25]) {
        for round in 0..24 {
            let mut bc = [0u64; 5];
            for i in 0..5 {
                bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
            }
            for i in 0..5 {
                let t = bc[(i + 4) % 5] ^ bc[(i + 1) % 5].rotate_left(1);
                for j in (0..25).step_by(5) {
                    st[j + i] ^= t;
                }
            }
            let mut t = st[1];
            for i in 0..24 {
                let j = PILN[i];
                let tmp = st[j];
                st[j] = t.rotate_left(ROTC[i]);
                t = tmp;
            }
            for j in (0..25).step_by(5) {
                let mut bc = [0u64; 5];
                bc.copy_from_slice(&st[j..j + 5]);
                for i in 0..5 {
                    st[j + i] ^= (!bc[(i + 1) % 5]) & bc[(i + 2) % 5];
                }
            }
            st[0] ^= RC[round];
        }
    }

    fn sha3_256_32(input: [u64; 4]) -> [u64; 4] {
        let mut st = [0u64; 25];
        st[..4].copy_from_slice(&input);
        st[4] = 0x06;
        st[16] = 0x8000_0000_0000_0000;
        keccakf(&mut st);
        [st[0], st[1], st[2], st[3]]
    }

    /// Exakt vad kerneln gör per nonce, i Rust.
    fn kernel_mirror_hash(header76: &[u8; 76], nonce: u32) -> [u64; 4] {
        let lanes = header_lanes(header76);
        let mut st = [0u64; 25];
        st[..10].copy_from_slice(&lanes);
        st[9] |= (nonce as u64) << 32;
        st[10] = 0x06;
        st[16] = 0x8000_0000_0000_0000;
        keccakf(&mut st);
        let h = [st[0], st[1], st[2], st[3]];
        sha3_256_32(sha3_256_32(h))
    }

    fn limbs_of_hash(hash: &[u8; 32]) -> [u64; 4] {
        let mut l = [0u64; 4];
        for (k, limb) in l.iter_mut().enumerate() {
            *limb = u64::from_le_bytes(hash[k * 8..k * 8 + 8].try_into().unwrap());
        }
        l
    }

    fn genesis_header() -> BlockHeader {
        let mut merkle: [u8; 32] =
            hex::decode("8e1df52fddd25c460304ff8ea7bcb570850bf0b0c027eecf8ebf8ab17d3e93b1")
                .unwrap()
                .try_into()
                .unwrap();
        merkle.reverse();
        BlockHeader {
            version: 1,
            prev_hash: [0u8; 32],
            merkle_root: merkle,
            time: 1_777_245_555,
            bits: 0x1d00ffff,
            nonce: 2_442_659_435,
        }
    }

    /// Enkel deterministisk PRNG för testheadrar (ingen rand-dependency).
    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn kernel_mirror_matches_cpu_reference_genesis() {
        let g = genesis_header();
        let ser = g.serialize();
        let header76: [u8; 76] = ser[..76].try_into().unwrap();
        let expected = limbs_of_hash(&sha3t(&ser));
        assert_eq!(kernel_mirror_hash(&header76, g.nonce), expected);
    }

    #[test]
    fn kernel_mirror_matches_cpu_reference_random() {
        let mut seed = 0xbc3_0001u64;
        for _ in 0..50 {
            let mut header80 = [0u8; 80];
            for b in header80.iter_mut() {
                *b = xorshift(&mut seed) as u8;
            }
            let header76: [u8; 76] = header80[..76].try_into().unwrap();
            let nonce = u32::from_le_bytes(header80[76..80].try_into().unwrap());
            let expected = limbs_of_hash(&sha3t(&header80));
            assert_eq!(kernel_mirror_hash(&header76, nonce), expected, "header {header80:02x?}");
        }
    }

    #[test]
    fn target_limbs_comparison_matches_hash_meets_target() {
        // Limb-jämförelsen (som kerneln gör) ska ge samma svar som
        // consensus::hash_meets_target för slumpade hashar/targets.
        let mut seed = 0xbc3_0002u64;
        let targets = [
            compact_to_target(0x1d00ffff).unwrap(),
            target_for_difficulty(16.0),
            target_for_difficulty(0.001), // högt target — många träffar
        ];
        for target in targets {
            let t = target_limbs(&target);
            for _ in 0..2000 {
                let mut hash = [0u8; 32];
                for b in hash.iter_mut() {
                    *b = xorshift(&mut seed) as u8;
                }
                // Gör vissa hashar små så båda grenarna övas.
                if seed % 3 == 0 {
                    for b in hash[4..].iter_mut() {
                        *b = 0;
                    }
                }
                let h = limbs_of_hash(&hash);
                let kernel_ok = if h[3] != t[3] {
                    h[3] < t[3]
                } else if h[2] != t[2] {
                    h[2] < t[2]
                } else if h[1] != t[1] {
                    h[1] < t[1]
                } else if h[0] != t[0] {
                    h[0] < t[0]
                } else {
                    true
                };
                assert_eq!(kernel_ok, hash_meets_target(&hash, &target));
            }
        }
    }

    #[test]
    fn header_lanes_roundtrip() {
        let mut header76 = [0u8; 76];
        for (i, b) in header76.iter_mut().enumerate() {
            *b = i as u8;
        }
        let lanes = header_lanes(&header76);
        // lane 9 = bytes 72..75 + nollad nonce.
        assert_eq!(lanes[9], u64::from_le_bytes([72, 73, 74, 75, 0, 0, 0, 0]));
        assert_eq!(lanes[0], u64::from_le_bytes([0, 1, 2, 3, 4, 5, 6, 7]));
    }

    #[test]
    fn sha3_vbit_headers_use_sha3t() {
        // Sanity: jobb från poolen har alltid versionsbiten satt, så
        // GPU-vägen (som alltid kör sha3t) matchar BlockHeader::hash.
        let mut h = genesis_header();
        h.version |= SHA3_VBIT;
        let ser = h.serialize();
        assert_eq!(h.hash(), sha3t(&ser));
    }
}
