// SHA3-256t kernel (triple NIST SHA3-256) - SHARED source for CUDA and OpenCL.
//
// The same file is compiled by NVRTC (as CUDA C++) and by the OpenCL runtime
// (as OpenCL C). All differences are encapsulated in the macros below - the
// keccak core itself is identical, so bit-exactness verified for one backend
// holds for both.
//
// Hash scheme (see src/consensus.rs for the CPU reference):
//   - The header is 80 bytes and the SHA3-256 rate is 136 bytes -> ONE
//     absorb block.
//   - Padding (NIST SHA3, not raw keccak): 0x06 at byte 80, 0x80 at byte 135.
//   - Rounds 2 and 3 hash 32 bytes: 0x06 at byte 32, 0x80 at byte 135.
//   - Exactly 3 keccak-f[1600] permutations per nonce in total.
//
// Values are packed as u64 lanes in little-endian (lane i = bytes 8i..8i+7),
// so the whole header fits in lanes 0..9 and the nonce is the high half of
// lane 9.

#if defined(__OPENCL_VERSION__)
  typedef ulong u64;
  typedef uint  u32;
  #define KERNEL_FN   __kernel
  #define GLOBAL      __global
  #define CONST_ARR   __constant
  #define DEVICE_FN   static
  #define GLOBAL_ID() ((u32)get_global_id(0))
  #define ATOMIC_INC_U32(p) atomic_inc(p)
  #define ROTL64(x, n) rotate((u64)(x), (u64)(n))
  // No unrolling here: the gain is measured on NVIDIA, and on AMD full
  // unrolling can instead drive up the register pressure. Measure before
  // it is turned on for OpenCL.
  #define UNROLL_ROUNDS
#else /* CUDA (NVRTC) */
  typedef unsigned long long u64;
  typedef unsigned int u32;
  #define KERNEL_FN   extern "C" __global__
  #define GLOBAL
  #define CONST_ARR   __constant__
  #define DEVICE_FN   static __device__ __forceinline__
  #define GLOBAL_ID() ((u32)(blockIdx.x * blockDim.x + threadIdx.x))
  #define ATOMIC_INC_U32(p) atomicAdd(p, 1u)
  #define ROTL64(x, n) (((u64)(x) << (n)) | ((u64)(x) >> (64 - (n))))
  // Full unrolling of the 24 rounds. Two effects, both measured: the five
  // overhead instructions per round (round constant, counter, comparison,
  // branch) disappear, and ptxas comes down from 80 to 64 registers, which
  // raises occupancy from 25 to 32 warps per SM. Partial unrolling (2, 4, 8)
  // is WORSE than none at all - it keeps both the loop and the 80 registers.
  #define UNROLL_ROUNDS _Pragma("unroll")
#endif

// Three-way XOR. Ampere/Ada have LOP3, which computes an ARBITRARY function
// of three inputs in ONE instruction (immLut 0x96 = a^b^c). ptxas finds some
// of these patterns by itself, but not all - writing them explicitly takes
// theta from 50 down to 35 logic operations per round. On AMD (RDNA2+) the
// compiler pattern-matches `a^b^c` to v_xor3_b32, so the OpenCL branch needs
// no asm.
#if defined(__OPENCL_VERSION__)
  #define XOR3(a, b, c) ((a) ^ (b) ^ (c))
#else
DEVICE_FN u64 xor3_lop3(u64 a, u64 b, u64 c) {
  u64 r;
  // One line: PTX statements end with a semicolon, so no line breaks are
  // needed - and then there are no escape sequences that can break.
  asm("{ .reg .b32 al, ah, bl, bh, cl, ch, rl, rh; mov.b64 {al,ah}, %1; mov.b64 {bl,bh}, %2; mov.b64 {cl,ch}, %3; lop3.b32 rl, al, bl, cl, 0x96; lop3.b32 rh, ah, bh, ch, 0x96; mov.b64 %0, {rl,rh}; }" : "=l"(r) : "l"(a), "l"(b), "l"(c));
  return r;
}
  #define XOR3(a, b, c) xor3_lop3((a), (b), (c))
#endif

CONST_ARR u64 KECCAK_RC[24] = {
  0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
  0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
  0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
  0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
  0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
  0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
  0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
  0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

// IMPORTANT for performance: the round function is FULLY unrolled with
// constant indices - dynamic indexing of st[] makes the state spill to
// local memory and the kernel becomes ~100x slower. (Verified empirically:
// the compact table-driven variant gave 4 MH/s instead of the GH/s class.)
DEVICE_FN void keccakf(u64 st[25]) {
  u64 bc0, bc1, bc2, bc3, bc4, t, tmp;
  UNROLL_ROUNDS
  for (int round = 0; round < 24; round++) {
    // Theta. The column sums as XOR3 pairs; the application folds in both
    // C[x-1] and ROTL(C[x+1],1) in the same instruction, so D[x] is never
    // materialized. `t` holds only the rotation - one live value at a
    // time, just as before, so the register pressure is unchanged.
    bc0 = XOR3(st[0], st[5], st[10]);  bc0 = XOR3(bc0, st[15], st[20]);
    bc1 = XOR3(st[1], st[6], st[11]);  bc1 = XOR3(bc1, st[16], st[21]);
    bc2 = XOR3(st[2], st[7], st[12]);  bc2 = XOR3(bc2, st[17], st[22]);
    bc3 = XOR3(st[3], st[8], st[13]);  bc3 = XOR3(bc3, st[18], st[23]);
    bc4 = XOR3(st[4], st[9], st[14]);  bc4 = XOR3(bc4, st[19], st[24]);
    t = ROTL64(bc1, 1);
    st[0] = XOR3(st[0], bc4, t); st[5] = XOR3(st[5], bc4, t); st[10] = XOR3(st[10], bc4, t);
    st[15] = XOR3(st[15], bc4, t); st[20] = XOR3(st[20], bc4, t);
    t = ROTL64(bc2, 1);
    st[1] = XOR3(st[1], bc0, t); st[6] = XOR3(st[6], bc0, t); st[11] = XOR3(st[11], bc0, t);
    st[16] = XOR3(st[16], bc0, t); st[21] = XOR3(st[21], bc0, t);
    t = ROTL64(bc3, 1);
    st[2] = XOR3(st[2], bc1, t); st[7] = XOR3(st[7], bc1, t); st[12] = XOR3(st[12], bc1, t);
    st[17] = XOR3(st[17], bc1, t); st[22] = XOR3(st[22], bc1, t);
    t = ROTL64(bc4, 1);
    st[3] = XOR3(st[3], bc2, t); st[8] = XOR3(st[8], bc2, t); st[13] = XOR3(st[13], bc2, t);
    st[18] = XOR3(st[18], bc2, t); st[23] = XOR3(st[23], bc2, t);
    t = ROTL64(bc0, 1);
    st[4] = XOR3(st[4], bc3, t); st[9] = XOR3(st[9], bc3, t); st[14] = XOR3(st[14], bc3, t);
    st[19] = XOR3(st[19], bc3, t); st[24] = XOR3(st[24], bc3, t);

    // Rho + Pi (Saarinen's ordering, unrolled: (lane, rot) per step)
    t = st[1];
    tmp = st[10]; st[10] = ROTL64(t, 1);  t = tmp;
    tmp = st[7];  st[7]  = ROTL64(t, 3);  t = tmp;
    tmp = st[11]; st[11] = ROTL64(t, 6);  t = tmp;
    tmp = st[17]; st[17] = ROTL64(t, 10); t = tmp;
    tmp = st[18]; st[18] = ROTL64(t, 15); t = tmp;
    tmp = st[3];  st[3]  = ROTL64(t, 21); t = tmp;
    tmp = st[5];  st[5]  = ROTL64(t, 28); t = tmp;
    tmp = st[16]; st[16] = ROTL64(t, 36); t = tmp;
    tmp = st[8];  st[8]  = ROTL64(t, 45); t = tmp;
    tmp = st[21]; st[21] = ROTL64(t, 55); t = tmp;
    tmp = st[24]; st[24] = ROTL64(t, 2);  t = tmp;
    tmp = st[4];  st[4]  = ROTL64(t, 14); t = tmp;
    tmp = st[15]; st[15] = ROTL64(t, 27); t = tmp;
    tmp = st[23]; st[23] = ROTL64(t, 41); t = tmp;
    tmp = st[19]; st[19] = ROTL64(t, 56); t = tmp;
    tmp = st[13]; st[13] = ROTL64(t, 8);  t = tmp;
    tmp = st[12]; st[12] = ROTL64(t, 25); t = tmp;
    tmp = st[2];  st[2]  = ROTL64(t, 43); t = tmp;
    tmp = st[20]; st[20] = ROTL64(t, 62); t = tmp;
    tmp = st[14]; st[14] = ROTL64(t, 18); t = tmp;
    tmp = st[22]; st[22] = ROTL64(t, 39); t = tmp;
    tmp = st[9];  st[9]  = ROTL64(t, 61); t = tmp;
    tmp = st[6];  st[6]  = ROTL64(t, 20); t = tmp;
    st[1] = ROTL64(t, 44);

    // Chi, row by row
    bc0 = st[0]; bc1 = st[1]; bc2 = st[2]; bc3 = st[3]; bc4 = st[4];
    st[0] ^= (~bc1) & bc2; st[1] ^= (~bc2) & bc3; st[2] ^= (~bc3) & bc4;
    st[3] ^= (~bc4) & bc0; st[4] ^= (~bc0) & bc1;
    bc0 = st[5]; bc1 = st[6]; bc2 = st[7]; bc3 = st[8]; bc4 = st[9];
    st[5] ^= (~bc1) & bc2; st[6] ^= (~bc2) & bc3; st[7] ^= (~bc3) & bc4;
    st[8] ^= (~bc4) & bc0; st[9] ^= (~bc0) & bc1;
    bc0 = st[10]; bc1 = st[11]; bc2 = st[12]; bc3 = st[13]; bc4 = st[14];
    st[10] ^= (~bc1) & bc2; st[11] ^= (~bc2) & bc3; st[12] ^= (~bc3) & bc4;
    st[13] ^= (~bc4) & bc0; st[14] ^= (~bc0) & bc1;
    bc0 = st[15]; bc1 = st[16]; bc2 = st[17]; bc3 = st[18]; bc4 = st[19];
    st[15] ^= (~bc1) & bc2; st[16] ^= (~bc2) & bc3; st[17] ^= (~bc3) & bc4;
    st[18] ^= (~bc4) & bc0; st[19] ^= (~bc0) & bc1;
    bc0 = st[20]; bc1 = st[21]; bc2 = st[22]; bc3 = st[23]; bc4 = st[24];
    st[20] ^= (~bc1) & bc2; st[21] ^= (~bc2) & bc3; st[22] ^= (~bc3) & bc4;
    st[23] ^= (~bc4) & bc0; st[24] ^= (~bc0) & bc1;

    // Iota
    st[0] ^= KECCAK_RC[round];
  }
}

// SHA3-256 of 32 bytes (lanes 0..3) -> 32 bytes. Padding: 0x06 at byte 32
// (lane 4, byte 0) and 0x80 at byte 135 (lane 16, byte 7).
DEVICE_FN void sha3_256_32(u64 out[4], const u64 in[4]) {
  u64 st[25];
  for (int i = 0; i < 25; i++)
    st[i] = 0;
  st[0] = in[0];
  st[1] = in[1];
  st[2] = in[2];
  st[3] = in[3];
  st[4] = 0x06UL;
  st[16] = 0x8000000000000000UL;
  keccakf(st);
  out[0] = st[0];
  out[1] = st[1];
  out[2] = st[2];
  out[3] = st[3];
}

// Grind the nonce space [start_nonce, start_nonce + nonce_count).
//
//   hdr_lanes: 10 u64 - the 80-byte header (nonce field = 0) as LE lanes.
//   t0..t3:    share target as four u64 limbs; t3 most significant.
//              (The hash is read as a little-endian 256-bit number; limb k =
//               u64 from hash bytes 8k..8k+7, cf.
//               consensus::hash_meets_target.)
//   hits:      hits[0] = atomic hit counter, hits[1..1+max_hits] = nonces.
KERNEL_FN void sha3t_scan(GLOBAL const u64 *hdr_lanes,
                          u32 start_nonce,
                          u32 nonce_count,
                          u64 t0, u64 t1, u64 t2, u64 t3,
                          GLOBAL u32 *hits,
                          u32 max_hits) {
  u32 gid = GLOBAL_ID();
  if (gid >= nonce_count)
    return;
  u32 nonce = start_nonce + gid;

  // Round 1: the 80-byte header. The nonce is bytes 76..79 = the high half
  // of lane 9.
  u64 st[25];
  for (int i = 0; i < 25; i++)
    st[i] = 0;
  for (int i = 0; i < 10; i++)
    st[i] = hdr_lanes[i];
  st[9] |= (u64)nonce << 32;
  st[10] = 0x06UL;                  // padding byte 80
  st[16] = 0x8000000000000000UL;    // padding byte 135
  keccakf(st);

  // Rounds 2 and 3: the 32-byte intermediate hash.
  u64 h[4], h2[4];
  h[0] = st[0]; h[1] = st[1]; h[2] = st[2]; h[3] = st[3];
  sha3_256_32(h2, h);
  sha3_256_32(h, h2);

  // hash <= target (little-endian 256-bit number, limb 3 most significant).
  bool ok;
  if (h[3] != t3)      ok = h[3] < t3;
  else if (h[2] != t2) ok = h[2] < t2;
  else if (h[1] != t1) ok = h[1] < t1;
  else if (h[0] != t0) ok = h[0] < t0;
  else                 ok = true;

  if (ok) {
    u32 idx = ATOMIC_INC_U32(&hits[0]);
    if (idx < max_hits)
      hits[idx + 1] = nonce;
  }
}
