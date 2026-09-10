// SHA3-256t kernel (triple NIST SHA3-256) - SHARED source for CUDA and OpenCL.
//
// The same file is compiled by nvcc (as CUDA C++, to PTX at build time) and
// by the OpenCL runtime (as OpenCL C). All differences are encapsulated in
// the macros below - the keccak core itself is identical, so bit-exactness
// verified for one backend holds for both.
//
// Hash scheme (see src/consensus.rs for the CPU reference):
//   - The header is 80 bytes and the SHA3-256 rate is 136 bytes -> ONE
//     absorb block.
//   - Padding (NIST SHA3, not raw keccak): 0x06 at byte 80, 0x80 at byte 135.
//   - Rounds 2 and 3 hash 32 bytes: 0x06 at byte 32, 0x80 at byte 135.
//   - Exactly 3 keccak-f[1600] permutations per nonce in total.
//
// LANE REPRESENTATION - BIT-INTERLEAVED. Keccak works on 64-bit lanes and a
// GPU has 32-bit registers. The obvious split (low word, high word) makes
// every 64-bit rotate two funnel shifts. Instead each lane L is kept as two
// words e and o, where bit i of e is bit 2i of L and bit i of o is bit 2i+1:
//
//   XOR, AND, NOT      word by word, unchanged
//   rotl64(L, 2k)      e' = rotl32(e, k),   o' = rotl32(o, k)
//   rotl64(L, 2k+1)    e' = rotl32(o, k+1), o' = rotl32(e, k)
//
// so a rotate by 1 is ONE instruction (o' = e is a register rename). Keccak
// rotates by 1 six times per round - the five theta parities and one rho
// lane - which is six of ~175 instructions per round, and the measured gain
// is +4.6 % over the word-split kernel it replaced (12 160 SASS per hash
// instead of 12 592, same 64 registers, RTX 3050 Ti: 165.5 vs 158.2 MH/s).
//
// The host does the interleaving of everything that is constant per launch:
// hdr_lanes[i] arrives as (o << 32) | e of header lane i (lane 9 with its
// nonce half zero), and t3 as (o << 32) | e of the target's most significant
// limb. The nonce itself is interleaved in the kernel (compress_even). The
// round constants are stored pre-interleaved (RC_E, RC_O); the CPU tests in
// backend/mod.rs check them against the 64-bit constants.
//
// Only lane 3 of the final digest - the most significant 64 bits of the hash
// read as a little-endian number - leaves the kernel, and h3 <= t3 decides.
// That is exact except when h3 == t3 (probability 2^-64), where the lower
// limbs would decide; such a nonce is reported as a hit and rejected by the
// CPU re-verification every hit goes through anyway.

#if defined(__OPENCL_VERSION__)
  typedef ulong u64;
  typedef uint  u32;
  #define KERNEL_FN   __kernel
  #define GLOBAL      __global
  #define CONST_ARR   __constant
  #define DEVICE_FN   static
  #define GLOBAL_ID() ((u32)get_global_id(0))
  #define ATOMIC_INC_U32(p) atomic_inc(p)
  #define ROTL32(x, k) rotate((u32)(x), (u32)(k))
  #define CLZ32(x)     ((int)clz((u32)(x)))
  // The compiler pattern-matches a^b^c to v_xor3_b32 on AMD; no asm needed.
  #define XOR3(a, b, c) ((a) ^ (b) ^ (c))
  // No unrolling here: the gain is measured on NVIDIA, and on AMD full
  // unrolling can instead drive up the register pressure. Measure before
  // it is turned on for OpenCL.
  #define UNROLL_ROUNDS
#else /* CUDA (nvcc) */
  typedef unsigned long long u64;
  typedef unsigned int u32;
  #define KERNEL_FN   extern "C" __global__
  #define GLOBAL
  #define CONST_ARR   __constant__
  #define DEVICE_FN   static __device__ __forceinline__
  #define GLOBAL_ID() ((u32)(blockIdx.x * blockDim.x + threadIdx.x))
  #define ATOMIC_INC_U32(p) atomicAdd(p, 1u)
  // A funnel shift of a word with itself is a 32-bit rotate: one SHF.
  #define ROTL32(x, k) __funnelshift_l((x), (x), (k))
  #define CLZ32(x)     __clz((int)(x))
  // Three-way XOR as one LOP3 (immLut 0x96 = a^b^c). ptxas finds some of
  // these by itself, but not all - written explicitly, theta's column sums
  // and application are one instruction per word.
  DEVICE_FN u32 xor3_lop3(u32 a, u32 b, u32 c) {
    u32 r;
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
  }
  #define XOR3(a, b, c) xor3_lop3((a), (b), (c))
  // Full unrolling of the 24 rounds. Two effects, both measured: the
  // per-round overhead (round constant, counter, comparison, branch)
  // disappears, and ptxas fits the kernel in 64 registers, which is 32
  // warps per SM. Partial unrolling (2, 4, 8) is WORSE than none at all.
  #define UNROLL_ROUNDS _Pragma("unroll")
#endif

// Rotate an interleaved lane by a compile-time n in 1..62. Every use has a
// literal n, so the selects fold and each expands to at most two rotates.
#define RK(x, k)    ((k) == 0 ? (x) : ROTL32((x), (k)))
#define RE(e, o, n) (((n) & 1) ? RK((o), ((n) >> 1) + 1) : RK((e), (n) >> 1))
#define RO(e, o, n) (((n) & 1) ? RK((e), (n) >> 1) : RK((o), (n) >> 1))

// The padding bytes in interleaved form. 0x06 has bits 1 and 2 set: bit 1
// is odd -> o bit 0, bit 2 is even -> e bit 1. 0x80 at byte 135 is bit 63
// of lane 16: odd -> o bit 31.
#define PAD_START_E 0x2u
#define PAD_START_O 0x1u
#define PAD_END_O   0x80000000u

// Keccak round constants, interleaved: RC_E[r] holds the even bits of
// RC[r], RC_O[r] the odd bits. Derived from the standard 64-bit table and
// pinned by the test `interleaved_round_constants_match_the_64_bit_table`.
CONST_ARR u32 RC_E[24] = {
  0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000001u,
  0x00000001u, 0x00000001u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u,
  0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u, 0x00000000u, 0x00000000u,
  0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x00000001u, 0x00000000u
};
CONST_ARR u32 RC_O[24] = {
  0x00000000u, 0x00000089u, 0x8000008bu, 0x80008080u, 0x0000008bu, 0x00008000u,
  0x80008088u, 0x80000082u, 0x0000000bu, 0x0000000au, 0x00008082u, 0x00008003u,
  0x0000808bu, 0x8000000bu, 0x8000008au, 0x80000081u, 0x80000081u, 0x80000008u,
  0x00000083u, 0x80008003u, 0x80008088u, 0x80000088u, 0x00008000u, 0x80008082u
};

// Gather the even bits of x into its low 16 bits (for the odd bits pass
// x >> 1). This is how the nonce enters the interleaved lane 9.
DEVICE_FN u32 compress_even(u32 x) {
  x &= 0x55555555u;
  x = (x | (x >> 1)) & 0x33333333u;
  x = (x | (x >> 2)) & 0x0f0f0f0fu;
  x = (x | (x >> 4)) & 0x00ff00ffu;
  x = (x | (x >> 8)) & 0x0000ffffu;
  return x;
}

// Chi on one row of one word set: a ^= ~b & c along the row. ptxas fuses
// each of these into a single LOP3.
#define CHI_ROW(w, a, b, c, d, f) { \
  u32 x0 = w[a], x1 = w[b], x2 = w[c], x3 = w[d], x4 = w[f]; \
  w[a] ^= (~x1) & x2; w[b] ^= (~x2) & x3; w[c] ^= (~x3) & x4; \
  w[d] ^= (~x4) & x0; w[f] ^= (~x0) & x1; }
#define CHI_ALL(e, o) \
  CHI_ROW(e, 0, 1, 2, 3, 4)      CHI_ROW(o, 0, 1, 2, 3, 4) \
  CHI_ROW(e, 5, 6, 7, 8, 9)      CHI_ROW(o, 5, 6, 7, 8, 9) \
  CHI_ROW(e, 10, 11, 12, 13, 14) CHI_ROW(o, 10, 11, 12, 13, 14) \
  CHI_ROW(e, 15, 16, 17, 18, 19) CHI_ROW(o, 15, 16, 17, 18, 19) \
  CHI_ROW(e, 20, 21, 22, 23, 24) CHI_ROW(o, 20, 21, 22, 23, 24)

// IMPORTANT for performance: every index into e[] and o[] below is a
// constant. Dynamic indexing makes the state spill to local memory and the
// kernel becomes ~100x slower (verified: a table-driven variant gave 4 MH/s
// instead of the GH/s class).
DEVICE_FN void keccak_round(u32 e[25], u32 o[25], u32 rce, u32 rco) {
  u32 ce0, ce1, ce2, ce3, ce4, co0, co1, co2, co3, co4, t, te, to, tme, tmo;

  // Theta. Column parities as XOR3 pairs, then column x gets
  // D[x] = C[x-1] ^ rotl64(C[x+1], 1), which in interleaved form is
  //   e words: C[x-1]_e ^ rotl32(C[x+1]_o, 1)   - one rotate per column
  //   o words: C[x-1]_o ^ C[x+1]_e              - no rotate at all
  // and the application folds both terms into one XOR3 per word, so D is
  // never materialized.
  ce0 = XOR3(e[0], e[5], e[10]);  ce0 = XOR3(ce0, e[15], e[20]);
  co0 = XOR3(o[0], o[5], o[10]);  co0 = XOR3(co0, o[15], o[20]);
  ce1 = XOR3(e[1], e[6], e[11]);  ce1 = XOR3(ce1, e[16], e[21]);
  co1 = XOR3(o[1], o[6], o[11]);  co1 = XOR3(co1, o[16], o[21]);
  ce2 = XOR3(e[2], e[7], e[12]);  ce2 = XOR3(ce2, e[17], e[22]);
  co2 = XOR3(o[2], o[7], o[12]);  co2 = XOR3(co2, o[17], o[22]);
  ce3 = XOR3(e[3], e[8], e[13]);  ce3 = XOR3(ce3, e[18], e[23]);
  co3 = XOR3(o[3], o[8], o[13]);  co3 = XOR3(co3, o[18], o[23]);
  ce4 = XOR3(e[4], e[9], e[14]);  ce4 = XOR3(ce4, e[19], e[24]);
  co4 = XOR3(o[4], o[9], o[14]);  co4 = XOR3(co4, o[19], o[24]);
  t = ROTL32(co1, 1);
  e[0] = XOR3(e[0], ce4, t); e[5] = XOR3(e[5], ce4, t); e[10] = XOR3(e[10], ce4, t);
  e[15] = XOR3(e[15], ce4, t); e[20] = XOR3(e[20], ce4, t);
  o[0] = XOR3(o[0], co4, ce1); o[5] = XOR3(o[5], co4, ce1); o[10] = XOR3(o[10], co4, ce1);
  o[15] = XOR3(o[15], co4, ce1); o[20] = XOR3(o[20], co4, ce1);
  t = ROTL32(co2, 1);
  e[1] = XOR3(e[1], ce0, t); e[6] = XOR3(e[6], ce0, t); e[11] = XOR3(e[11], ce0, t);
  e[16] = XOR3(e[16], ce0, t); e[21] = XOR3(e[21], ce0, t);
  o[1] = XOR3(o[1], co0, ce2); o[6] = XOR3(o[6], co0, ce2); o[11] = XOR3(o[11], co0, ce2);
  o[16] = XOR3(o[16], co0, ce2); o[21] = XOR3(o[21], co0, ce2);
  t = ROTL32(co3, 1);
  e[2] = XOR3(e[2], ce1, t); e[7] = XOR3(e[7], ce1, t); e[12] = XOR3(e[12], ce1, t);
  e[17] = XOR3(e[17], ce1, t); e[22] = XOR3(e[22], ce1, t);
  o[2] = XOR3(o[2], co1, ce3); o[7] = XOR3(o[7], co1, ce3); o[12] = XOR3(o[12], co1, ce3);
  o[17] = XOR3(o[17], co1, ce3); o[22] = XOR3(o[22], co1, ce3);
  t = ROTL32(co4, 1);
  e[3] = XOR3(e[3], ce2, t); e[8] = XOR3(e[8], ce2, t); e[13] = XOR3(e[13], ce2, t);
  e[18] = XOR3(e[18], ce2, t); e[23] = XOR3(e[23], ce2, t);
  o[3] = XOR3(o[3], co2, ce4); o[8] = XOR3(o[8], co2, ce4); o[13] = XOR3(o[13], co2, ce4);
  o[18] = XOR3(o[18], co2, ce4); o[23] = XOR3(o[23], co2, ce4);
  t = ROTL32(co0, 1);
  e[4] = XOR3(e[4], ce3, t); e[9] = XOR3(e[9], ce3, t); e[14] = XOR3(e[14], ce3, t);
  e[19] = XOR3(e[19], ce3, t); e[24] = XOR3(e[24], ce3, t);
  o[4] = XOR3(o[4], co3, ce0); o[9] = XOR3(o[9], co3, ce0); o[14] = XOR3(o[14], co3, ce0);
  o[19] = XOR3(o[19], co3, ce0); o[24] = XOR3(o[24], co3, ce0);

  // Rho + Pi, Saarinen's ordering: the value at position 1 moves to 10 with
  // rotation 1, the value that was at 10 moves to 7 with rotation 3, ...
#define STEP(pos, n) \
  tme = e[pos]; tmo = o[pos]; \
  e[pos] = RE(te, to, n); o[pos] = RO(te, to, n); \
  te = tme; to = tmo;
  te = e[1]; to = o[1];
  STEP(10, 1)  STEP(7, 3)   STEP(11, 6)  STEP(17, 10) STEP(18, 15) STEP(3, 21)
  STEP(5, 28)  STEP(16, 36) STEP(8, 45)  STEP(21, 55) STEP(24, 2)  STEP(4, 14)
  STEP(15, 27) STEP(23, 41) STEP(19, 56) STEP(13, 8)  STEP(12, 25) STEP(2, 43)
  STEP(20, 62) STEP(14, 18) STEP(22, 39) STEP(9, 61)  STEP(6, 20)
  e[1] = RE(te, to, 44); o[1] = RO(te, to, 44);
#undef STEP

  CHI_ALL(e, o)

  // Iota
  e[0] ^= rce;
  o[0] ^= rco;
}

DEVICE_FN void keccakf(u32 e[25], u32 o[25]) {
  UNROLL_ROUNDS
  for (int round = 0; round < 24; round++)
    keccak_round(e, o, RC_E[round], RC_O[round]);
}

// SHA3-256 of 32 bytes (lanes 0..3), in place: lanes 0..3 hold the input on
// entry and the digest on return. Padding: 0x06 at byte 32 (lane 4) and
// 0x80 at byte 135 (lane 16).
DEVICE_FN void sha3_256_32(u32 e[25], u32 o[25]) {
  u32 e2[25], o2[25];
  for (int i = 0; i < 25; i++) {
    e2[i] = 0;
    o2[i] = 0;
  }
  for (int i = 0; i < 4; i++) {
    e2[i] = e[i];
    o2[i] = o[i];
  }
  e2[4] = PAD_START_E;
  o2[4] = PAD_START_O;
  o2[16] = PAD_END_O;
  keccakf(e2, o2);
  for (int i = 0; i < 4; i++) {
    e[i] = e2[i];
    o[i] = o2[i];
  }
}

// SHA3-256 of 32 bytes, returning only lane 3 of the digest. 23 full rounds,
// then the part of round 24 that lane (3,0) depends on: theta needs every
// column parity, lane 3 of row 0 is chi(B[3,0], B[4,0], B[0,0]) with
// B[3,0] = rotl(A[3,3], 21) = lane 18, B[4,0] = rotl(A[4,4], 14) = lane 24
// and B[0,0] = lane 0, all after theta. Lane 3 gets no iota.
DEVICE_FN void sha3_256_32_lane3(const u32 in_e[4], const u32 in_o[4],
                                 u32 *out_e, u32 *out_o) {
  u32 e[25], o[25];
  for (int i = 0; i < 25; i++) {
    e[i] = 0;
    o[i] = 0;
  }
  for (int i = 0; i < 4; i++) {
    e[i] = in_e[i];
    o[i] = in_o[i];
  }
  e[4] = PAD_START_E;
  o[4] = PAD_START_O;
  o[16] = PAD_END_O;
  UNROLL_ROUNDS
  for (int round = 0; round < 23; round++)
    keccak_round(e, o, RC_E[round], RC_O[round]);

  u32 ce0, ce1, ce2, ce3, ce4, co0, co1, co2, co3, co4;
  ce0 = XOR3(e[0], e[5], e[10]);  ce0 = XOR3(ce0, e[15], e[20]);
  co0 = XOR3(o[0], o[5], o[10]);  co0 = XOR3(co0, o[15], o[20]);
  ce1 = XOR3(e[1], e[6], e[11]);  ce1 = XOR3(ce1, e[16], e[21]);
  co1 = XOR3(o[1], o[6], o[11]);  co1 = XOR3(co1, o[16], o[21]);
  ce2 = XOR3(e[2], e[7], e[12]);  ce2 = XOR3(ce2, e[17], e[22]);
  co2 = XOR3(o[2], o[7], o[12]);  co2 = XOR3(co2, o[17], o[22]);
  ce3 = XOR3(e[3], e[8], e[13]);  ce3 = XOR3(ce3, e[18], e[23]);
  co3 = XOR3(o[3], o[8], o[13]);  co3 = XOR3(co3, o[18], o[23]);
  ce4 = XOR3(e[4], e[9], e[14]);  ce4 = XOR3(ce4, e[19], e[24]);
  co4 = XOR3(o[4], o[9], o[14]);  co4 = XOR3(co4, o[19], o[24]);
  // Theta on lanes 0 (column 0: C4, C1), 18 (column 3: C2, C4) and 24
  // (column 4: C3, C0).
  u32 s0e  = XOR3(e[0],  ce4, ROTL32(co1, 1));
  u32 s0o  = XOR3(o[0],  co4, ce1);
  u32 s18e = XOR3(e[18], ce2, ROTL32(co4, 1));
  u32 s18o = XOR3(o[18], co2, ce4);
  u32 s24e = XOR3(e[24], ce3, ROTL32(co0, 1));
  u32 s24o = XOR3(o[24], co3, ce0);
  u32 b3e = RE(s18e, s18o, 21), b3o = RO(s18e, s18o, 21);
  u32 b4e = RE(s24e, s24o, 14), b4o = RO(s24e, s24o, 14);
  *out_e = b3e ^ ((~b4e) & s0e);
  *out_o = b3o ^ ((~b4o) & s0o);
}

// Grind the nonce space [start_nonce, start_nonce + nonce_count).
//
//   hdr_lanes: 10 u64 - the 80-byte header (nonce field = 0) as LE lanes,
//              each INTERLEAVED: (o << 32) | e. See backend::kernel_lanes.
//   t3:        the most significant u64 limb of the share target (limb 3 of
//              the hash read as a little-endian 256-bit number), interleaved
//              the same way. See backend::kernel_target.
//   hits:      hits[0] = atomic hit counter, hits[1..1+max_hits] = nonces.
KERNEL_FN void sha3t_scan(GLOBAL const u64 *hdr_lanes,
                          u32 start_nonce,
                          u32 nonce_count,
                          u64 t3,
                          GLOBAL u32 *hits,
                          u32 max_hits) {
  u32 gid = GLOBAL_ID();
  if (gid >= nonce_count)
    return;
  u32 nonce = start_nonce + gid;

  // Round 1: the 80-byte header. The nonce is bytes 76..79 = the high word
  // of lane 9, so its even bits land in e[9] bits 16..31 and its odd bits in
  // o[9] bits 16..31.
  u32 e[25], o[25];
  for (int i = 0; i < 25; i++) {
    e[i] = 0;
    o[i] = 0;
  }
  for (int i = 0; i < 10; i++) {
    u64 l = hdr_lanes[i];
    e[i] = (u32)l;
    o[i] = (u32)(l >> 32);
  }
  e[9] |= compress_even(nonce) << 16;
  o[9] |= compress_even(nonce >> 1) << 16;
  e[10] = PAD_START_E;   // padding byte 80
  o[10] = PAD_START_O;
  o[16] = PAD_END_O;     // padding byte 135
  keccakf(e, o);

  // Rounds 2 and 3: the 32-byte intermediate hash. Only lane 3 comes out of
  // the last one.
  sha3_256_32(e, o);
  u32 h3e, h3o;
  sha3_256_32_lane3(e, o, &h3e, &h3o);

  // h3 <= t3, decided in interleaved form: the highest bit where they
  // differ decides, and h3 < t3 iff t3 has a 1 there. Odd word index p is
  // real bit 2p+1 and even word index p is real bit 2p, so on equal indices
  // the odd word wins.
  u32 te = (u32)t3, to = (u32)(t3 >> 32);
  u32 xe = h3e ^ te, xo = h3o ^ to;
  bool ok;
  if ((xe | xo) == 0u) {
    ok = true;
  } else {
    int pe = 31 - CLZ32(xe);   // -1 when xe == 0
    int po = 31 - CLZ32(xo);
    ok = (po >= pe) ? (((to >> po) & 1u) != 0u) : (((te >> pe) & 1u) != 0u);
  }

  if (ok) {
    u32 idx = ATOMIC_INC_U32(&hits[0]);
    if (idx < max_hits)
      hits[idx + 1] = nonce;
  }
}
