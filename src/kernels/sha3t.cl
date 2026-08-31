// SHA3-256t-kernel (trippel NIST SHA3-256) — DELAD källa för CUDA och OpenCL.
//
// Samma fil kompileras av NVRTC (som CUDA C++) och av OpenCL-runtimen (som
// OpenCL C). Alla skillnader kapslas i makronen nedan — själva keccak-kärnan
// är identisk, så bitexakthet verifierad för den ena backenden gäller båda.
//
// Hashschema (se src/consensus.rs för CPU-referensen):
//   - Headern är 80 bytes och SHA3-256-raten är 136 bytes ⇒ ETT absorb-block.
//   - Padding (NIST SHA3, inte rå keccak): 0x06 på byte 80, 0x80 på byte 135.
//   - Varv 2 och 3 hashar 32 bytes: 0x06 på byte 32, 0x80 på byte 135.
//   - Totalt exakt 3 keccak-f[1600]-permutationer per nonce.
//
// Värden packas som u64-lanes i little-endian (lane i = bytes 8i..8i+7), så
// hela headern får plats i lane 0..9 och noncen är höga halvan av lane 9.

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
#endif

// Trevags-XOR. Ampere/Ada har LOP3, som raknar ut en GODTYCKLIG funktion av
// tre indata i EN instruktion (immLut 0x96 = a^b^c). ptxas hittar en del av
// dessa monster sjalv, men inte alla — att skriva dem explicit tar theta fran
// 50 till 35 logikoperationer per varv. Pa AMD (RDNA2+) monstermatchar
// kompilatorn `a^b^c` till v_xor3_b32, sa OpenCL-grenen behover ingen asm.
#if defined(__OPENCL_VERSION__)
  #define XOR3(a, b, c) ((a) ^ (b) ^ (c))
#else
DEVICE_FN u64 xor3_lop3(u64 a, u64 b, u64 c) {
  u64 r;
  // En rad: PTX-satser avslutas med semikolon, sa inga radbrytningar
  // behovs — och da finns inga escape-sekvenser som kan ga sonder.
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

// VIKTIGT för prestanda: rundfunktionen är HELT utrullad med konstanta
// index — dynamisk indexering av st[] gör att tillståndet spiller till
// local memory och kerneln blir ~100× långsammare. (Verifierat empiriskt:
// den kompakta tabellstyrda varianten gav 4 MH/s i stället för GH/s-klass.)
DEVICE_FN void keccakf(u64 st[25]) {
  u64 bc0, bc1, bc2, bc3, bc4, t, tmp;
  for (int round = 0; round < 24; round++) {
    // Theta. Kolumnsummorna som XOR3-par; appliceringen vager in bade
    // C[x-1] och ROTL(C[x+1],1) i samma instruktion, sa D[x] aldrig
    // materialiseras. `t` haller bara rotationen — en levande vardet i
    // taget, precis som forr, sa registertrycket ar oforandrat.
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

    // Rho + Pi (Saarinens ordning, utrullad: (lane, rot) per steg)
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

    // Chi, rad för rad
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

// SHA3-256 av 32 bytes (lane 0..3) → 32 bytes. Padding: 0x06 på byte 32
// (lane 4, byte 0) och 0x80 på byte 135 (lane 16, byte 7).
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

// Grinda nonce-rymden [start_nonce, start_nonce + nonce_count).
//
//   hdr_lanes: 10 u64 — 80-bytesheadern (nonce-fältet = 0) som LE-lanes.
//   t0..t3:    share-target som fyra u64-limbar; t3 mest signifikant.
//              (Hashen tolkas som little-endian 256-bitars tal; limb k = u64
//               ur hashbytes 8k..8k+7, jfr consensus::hash_meets_target.)
//   hits:      hits[0] = atomisk träffräknare, hits[1..1+max_hits] = noncer.
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

  // Varv 1: 80-bytesheadern. Noncen är bytes 76..79 = höga halvan av lane 9.
  u64 st[25];
  for (int i = 0; i < 25; i++)
    st[i] = 0;
  for (int i = 0; i < 10; i++)
    st[i] = hdr_lanes[i];
  st[9] |= (u64)nonce << 32;
  st[10] = 0x06UL;                  // paddingbyte 80
  st[16] = 0x8000000000000000UL;    // paddingbyte 135
  keccakf(st);

  // Varv 2 och 3: 32-bytes mellanhash.
  u64 h[4], h2[4];
  h[0] = st[0]; h[1] = st[1]; h[2] = st[2]; h[3] = st[3];
  sha3_256_32(h2, h);
  sha3_256_32(h, h2);

  // hash ≤ target (little-endian 256-bitars tal, limb 3 mest signifikant).
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
