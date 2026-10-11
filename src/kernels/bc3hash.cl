// BC3Hash kernel - SHARED source for CUDA and OpenCL.
//
// GENERATED FILE: kernel_head.cl + consts.h + kernel_body.cl from
// miner-perf-lab/bc3gpu (consts.h comes from the node's constants.hpp via
// gen_consts.cpp). Edit the parts there, not this file.
//
// BC3Hash (bitcoiniii-core src/crypto/bc3) of an 80-byte block header:
//   six standard hashes of tag || header, BC3Mix of the SHA3-512 one, and a
//   final SHA3-256 over all of them. BC3Mix is the expensive part: modular
//   exponentiations with fixed exponents mod 2^(64L)-C, table powers of fixed
//   bases, 64 secp256k1 scalar multiplications from precomputed tables and 64
//   inversions mod 2^256-36113, in four dependent stages.
//
// Every modular result is the canonical residue, so any correct way of
// computing it gives the node's bits; the CPU reference re-verifies every hit.

#if defined(__OPENCL_VERSION__)
  typedef ulong u64;
  typedef uint  u32;
  typedef uchar u8;
  typedef long  i64;
  typedef int   i32;
  #define KERNEL_FN   __kernel
  #define GLOBAL      __global
  #define CONST_ARR   __constant
  #define DEVICE_FN   static inline
  #define DEVICE_BIG  static
  #define GLOBAL_ID() ((u32)get_global_id(0))
  #define ATOMIC_INC_U32(p) atomic_inc(p)
  #define MULHI64(a, b) mul_hi((ulong)(a), (ulong)(b))
  #define CONST_PTR   __constant
#else /* CUDA (nvcc) */
  typedef unsigned long long u64;
  typedef unsigned int       u32;
  typedef unsigned char      u8;
  typedef long long          i64;
  typedef int                i32;
  #define KERNEL_FN   extern "C" __global__
  #define GLOBAL
  #define CONST_ARR   __constant__
  #define DEVICE_FN   static __device__ __forceinline__
  #define DEVICE_BIG  static __device__ __noinline__
  #define GLOBAL_ID() (blockIdx.x * blockDim.x + threadIdx.x)
  #define ATOMIC_INC_U32(p) atomicAdd((p), 1u)
  #define MULHI64(a, b) __umul64hi((a), (b))
  #define CONST_PTR
#endif

// Window of the secp256k1 tables (bits per digit). The host builds the table
// with the same value; see BC3_TABLE_* below for the buffer layout.
#ifndef BC3_WINDOW
  #define BC3_WINDOW 10
#endif
#define BC3_DIGITS  ((256 + BC3_WINDOW - 1) / BC3_WINDOW)
#define BC3_ENTRIES ((1u << BC3_WINDOW) - 1u)
#define BC3_CURVE_POINTS (BC3_DIGITS * BC3_ENTRIES)

// One table buffer, in u64 units:
//   pow64   8 rows x 256 x 1 limb      (g64^(b*256^i) mod 2^64-1469)
//   pow128  16 rows x 256 x 2 limbs    (mod 2^128-15449)
//   pow256  32 rows x 256 x 4 limbs    (mod 2^256-36113)
//   curves  4 x BC3_CURVE_POINTS x (x[4], y[4])   affine, b*2^(w*i)*P
#define BC3_TABLE_POW64   0
#define BC3_TABLE_POW128  (BC3_TABLE_POW64 + 8 * 256 * 1)
#define BC3_TABLE_POW256  (BC3_TABLE_POW128 + 16 * 256 * 2)
#define BC3_TABLE_CURVE   (BC3_TABLE_POW256 + 32 * 256 * 4)
#define BC3_TABLE_U64     (BC3_TABLE_CURVE + 4 * BC3_CURVE_POINTS * 8)
CONST_ARR i32 BC3_MODINV_SECP[9] = {-0x3D1, -4, 0, 0, 0, 0, 0, 0, 65536};
#define BC3_MODINV_SECP_INV30 0x2ddacacfu
CONST_ARR i32 BC3_MODINV_F256[9] = {-36113, 0, 0, 0, 0, 0, 0, 0, 65536};
#define BC3_MODINV_F256_INV30 0x0f6bfc0fu
// ---- GENERERAT av miner-perf-lab/bc3gpu/gen_consts.cpp ur nodens constants.hpp ----
CONST_ARR u64 BC3_BASE[8] = {
  0xf70c6462eb2ae2d6ULL,
  0x927cb0e2c353f73eULL,
  0xbf2f9516fc9c4330ULL,
  0xc15671f266593ff5ULL,
  0x762c297f71890b5eULL,
  0x108c447e7ebcd488ULL,
  0xc24818761058e50eULL,
  0x36fd1f3c7fa624b2ULL,
};
CONST_ARR u64 BC3_ROUND64[32] = {
  0x80b4847ee1d126d5ULL,
  0xc52a0d9a387a457bULL,
  0x22e65c431bed21b2ULL,
  0x14667805c6955887ULL,
  0xe17b3ee0aa3cf44eULL,
  0xd2cd9d698d6a1671ULL,
  0xdb425c815c0704feULL,
  0x066c4ffd688e12a5ULL,
  0x419215840b275a8cULL,
  0x369230ade1d36976ULL,
  0x6bab679cb1b90894ULL,
  0x2c1c295d2e646b3bULL,
  0x3e59f4830d13644dULL,
  0x1b0031adae0acaf8ULL,
  0x3796b541775d8e24ULL,
  0x2f3efa4806170292ULL,
  0x16075331aa74e4e4ULL,
  0xdae9d5b3ec6cefedULL,
  0x2820ef5c65f86f0bULL,
  0x7588f2cff28b3fc5ULL,
  0xa37049d63b598ca5ULL,
  0x3edd8b4d6ca62563ULL,
  0x43e59de592ba46a0ULL,
  0xcf164eec37b7f680ULL,
  0xffeeebd5c0653331ULL,
  0xebf298d3d723ccdaULL,
  0xb2c336ac814e438dULL,
  0xd0a3b74e43a31fb5ULL,
  0x1a6309def1115850ULL,
  0x156d245c87623bedULL,
  0x81971caeb07b6180ULL,
  0x08cd67e5ce559832ULL,
};
CONST_ARR u64 BC3_ROUND128[32] = {
  0xb446f99f1909aa84ULL, 0x72d2dabbaac34eefULL,
  0x711549a20ca98933ULL, 0xd15a623dd1654d61ULL,
  0xc2399226e67a0018ULL, 0xf80af0b8c6f423c7ULL,
  0x05cc4bf907121fedULL, 0x2be1771ab79765a4ULL,
  0x49a987a99529b720ULL, 0x75ddc3d8e0566ca8ULL,
  0xa1d91327ebdd2c4dULL, 0x60826fc8e665ede7ULL,
  0x5f7e14cdd2cb98b8ULL, 0xaf5f60a2275a84b4ULL,
  0xb6a7e0025fb3c250ULL, 0xa28ab2aa854ec5beULL,
  0x9680b307838b4544ULL, 0x95481fcea93b3421ULL,
  0x8da650a396026d41ULL, 0xa1d6048ac9f8f630ULL,
  0x37448d77338dea6bULL, 0xf9556fbf52a88c13ULL,
  0xeba00bae9a3aa372ULL, 0xead719a4044d895dULL,
  0x5dc6d29da087752fULL, 0xe6bc94f7926414a3ULL,
  0xd9e1d13da4ef534cULL, 0xdc248964f9a21c16ULL,
  0xe87b49495b47cfddULL, 0xee3ca1fb0e50d3f5ULL,
  0x82a524368adf3893ULL, 0x49b06366a722cff5ULL,
};
CONST_ARR u64 BC3_ROUND256[64] = {
  0x37861eaa70b39a41ULL, 0x305580c292edc126ULL, 0x396410c3827e8d0aULL, 0x6111f39bfd11b164ULL,
  0x78c70bc875a9669aULL, 0x5f0693792b60006bULL, 0x464535d0a21a31dbULL, 0xe1490e4bf73d5aeaULL,
  0xaffc74849b7bb962ULL, 0xd7de1078d67685a5ULL, 0xd1a5ba1b790d2374ULL, 0xfe00099f5049ad0eULL,
  0xee761ec225206318ULL, 0x9037f4db5550af11ULL, 0xcf7cd1d6d31f3f07ULL, 0xd1d70fddb21d577bULL,
  0xcd11d4fe89327b8fULL, 0xb0fe38675d26281bULL, 0x96caa96b03cd8242ULL, 0xcf0e2c6afd6fadfbULL,
  0xdf19dafcdd5bb1f0ULL, 0xd282b4ab122fce31ULL, 0xafe08dc41326e314ULL, 0x3bdd3a3c74d68e63ULL,
  0x29675e8674bbc2c4ULL, 0xdfbd4c541ceb948eULL, 0x8a128eccabd57250ULL, 0x351c9afb5cb7987bULL,
  0x456167a9a6a691eaULL, 0xf0b9328f0ba9cbd4ULL, 0xa8a181aa42a58c66ULL, 0x0fb1b2ec6b8c8d10ULL,
  0x60a6f3adc03159d4ULL, 0x4a29260de0268f70ULL, 0x9763d26dcdcab4ffULL, 0xe38b86a71f2ae6d6ULL,
  0xf83a498bfecffb06ULL, 0x9f8a6e614127b710ULL, 0xc55cb557f99c60d8ULL, 0x14be1b97b838aca8ULL,
  0xfb080b788e02f0e1ULL, 0x207b6d170dc53239ULL, 0xe89ba2d35ffeada6ULL, 0x79c5d2a0e9c9c3d2ULL,
  0x57b9e4e324f2b1bbULL, 0x8f05d8e3a804bc89ULL, 0xb7c03690e3234b46ULL, 0x40c94e657e14192dULL,
  0x1138790ee09cd03bULL, 0x49d9fa3b78cd94b8ULL, 0xffd351652d7d7316ULL, 0x474cd214da0634afULL,
  0x12e40443ebc81f63ULL, 0x66daf436f0f55faeULL, 0xa0445bc6296f9bbcULL, 0x900ebe45b91037f8ULL,
  0x6654e39eedc737e1ULL, 0xe839b455dd7b5fd4ULL, 0x023e1571bb8d4fa3ULL, 0x5aef38a1b25c2be9ULL,
  0xdbabd97368b8a187ULL, 0x459daab2788d2031ULL, 0x1b80fd175ae409fdULL, 0x73949255913776deULL,
};
CONST_ARR u64 BC3_FINAL[256] = {
  0x817083395555c30fULL, 0xd95a85dfc873ac54ULL, 0xcf6ba04a683d7a60ULL, 0x9955b67ba23081f7ULL,
  0x4651c0dce4e415c5ULL, 0xceef42e7fa66331fULL, 0xa0a54813a9d3ea9dULL, 0x99784555ee242dfdULL,
  0xbaa8ebfecbf90b46ULL, 0x4e362ea94f962c31ULL, 0x84c0636bf6818ffbULL, 0x170f26673e7dbb0aULL,
  0x16c48f46b4f733fcULL, 0x51f91ea5843f1e83ULL, 0x178921653f63d8d2ULL, 0x07aa11fa2a5fcfeeULL,
  0x3b7176a8440fbb27ULL, 0x9a2f60d493f872c9ULL, 0x3975e3b82d679f5fULL, 0xd735370d48bfdcceULL,
  0x3e722447a516b4c9ULL, 0x23c8b930c24ab561ULL, 0x35864c083059205eULL, 0xd899501de756bcb1ULL,
  0xb012027930f5ab63ULL, 0x71874f942d6eb59fULL, 0xcbbc6a933900dacdULL, 0x180e6ad999dd9db2ULL,
  0xee639bc6dd332995ULL, 0x61033b4fe9244710ULL, 0xe53869e7ce97778fULL, 0xdcb0af221294f003ULL,
  0x5906bfd8ad93e48aULL, 0x362deef1f1c0ba8dULL, 0x00c5c514e05a2f81ULL, 0x912a7b68f34f6eeeULL,
  0x80991ddb2c3b113aULL, 0x76d17a95f7e7d29aULL, 0xf8c31b7e33458a44ULL, 0xf5befb974f5feb7dULL,
  0x64ae619444ed9fe8ULL, 0xc2d35daf57dbcadaULL, 0x5ccdfdfd712c421bULL, 0xf1313f1399472a6fULL,
  0xc99f669e768f23eaULL, 0x6480fdefa405b9f6ULL, 0x52bf81f3a739e396ULL, 0x68877600cbbaff50ULL,
  0xd890bb67328fdf7fULL, 0x58390731d6a3b913ULL, 0x890645bf615e27c0ULL, 0x673f37f85f3b2f13ULL,
  0xeb0e5619f25c48a6ULL, 0xf39b9e5197060f27ULL, 0xc0772ed24b553f10ULL, 0x0650a4eca8b60dacULL,
  0x2a868b7a9f434519ULL, 0x3caeb76d0615e5d5ULL, 0x6c661d95765377feULL, 0x3731b934e3849714ULL,
  0x771aa47493332e30ULL, 0xb3bfef97dccb5bd2ULL, 0x7775842bab918482ULL, 0x101d01fe3f98e74bULL,
  0x143ac89dfe80eab6ULL, 0x59ed4354c04ee809ULL, 0x75f4d19ee9215fccULL, 0x4ae24cc98f987ef6ULL,
  0xd89e4848e896d1b6ULL, 0x5207cad331ea1de1ULL, 0x498969c35a4b8cc6ULL, 0x4c9356eba43c4324ULL,
  0x80fce2f269aeaef4ULL, 0x5a1a6eb16dd70d6fULL, 0xd19c79d0fa8f3269ULL, 0xbbbdd512ae282ccaULL,
  0xb7005ecd7a1d5c1bULL, 0xb0a69d436d746990ULL, 0xb5ebf69f31961744ULL, 0x32828e0b7e350cf8ULL,
  0xec8afde7284bc232ULL, 0x678aeea88c1ed974ULL, 0x83be5b9af0657182ULL, 0x755820e7ff973146ULL,
  0x70b3978e5901b6a0ULL, 0x86b1b3ad728fccc2ULL, 0x2abc0943fdb61c45ULL, 0x1c82c1614ebf98acULL,
  0xa9c3d061fa3df727ULL, 0x73dd3171e403304fULL, 0xbde528cffd3eebb2ULL, 0x4a30eeef58c74af4ULL,
  0xa110a69caf6fb820ULL, 0x56acf773f839b3f5ULL, 0xddc0054536cb2edeULL, 0x4ea5259761fe70b5ULL,
  0x37df9a34c054fcf6ULL, 0xd51be7f84b164ae3ULL, 0x827dfbd303eca5f2ULL, 0x6da17ae5e890f07aULL,
  0xe418212ddff2d318ULL, 0x153585c78ab82ac1ULL, 0x0f54ccf511035d2dULL, 0xafd1dc3c5ea81907ULL,
  0x4f7938a15b6374a4ULL, 0x93de10ef5eae66c8ULL, 0xef7bf20247c0fc1cULL, 0x5590f6b776f76817ULL,
  0x55dbfe0727d6e5f5ULL, 0x9344e69a351acc57ULL, 0x5efde95b19694c30ULL, 0xa1357f946b693c2fULL,
  0xb437b1cd798dd2f2ULL, 0xb9c2b941aa464b02ULL, 0xa8441df79eb8fbd2ULL, 0x4e38c422f62b5823ULL,
  0xcbb9a8c91faec809ULL, 0xc349f4c32d679fc5ULL, 0x240cb209683469b9ULL, 0xee939c7637580417ULL,
  0x086b80cdfed83435ULL, 0xce919488b37d1627ULL, 0x7da34fefe118b158ULL, 0xcd9879979778216eULL,
  0xf95c27f9e489a029ULL, 0x5a45e98cb3c17adfULL, 0xed803a6996e7dae5ULL, 0x7fc58ed8b6f4b34bULL,
  0x585de1ae6fc3441dULL, 0xa1eebae3675b0ec6ULL, 0x4a22305be4a5c246ULL, 0x3710e04efbd62c6dULL,
  0xbd71610480d26760ULL, 0xf55b8de6dcdc7351ULL, 0x417561bead79891cULL, 0xf1cd05d64625d55fULL,
  0x4754d1054703346eULL, 0x9ebcbeb575f472e2ULL, 0x6e3cdf7dafbd8ae6ULL, 0x273cc20d22bae544ULL,
  0x9b9b7fcf81630a72ULL, 0xa89ee2553cc12703ULL, 0xeabab1c45a45a615ULL, 0xfb85f5fa5dfcf868ULL,
  0x85c885725555a6a0ULL, 0x63375462b0df01d5ULL, 0x93772851d0a7b22bULL, 0x76d12280182f5e00ULL,
  0xe888e8e8a61c2841ULL, 0x3370cc67f7408465ULL, 0x4634344f6e8a031dULL, 0xaf0c661b32eb31b4ULL,
  0x3757cdcb26d95c2cULL, 0x53e509d0f32aeb55ULL, 0x1304b17927b2dfcbULL, 0xa7a2320511061575ULL,
  0x3ecc8f7495140f28ULL, 0x8b596ca141685444ULL, 0x5be4561a5f229a18ULL, 0xd387d84911209f0dULL,
  0x5e66560bf72e1a5fULL, 0x32b504d96136384dULL, 0x31c69fa0540a4aecULL, 0x1563bfdefacc80b1ULL,
  0x172c9c5613a717ebULL, 0x1f805827a347ea9dULL, 0xae0070ee2f6afb3aULL, 0xe1f77bb83abe2e45ULL,
  0xfb1fe0f95b210f80ULL, 0xe7580352e5c95776ULL, 0x1646e12c209284c7ULL, 0x3f97105dc086f1d1ULL,
  0x4e1d420a8b183fbdULL, 0xda82b46684070428ULL, 0xb6b2c089f87fa9f6ULL, 0x2db485e568750050ULL,
  0x0ac96805e1e334a7ULL, 0xb8279822c219eec5ULL, 0xff3b2fce537e028aULL, 0x405fff4a93c4f70fULL,
  0x661cbe2703641379ULL, 0xbdb957c4d7659545ULL, 0x4ffb42e3c3f2099eULL, 0xa901c8e4585ed757ULL,
  0x4cabcc116b5d19adULL, 0x8e9472e15d4413bdULL, 0x0001a0f2e798b51cULL, 0x0e5b32c400c04baeULL,
  0x290b6cf63894f77bULL, 0xd0cb3f0ad7eea67dULL, 0x0b556e145fa979a1ULL, 0x26446d2b4cfa5b7bULL,
  0xd98e63688809a927ULL, 0xc078c4efc627b530ULL, 0xd0f8c452ed4747afULL, 0xe42fb0dd2e00b32fULL,
  0x5751cc98b1812ebcULL, 0x9bf2f9c7a5592224ULL, 0x72dc31f9827929acULL, 0x388b5e4c9933b600ULL,
  0x46a96722fc856d15ULL, 0x33abd57d1fccb4f6ULL, 0xf5a32e8628e36127ULL, 0x8437e139f1a563acULL,
  0x0f1b47ab2c1bb213ULL, 0xfe1245392c2940e4ULL, 0x52c82ccd0bfd2dabULL, 0x5a8ec26dffa90c5dULL,
  0x7b2923f08ce3c5baULL, 0x8ba684b4fc24483fULL, 0x8879172903cf3236ULL, 0x49a7d025205daca7ULL,
  0xf6e267857aacea89ULL, 0x32422cc5a16f71acULL, 0xc3b2ef06d2cdffe3ULL, 0x1a689e0246346b2eULL,
  0xbe82aa7ee027adc1ULL, 0xbffaa5e9f9b857f4ULL, 0x9de97718e6cb3629ULL, 0xeaaab1036f119011ULL,
  0x8bc731fe9c5e954bULL, 0xfa3d55e426003402ULL, 0x35b4884bc6e3aa8eULL, 0x4967e0525cf1f050ULL,
  0xf9651571f3e4df9eULL, 0x3fe3c4f2723e8938ULL, 0x0b0d62835ce112a9ULL, 0x60be27cc1bb8fdc1ULL,
  0xa03d6d8b8b021999ULL, 0x0653ae9e3125479eULL, 0x2810de3c1cd36bdfULL, 0xc3c09a1a2af40fccULL,
  0xd962e9576c6a2ef1ULL, 0x0d8c75a7e1196c9eULL, 0x1fa26f5be2a4f258ULL, 0x930873d43406c399ULL,
  0xa21027269105a62eULL, 0x50bcd2d7d6509bc7ULL, 0x494f1e91ff17f2f1ULL, 0xe969ccbf6d73649eULL,
  0xc3e22109f6a96e98ULL, 0x035949c07b738144ULL, 0x56e24a3016f6addfULL, 0xed69c264a65d7920ULL,
  0x59c6040b58498838ULL, 0xa45a8e51d4bb9110ULL, 0xf721cbe37c81042bULL, 0xf40f828787f7af6bULL,
  0xaa75827ad27b2e72ULL, 0x4778f53847660432ULL, 0xcf20ee66047fb084ULL, 0xa85f0ba124516696ULL,
  0xafa82a2ead66dad2ULL, 0x87c0dc112fde6941ULL, 0xd866934046e7fe7bULL, 0x07725b2a7c500486ULL,
};
CONST_ARR u64 BC3_E64[1] = {0x08cc1751c4763fa1ULL};
CONST_ARR u64 BC3_E128[2] = {0xfa5aa27fac6a65b3ULL, 0x928a70e905d1724aULL};
CONST_ARR u64 BC3_E256[4] = {0xa30182cddcf6c2a9ULL, 0x2d97ec3a6a936438ULL, 0xb27c7c729029c0a7ULL, 0x77f21ca678d771adULL};
CONST_ARR u64 BC3_INVALID_X[4] = {0xfffffffefffffc2fULL, 0xffffffffffffffffULL, 0xffffffffffffffffULL, 0xffffffffffffffffULL};
CONST_ARR u64 BC3_INVALID_Y[4] = {0xb0d1596d3e58acfdULL, 0xcca6988067678d8dULL, 0xae3609da2bb48f73ULL, 0x0f63cbbb426c5da1ULL};
// ---- slut på genererat ----

// =========================================================================
// Multi-limb integers. Limbs are u64, least significant first - the same
// layout as UInt<L>::v in uint.hpp, so the port of the mix reads like the
// reference. "+" between values is addition mod 2^(64L), as in UInt.
// =========================================================================

typedef struct { u64 v[2]; } u128;
typedef struct { u64 v[4]; } u256;

DEVICE_FN u128 mk128(u64 lo, u64 hi) { u128 r; r.v[0] = lo; r.v[1] = hi; return r; }
DEVICE_FN u256 mk256(u64 a, u64 b, u64 c, u64 d) { u256 r; r.v[0] = a; r.v[1] = b; r.v[2] = c; r.v[3] = d; return r; }
DEVICE_FN u256 c256(CONST_PTR const u64* p) { return mk256(p[0], p[1], p[2], p[3]); }

DEVICE_FN u64 ror64(u64 x, u32 n) { return (x >> n) | (x << ((64u - n) & 63u)); }
DEVICE_FN u64 rol64(u64 x, u32 n) { return ror64(x, (64u - n) & 63u); }

DEVICE_FN u128 add128(u128 a, u128 b) {
  u128 r;
  r.v[0] = a.v[0] + b.v[0];
  r.v[1] = a.v[1] + b.v[1] + (u64)(r.v[0] < a.v[0]);
  return r;
}
DEVICE_FN u256 add256(u256 a, u256 b) {
  u256 r;
  u64 c = 0;
  for (int i = 0; i < 4; i++) {
    u64 s = a.v[i] + b.v[i];
    u64 c1 = (u64)(s < a.v[i]);
    u64 s2 = s + c;
    c = c1 | (u64)(s2 < s);
    r.v[i] = s2;
  }
  return r;
}
DEVICE_FN u128 xor128(u128 a, u128 b) { return mk128(a.v[0] ^ b.v[0], a.v[1] ^ b.v[1]); }
DEVICE_FN u128 and128(u128 a, u128 b) { return mk128(a.v[0] & b.v[0], a.v[1] & b.v[1]); }
DEVICE_FN u128 or128(u128 a, u128 b) { return mk128(a.v[0] | b.v[0], a.v[1] | b.v[1]); }
DEVICE_FN u128 not128(u128 a) { return mk128(~a.v[0], ~a.v[1]); }
DEVICE_FN u256 xor256(u256 a, u256 b) { return mk256(a.v[0] ^ b.v[0], a.v[1] ^ b.v[1], a.v[2] ^ b.v[2], a.v[3] ^ b.v[3]); }
DEVICE_FN u256 and256(u256 a, u256 b) { return mk256(a.v[0] & b.v[0], a.v[1] & b.v[1], a.v[2] & b.v[2], a.v[3] & b.v[3]); }
DEVICE_FN u256 or256(u256 a, u256 b) { return mk256(a.v[0] | b.v[0], a.v[1] | b.v[1], a.v[2] | b.v[2], a.v[3] | b.v[3]); }
DEVICE_FN u256 not256(u256 a) { return mk256(~a.v[0], ~a.v[1], ~a.v[2], ~a.v[3]); }

// ror/shr by n bits; n is a literal at every call site, so after inlining
// the limb indices and shift counts are constants.
DEVICE_FN u128 ror128(u128 a, u32 n) {
  u32 q = (n >> 6) & 1u, s = n & 63u;
  u128 r;
  for (u32 i = 0; i < 2; i++) {
    u64 lo = a.v[(i + q) & 1u] >> s;
    u64 hi = s ? (a.v[(i + q + 1) & 1u] << (64u - s)) : 0;
    r.v[i] = lo | hi;
  }
  return r;
}
DEVICE_FN u128 rol128(u128 a, u32 n) { return ror128(a, (128u - (n & 127u)) & 127u); }
DEVICE_FN u128 shr128(u128 a, u32 n) {
  u32 q = n >> 6, s = n & 63u;
  u128 r;
  for (u32 i = 0; i < 2; i++) {
    u64 lo = (i + q < 2) ? (a.v[i + q] >> s) : 0;
    u64 hi = (s && i + q + 1 < 2) ? (a.v[i + q + 1] << (64u - s)) : 0;
    r.v[i] = lo | hi;
  }
  return r;
}
DEVICE_FN u256 ror256(u256 a, u32 n) {
  u32 q = (n >> 6) & 3u, s = n & 63u;
  u256 r;
  for (u32 i = 0; i < 4; i++) {
    u64 lo = a.v[(i + q) & 3u] >> s;
    u64 hi = s ? (a.v[(i + q + 1) & 3u] << (64u - s)) : 0;
    r.v[i] = lo | hi;
  }
  return r;
}
DEVICE_FN u256 rol256(u256 a, u32 n) { return ror256(a, (256u - (n & 255u)) & 255u); }
DEVICE_FN u256 shr256(u256 a, u32 n) {
  u32 q = n >> 6, s = n & 63u;
  u256 r;
  for (u32 i = 0; i < 4; i++) {
    u64 lo = (i + q < 4) ? (a.v[i + q] >> s) : 0;
    u64 hi = (s && i + q + 1 < 4) ? (a.v[i + q + 1] << (64u - s)) : 0;
    r.v[i] = lo | hi;
  }
  return r;
}

// ---- The mixing functions of uint.hpp, per width ----

DEVICE_FN u64 ch64(u64 x, u64 y, u64 z) { return z ^ (x & (y ^ z)); }
DEVICE_FN u64 maj64(u64 x, u64 y, u64 z) { return (x & y) | (z & (x | y)); }
DEVICE_FN u64 Sig0_64(u64 x) { return ror64(x, 28) ^ ror64(x, 34) ^ ror64(x, 39); }
DEVICE_FN u64 Sig1_64(u64 x) { return ror64(x, 14) ^ ror64(x, 18) ^ ror64(x, 41); }
DEVICE_FN u64 sig0_64(u64 x) { return ror64(x, 1) ^ ror64(x, 8) ^ (x >> 7); }
DEVICE_FN u64 sig1_64(u64 x) { return ror64(x, 19) ^ ror64(x, 61) ^ (x >> 6); }
DEVICE_FN u64 beta64(u64 x) { return x ^ Sig0_64(x) ^ Sig1_64(x) ^ sig0_64(x) ^ sig1_64(x) ^ rol64(x, 51); }
DEVICE_FN u64 gamma64(u64 x, u64 y) {
  return x ^ beta64(y) ^ ror64(y, 30) ^ rol64(sig1_64(x), 38) ^ (y >> 12);
}

DEVICE_FN u128 ch128(u128 x, u128 y, u128 z) { return xor128(z, and128(x, xor128(y, z))); }
DEVICE_FN u128 maj128(u128 x, u128 y, u128 z) { return or128(and128(x, y), and128(z, or128(x, y))); }
DEVICE_FN u128 Sig0_128(u128 x) { return xor128(xor128(ror128(x, 57), ror128(x, 63)), ror128(x, 92)); }
DEVICE_FN u128 Sig1_128(u128 x) { return xor128(xor128(ror128(x, 35), ror128(x, 44)), ror128(x, 51)); }
DEVICE_FN u128 sig0_128(u128 x) { return xor128(xor128(ror128(x, 42), ror128(x, 83)), shr128(x, 20)); }
DEVICE_FN u128 sig1_128(u128 x) { return xor128(xor128(ror128(x, 24), ror128(x, 119)), shr128(x, 18)); }
DEVICE_FN u128 beta128(u128 x) {
  u128 r = xor128(x, Sig0_128(x));
  r = xor128(r, Sig1_128(x));
  r = xor128(r, sig0_128(x));
  r = xor128(r, sig1_128(x));
  return xor128(r, rol128(x, 54));
}
DEVICE_FN u128 gamma128(u128 x, u128 y) {
  u128 r = xor128(x, beta128(y));
  r = xor128(r, ror128(y, 48));
  r = xor128(r, rol128(sig1_128(x), 99));
  return xor128(r, shr128(y, 22));
}

DEVICE_FN u256 ch256(u256 x, u256 y, u256 z) { return xor256(z, and256(x, xor256(y, z))); }
DEVICE_FN u256 maj256(u256 x, u256 y, u256 z) { return or256(and256(x, y), and256(z, or256(x, y))); }
DEVICE_FN u256 Sig0_256(u256 x) { return xor256(xor256(ror256(x, 113), ror256(x, 146)), ror256(x, 171)); }
DEVICE_FN u256 Sig1_256(u256 x) { return xor256(xor256(ror256(x, 88), ror256(x, 97)), ror256(x, 201)); }
DEVICE_FN u256 sig0_256(u256 x) { return xor256(xor256(ror256(x, 131), ror256(x, 57)), shr256(x, 56)); }
DEVICE_FN u256 sig1_256(u256 x) { return xor256(xor256(ror256(x, 53), ror256(x, 247)), shr256(x, 26)); }
DEVICE_FN u256 beta256(u256 x) {
  u256 r = xor256(x, Sig0_256(x));
  r = xor256(r, Sig1_256(x));
  r = xor256(r, sig0_256(x));
  r = xor256(r, sig1_256(x));
  return xor256(r, rol256(x, 81));
}
DEVICE_FN u256 gamma256(u256 x, u256 y) {
  u256 r = xor256(x, beta256(y));
  r = xor256(r, ror256(y, 58));
  r = xor256(r, rol256(sig1_256(x), 183));
  return xor256(r, shr256(y, 32));
}

// =========================================================================
// Arithmetic modulo p = 2^(64L) - C (Mod<L, C> in uint.hpp; also secp256k1's
// field with C = 2^32 + 977). Products are reduced by folding the high half
// times C twice, then one conditional subtraction - the canonical residue.
// =========================================================================

#define C_M64   59ULL
#define C_F64   1469ULL
#define C_M128  159ULL
#define C_F128  15449ULL
#define C_M256  189ULL
#define C_F256  36113ULL
#define C_SECP  0x1000003D1ULL

// x mod p for x < 2^(64L): x >= p exactly when x + C overflows.
DEVICE_FN u64 norm64(u64 x, u64 C) { u64 y = x + C; return (y < x) ? y : x; }
DEVICE_FN u128 norm128(u128 x, u64 C) {
  u128 y; y.v[0] = x.v[0] + C;
  u64 c = (u64)(y.v[0] < x.v[0]);
  y.v[1] = x.v[1] + c;
  c = (u64)(y.v[1] < c);
  return c ? y : x;
}
DEVICE_FN u256 norm256(u256 x, u64 C) {
  u256 y;
  u64 c = C;
  for (int i = 0; i < 4; i++) { y.v[i] = x.v[i] + c; c = (u64)(y.v[i] < c); }
  return c ? y : x;
}

DEVICE_FN u64 mulmod64(u64 a, u64 b, u64 C) {
  u64 lo = a * b, hi = MULHI64(a, b);
  // lo + hi*C: hi*C < 2^64 * 2^11
  u64 plo = hi * C, phi = MULHI64(hi, C);
  u64 s = lo + plo;
  u64 carry = phi + (u64)(s < plo);
  // second fold: carry*C < 2^22
  u64 s2 = s + carry * C;
  if (s2 < s) s2 += C;
  return norm64(s2, C);
}

// Reduce a 2L-limb product t modulo 2^(64L) - C. C < 2^34.
#define REDUCE_BODY(L)                                                         \
  u64 carry = 0;                                                               \
  for (int i = 0; i < (L); i++) {                                              \
    u64 plo = t[(L) + i] * C, phi = MULHI64(t[(L) + i], C);                    \
    u64 s = t[i] + plo;                                                        \
    u64 c = (u64)(s < plo);                                                    \
    s += carry;                                                                \
    c += (u64)(s < carry);                                                     \
    r[i] = s;                                                                  \
    carry = phi + c;                                                           \
  }                                                                            \
  {                                                                            \
    u64 plo = carry * C, phi = MULHI64(carry, C);                              \
    u64 s = r[0] + plo;                                                        \
    u64 c = (u64)(s < plo);                                                    \
    r[0] = s;                                                                  \
    s = r[1] + phi;                                                            \
    u64 c2 = (u64)(s < phi);                                                   \
    s += c;                                                                    \
    c2 += (u64)(s < c);                                                        \
    r[1] = s;                                                                  \
    c = c2;                                                                    \
    for (int i = 2; i < (L); i++) { r[i] += c; c = (u64)(r[i] < c); }          \
    if (c) {                                                                   \
      u64 cc = C;                                                              \
      for (int i = 0; i < (L); i++) { r[i] += cc; cc = (u64)(r[i] < cc); }     \
    }                                                                          \
  }

DEVICE_FN u128 mulmod128(u128 a, u128 b, u64 C) {
  u64 t[4] = {0, 0, 0, 0};
  for (int i = 0; i < 2; i++) {
    u64 carry = 0;
    for (int j = 0; j < 2; j++) {
      u64 lo = a.v[i] * b.v[j], hi = MULHI64(a.v[i], b.v[j]);
      u64 s = t[i + j] + lo;
      u64 c = (u64)(s < lo);
      s += carry;
      c += (u64)(s < carry);
      t[i + j] = s;
      carry = hi + c;
    }
    t[i + 2] = carry;
  }
  u64 r[2];
  REDUCE_BODY(2)
  return norm128(mk128(r[0], r[1]), C);
}

// ---- 256-bit products on 32-bit limbs ----
//
// The GPU multiplies 32x32 -> 64 natively (IMAD.WIDE); a 64-bit limb product
// costs several of those plus carry compares. So the 256-bit arithmetic -
// nearly all of BC3Mix's work - runs on eight 32-bit limbs with 64-bit
// accumulators, and squaring computes each cross product once.

DEVICE_FN void split256(u256 a, u32* x) {
  for (int i = 0; i < 4; i++) { x[2 * i] = (u32)a.v[i]; x[2 * i + 1] = (u32)(a.v[i] >> 32); }
}
DEVICE_FN u256 join256(const u32* x) {
  return mk256((u64)x[0] | ((u64)x[1] << 32), (u64)x[2] | ((u64)x[3] << 32), (u64)x[4] | ((u64)x[5] << 32),
               (u64)x[6] | ((u64)x[7] << 32));
}

#if !defined(__OPENCL_VERSION__) && defined(BC3_PTX_MUL)
// CUDA, opt-in (-DBC3_PTX_MUL): the same products with the hardware carry
// chain. MEASURED SLOWER on sm_86 (105 vs 111 kH/s): Ampere has no carry flag
// for IMAD, so ptxas emulates the chain. Kept for other architectures to try.
// One asm block per
// row, so no carry flag ever has to survive between asm statements: row i
// adds a_i * b into t[i .. i+8], where t[i+8] is still zero.
#define BC3_PTX_ROW(ai, b, t0, t1, t2, t3, t4, t5, t6, t7, t8)                 \
  asm("mad.lo.cc.u32  %0, %9, %10, %0;\n\t"                                    \
      "madc.lo.cc.u32 %1, %9, %11, %1;\n\t"                                    \
      "madc.lo.cc.u32 %2, %9, %12, %2;\n\t"                                    \
      "madc.lo.cc.u32 %3, %9, %13, %3;\n\t"                                    \
      "madc.lo.cc.u32 %4, %9, %14, %4;\n\t"                                    \
      "madc.lo.cc.u32 %5, %9, %15, %5;\n\t"                                    \
      "madc.lo.cc.u32 %6, %9, %16, %6;\n\t"                                    \
      "madc.lo.cc.u32 %7, %9, %17, %7;\n\t"                                    \
      "addc.u32       %8, 0, 0;\n\t"                                           \
      "mad.hi.cc.u32  %1, %9, %10, %1;\n\t"                                    \
      "madc.hi.cc.u32 %2, %9, %11, %2;\n\t"                                    \
      "madc.hi.cc.u32 %3, %9, %12, %3;\n\t"                                    \
      "madc.hi.cc.u32 %4, %9, %13, %4;\n\t"                                    \
      "madc.hi.cc.u32 %5, %9, %14, %5;\n\t"                                    \
      "madc.hi.cc.u32 %6, %9, %15, %6;\n\t"                                    \
      "madc.hi.cc.u32 %7, %9, %16, %7;\n\t"                                    \
      "madc.hi.u32    %8, %9, %17, %8;"                                        \
      : "+r"(t0), "+r"(t1), "+r"(t2), "+r"(t3), "+r"(t4), "+r"(t5), "+r"(t6),  \
        "+r"(t7), "=r"(t8)                                                     \
      : "r"(ai), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]),        \
        "r"(b[5]), "r"(b[6]), "r"(b[7]))

DEVICE_FN void mul8(const u32* a, const u32* b, u32* t) {
  u32 t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8, t9, t10, t11, t12, t13, t14, t15;
  BC3_PTX_ROW(a[0], b, t0, t1, t2, t3, t4, t5, t6, t7, t8);
  BC3_PTX_ROW(a[1], b, t1, t2, t3, t4, t5, t6, t7, t8, t9);
  BC3_PTX_ROW(a[2], b, t2, t3, t4, t5, t6, t7, t8, t9, t10);
  BC3_PTX_ROW(a[3], b, t3, t4, t5, t6, t7, t8, t9, t10, t11);
  BC3_PTX_ROW(a[4], b, t4, t5, t6, t7, t8, t9, t10, t11, t12);
  BC3_PTX_ROW(a[5], b, t5, t6, t7, t8, t9, t10, t11, t12, t13);
  BC3_PTX_ROW(a[6], b, t6, t7, t8, t9, t10, t11, t12, t13, t14);
  BC3_PTX_ROW(a[7], b, t7, t8, t9, t10, t11, t12, t13, t14, t15);
  t[0] = t0; t[1] = t1; t[2] = t2; t[3] = t3; t[4] = t4; t[5] = t5; t[6] = t6; t[7] = t7;
  t[8] = t8; t[9] = t9; t[10] = t10; t[11] = t11; t[12] = t12; t[13] = t13; t[14] = t14; t[15] = t15;
}
#else
// t[0..16) = a * b
DEVICE_FN void mul8(const u32* a, const u32* b, u32* t) {
  for (int i = 0; i < 16; i++) t[i] = 0;
  for (int i = 0; i < 8; i++) {
    u64 c = 0;
    for (int j = 0; j < 8; j++) {
      c += (u64)a[i] * b[j] + t[i + j];
      t[i + j] = (u32)c;
      c >>= 32;
    }
    t[i + 8] = (u32)c;
  }
}

#endif

// t[0..16) = a^2: 28 cross products doubled, plus 8 squares.
DEVICE_FN void sqr8(const u32* a, u32* t) {
  for (int i = 0; i < 16; i++) t[i] = 0;
  for (int i = 0; i < 7; i++) {
    u64 c = 0;
    for (int j = i + 1; j < 8; j++) {
      c += (u64)a[i] * a[j] + t[i + j];
      t[i + j] = (u32)c;
      c >>= 32;
    }
    t[i + 8] = (u32)c;
  }
  u32 top = 0;
  for (int k = 0; k < 16; k++) {
    u32 x = t[k];
    t[k] = (x << 1) | top;
    top = x >> 31;
  }
  u64 c = 0;
  for (int i = 0; i < 8; i++) {
    u64 sq = (u64)a[i] * a[i];
    c += (u64)t[2 * i] + (sq & 0xffffffffULL);
    t[2 * i] = (u32)c;
    c >>= 32;
    c += (u64)t[2 * i + 1] + (sq >> 32);
    t[2 * i + 1] = (u32)c;
    c >>= 32;
  }
}

// t mod 2^256 - C for C < 2^16 (M256, F256), canonical.
DEVICE_FN u256 reduce8_small(const u32* t, u32 C) {
  u32 r[8];
  u64 c = 0;
  for (int i = 0; i < 8; i++) {
    c += (u64)t[8 + i] * C + t[i];
    r[i] = (u32)c;
    c >>= 32;
  }
  // c <= C: fold it once more, then at most one more carry.
  u64 s = (u64)r[0] + c * C;
  r[0] = (u32)s;
  s >>= 32;
  for (int i = 1; i < 8; i++) { s += r[i]; r[i] = (u32)s; s >>= 32; }
  if (s) {
    s = (u64)r[0] + C;
    r[0] = (u32)s;
    s >>= 32;
    for (int i = 1; i < 8; i++) { s += r[i]; r[i] = (u32)s; s >>= 32; }
  }
  // r >= p exactly when r + C overflows 2^256.
  u32 y[8];
  s = (u64)r[0] + C;
  y[0] = (u32)s;
  s >>= 32;
  for (int i = 1; i < 8; i++) { s += r[i]; y[i] = (u32)s; s >>= 32; }
  return s ? join256(y) : join256(r);
}

// t mod p for secp256k1's p = 2^256 - (2^32 + 977), canonical.
DEVICE_FN u256 reduce8_secp(const u32* t) {
  u32 r[8];
  // hi * (2^32 + 977) = hi*977 + (hi << 32)
  u64 c = 0;
  for (int i = 0; i < 8; i++) {
    c += (u64)t[8 + i] * 977u + t[i];
    if (i > 0) c += t[8 + i - 1];
    r[i] = (u32)c;
    c >>= 32;
  }
  c += t[15];  // the top limb shifted out by << 32
  // c < 2^33: fold c * (2^32 + 977) into r.
  u64 v0 = c * 977u;
  u64 s = (u64)r[0] + (v0 & 0xffffffffULL);
  r[0] = (u32)s;
  s >>= 32;
  s += (u64)r[1] + (v0 >> 32) + (c & 0xffffffffULL);
  r[1] = (u32)s;
  s >>= 32;
  s += (u64)r[2] + (c >> 32);
  r[2] = (u32)s;
  s >>= 32;
  for (int i = 3; i < 8; i++) { s += r[i]; r[i] = (u32)s; s >>= 32; }
  if (s) {
    s = (u64)r[0] + 977u;
    r[0] = (u32)s;
    s >>= 32;
    s += (u64)r[1] + 1u;
    r[1] = (u32)s;
    s >>= 32;
    for (int i = 2; i < 8; i++) { s += r[i]; r[i] = (u32)s; s >>= 32; }
  }
  u32 y[8];
  s = (u64)r[0] + 977u;
  y[0] = (u32)s;
  s >>= 32;
  s += (u64)r[1] + 1u;
  y[1] = (u32)s;
  s >>= 32;
  for (int i = 2; i < 8; i++) { s += r[i]; y[i] = (u32)s; s >>= 32; }
  return s ? join256(y) : join256(r);
}

DEVICE_FN u256 mulmod256(u256 a, u256 b, u64 C) {
  u32 x[8], y[8], t[16];
  split256(a, x);
  split256(b, y);
  mul8(x, y, t);
  return C == C_SECP ? reduce8_secp(t) : reduce8_small(t, (u32)C);
}
DEVICE_FN u256 sqrmod256(u256 a, u64 C) {
  u32 x[8], t[16];
  split256(a, x);
  sqr8(x, t);
  return C == C_SECP ? reduce8_secp(t) : reduce8_small(t, (u32)C);
}

#define EBIT(E, n) ((u32)(((E)[(n) >> 6] >> ((n) & 63)) & 1ULL))

// a^E mod p for a fixed exponent E (FixedExponent in uint.hpp): sliding
// window of 4 bits over odd powers. E is uniform across threads, so the
// schedule branches identically in every lane.
DEVICE_BIG u64 expfix64(u64 a, CONST_PTR const u64* E, u64 C) {
  u64 odd[8];
  odd[0] = norm64(a, C);
  u64 a2 = mulmod64(odd[0], odd[0], C);
  for (int i = 1; i < 8; i++) odd[i] = mulmod64(odd[i - 1], a2, C);
  int i = 63;
  while (i >= 0 && !EBIT(E, i)) i--;
  u64 r = 1;
  int started = 0;
  while (i >= 0) {
    if (!EBIT(E, i)) { r = mulmod64(r, r, C); i--; continue; }
    int low = i >= 3 ? i - 3 : 0;
    while (!EBIT(E, low)) low++;
    u32 value = 0;
    for (int j = i; j >= low; j--) value = (value << 1) | EBIT(E, j);
    if (started) {
      for (int j = i; j >= low; j--) r = mulmod64(r, r, C);
      r = mulmod64(r, odd[value >> 1], C);
    } else {
      r = odd[value >> 1];
      started = 1;
    }
    i = low - 1;
  }
  return r;
}

DEVICE_BIG u128 expfix128(u128 a, CONST_PTR const u64* E, u64 C) {
  u128 odd[8];
  odd[0] = norm128(a, C);
  u128 a2 = mulmod128(odd[0], odd[0], C);
  for (int i = 1; i < 8; i++) odd[i] = mulmod128(odd[i - 1], a2, C);
  int i = 127;
  while (i >= 0 && !EBIT(E, i)) i--;
  u128 r = mk128(1, 0);
  int started = 0;
  while (i >= 0) {
    if (!EBIT(E, i)) { r = mulmod128(r, r, C); i--; continue; }
    int low = i >= 3 ? i - 3 : 0;
    while (!EBIT(E, low)) low++;
    u32 value = 0;
    for (int j = i; j >= low; j--) value = (value << 1) | EBIT(E, j);
    if (started) {
      for (int j = i; j >= low; j--) r = mulmod128(r, r, C);
      r = mulmod128(r, odd[value >> 1], C);
    } else {
      r = odd[value >> 1];
      started = 1;
    }
    i = low - 1;
  }
  return r;
}

DEVICE_BIG u256 expfix256(u256 a, CONST_PTR const u64* E, u64 C) {
  u256 odd[8];
  odd[0] = norm256(a, C);
  u256 a2 = sqrmod256(odd[0], C);
  for (int i = 1; i < 8; i++) odd[i] = mulmod256(odd[i - 1], a2, C);
  int i = 255;
  while (i >= 0 && !EBIT(E, i)) i--;
  u256 r = mk256(1, 0, 0, 0);
  int started = 0;
  while (i >= 0) {
    if (!EBIT(E, i)) { r = sqrmod256(r, C); i--; continue; }
    int low = i >= 3 ? i - 3 : 0;
    while (!EBIT(E, low)) low++;
    u32 value = 0;
    for (int j = i; j >= low; j--) value = (value << 1) | EBIT(E, j);
    if (started) {
      for (int j = i; j >= low; j--) r = sqrmod256(r, C);
      r = mulmod256(r, odd[value >> 1], C);
    } else {
      r = odd[value >> 1];
      started = 1;
    }
    i = low - 1;
  }
  return r;
}

// g^e = product of table[i][byte i of e] (FixedPower in uint.hpp). The
// tables come from the reference implementation (ref_pow_table).
DEVICE_FN u64 powtab64(GLOBAL const u64* tab, u64 e) {
  u64 r = tab[(e & 255u)];
  for (int i = 1; i < 8; i++) r = mulmod64(r, tab[i * 256 + ((e >> (8 * i)) & 255u)], C_F64);
  return r;
}
DEVICE_FN u128 powtab128(GLOBAL const u64* tab, u128 e) {
  u32 b = (u32)(e.v[0] & 255u);
  u128 r = mk128(tab[b * 2], tab[b * 2 + 1]);
  for (int i = 1; i < 16; i++) {
    b = (u32)((e.v[i >> 3] >> (8 * (i & 7))) & 255u);
    u32 k = (u32)(i * 256 + b) * 2u;
    r = mulmod128(r, mk128(tab[k], tab[k + 1]), C_F128);
  }
  return r;
}
DEVICE_FN u256 tab256(GLOBAL const u64* tab, u32 row, u32 b) {
  u32 k = (row * 256u + b) * 4u;
  return mk256(tab[k], tab[k + 1], tab[k + 2], tab[k + 3]);
}
DEVICE_BIG u256 powtab256(GLOBAL const u64* tab, u256 e) {
  u256 r = tab256(tab, 0, (u32)(e.v[0] & 255u));
  for (u32 i = 1; i < 32; i++)
    r = mulmod256(r, tab256(tab, i, (u32)((e.v[i >> 3] >> (8 * (i & 7))) & 255u)), C_F256);
  return r;
}

// =========================================================================
// secp256k1: field mod p, scalar multiplication from the precomputed tables
// (CurveTable::operator() in mix.cpp), affine output.
// =========================================================================

CONST_ARR u64 SECP_N[4] = {0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};
CONST_ARR u64 SECP_PM2[4] = {0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
// F256 = 2^256 - 36113; inverses by Fermat: a^(p-2).
CONST_ARR u64 F256_PM2[4] = {0xFFFFFFFFFFFF72EDULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};

DEVICE_FN int iszero256(u256 a) { return (a.v[0] | a.v[1] | a.v[2] | a.v[3]) == 0; }
DEVICE_FN int ge256(u256 a, u256 b) {
  for (int i = 3; i >= 0; i--)
    if (a.v[i] != b.v[i]) return a.v[i] > b.v[i];
  return 1;
}
DEVICE_FN u256 sub256(u256 a, u256 b) {
  u256 r;
  u64 borrow = 0;
  for (int i = 0; i < 4; i++) {
    u64 d = a.v[i] - b.v[i];
    u64 b1 = (u64)(a.v[i] < b.v[i]);
    u64 d2 = d - borrow;
    borrow = b1 | (u64)(d < borrow);
    r.v[i] = d2;
  }
  return r;
}

DEVICE_FN u256 fmul(u256 a, u256 b) { return mulmod256(a, b, C_SECP); }
DEVICE_FN u256 fsq(u256 a) { return sqrmod256(a, C_SECP); }
// a + b mod p for canonical a, b.
DEVICE_FN u256 fadd(u256 a, u256 b) {
  u256 s;
  u64 c = 0;
  for (int i = 0; i < 4; i++) {
    u64 t = a.v[i] + b.v[i];
    u64 c1 = (u64)(t < a.v[i]);
    u64 t2 = t + c;
    c = c1 | (u64)(t2 < t);
    s.v[i] = t2;
  }
  u256 y;
  u64 cc = C_SECP;
  for (int i = 0; i < 4; i++) { y.v[i] = s.v[i] + cc; cc = (u64)(y.v[i] < cc); }
  return (c | cc) ? y : s;
}
// a - b mod p for canonical a, b.
DEVICE_FN u256 fsub(u256 a, u256 b) {
  u256 d;
  u64 borrow = 0;
  for (int i = 0; i < 4; i++) {
    u64 t = a.v[i] - b.v[i];
    u64 b1 = (u64)(a.v[i] < b.v[i]);
    u64 t2 = t - borrow;
    borrow = b1 | (u64)(t < borrow);
    d.v[i] = t2;
  }
  if (borrow) {
    // a - b + p = (a - b + 2^256) - C
    u64 bb = C_SECP;
    for (int i = 0; i < 4; i++) {
      u64 t = d.v[i] - bb;
      bb = (u64)(d.v[i] < bb);
      d.v[i] = t;
    }
  }
  return d;
}

// Jacobian accumulator += affine point. The degenerate cases (equal points,
// opposite points) cannot occur for the table sums in BC3Mix - every partial
// sum is a smaller multiple of the base than the next table point - but they
// are handled exactly, as libsecp256k1's addition does.
DEVICE_FN void jac_double(u256* X, u256* Y, u256* Z) {
  u256 a = fsq(*X), b = fsq(*Y), c = fsq(b);
  u256 d = fsub(fsub(fsq(fadd(*X, b)), a), c);
  d = fadd(d, d);
  u256 e = fadd(fadd(a, a), a);
  u256 f = fsq(e);
  u256 x3 = fsub(f, fadd(d, d));
  u256 c8 = fadd(c, c); c8 = fadd(c8, c8); c8 = fadd(c8, c8);
  u256 y3 = fsub(fmul(e, fsub(d, x3)), c8);
  u256 z3 = fmul(*Y, *Z); z3 = fadd(z3, z3);
  *X = x3; *Y = y3; *Z = z3;
}
DEVICE_FN void jac_add_affine(u256* X, u256* Y, u256* Z, int* inf, u256 x2, u256 y2) {
  if (*inf) { *X = x2; *Y = y2; *Z = mk256(1, 0, 0, 0); *inf = 0; return; }
  u256 z1z1 = fsq(*Z);
  u256 u2 = fmul(x2, z1z1);
  u256 s2 = fmul(fmul(y2, *Z), z1z1);
  u256 h = fsub(u2, *X);
  u256 r = fsub(s2, *Y);
  if (iszero256(h)) {
    if (iszero256(r)) jac_double(X, Y, Z);
    else *inf = 1;
    return;
  }
  u256 hh = fsq(h);
  u256 hhh = fmul(h, hh);
  u256 v = fmul(*X, hh);
  u256 x3 = fsub(fsub(fsq(r), hhh), fadd(v, v));
  u256 y3 = fsub(fmul(r, fsub(v, x3)), fmul(*Y, hhh));
  u256 z3 = fmul(*Z, h);
  *X = x3; *Y = y3; *Z = z3;
}

DEVICE_FN u256 sqrn256(u256 a, int n, u64 C) {
  for (int i = 0; i < n; i++) a = sqrmod256(a, C);
  return a;
}

// ---- Modular inversion by safegcd (Bernstein-Yang divsteps) ----
//
// A port of libsecp256k1's constant-time modinv32 (src/modinv32_impl.h,
// MIT, Peter Dettman / Pieter Wuille): 20 batches of 30 divsteps on signed
// 30-bit limbs. Constant-time here means every lane of a warp takes the same
// path. It replaces Fermat's 255 squarings with about a quarter of the
// instructions, and the inverse is the same unique residue.
typedef struct { i32 v[9]; } s30;

#define M30 0x3FFFFFFF

DEVICE_FN s30 to_s30(u256 x) {
  s30 r;
  for (int i = 0; i < 9; i++) {
    u32 o = 30u * (u32)i, limb = o >> 6, sh = o & 63u;
    u64 val = x.v[limb] >> sh;
    if (sh > 34u && limb + 1u < 4u) val |= x.v[limb + 1u] << (64u - sh);
    r.v[i] = (i32)(val & (i < 8 ? (u64)M30 : 0xFFFFULL));
  }
  return r;
}

// Limbs in [0, 2^30) (the normalized output).
DEVICE_FN u256 from_s30(s30 a) {
  u256 r = mk256(0, 0, 0, 0);
  for (int i = 0; i < 9; i++) {
    u64 val = (u64)(u32)a.v[i];
    u32 o = 30u * (u32)i, limb = o >> 6, sh = o & 63u;
    r.v[limb] |= val << sh;
    if (sh > 34u && limb + 1u < 4u) r.v[limb + 1u] |= val >> (64u - sh);
  }
  return r;
}

// 30 divsteps on the bottom limbs; the transition matrix [u v; q r] scaled by 2^30.
DEVICE_FN i32 divsteps_30(i32 zeta, u32 f0, u32 g0, i32* tu, i32* tv, i32* tq, i32* tr) {
  u32 u = 1, v = 0, q = 0, r = 1, f = f0, g = g0;
  for (int i = 0; i < 30; i++) {
    u32 mask1 = (u32)(zeta >> 31);
    u32 mask2 = 0u - (g & 1u);
    u32 x = (f ^ mask1) - mask1;
    u32 y = (u ^ mask1) - mask1;
    u32 z = (v ^ mask1) - mask1;
    g += x & mask2;
    q += y & mask2;
    r += z & mask2;
    mask1 &= mask2;
    zeta = (i32)(((u32)zeta ^ mask1) - 1u);
    f += g & mask1;
    u += q & mask1;
    v += r & mask1;
    g >>= 1;
    u <<= 1;
    v <<= 1;
  }
  *tu = (i32)u;
  *tv = (i32)v;
  *tq = (i32)q;
  *tr = (i32)r;
  return zeta;
}

// (t / 2^30) * [d, e] mod the modulus; d, e stay in (-2*modulus, modulus).
DEVICE_FN void update_de_30(s30* d, s30* e, i32 u, i32 v, i32 q, i32 r, CONST_PTR const i32* M, u32 inv30) {
  i32 sd = d->v[8] >> 31, se = e->v[8] >> 31;
  i32 md = (u & sd) + (v & se);
  i32 me = (q & sd) + (r & se);
  i32 di = d->v[0], ei = e->v[0];
  i64 cd = (i64)u * di + (i64)v * ei;
  i64 ce = (i64)q * di + (i64)r * ei;
  md = (i32)((u32)md - ((inv30 * (u32)cd + (u32)md) & (u32)M30));
  me = (i32)((u32)me - ((inv30 * (u32)ce + (u32)me) & (u32)M30));
  cd += (i64)M[0] * md;
  ce += (i64)M[0] * me;
  cd >>= 30;
  ce >>= 30;
  for (int i = 1; i < 9; i++) {
    di = d->v[i];
    ei = e->v[i];
    cd += (i64)u * di + (i64)v * ei;
    ce += (i64)q * di + (i64)r * ei;
    cd += (i64)M[i] * md;
    ce += (i64)M[i] * me;
    d->v[i - 1] = (i32)cd & M30;
    cd >>= 30;
    e->v[i - 1] = (i32)ce & M30;
    ce >>= 30;
  }
  d->v[8] = (i32)cd;
  e->v[8] = (i32)ce;
}

// (t / 2^30) * [f, g].
DEVICE_FN void update_fg_30(s30* f, s30* g, i32 u, i32 v, i32 q, i32 r) {
  i32 fi = f->v[0], gi = g->v[0];
  i64 cf = (i64)u * fi + (i64)v * gi;
  i64 cg = (i64)q * fi + (i64)r * gi;
  cf >>= 30;
  cg >>= 30;
  for (int i = 1; i < 9; i++) {
    fi = f->v[i];
    gi = g->v[i];
    cf += (i64)u * fi + (i64)v * gi;
    cg += (i64)q * fi + (i64)r * gi;
    f->v[i - 1] = (i32)cf & M30;
    cf >>= 30;
    g->v[i - 1] = (i32)cg & M30;
    cg >>= 30;
  }
  f->v[8] = (i32)cf;
  g->v[8] = (i32)cg;
}

// r from (-2*modulus, modulus), negated if sign < 0, to [0, modulus).
DEVICE_FN void normalize_30(s30* r, i32 sign, CONST_PTR const i32* M) {
  i32 cond_add = r->v[8] >> 31;
  for (int i = 0; i < 9; i++) r->v[i] += M[i] & cond_add;
  i32 cond_negate = sign >> 31;
  for (int i = 0; i < 9; i++) r->v[i] = (r->v[i] ^ cond_negate) - cond_negate;
  for (int i = 0; i < 8; i++) { r->v[i + 1] += r->v[i] >> 30; r->v[i] &= M30; }
  cond_add = r->v[8] >> 31;
  for (int i = 0; i < 9; i++) r->v[i] += M[i] & cond_add;
  for (int i = 0; i < 8; i++) { r->v[i + 1] += r->v[i] >> 30; r->v[i] &= M30; }
}

// x^-1 mod the modulus, x in [0, modulus); 0 gives 0.
DEVICE_BIG u256 modinv256(u256 x, CONST_PTR const i32* M, u32 inv30) {
  s30 d, e, f, g;
  for (int i = 0; i < 9; i++) { d.v[i] = 0; e.v[i] = 0; f.v[i] = M[i]; }
  e.v[0] = 1;
  g = to_s30(x);
  i32 zeta = -1;
  for (int i = 0; i < 20; i++) {
    i32 tu, tv, tq, tr;
    zeta = divsteps_30(zeta, (u32)f.v[0], (u32)g.v[0], &tu, &tv, &tq, &tr);
    update_de_30(&d, &e, tu, tv, tq, tr, M, inv30);
    update_fg_30(&f, &g, tu, tv, tq, tr);
  }
  normalize_30(&d, f.v[8], M);
  return from_s30(d);
}

// a^(p-2) mod secp256k1's p: libsecp256k1's addition chain (secp256k1_fe_inv),
// 255 squarings and 15 multiplications. a = 0 gives 0.
DEVICE_BIG u256 inv_secp_fermat(u256 a) {
  u256 x2 = fmul(fsq(a), a);
  u256 x3 = fmul(fsq(x2), a);
  u256 x6 = fmul(sqrn256(x3, 3, C_SECP), x3);
  u256 x9 = fmul(sqrn256(x6, 3, C_SECP), x3);
  u256 x11 = fmul(sqrn256(x9, 2, C_SECP), x2);
  u256 x22 = fmul(sqrn256(x11, 11, C_SECP), x11);
  u256 x44 = fmul(sqrn256(x22, 22, C_SECP), x22);
  u256 x88 = fmul(sqrn256(x44, 44, C_SECP), x44);
  u256 x176 = fmul(sqrn256(x88, 88, C_SECP), x88);
  u256 x220 = fmul(sqrn256(x176, 44, C_SECP), x44);
  u256 x223 = fmul(sqrn256(x220, 3, C_SECP), x3);
  u256 t = fmul(sqrn256(x223, 23, C_SECP), x22);
  t = fmul(sqrn256(t, 5, C_SECP), a);
  t = fmul(sqrn256(t, 3, C_SECP), x2);
  return fmul(sqrn256(t, 2, C_SECP), a);
}

// a^(p-2) mod p = 2^256 - 36113 (inverse256 in mix.cpp; 0 gives 0).
// p - 2 = (2^240 - 1) * 2^16 + 0x72ED: build a^(2^240-1) from blocks of
// ones, then the low 16 bits 0111 0010 1110 1101 with a^7, a^3 and a.
// 255 squarings, 14 multiplications.
DEVICE_BIG u256 inv_f256_fermat(u256 a) {
  a = norm256(a, C_F256);
  u256 x2 = mulmod256(sqrmod256(a, C_F256), a, C_F256);
  u256 x3 = mulmod256(sqrmod256(x2, C_F256), a, C_F256);
  u256 x6 = mulmod256(sqrn256(x3, 3, C_F256), x3, C_F256);
  u256 x12 = mulmod256(sqrn256(x6, 6, C_F256), x6, C_F256);
  u256 x15 = mulmod256(sqrn256(x12, 3, C_F256), x3, C_F256);
  u256 x30 = mulmod256(sqrn256(x15, 15, C_F256), x15, C_F256);
  u256 x60 = mulmod256(sqrn256(x30, 30, C_F256), x30, C_F256);
  u256 x120 = mulmod256(sqrn256(x60, 60, C_F256), x60, C_F256);
  u256 r = mulmod256(sqrn256(x120, 120, C_F256), x120, C_F256);  // a^(2^240 - 1)
  r = mulmod256(sqrn256(r, 4, C_F256), x3, C_F256);                // 0111
  r = mulmod256(sqrn256(r, 3, C_F256), a, C_F256);                 // 001
  r = sqrmod256(r, C_F256);                                         //    0
  r = mulmod256(sqrn256(r, 3, C_F256), x3, C_F256);                // 111
  r = sqrmod256(r, C_F256);                                         //    0
  r = mulmod256(sqrn256(r, 2, C_F256), x2, C_F256);                // 11
  r = mulmod256(sqrn256(r, 2, C_F256), a, C_F256);                 //   01
  return r;
}

// The inversions used by BC3Mix: safegcd. The Fermat chains above stay as
// the slower reference they were measured against.
DEVICE_FN u256 inv_secp(u256 a) { return modinv256(a, BC3_MODINV_SECP, BC3_MODINV_SECP_INV30); }
DEVICE_FN u256 inv_f256(u256 a) { return modinv256(norm256(a, C_F256), BC3_MODINV_F256, BC3_MODINV_F256_INV30); }

// The table sum of curve_multiply(point, k) in Jacobian coordinates.
// *invalid is set when k = 0 mod n - the reference then returns the fixed
// "invalid point" constants - and Z is set to 1 so a batch inversion can
// include it harmlessly.
DEVICE_BIG void curve_sum(GLOBAL const u64* tab, u256 k, u256* X, u256* Y, u256* Z, int* invalid) {
  u256 n = c256(SECP_N);
  if (ge256(k, n)) k = sub256(k, n);
  *invalid = iszero256(k);
  *X = mk256(0, 0, 0, 0);
  *Y = *X;
  *Z = mk256(1, 0, 0, 0);
  int inf = 1;
  for (u32 i = 0; i < BC3_DIGITS; i++) {
    u32 pos = i * BC3_WINDOW, limb = pos >> 6, shift = pos & 63u;
    u64 b = k.v[limb] >> shift;
    if (shift && limb + 1 < 4) b |= k.v[limb + 1] << (64u - shift);
    b &= (u64)BC3_ENTRIES;
    if (b) {
      GLOBAL const u64* p = tab + ((u64)(i * BC3_ENTRIES) + b - 1) * 8u;
      jac_add_affine(X, Y, Z, &inf, mk256(p[0], p[1], p[2], p[3]), mk256(p[4], p[5], p[6], p[7]));
    }
  }
  // k in [1, n-1]: the sum is never the point at infinity.
}

DEVICE_FN void jac_to_affine(u256 X, u256 Y, u256 zi, int invalid, u256* ox, u256* oy) {
  if (invalid) { *ox = c256(BC3_INVALID_X); *oy = c256(BC3_INVALID_Y); return; }
  u256 zi2 = fsq(zi);
  *ox = fmul(X, zi2);
  *oy = fmul(fmul(Y, zi2), zi);
}

// curve_multiply(point, k): (x, y) of k*P from the table of P.
DEVICE_BIG void curve_mul(GLOBAL const u64* tab, u256 k, u256* ox, u256* oy) {
  u256 X, Y, Z;
  int invalid;
  curve_sum(tab, k, &X, &Y, &Z, &invalid);
  jac_to_affine(X, Y, inv_secp(Z), invalid, ox, oy);
}

// Four scalar multiplications sharing ONE field inversion (Montgomery's
// trick): 1 inversion + 9 multiplications instead of 4 inversions.
DEVICE_BIG void curve_mul4(GLOBAL const u64* tc, u256 k0, u256 k1, u256 k2, u256 k3, u256* out) {
  u256 X[4], Y[4], Z[4];
  int inv[4];
  curve_sum(tc + 0 * (u64)BC3_CURVE_POINTS * 8u, k0, &X[0], &Y[0], &Z[0], &inv[0]);
  curve_sum(tc + 1 * (u64)BC3_CURVE_POINTS * 8u, k1, &X[1], &Y[1], &Z[1], &inv[1]);
  curve_sum(tc + 2 * (u64)BC3_CURVE_POINTS * 8u, k2, &X[2], &Y[2], &Z[2], &inv[2]);
  curve_sum(tc + 3 * (u64)BC3_CURVE_POINTS * 8u, k3, &X[3], &Y[3], &Z[3], &inv[3]);
  u256 p01 = fmul(Z[0], Z[1]);
  u256 p012 = fmul(p01, Z[2]);
  u256 all = fmul(p012, Z[3]);
  u256 t = inv_secp(all);           // (Z0 Z1 Z2 Z3)^-1
  u256 z3i = fmul(t, p012);
  t = fmul(t, Z[3]);                // (Z0 Z1 Z2)^-1
  u256 z2i = fmul(t, p01);
  t = fmul(t, Z[2]);                // (Z0 Z1)^-1
  u256 z1i = fmul(t, Z[0]);
  u256 z0i = fmul(t, Z[1]);
  jac_to_affine(X[0], Y[0], z0i, inv[0], &out[0], &out[1]);
  jac_to_affine(X[1], Y[1], z1i, inv[1], &out[2], &out[3]);
  jac_to_affine(X[2], Y[2], z2i, inv[2], &out[4], &out[5]);
  jac_to_affine(X[3], Y[3], z3i, inv[3], &out[6], &out[7]);
}

// =========================================================================
// BC3Mix of 64 bytes (mix512 with folded = 0).
// =========================================================================

DEVICE_FN u64 load_be64(const u8* p) {
  u64 v = 0;
  for (int i = 0; i < 8; i++) v = (v << 8) | (u64)p[i];
  return v;
}

DEVICE_BIG void bc3mix512(GLOBAL const u64* tables, const u8* in, u8* out) {
  GLOBAL const u64* T64 = tables + BC3_TABLE_POW64;
  GLOBAL const u64* T128 = tables + BC3_TABLE_POW128;
  GLOBAL const u64* T256 = tables + BC3_TABLE_POW256;
  GLOBAL const u64* TC = tables + BC3_TABLE_CURVE;

  // ---- message expansion ----
  u64 m64[32];
  for (int i = 0; i < 8; i++) m64[i] = load_be64(in + 8 * i);
  u64 c1 = m64[0] ^ BC3_BASE[0], c2 = m64[1] ^ BC3_BASE[1], c3 = m64[2] ^ BC3_BASE[2], c4 = m64[3] ^ BC3_BASE[3];
  u64 c5 = m64[4] ^ BC3_BASE[4], c6 = m64[5] ^ BC3_BASE[5], c7 = m64[6] ^ BC3_BASE[6], c8 = m64[7] ^ BC3_BASE[7];
  for (int i = 8; i < 32; i++)
    m64[i] = gamma64(m64[i - 8], m64[i - 4]) + gamma64(m64[i - 7], m64[i - 3]) + gamma64(m64[i - 6], m64[i - 2]) +
             gamma64(m64[i - 5], m64[i - 1]);
  u128 m128[24];  // m128[i] of the reference is m128[i - 24] here
  for (int i = 24; i < 32; i++) m128[i - 24] = mk128(m64[i], 0);
  for (int i = 32; i < 48; i++)
    m128[i - 24] = add128(add128(beta128(m128[i - 32]), beta128(m128[i - 31])),
                          add128(beta128(m128[i - 28]), beta128(m128[i - 27])));
  u256 m256[23];  // m256[i] of the reference is m256[i - 41] here
  for (int i = 41; i < 48; i++) m256[i - 41] = mk256(m128[i - 24].v[0], m128[i - 24].v[1], 0, 0);
  for (int i = 48; i < 64; i++)
    m256[i - 41] = add256(add256(beta256(m256[i - 48]), beta256(m256[i - 47])),
                          add256(beta256(m256[i - 43]), beta256(m256[i - 42])));

  // ---- stage 1: 64-bit lanes ----
  for (int i = 0; i < 32; i++) {
    u64 p1 = expfix64(c1 ^ sig0_64(c2), BC3_E64, C_M64);
    u64 p2 = expfix64(c3 ^ sig1_64(c4), BC3_E64, C_M64);
    u64 d1 = p1 + ror64(c6, 7) + rol64(c5, 13);
    u64 d2 = p2 + ch64(c5, c6, c7) + maj64(c1, c4, c8);
    u64 d3 = c8 + Sig0_64(c7) + BC3_ROUND64[i];
    u64 d4 = c5 + Sig1_64(c1) + d1 + m64[i] + powtab64(T64, d3);
    u64 n8 = c7;
    u64 n7 = c6 + d1 + sig0_64(mulmod64(d1, d3, C_M64));
    u64 n6 = c5 + sig1_64(d2 ^ d3);
    u64 n1 = gamma64(d1, d2) + (d3 ^ d4) + beta64(mulmod64(d2, d4, C_M64));
    c8 = n8; c7 = n7; c6 = n6;
    c5 = c4;
    c4 = c3 + d4;
    c3 = c2;
    c2 = c1 ^ d3;
    c1 = n1;
  }

  // ---- stage 2: 128-bit lanes ----
  u128 f1 = mk128(c5, c1), f2 = mk128(c6, c2), f3 = mk128(c7, c3), f4 = mk128(c8, c4);
  for (int i = 0; i < 16; i++) {
    u128 pa = expfix128(xor128(f1, beta128(f3)), BC3_E128, C_M128);
    u128 pb = expfix128(xor128(f2, beta128(f4)), BC3_E128, C_M128);
    u128 a = add128(add128(pa, sig0_128(xor128(f2, f4))), ch128(f2, f3, f4));
    u128 b = add128(add128(pb, sig1_128(xor128(f1, f3))), maj128(f1, f2, f3));
    u128 c = add128(add128(rol128(not128(f1), 29), ror128(not128(f2), 7)), f3);
    c = add128(add128(c, not128(f4)), m128[8 + i]);
    c = add128(c, powtab128(T128, f3));
    u128 d = add128(add128(rol128(not128(f3), 29), ror128(f4, 7)), f2);
    d = add128(add128(d, not128(f1)), mk128(BC3_ROUND128[2 * i], BC3_ROUND128[2 * i + 1]));
    u128 nf4 = add128(f3, a);
    u128 nf2 = add128(add128(f1, c), beta128(d));
    u128 nf1 = add128(b, Sig0_128(xor128(xor128(a, c), d)));
    f4 = nf4;
    f3 = f2;
    f2 = nf2;
    f1 = nf1;
  }

  // ---- stage 3: 256-bit lanes and the curves ----
  u256 h1 = mk256(f3.v[0], f3.v[1], f1.v[0], f1.v[1]);
  u256 h2 = mk256(f4.v[0], f4.v[1], f2.v[0], f2.v[1]);
  for (int i = 0; i < 16; i++) {
    u256 p1 = expfix256(xor256(rol256(h1, 11), h2), BC3_E256, C_M256);
    u256 p2 = expfix256(xor256(h1, beta256(not256(h2))), BC3_E256, C_M256);
    u256 j1 = add256(p1, beta256(xor256(h1, h2)));
    u256 j2 = add256(p2, gamma256(h1, h2));
    u256 q1 = expfix256(j1, BC3_E256, C_M256);
    u256 q2 = expfix256(j2, BC3_E256, C_M256);
    u256 j3 = add256(add256(gamma256(j1, j2), q1), c256(BC3_ROUND256 + 4 * i));
    u256 j4 = add256(add256(gamma256(j3, beta256(h2)), q2), add256(m256[7 + i], ch256(j1, j2, j3)));
    u256 pts[8];
    curve_mul4(TC, h1, h2, j1, j3, pts);
    u256 j5 = pts[0], j6 = pts[1], j7 = pts[2], j8 = pts[3], j9 = pts[4], j10 = pts[5], j11 = pts[6], j12 = pts[7];
    u256 nh2 = add256(h1, gamma256(j1, j3));
    nh2 = add256(nh2, maj256(j2, j3, j4));
    nh2 = add256(nh2, j5);
    nh2 = add256(nh2, j8);
    nh2 = add256(nh2, ch256(j1, j6, j7));
    nh2 = add256(nh2, beta256(j10));
    nh2 = add256(nh2, j11);
    u256 nh1 = add256(beta256(j2), gamma256(j3, j4));
    nh1 = add256(nh1, Sig1_256(mulmod256(j1, j2, C_M256)));
    nh1 = add256(nh1, maj256(j7, j6, j5));
    nh1 = add256(nh1, Sig0_256(j9));
    nh1 = add256(nh1, j12);
    h1 = nh1;
    h2 = nh2;
  }

  // ---- stage 4: table powers and inversions mod 2^256 - 36113 ----
  u256 u1 = h1, u2 = h2;
  for (int i = 0; i < 64; i++) {
    u256 fc = c256(BC3_FINAL + 4 * i);
    u256 k1 = add256(powtab256(T256, u1), beta256(u2));
    u256 k2 = add256(add256(powtab256(T256, u2), gamma256(u1, u2)), ch256(k1, u2, u1));
    u256 k3 = add256(add256(inv_f256(k2), beta256(u1)), beta256(xor256(xor256(u2, k1), fc)));
    u256 nu2 = add256(add256(xor256(k1, u1), maj256(k1, k2, u2)), beta256(k3));
    u256 nu1 = add256(beta256(k1), k2);
    u2 = nu2;
    u1 = nu1;
  }
  u256 res = add256(u1, beta256(u2));
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 8; j++) out[i * 8 + j] = (u8)(res.v[3 - i] >> (56 - 8 * j));
}

// =========================================================================
// Standard hashes (standard_hashes.cpp), one-shot over private buffers.
// =========================================================================

CONST_ARR u64 KECCAK_RC[24] = {
  0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
  0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
  0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
  0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
  0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
  0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
// rho offset of lane x + 5y, and the pi destination of lane x + 5y.
CONST_ARR u32 KECCAK_RHO[25] = {0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14};
CONST_ARR u32 KECCAK_PI[25] = {0, 10, 20, 5, 15, 16, 1, 11, 21, 6, 7, 17, 2, 12, 22, 23, 8, 18, 3, 13, 14, 24, 9, 19, 4};

DEVICE_BIG void keccakf(u64* a) {
  for (int round = 0; round < 24; round++) {
    u64 c[5], b[25];
    for (int x = 0; x < 5; x++) c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
    for (int y = 0; y < 5; y++)
      for (int x = 0; x < 5; x++) {
        u64 t = a[x + 5 * y] ^ c[(x + 4) % 5] ^ rol64(c[(x + 1) % 5], 1);
        b[KECCAK_PI[x + 5 * y]] = rol64(t, KECCAK_RHO[x + 5 * y]);
      }
    for (int y = 0; y < 5; y++)
      for (int x = 0; x < 5; x++)
        a[x + 5 * y] = b[x + 5 * y] ^ ((~b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]);
    a[0] ^= KECCAK_RC[round];
  }
}

// SHA3 with output `outlen` bytes (rate = 200 - 2*outlen).
DEVICE_BIG void sha3(const u8* msg, u32 len, u32 outlen, u8* out) {
  u64 a[25];
  for (int i = 0; i < 25; i++) a[i] = 0;
  u32 rate = 200u - 2u * outlen, pos = 0;
  for (u32 i = 0; i < len; i++) {
    a[pos >> 3] ^= (u64)msg[i] << (8u * (pos & 7u));
    if (++pos == rate) { keccakf(a); pos = 0; }
  }
  a[pos >> 3] ^= 6ULL << (8u * (pos & 7u));
  a[rate / 8u - 1u] ^= 0x80ULL << 56;
  keccakf(a);
  for (u32 i = 0; i < outlen; i++) out[i] = (u8)(a[i >> 3] >> (8u * (i & 7u)));
}

CONST_ARR u64 SHA512_K[80] = {
  0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL,
  0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL, 0x12835b0145706fbeULL,
  0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL, 0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL,
  0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
  0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL, 0x983e5152ee66dfabULL,
  0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
  0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL,
  0x53380d139d95b3dfULL, 0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
  0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL,
  0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL, 0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL,
  0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL,
  0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
  0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL, 0xca273eceea26619cULL,
  0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL,
  0x113f9804bef90daeULL, 0x1b710b35131c471bULL, 0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL,
  0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL};
CONST_ARR u64 SHA512_IV[8] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
                              0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
CONST_ARR u32 BLAKE2S_IV[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

DEVICE_BIG void sha512_block(u64* h, const u8* blk) {
  u64 w[80];
  for (int i = 0; i < 16; i++) w[i] = load_be64(blk + 8 * i);
  for (int i = 16; i < 80; i++) {
    u64 a = w[i - 15], b = w[i - 2];
    w[i] = w[i - 16] + w[i - 7] + (ror64(a, 1) ^ ror64(a, 8) ^ (a >> 7)) + (ror64(b, 19) ^ ror64(b, 61) ^ (b >> 6));
  }
  u64 s[8];
  for (int i = 0; i < 8; i++) s[i] = h[i];
  for (int i = 0; i < 80; i++) {
    u64 head = s[7] + (ror64(s[4], 14) ^ ror64(s[4], 18) ^ ror64(s[4], 41)) + (s[6] ^ (s[4] & (s[5] ^ s[6]))) +
               SHA512_K[i] + w[i];
    u64 extra = (ror64(s[0], 28) ^ ror64(s[0], 34) ^ ror64(s[0], 39)) + ((s[0] & s[1]) | (s[2] & (s[0] | s[1])));
    for (int j = 7; j > 0; j--) s[j] = s[j - 1];
    s[4] += head;
    s[0] = head + extra;
  }
  for (int i = 0; i < 8; i++) h[i] += s[i];
}

// len < 2^61; at most two final blocks.
DEVICE_BIG void sha512(const u8* msg, u32 len, u8* out) {
  u64 h[8];
  for (int i = 0; i < 8; i++) h[i] = SHA512_IV[i];
  u32 done = 0;
  while (len - done >= 128) { sha512_block(h, msg + done); done += 128; }
  u8 blk[128];
  u32 rest = len - done;
  for (u32 i = 0; i < 128; i++) blk[i] = 0;
  for (u32 i = 0; i < rest; i++) blk[i] = msg[done + i];
  blk[rest] = 0x80;
  if (rest + 1 > 112) {
    sha512_block(h, blk);
    for (u32 i = 0; i < 128; i++) blk[i] = 0;
  }
  u64 bits = (u64)len << 3;
  for (int i = 0; i < 8; i++) blk[120 + i] = (u8)(bits >> (56 - 8 * i));
  sha512_block(h, blk);
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++) out[8 * i + j] = (u8)(h[i] >> (56 - 8 * j));
}

DEVICE_FN u32 rol32(u32 x, u32 n) { return (x << n) | (x >> ((32u - n) & 31u)); }
DEVICE_FN u32 ror32(u32 x, u32 n) { return (x >> n) | (x << ((32u - n) & 31u)); }
DEVICE_FN u32 load_le32(const u8* p) { return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24); }
DEVICE_FN u64 load_le64(const u8* p) { return (u64)load_le32(p) | ((u64)load_le32(p + 4) << 32); }

CONST_ARR u8 RMD_WORD[2][80] = {
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
   3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12, 1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
   4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13},
  {5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, 6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
   15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13, 8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
   12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11}};
CONST_ARR u8 RMD_ROT[2][80] = {
  {11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8, 7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
   11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5, 11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
   9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6},
  {8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6, 9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
   9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5, 15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
   8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11}};
CONST_ARR u32 RMD_K[2][5] = {{0u, 0x5a827999u, 0x6ed9eba1u, 0x8f1bbcdcu, 0xa953fd4eu},
                             {0x50a28be6u, 0x5c4dd124u, 0x6d703ef3u, 0x7a6d76e9u, 0u}};

DEVICE_FN u32 rmd_f(u32 round, u32 x, u32 y, u32 z) {
  switch (round) {
  case 0: return x ^ y ^ z;
  case 1: return z ^ (x & (y ^ z));
  case 2: return (x | ~y) ^ z;
  case 3: return y ^ (z & (x ^ y));
  default: return x ^ (y | ~z);
  }
}

DEVICE_BIG void ripemd160_block(u32* chain, const u8* blk) {
  u32 m[16];
  for (int i = 0; i < 16; i++) m[i] = load_le32(blk + 4 * i);
  u32 s[2][5];
  for (int p = 0; p < 2; p++)
    for (int i = 0; i < 5; i++) s[p][i] = chain[i];
  for (u32 i = 0; i < 80; i++)
    for (u32 p = 0; p < 2; p++) {
      u32 round = p ? 4u - i / 16u : i / 16u;
      u32 sum = s[p][0] + rmd_f(round, s[p][1], s[p][2], s[p][3]) + m[RMD_WORD[p][i]] + RMD_K[p][i / 16u];
      u32 n0 = s[p][4], n1 = rol32(sum, RMD_ROT[p][i]) + s[p][4], n2 = s[p][1], n3 = rol32(s[p][2], 10), n4 = s[p][3];
      s[p][0] = n0; s[p][1] = n1; s[p][2] = n2; s[p][3] = n3; s[p][4] = n4;
    }
  u32 before[5];
  for (int i = 0; i < 5; i++) before[i] = chain[i];
  for (int i = 0; i < 5; i++) chain[i] = before[(i + 1) % 5] + s[0][(i + 2) % 5] + s[1][(i + 3) % 5];
}

DEVICE_BIG void ripemd160(const u8* msg, u32 len, u8* out) {
  u32 h[5] = {0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u, 0xc3d2e1f0u};
  u32 done = 0;
  while (len - done >= 64) { ripemd160_block(h, msg + done); done += 64; }
  u8 blk[64];
  u32 rest = len - done;
  for (u32 i = 0; i < 64; i++) blk[i] = 0;
  for (u32 i = 0; i < rest; i++) blk[i] = msg[done + i];
  blk[rest] = 0x80;
  if (rest + 1 > 56) {
    ripemd160_block(h, blk);
    for (u32 i = 0; i < 64; i++) blk[i] = 0;
  }
  u64 bits = (u64)len << 3;
  for (int i = 0; i < 8; i++) blk[56 + i] = (u8)(bits >> (8 * i));
  ripemd160_block(h, blk);
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 4; j++) out[4 * i + j] = (u8)(h[i] >> (8 * j));
}

CONST_ARR u8 BLAKE2_SIGMA[10][16] = {
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
  {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
  {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
  {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
  {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0}};

DEVICE_BIG void blake2s_block(u32* h, const u8* blk, u32 counter, int last) {
  u32 m[16], v[16];
  for (int i = 0; i < 16; i++) m[i] = load_le32(blk + 4 * i);
  for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = BLAKE2S_IV[i]; }
  v[12] ^= counter;
  if (last) v[14] = ~v[14];
  for (int round = 0; round < 10; round++)
    for (int dia = 0; dia < 2; dia++)
      for (int col = 0; col < 4; col++) {
        int ia = col, ib = 4 + (col + dia) % 4, ic = 8 + (col + 2 * dia) % 4, id = 12 + (col + 3 * dia) % 4;
        int pr = 8 * dia + 2 * col;
        v[ia] += v[ib] + m[BLAKE2_SIGMA[round][pr]];
        v[id] = ror32(v[id] ^ v[ia], 16);
        v[ic] += v[id];
        v[ib] = ror32(v[ib] ^ v[ic], 12);
        v[ia] += v[ib] + m[BLAKE2_SIGMA[round][pr + 1]];
        v[id] = ror32(v[id] ^ v[ia], 8);
        v[ic] += v[id];
        v[ib] = ror32(v[ib] ^ v[ic], 7);
      }
  for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

// Unkeyed BLAKE2s-256 with an 8-byte personalization; len > 0.
DEVICE_BIG void blake2s(CONST_PTR const u8* person, const u8* msg, u32 len, u8* out) {
  u32 h[8];
  for (int i = 0; i < 8; i++) h[i] = BLAKE2S_IV[i];
  h[0] ^= 0x01010000u + 32u;
  h[6] ^= (u32)person[0] | ((u32)person[1] << 8) | ((u32)person[2] << 16) | ((u32)person[3] << 24);
  h[7] ^= (u32)person[4] | ((u32)person[5] << 8) | ((u32)person[6] << 16) | ((u32)person[7] << 24);
  u32 done = 0;
  while (len - done > 64) { blake2s_block(h, msg + done, done + 64u, 0); done += 64; }
  u8 blk[64];
  for (u32 i = 0; i < 64; i++) blk[i] = 0;
  for (u32 i = 0; i < len - done; i++) blk[i] = msg[done + i];
  blake2s_block(h, blk, len, 1);
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 4; j++) out[4 * i + j] = (u8)(h[i] >> (8 * j));
}

DEVICE_BIG void blake2b_block(u64* h, const u8* blk, u64 counter, int last) {
  u64 m[16], v[16];
  for (int i = 0; i < 16; i++) m[i] = load_le64(blk + 8 * i);
  for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = SHA512_IV[i]; }
  v[12] ^= counter;
  if (last) v[14] = ~v[14];
  for (int round = 0; round < 12; round++)
    for (int dia = 0; dia < 2; dia++)
      for (int col = 0; col < 4; col++) {
        int ia = col, ib = 4 + (col + dia) % 4, ic = 8 + (col + 2 * dia) % 4, id = 12 + (col + 3 * dia) % 4;
        int pr = 8 * dia + 2 * col;
        v[ia] += v[ib] + m[BLAKE2_SIGMA[round % 10][pr]];
        v[id] = ror64(v[id] ^ v[ia], 32);
        v[ic] += v[id];
        v[ib] = ror64(v[ib] ^ v[ic], 24);
        v[ia] += v[ib] + m[BLAKE2_SIGMA[round % 10][pr + 1]];
        v[id] = ror64(v[id] ^ v[ia], 16);
        v[ic] += v[id];
        v[ib] = ror64(v[ib] ^ v[ic], 63);
      }
  for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

// Unkeyed BLAKE2b-512 with a 16-byte personalization; len > 0.
DEVICE_BIG void blake2b(CONST_PTR const u8* person, const u8* msg, u32 len, u8* out) {
  u64 h[8];
  for (int i = 0; i < 8; i++) h[i] = SHA512_IV[i];
  h[0] ^= 0x01010000ULL + 64ULL;
  u64 p0 = 0, p1 = 0;
  for (int i = 0; i < 8; i++) { p0 |= (u64)person[i] << (8 * i); p1 |= (u64)person[8 + i] << (8 * i); }
  h[6] ^= p0;
  h[7] ^= p1;
  u32 done = 0;
  while (len - done > 128) { blake2b_block(h, msg + done, (u64)done + 128u, 0); done += 128; }
  u8 blk[128];
  for (u32 i = 0; i < 128; i++) blk[i] = 0;
  for (u32 i = 0; i < len - done; i++) blk[i] = msg[done + i];
  blake2b_block(h, blk, len, 1);
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++) out[8 * i + j] = (u8)(h[i] >> (8 * j));
}

// =========================================================================
// BC3Hash (hash.cpp): domain-tagged standard hashes around BC3Mix.
// =========================================================================

CONST_ARR u8 BC3_DOMAIN[47] = {'B', 'C', '3', 'H', 'a', 's', 'h', '-', 'B', 'l', 'o', 'c', 'k', 'H', 'a', 's',
                               'h', '-', 'A', 'l', 'g', 'o', 'r', 'i', 't', 'h', 'm', '/', 'B', 'i', 't', 'c',
                               'o', 'i', 'n', 'I', 'I', 'I', '-', 'P', 'o', 'W', '/', '2', '0', '2', '6'};
CONST_ARR u8 BC3_PERSON_S[8] = {0x0a, 0x20, 0xe8, 0x11, 0x61, 0x09, 0x08, 0xba};
CONST_ARR u8 BC3_PERSON_B[16] = {0x1e, 0x80, 0xf2, 0xfd, 0xdc, 0x58, 0x18, 0xd6,
                                 0xfe, 0x9b, 0x58, 0x1e, 0x3b, 0xd4, 0x93, 0x0b};
// Suffixes, padded to 11: "/SHA3-512" "/SHA3-384" "/SHA-512" "/RIPEMD-160" "/BLAKE2s" "/BLAKE2b" "/SHA3-256"
CONST_ARR u8 BC3_SUFFIX[7][11] = {{'/', 'S', 'H', 'A', '3', '-', '5', '1', '2'},
                                  {'/', 'S', 'H', 'A', '3', '-', '3', '8', '4'},
                                  {'/', 'S', 'H', 'A', '-', '5', '1', '2'},
                                  {'/', 'R', 'I', 'P', 'E', 'M', 'D', '-', '1', '6', '0'},
                                  {'/', 'B', 'L', 'A', 'K', 'E', '2', 's'},
                                  {'/', 'B', 'L', 'A', 'K', 'E', '2', 'b'},
                                  {'/', 'S', 'H', 'A', '3', '-', '2', '5', '6'}};
CONST_ARR u8 BC3_SUFFIX_LEN[7] = {9, 9, 8, 11, 8, 8, 9};

// buf = domain || suffix k, returns the length.
DEVICE_FN u32 bc3_tag(u8* buf, int k) {
  for (int i = 0; i < 47; i++) buf[i] = BC3_DOMAIN[i];
  u32 n = BC3_SUFFIX_LEN[k];
  for (u32 i = 0; i < n; i++) buf[47 + i] = BC3_SUFFIX[k][i];
  return 47u + n;
}

DEVICE_BIG void bc3hash80(GLOBAL const u64* tables, const u8* header, u8* out) {
  u8 msg[58 + 80];
  u8 preimage[64], mix[32], s384[48], s512[64], r160[20], bs[32], bb[64];
  u32 n;
  n = bc3_tag(msg, 0);
  for (int i = 0; i < 80; i++) msg[n + i] = header[i];
  sha3(msg, n + 80u, 64u, preimage);
  n = bc3_tag(msg, 1);
  for (int i = 0; i < 80; i++) msg[n + i] = header[i];
  sha3(msg, n + 80u, 48u, s384);
  n = bc3_tag(msg, 2);
  for (int i = 0; i < 80; i++) msg[n + i] = header[i];
  sha512(msg, n + 80u, s512);
  n = bc3_tag(msg, 3);
  for (int i = 0; i < 80; i++) msg[n + i] = header[i];
  ripemd160(msg, n + 80u, r160);
  n = bc3_tag(msg, 4);
  for (int i = 0; i < 80; i++) msg[n + i] = header[i];
  blake2s(BC3_PERSON_S, msg, n + 80u, bs);
  n = bc3_tag(msg, 5);
  for (int i = 0; i < 80; i++) msg[n + i] = header[i];
  blake2b(BC3_PERSON_B, msg, n + 80u, bb);
  bc3mix512(tables, preimage, mix);
  u8 fin[56 + 48 + 32 + 64 + 20 + 32 + 64];
  n = bc3_tag(fin, 6);
  for (int i = 0; i < 48; i++) fin[n++] = s384[i];
  for (int i = 0; i < 32; i++) fin[n++] = mix[i];
  for (int i = 0; i < 64; i++) fin[n++] = s512[i];
  for (int i = 0; i < 20; i++) fin[n++] = r160[i];
  for (int i = 0; i < 32; i++) fin[n++] = bs[i];
  for (int i = 0; i < 64; i++) fin[n++] = bb[i];
  sha3(fin, n, 32u, out);
}

// =========================================================================
// Kernels.
// =========================================================================

// Test/verification: BC3Mix of `count` 64-byte inputs.
KERNEL_FN void bc3mix_batch(GLOBAL const u64* tables, GLOBAL const u8* in, u32 count, GLOBAL u8* out) {
  u32 gid = GLOBAL_ID();
  if (gid >= count) return;
  u8 buf[64], res[32];
  for (int i = 0; i < 64; i++) buf[i] = in[gid * 64u + i];
  bc3mix512(tables, buf, res);
  for (int i = 0; i < 32; i++) out[gid * 32u + i] = res[i];
}

// Test/verification: BC3Hash of `count` 80-byte headers.
KERNEL_FN void bc3hash_batch(GLOBAL const u64* tables, GLOBAL const u8* in, u32 count, GLOBAL u8* out) {
  u32 gid = GLOBAL_ID();
  if (gid >= count) return;
  u8 hdr[80], res[32];
  for (int i = 0; i < 80; i++) hdr[i] = in[gid * 80u + i];
  bc3hash80(tables, hdr, res);
  for (int i = 0; i < 32; i++) out[gid * 32u + i] = res[i];
}

// Mining: BC3Hash of the header with nonce start_nonce + gid; report the
// nonce when the hash, read as a little-endian 256-bit number, is <= target
// (target[0..4) are its u64 limbs, least significant first). hits[0] is the
// atomic hit counter, hits[1..1+max_hits] the nonces - as sha3t_scan.
KERNEL_FN void bc3hash_scan(GLOBAL const u64* tables, GLOBAL const u8* header76, u32 start_nonce, u32 nonce_count,
                            GLOBAL const u64* target, GLOBAL u32* hits, u32 max_hits) {
  u32 gid = GLOBAL_ID();
  if (gid >= nonce_count) return;
  u32 nonce = start_nonce + gid;
  u8 hdr[80], res[32];
  for (int i = 0; i < 76; i++) hdr[i] = header76[i];
  hdr[76] = (u8)nonce;
  hdr[77] = (u8)(nonce >> 8);
  hdr[78] = (u8)(nonce >> 16);
  hdr[79] = (u8)(nonce >> 24);
  bc3hash80(tables, hdr, res);
  int ok = 1;
  for (int i = 3; i >= 0; i--) {
    u64 h = load_le64(res + 8 * i);
    if (h != target[i]) { ok = h < target[i]; break; }
  }
  if (ok) {
    u32 idx = ATOMIC_INC_U32(&hits[0]);
    if (idx < max_hits) hits[idx + 1] = nonce;
  }
}
