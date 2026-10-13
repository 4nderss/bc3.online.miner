// C interface to the node's BC3Hash reference (vendor/bc3). Exceptions must
// not cross an extern "C" boundary, so every entry point catches them and
// returns -1 instead.
#include <bc3/bc3.hpp>

// For the GPU tables: the node's constants and its FixedPower, both header
// only, and the public secp256k1 API.
#include "constants.hpp"
#include <secp256k1.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace {
using namespace bc3::detail;

// One FixedPower table - 8L rows x 256 values x L limbs, least significant
// limb first - exactly as mix.cpp builds it for its own pow64/128/256. On the
// heap: the 256-bit one is 256 KiB, too much for a thread's stack.
template <class M>
std::size_t power_table(const typename M::U& base, std::uint64_t* out)
{
    const auto table = std::make_unique<FixedPower<M>>(base);
    std::size_t k = 0;
    for (const auto& row : table->table)
        for (const auto& value : row)
            for (const auto limb : value.v)
                out[k++] = limb;
    return k;
}

std::size_t curve_points(unsigned window)
{
    const unsigned digits = (256 + window - 1) / window;
    return std::size_t(digits) * ((std::size_t{1} << window) - 1);
}

// The points b * 2^(w*i) * P, b = 1 .. 2^w - 1, in affine x[4], y[4] limbs,
// built with the same secp256k1 calls as CurveTable in mix.cpp; the last row
// is padded with zeros. 0 = ok, -1 = secp256k1 refused something.
int curve_table(const U256& px, const U256& py, unsigned window, std::uint64_t* out)
{
    struct Context {
        secp256k1_context* p = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
        ~Context() { secp256k1_context_destroy(p); }
    } ctx;
    if (!ctx.p) return -1;
    const unsigned digits = (256 + window - 1) / window;
    const unsigned entries = (1u << window) - 1;
    const unsigned last_entries = (1u << (256 - window * (digits - 1))) - 1;
    std::array<unsigned char, 65> encoded{};
    encoded[0] = 4;
    px.store(encoded.data() + 1);
    py.store(encoded.data() + 33);
    secp256k1_pubkey base;
    if (!secp256k1_ec_pubkey_parse(ctx.p, &base, encoded.data(), 65)) return -1;
    std::vector<secp256k1_pubkey> row(entries);
    std::size_t k = 0;
    for (unsigned i = 0; i < digits; ++i) {
        const unsigned count = i + 1 == digits ? last_entries : entries;
        row[0] = base;
        for (unsigned j = 1; j < count; ++j) {
            const secp256k1_pubkey* points[] = {&row[j - 1], &base};
            if (!secp256k1_ec_pubkey_combine(ctx.p, &row[j], points, 2)) return -1;
        }
        for (unsigned j = 0; j < entries; ++j) {
            if (j < count) {
                std::size_t length = 65;
                if (!secp256k1_ec_pubkey_serialize(ctx.p, encoded.data(), &length, &row[j], SECP256K1_EC_UNCOMPRESSED))
                    return -1;
                const auto x = U256::load(encoded.data() + 1), y = U256::load(encoded.data() + 33);
                for (std::size_t l = 0; l < 4; ++l) {
                    out[k + l] = x.v[l];
                    out[k + 4 + l] = y.v[l];
                }
            } else {
                std::memset(out + k, 0, 8 * sizeof(std::uint64_t));
            }
            k += 8;
        }
        if (i + 1 < digits) {
            const secp256k1_pubkey* points[] = {&row[entries - 1], &row[0]};
            if (!secp256k1_ec_pubkey_combine(ctx.p, &base, points, 2)) return -1;
        }
    }
    return 0;
}
// One row of a curve's table on its own: b * 2^(w*row) * P for b = 1 ..
// 2^w - 1 (the last row only up to its bits, padded with zeros), the same
// points as curve_table's row - whose base comes from the row before it, here
// straight from P by one scalar multiplication, so that every row of every
// curve can be built at once. 0 = ok, -1 = secp256k1 refused something.
int curve_row(const U256& px, const U256& py, unsigned window, unsigned row, std::uint64_t* out)
{
    struct Context {
        secp256k1_context* p = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
        ~Context() { secp256k1_context_destroy(p); }
    } ctx;
    if (!ctx.p) return -1;
    const unsigned digits = (256 + window - 1) / window;
    const unsigned entries = (1u << window) - 1;
    const unsigned last_entries = (1u << (256 - window * (digits - 1))) - 1;
    const unsigned count = row + 1 == digits ? last_entries : entries;
    std::array<unsigned char, 65> encoded{};
    encoded[0] = 4;
    px.store(encoded.data() + 1);
    py.store(encoded.data() + 33);
    secp256k1_pubkey base;
    if (!secp256k1_ec_pubkey_parse(ctx.p, &base, encoded.data(), 65)) return -1;
    if (row) {
        // 2^(w*row) as a big-endian scalar; w*row <= 255 - w, below n.
        std::array<unsigned char, 32> scalar{};
        const unsigned bit = window * row;
        scalar[31 - bit / 8] = static_cast<unsigned char>(1u << (bit % 8));
        if (!secp256k1_ec_pubkey_tweak_mul(ctx.p, &base, scalar.data())) return -1;
    }
    secp256k1_pubkey point = base;
    std::size_t k = 0;
    for (unsigned j = 0; j < entries; ++j, k += 8) {
        if (j >= count) {
            std::memset(out + k, 0, 8 * sizeof(std::uint64_t));
            continue;
        }
        if (j) {
            const secp256k1_pubkey* points[] = {&point, &base};
            secp256k1_pubkey next;
            if (!secp256k1_ec_pubkey_combine(ctx.p, &next, points, 2)) return -1;
            point = next;
        }
        std::size_t length = 65;
        if (!secp256k1_ec_pubkey_serialize(ctx.p, encoded.data(), &length, &point, SECP256K1_EC_UNCOMPRESSED)) return -1;
        const auto x = U256::load(encoded.data() + 1), y = U256::load(encoded.data() + 33);
        for (std::size_t l = 0; l < 4; ++l) {
            out[k + l] = x.v[l];
            out[k + 4 + l] = y.v[l];
        }
    }
    return 0;
}
} // namespace

extern "C" {
// Rows of a curve table at window `window`: how many, and u64s in each.
unsigned bc3hash_gpu_rows(unsigned window) noexcept { return (256 + window - 1) / window; }
std::size_t bc3hash_gpu_row_u64(unsigned window) noexcept { return std::size_t((1u << window) - 1) * 8; }

// Fill row `row` of curve `curve` (0..3: H, J, K, L). 0 = ok, -1 = failed,
// -2 = wrong window, curve, row or size.
int bc3hash_gpu_curve_row(unsigned window, int curve, unsigned row, std::uint64_t* out, std::size_t len) noexcept
{
    try {
        if (window < 2 || window > 24 || curve < 0 || curve > 3 || row >= bc3hash_gpu_rows(window) ||
            len != bc3hash_gpu_row_u64(window))
            return -2;
        using namespace bc3::detail::constants;
        const U256 xs[4] = {H_x, J_x, K_x, L_x};
        const U256 ys[4] = {H_y, J_y, K_y, L_y};
        return curve_row(xs[curve], ys[curve], window, row, out);
    } catch (...) {
        return -1;
    }
}


// The GPU table buffer for a curve window of `window` bits comes in five
// parts, in this order - the BC3_TABLE_* layout of src/kernels/bc3hash.cl:
//   part 0     pow64 (8 x 256 x 1), pow128 (16 x 256 x 2), pow256 (32 x 256 x 4)
//   parts 1-4  curves H, J, K, L: curve_points x 8 each
// so that the caller can build them on five threads.
std::size_t bc3hash_gpu_part_u64(unsigned window, int part) noexcept
{
    return part == 0 ? 8 * 256 * 1 + 16 * 256 * 2 + 32 * 256 * 4 : curve_points(window) * 8;
}

// Fill one part. 0 = ok, -1 = failed, -2 = wrong window, part or size.
int bc3hash_gpu_table_part(unsigned window, int part, std::uint64_t* out, std::size_t len) noexcept
{
    try {
        if (window < 2 || window > 20 || part < 0 || part > 4 || len != bc3hash_gpu_part_u64(window, part))
            return -2;
        using namespace bc3::detail::constants;
        if (part == 0) {
            std::size_t k = 0;
            k += power_table<F64>(g64, out + k);
            k += power_table<F128>(g128, out + k);
            k += power_table<F256>(g256, out + k);
            return k == len ? 0 : -1;
        }
        const U256 xs[4] = {H_x, J_x, K_x, L_x};
        const U256 ys[4] = {H_y, J_y, K_y, L_y};
        return curve_table(xs[part - 1], ys[part - 1], window, out);
    } catch (...) {
        return -1;
    }
}

// Select the table size and build the tables. 0 = ok, -1 = the tables already
// exist in the other mode, or there was not enough memory for them.
int bc3hash_initialize(int fast) noexcept
{
    try {
        bc3::configure(fast ? bc3::Mode::fast : bc3::Mode::compact);
        bc3::initialize();
        return 0;
    } catch (...) {
        return -1;
    }
}

int bc3hash_hash(const std::uint8_t* data, std::size_t len, std::uint8_t* out32) noexcept
{
    try {
        const auto d = bc3::bc3hash(data, len);
        std::memcpy(out32, d.data(), d.size());
        return 0;
    } catch (...) {
        return -1;
    }
}

int bc3hash_mix(const std::uint8_t* data, std::size_t len, std::uint8_t* out32) noexcept
{
    try {
        const auto d = bc3::bc3mix(data, len);
        std::memcpy(out32, d.data(), d.size());
        return 0;
    } catch (...) {
        return -1;
    }
}

// Bytes of precomputed tables in the configured mode.
std::size_t bc3hash_table_bytes() noexcept
{
    return bc3::build_info().mix_table_bytes;
}

} // extern "C"
