// C interface to the node's BC3Hash reference (vendor/bc3). Exceptions must
// not cross an extern "C" boundary, so every entry point catches them and
// returns -1 instead.
#include <bc3/bc3.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" {

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
