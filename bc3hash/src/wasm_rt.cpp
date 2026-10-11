// What the node's reference code needs from a C/C++ runtime on wasm32, where
// the Rust target (wasm32-unknown-unknown) has none. Compiled for wasm32 only
// (build.rs); malloc and friends are in lib.rs, on Rust's allocator.
//
// Every function here is an error path the hash never takes once the tables
// exist: secp256k1's callbacks fire on invalid arguments, and the C++ throws
// only for impossible inputs or a failed allocation. wasm32 without the
// exception-handling proposal cannot unwind, so each of them traps: the
// instance stops with a RuntimeError the worker reports, which is the same
// outcome as a native build failing its self-test - no hashes at all rather
// than wrong ones.
#include <cstddef>

extern "C" {

void secp256k1_default_illegal_callback_fn(const char*, void*) { __builtin_trap(); }
void secp256k1_default_error_callback_fn(const char*, void*) { __builtin_trap(); }

[[noreturn]] void abort() { __builtin_trap(); }

// The Itanium ABI's throw: allocate the exception object, construct it, then
// __cxa_throw. Trapping at the allocation means nothing is ever constructed.
void* __cxa_allocate_exception(std::size_t) noexcept { __builtin_trap(); }
[[noreturn]] void __cxa_throw(void*, void*, void (*)(void*)) { __builtin_trap(); }

// libc++abi prints its fatal errors to stderr before it aborts. There is no
// stderr to print to; the abort that follows is the message.
void* stderr = nullptr;
std::size_t fwrite(const void*, std::size_t, std::size_t, void*) { return 0; }
int vfprintf(void*, const char*, __builtin_va_list) { return 0; }
int fputc(int c, void*) { return c; }

// Destructors of function-local statics (the tables) would run at exit, which
// a wasm instance never reaches.
int __cxa_atexit(void (*)(void*), void*, void*) { return 0; }

// For libc++abi's type_info comparisons.
int strcmp(const char* a, const char* b)
{
    while (*a && *a == *b) ++a, ++b;
    return (unsigned char)*a - (unsigned char)*b;
}

} // extern "C"
