# Working in this repository

Notes for anyone - human or agent - making changes here. Read this before the
first edit; it encodes decisions that are expensive to rediscover.

## Language

**Everything committed here is in English.** Code comments, doc comments,
commit messages, documentation, and every string a user can see. The project
had a mixed-language period; do not add to it.

**Write ASCII in source.** Comments, commit messages, identifiers and log
output: no typographic dashes, arrows, quotes or superscripts - use `-`,
`->`, `"` and `2^32`. This is not pedantry. The repository has already lost
characters twice to encoding round-trips, once through a tool that re-encoded
an entire UTF-8 file in place. ASCII cannot be damaged that way.

The exception is **text a user reads in the GUI**, where the correct character
is the right choice: a degree sign is a degree sign, and an em dash reads
better than a hyphen as an empty-value placeholder. Keep those; just never let
them migrate into comments or commit messages.

Comments here explain **why**, not what. A comment that restates the code is
worse than none. The valuable ones carry the measurement or the bug that
motivated the decision - keep those intact when you touch surrounding code, and
update the numbers if you invalidate them.

## Layout

```
src/                 mining core (the bc3-miner CLI)
  backend/           CUDA, OpenCL and the dynamic loaders for both
  kernels/sha3t.cl   GPU kernel - shared source for CUDA and OpenCL
  kernels/sha3t.ptx  GENERATED from sha3t.cl - never hand-edit
  consensus.rs       CPU reference implementation; the correctness oracle
bc3hash/             BC3Hash, the PoW of the second fork: the node's own C++
  vendor/            src/crypto/bc3 + src/secp256k1 of the node - never edit
  vectors/           the node's test vectors (doc/bc3/vectors)
wasm/                browser miner used on bc3.online
gui/dist/            Tauri frontend - plain HTML/CSS/JS, no build step
gui/src-tauri/       Tauri shell; runs the CLI as a sidecar
```

## Building

```
cargo build --release
```

CUDA and OpenCL are both in the default build. Neither is linked - both are
resolved at run time - so **no GPU SDK is needed to build**, and one binary
runs with CUDA on NVIDIA, OpenCL on AMD/Intel, or CPU alone.

`build.rs` compiles `sha3t.cl` to PTX when `nvcc` is on PATH, and otherwise
falls back to the checked-in `sha3t.ptx`. That fallback is the normal path:
NVRTC ships with the CUDA *Toolkit*, not the graphics driver, so the kernel must
be compiled ahead of time and embedded.

### BC3Hash and its C++

The version bits pick the PoW per job, exactly as the node's `GetHash` does:
bit 11 (`0x800`) is BC3Hash and wins over bit 12 (`0x1000`, SHA3-256t);
neither is SHA256d. BC3Hash is not ported - `bc3hash/` compiles the node's
reference code from `bc3hash/vendor/` (commit in `vendor/SOURCE_COMMIT`)
behind a small C shim of ours, `bc3hash/src/shim.cpp`. A port could differ
from the node in one bit and no test of ours would notice; the node's own code
plus the node's vectors cannot. To move to a newer node, copy the two
directories again, update `SOURCE_COMMIT` and the vectors, and run the tests.

- **A compiler can build it wrong and nothing fails.** GCC 13 (Ubuntu 24.04)
  at -O3 turns mix.cpp into code that runs and gets every hash wrong. The
  shares would look fine to the miner and all be rejected by the pool. Two
  defences, keep both: `bc3hash/build.rs` compiles the C++ with
  `-fno-strict-aliasing` and loop vectorisation off, in each compiler's
  spelling (cost: under 2 %), and `bc3hash::initialize` hashes four of the
  node's vectors before anything is mined. A build that fails them never
  mines BC3Hash; it says "this build computes BC3Hash incorrectly - report
  it" on stderr and as a GUI status, and SHA3-256t carries on.
- **Run the vector tests with every compiler that builds a release** -
  `cargo test` in `bc3hash/` and in the root, which needs `bc3hash/vectors/`.
  For the Windows build, cross-build the test binaries with cargo-xwin and run
  them under Wine; they pass there.
- It needs a C++20 compiler with `unsigned __int128`: GCC or Clang. MSVC's
  `cl.exe` cannot build it; on an MSVC target `bc3hash/build.rs` switches to
  `clang-cl` (cargo-xwin already uses it; GitHub's Windows images ship LLVM).
  The Windows binary then also imports `MSVCP140.dll`, from the same VC++
  redistributable as the `VCRUNTIME140.dll` it always needed.
- Native builds prefer `clang`/`clang++` when the environment names no
  compiler: ~10 % more hashrate than GCC 14, measured. The build warns when
  it has to fall back. -O3 stays: -O2 measured 20-25 % slower with either.
- There is no GPU kernel for it, so in `--backend auto` the CPU threads stand
  in for the GPU on BC3Hash jobs (`worker::CpuRole::GpuFallback`).
- The browser build (`wasm/`) shares `consensus.rs` but never has the
  `bc3hash` feature: no C++ in wasm32. It stays SHA3-256t only.
- `BC3HASH_VENDOR` and `BC3HASH_VECTORS` point the build and the tests at the
  same sources and vectors somewhere else.
- `cargo run --release --example bench -- fast 8` in `bc3hash/` measures the
  hash rate per thread; `cargo test --release -- --ignored bc3hash_worker_rate
  --nocapture` measures the same through the worker loop.

### Environment pitfalls

- The rustup **windows-gnu** toolchain ships `dlltool` and a linker but no
  assembler, so any dependency needing an import library fails to link. Use the
  **msvc** toolchain with the Visual Studio C++ build tools *and their libraries*
  installed - a Visual Studio install without `VC/Tools/MSVC/*/lib` cannot link
  either, and fails with `LNK1104: cannot open file 'msvcrt.lib'`.
- If neither works, push and let CI build. `.github/workflows/build.yml` produces
  the Windows package as a zip artifact with `SHA256SUMS`.

## Changing the GPU kernel

`sha3t.cl` is shared by both backends and compiled two different ways, so a
change there is a consensus change. The workflow that catches mistakes:

1. Edit `sha3t.cl`. Backend-specific differences belong in the macro block at
   the top of the file, not scattered through the round function. The lanes
   are bit-interleaved (see the note at the top of the kernel); the host side
   of that contract is `kernel_lanes` and `kernel_target` in
   `src/backend/mod.rs`, and the Rust mirror of the kernel in the tests there
   (`il_round` and friends) must follow every change to the kernel's math.
2. Regenerate the PTX with `nvcc --ptx -arch compute_52 -O3 -x cu`, **using the
   CUDA 12.0 toolkit**. The toolkit version decides the PTX ISA version, and a
   driver can only JIT the versions it knows: ISA 8.0 needs driver R525, ISA
   8.8 (CUDA 12.9) needs R575. Build it too new and the card is detected but
   the kernel never loads. `build.rs` fails the build above ISA 8.0 for exactly
   that reason. Commit the regenerated `sha3t.ptx` - the build embeds it.

   ```
   docker run --rm -v "$PWD:/work" -w /work nvidia/cuda:12.0.1-devel-ubuntu22.04 \
     /usr/local/cuda/bin/nvcc --ptx -arch compute_52 -O3 -x cu \
     src/kernels/sha3t.cl -o src/kernels/sha3t.ptx
   ```
3. **Verify bit-exactness against an independent reference**, not against the
   previous kernel. `cargo test` compares the GPU hit set to `consensus.rs` on
   random headers; `sha3_256` from any standard library works as a second
   opinion and does not share our assumptions.
4. Check the OpenCL branch still compiles. It is a different compiler with
   different rules, and it will not be exercised by a CUDA-only test.
5. Measure. `ptxas -arch=sm_89 -O3 -v` reports registers and spills;
   `nvdisasm -c` gives the SASS so you can count instructions per round.

The kernel is **ALU-bound and at the instruction-issue ceiling**. Occupancy,
batch size and memory access are not the lever - instruction count is. One
keccak round is ~168 SASS instructions on sm_86 (117 `LOP3`, 50 `SHF`), and
every one is accounted for in the README, which also lists what has been
measured NOT to help - the multiplier pipe, two nonces per thread, launch
bounds, block sizes, hand-pruning the last round. Before optimising, count;
do not guess. And measure with the kernels alternated in one process: on a
laptop the clock drifts more with heat than most changes are worth.

Register spills are catastrophic here, not merely slow: dynamic indexing of the
state array pushes it to local memory and costs roughly 100x. If `ptxas -v`
reports any spill, stop and fix that first.

## Testing

```
cargo test                              # CPU: consensus vectors, kernel mirror
cargo test --release -- --ignored       # GPU tests; need an actual GPU
```

The OpenCL test needs an OpenCL runtime but not a GPU - pocl provides a CPU one,
so it runs in plain Docker. Both Docker recipes are in the README.

## Releasing

Tag; CI does the rest. `release.yml` runs the tests, builds Windows and Linux,
writes one `SHA256SUMS` over both, attaches a build provenance attestation, and
opens a **draft** release. Publishing stays a human click.

The checksum and the attestation answer different questions and both have to
ship: the checksum says the download is intact, the attestation says it came
from this repository's workflow at a named commit. The README tells users to
check both.

Actions in both workflows are pinned by commit SHA rather than tag. Tags move,
and these jobs produce the binaries people download. When you bump one, check
the inputs you rely on still exist - a major bump is not free.

Antivirus flags mining binaries as a matter of course. That is one more reason
the checksums matter, and it is why the miner does not download or execute
anything at run time.

## Rules that are not up for negotiation

- **The miner never updates itself.** It holds the user's payout address. A
  binary that downloads and runs code turns a compromised release key into
  everyone's funds being redirected, silently. The GUI may *tell* the user a
  release exists and link to it; that is the whole scope.
- **Every GPU hit is re-verified on the CPU** against `consensus.rs` before
  submission. A kernel bug must never be able to submit an invalid share.
- **Secrets never enter the repository.** Credentials live outside the working
  tree; scrub tokens from any command output you paste or log.
- **Never hand-edit `sha3t.ptx`.** Regenerate it from the source.

## Commit messages

Subject line under ~72 characters, imperative, prefixed by area where it helps
(`GPU:`, `GUI:`, `CI:`, `Docs:`). Then a blank line and a body that explains the
reasoning: what was wrong, why this fix and not another, and what was measured.
Include the numbers. A reader six months from now needs the why, and the diff
already shows the what.
