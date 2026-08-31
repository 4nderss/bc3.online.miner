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
   the top of the file, not scattered through the round function.
2. Regenerate the PTX with `nvcc --ptx -arch compute_52 -O3 -x cu` (or NVRTC).
   Commit the regenerated `sha3t.ptx` - the build embeds it.
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
keccak round is 185 SASS instructions on sm_89, and every one is accounted for
in the README. Before optimising, count; do not guess.

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

Tag, let CI build, attach the zip and `SHA256SUMS`. The README tells users to
verify their download, so the checksums must ship with the package.

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
