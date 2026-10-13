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
  kernels/bc3hash.cl GPU kernel - one source for CUDA and OpenCL; from the
                     kernel lab, never edited here
  kernels/bc3hash_w*.ptx
                     GENERATED from bc3hash.cl, one per table window -
                     never hand-edit
  consensus.rs       the choice of PoW, headers, targets; the correctness
                     oracle, with bc3hash/
  tuning.rs          each GPU's BC3Hash table window, measured while it
                     mines (see "Window tuning")
  internal.rs        --benchmark (bench.rs) and --solve (solve.rs): only
                     with the internal-tools feature, never in a release
bc3hash/             BC3Hash, the PoW since the second fork: the node's own C++
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

`build.rs` compiles `bc3hash.cl` to PTX, once per table window, when `nvcc` is
on PATH, and otherwise falls back to the checked-in `bc3hash_w*.ptx`. That
fallback is the normal path: NVRTC ships with the CUDA *Toolkit*, not the
graphics driver, so the kernel must be compiled ahead of time and embedded.
The `cuda` and `opencl` features bring `bc3hash` with them - BC3Hash is the
only kernel either backend has.

Beside the PTX, `build.rs` embeds the kernel's **machine code** (SASS) for
sm_61, sm_75, sm_86, sm_89 and sm_120 - Pascal, Turing, Ampere, Ada and
Blackwell - from `src/kernels/sass/` (or `BC3_SASS_DIR`), deflated: ~45 MB of
cubins, 13.6 MB in the binary. A card with its own architecture, or an
earlier minor of its major (sm_86 code runs on sm_87), loads it in a second;
anything else JITs the PTX as before, 20-60 s where the driver has not cached
it. The cubins are never committed. `tools/sass.sh` makes them from the
committed PTX in the CUDA 12.9 image:

```
docker run --rm -v "$PWD:/work" -w /work nvidia/cuda:12.9.1-devel-ubuntu22.04 sh tools/sass.sh
```

and writes a manifest with the SHA-256 of each PTX; `build.rs` embeds a
window's code only while the PTX it embeds is that one. Both workflows build
the cubins in a `sass` job and set `BC3_REQUIRE_SASS=1`, which makes a build
without them fail. CUDA 12.9 because its machine code loads on every CUDA 12
driver (R525 and up, as the PTX needs) and it knows Blackwell; a CUDA 13
toolkit's would need R580. On sm_86 its ptxas makes what the driver's JIT
makes - 119,232 instructions both, 168 registers and 3360 bytes of stack,
568 lines in another order - and hashes as fast. The link API
(`cuLinkAddData` + `cuLinkComplete`) was tried for machine code made at run
time and dropped: it compiles relocatable code, 4056 bytes of stack and
1.4 % slower.

### BC3Hash and its C++

The version bits pick the PoW per job, exactly as the node's `GetHash` does:
bit 11 (`0x800`) is BC3Hash and wins over bit 12 (`0x1000`, SHA3-256t);
neither is SHA256d. Only BC3Hash is mined: the other two have no hash
function (`PowAlgo::hash_fn` is None), so every worker sits such a job out
(`shared::AlgoGate`) and the status says "NOT mining" and why - never hashed
with the wrong function into shares the pool rejects. The SHA3-256t kernel
and CPU path were removed once the chain was past the fork; `PowAlgo` keeps
all three so a job is still classified as the node classifies it.

BC3Hash is not ported - `bc3hash/` compiles the node's reference code from
`bc3hash/vendor/` (commit in `vendor/SOURCE_COMMIT`) behind a small C shim of
ours, `bc3hash/src/shim.cpp`. A port could differ from the node in one bit and
no test of ours would notice; the node's own code plus the node's vectors
cannot. To move to a newer node, copy the two directories again, update
`SOURCE_COMMIT` and the vectors, and run the tests.

- **A compiler can build it wrong and nothing fails.** GCC 13 (Ubuntu 24.04)
  at -O3 turns mix.cpp into code that runs and gets every hash wrong. The
  shares would look fine to the miner and all be rejected by the pool. Two
  defences, keep both: `bc3hash/build.rs` compiles the C++ with
  `-fno-strict-aliasing` and loop vectorisation off, in each compiler's
  spelling (cost: under 2 %), and `bc3hash::initialize` hashes four of the
  node's vectors before anything is mined. A build that fails them never
  mines BC3Hash; it says "this build computes BC3Hash incorrectly - report
  it" on stderr and as a GUI status, and mines nothing - a GPU needs the same
  reference for its self-test and for every hit.
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
- Where a card's BC3Hash kernel failed to load or failed the self-test, the
  CPU threads stand in for it in `--backend auto`
  (`worker::CpuRole::GpuFallback`). See "The BC3Hash GPU kernel" below.
- The browser build (`wasm/`) shares `consensus.rs` - so it refuses a job
  without bit 11 the same way - and has the `bc3hash` feature too: the same
  C++, compiled for wasm32 by clang against WASI's libc/libc++ headers
  (`wasm/build.sh` installs them; `bc3hash/build.rs` has the wasm32 flags).
  The module imports nothing: `bc3hash/src/wasm_rt.cpp`
  and `wasm_malloc.rs` stand in for the C runtime, every error path traps,
  and libc++ without threads gets a no-op `std::mutex`
  (`bc3hash/wasm/include/mutex` - one thread per wasm instance). secp256k1
  is built with its 32-bit field there (`USE_FORCE_WIDEMUL_INT64`): wasm has
  no 64x64->128 multiply, and that alone took the module from 165 to 240 H/s
  a thread. `wasm/test/vectors.mjs` runs the node's vectors against the
  built module - the host tests cannot vouch for the wasm32 build - and
  `build.sh` ships nothing that fails them.
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

## The BC3Hash GPU kernel

`src/kernels/bc3hash.cl` is generated outside this repository (the kernel lab,
from the node's constants) - copy it in whole, never edit it here. One source
serves both backends, with what differs between them in macros at the top of
the file. Its secp256k1 tables come in four window sizes - 16, 18, 19 and 20
bits - chosen at run time, per card, by the tuning (see "Window tuning"), so
it is built four times:

```
for w in 16 18 19 20; do
  nvcc --ptx -arch compute_52 -O3 -DBC3_WINDOW=$w -x cu \
    src/kernels/bc3hash.cl -o src/kernels/bc3hash_w$w.ptx
done
```

with the CUDA 12.0 toolkit (see "Changing the GPU kernel" for why).
OpenCL builds the same source
at run time with `-DBC3_WINDOW=N` (`cargo test --release opencl -- --ignored`
under pocl runs it, see the README). The two builds do not share all of their
arithmetic: under CUDA the multi-word products, squarings, reductions and the
secp256k1 field additions are inline PTX generated by the lab's `gen_ptx.js`,
while OpenCL compiles the C versions next to them in the file
(`-DBC3_C_ARITH` selects those under CUDA too). The lab checks the two against
each other on inputs that reach every rare carry before it publishes a kernel.

The PTX products are `mad.lo.cc`/`madc.hi.cc` chains, which ptxas fuses into
ONE `IMAD.WIDE.U32.X` per 32x32 product - the multiply, the 64-bit addition
into the accumulator pair and both carries - but only while the accumulator
words sit in an aligned register pair that never changes partners; when a
word changes its pairing ptxas moves registers instead. So every product goes
to an accumulator by the parity of its column i+j: E holds the even columns'
pairs, O the odd ones' one word down, a row's products of one parity are one
carry chain, and t = E + O at the end. The secp256k1 reduction adds its odd
products and t_hi shifted up a word as one more such chain (`{t[9+2m],
t[8+2m]}` is exactly the 64-bit addend of the product `t[9+2m] * 977`). A
256-bit squaring with its reduction went from 169 to 127 instructions and a
product from 240 to 145; the whole hash +23 % on an RTX 3050 Ti (172 -> 212
kH/s in the lab's alternated A/B). The form before - `mul.wide` plus two
2-input `IADD3` per product, the third input always `RZ` - was limited by the
ALU pipe, not the multiplier. Measured and dropped on top of it: nested
chains for the squaring's cross products (7 chains instead of 13, under 1 %),
two curve sums interleaved (-9 %, spills), 128 registers (-6 %).

The inversions (safegcd, 64 a hash mod 2^256 - 36113 and 16 for the curves)
are libsecp256k1's modinv32 with two changes: the divsteps' masked negations
are multiply-adds by m in {-1, 0, 1} (16 instructions a divstep instead of
26; the inversions 18 % faster, the hash +2.4 %), and both moduli's signed30
limbs are compile-time constants, so the products with their zero limbs
drop out. bc3mix512 is stages 1-3 (`bc3mix_s123`) plus stage 4, and
bc3hash80 the standard hashes (`bc3hash_pre`, `bc3hash_fin`) around the
mix: smaller frames, 3360 bytes of local memory a thread instead of 3760,
+1.9 %. Together 172 -> 222 kH/s on the RTX 3050 Ti, 1.82 -> 2.52 MH/s on
an RTX 4090 at window 16.

The kernel is **latency-bound**: each step is a chain of
dependent carries, and at 255 registers a thread a scheduler had only two
warps to switch between. Two changes fixed most of that: the independent
powers of a round run as pairs in one basic block (`expfix*x2`,
`powtab256x2`, as the reference's `pair()`), with lazy products - reduced to
n words but not compared against p until the end of a chain, so no branch
sits between two products - and the kernel declares `__launch_bounds__(64,
6)`: 168 registers, three warps a scheduler. Four warps (128 registers) made
the powers faster but the curve and table reads slower - their L1/L2 traffic
grows with the warps - and lost overall; an unrolled squaring loop cost 7.5 %
at this occupancy (instruction cache), hence `#pragma unroll 1` there. On an
RTX 3050 Ti that is 30 % more hashes a clock, +18 % in the lab's alternated
A/B, and +12 % in `--benchmark` (151 -> 169 kH/s at window 16): the laptop
is power-capped (35 W), and the busier kernel runs 14 % lower clocks at the
same power. A card with headroom gets more of the 30 %. The lab's `prof.cu`
profiles the phases with clock64; `ubench.cu` measures a product at 1-8
warps a scheduler. The host builds the matching tables
(`bc3hash::gpu_tables`), and every card self-tests every window before it
mines with it: the node's header vectors, random headers against the CPU
reference, and a scan range against the CPU's hit set. A card that fails
mines no BC3Hash; the CPU takes over. Launches are whole waves of the card
(`launch_granule`) - a partial last wave cost 14 % on an RTX 3050 Ti.

`bc3-miner --benchmark`, in an internal-tools build, measures it through the
real worker path with an in-process job source, compares the four windows on
every card with the kernel, and checks every share with the CPU reference. It
also prints how busy each GPU was: the device time of its launches (CUDA
events, OpenCL profiling) over the wall time between them.

Every GPU keeps two launches queued (`backend::QUEUE_DEPTH`), each with its
own buffers: the next one waits on the device while the host collects the
last and re-checks its hits - about a millisecond a BC3Hash hit. The hidden
`--gpu-queue 1` of an internal-tools build brings back the one-at-a-time
loop, to compare the two with `--benchmark`. Launches stay long (~400 ms, 8
waves): the first waves of every launch run in lockstep and are slower, 8.4 us
a hash at one or two waves a launch against 7.9 at eight on an RTX 3050 Ti
(`cuda_bc3hash_launch_sizes`). A stream per launch did not change that.

### Window tuning

`src/tuning.rs` decides, `gpu_worker::Tune` drives it. Every CUDA card
starts at window 16 and mines from its first launch. Then it picks the
largest window whose tables fit beside 1 GiB of free GPU memory and, while
they are built, 2 GiB of free host memory, builds them in the background
(`bc3::prebuild`: one build for every card that wants the same window, held
until the last lets go) and has the driver compile that window's kernel
ahead on a non-primary context of the card (`compile_bc3hash_ahead`), so the
switch later is a load from the driver's cache. Then it measures, in GPU
time from the launches' own events: 90 s of warm-up after the build, 16 for
a minute, the bigger window for a minute after 30 s of settling, 16 again.
The bigger window stays if it beat the mean of the two 16s by 0.5 %. A
switch ends the launch run as a new job would: what is in flight is
collected, the main loop switches (`switch_bc3`, self-test included) and
picks the same job up where the cursor stopped.

- **In a laptop, CPU and GPU share one power budget, and the GPU gets its
  share back slowly.** The table build, on every core, took an RTX 3050 Ti
  from 40 to 27 W, and it was back at 40 W only 90 s after the build; 16
  measured right after the build came out 20 % low, and window 19 looked
  11.5 % faster where it is 2-5 %. Hence the warm-up after the build, and
  hence 16 is measured twice: the mean cancels a steady drift, and two 16s
  more than 5 % apart (`MAX_DRIFT`) mean something else moved the rate - the
  choice holds for the run, and the next start measures again.
- **What is remembered**: a steady measurement, win or loss, and a bigger
  window that failed its switch - per card name, memory size, driver and
  miner version (`tuning::key`), in `tuning.json` in the user's cache
  directory (`BC3_TUNING_FILE` overrides it; the Docker image points
  `XDG_CACHE_HOME` at a directory its user may write). A new driver or
  miner version measures again.
- **The switch is quick because the card loads machine code** made for it
  (see "Building"). In 2.1.0 every card JITted the PTX: 23-26 s of idle card
  a switch on a host with three RTX 5090s, three switches a tuning, and 31 s
  at every start - the Docker image's user had no home for the driver's JIT
  cache. A card this build has no machine code for still relies on that
  cache: the compile ahead fills it, and the switch loads from it. It lives
  in `$HOME/.nv` unless `CUDA_CACHE_PATH` says otherwise, which the image
  sets. Under WSL2 the driver caches for root only.
- **OpenCL cards do not tune** (`MiningBackend::memory` is None there): a
  switch builds the program from source while the card waits, and OpenCL
  caps one buffer at `CL_DEVICE_MAX_MEM_ALLOC_SIZE`. Untried on AMD and
  Intel cards.
- The GUI learns each card's window from `Event::Tables`: the window, the
  tables' size, whether it is tuning, and what the measurement found.

## Changing the GPU kernel

`bc3hash.cl` is shared by both backends and compiled two different ways - to
PTX by nvcc when the miner is built, by the OpenCL runtime on the user's
machine - so a change there is a consensus change. It comes from the kernel
lab (see "The BC3Hash GPU kernel"); the workflow that catches mistakes here:

1. Copy the lab's `bc3hash.cl` in whole. The host side of its contract is
   `src/backend/bc3.rs` - the target limbs, the self-test - the tables from
   `bc3hash::gpu_tables`, and the argument order of `bc3hash_scan` and
   `bc3hash_batch` in `cuda.rs` and `opencl.rs`; they change with it.
2. Regenerate the four PTX files, **using the CUDA 12.0 toolkit**. The toolkit
   version decides the PTX ISA version, and a driver can only JIT the versions
   it knows: ISA 8.0 needs driver R525, ISA 8.8 (CUDA 12.9) needs R575. Build
   it too new and the card is detected but the kernel never loads. `build.rs`
   fails the build above ISA 8.0 for exactly that reason, and the CUDA backend
   refuses a driver older than R525 when it opens a card (`check_driver`), so
   that card goes to OpenCL. Commit the regenerated PTX - the build embeds it.
   Then run `tools/sass.sh` again before testing on a GPU: the machine code
   of the old PTX is no longer embedded (the manifest no longer matches), so
   the cards would JIT the new PTX - correct, but not what a release does.
   CI makes its own cubins; nothing of them is committed.
   Build it in `/work`: nvcc derives the names of internal functions from the
   source's absolute path, and the committed PTX was built there (CI's
   comparison ignores those names, see `build.yml`).

   ```
   docker run --rm -v "$PWD:/work" -w /work nvidia/cuda:12.0.1-devel-ubuntu22.04 \
     sh -c 'for w in 16 18 19 20; do /usr/local/cuda/bin/nvcc --ptx -arch compute_52 \
       -O3 -DBC3_WINDOW=$w -x cu src/kernels/bc3hash.cl -o src/kernels/bc3hash_w$w.ptx; done'
   ```
3. **Verify bit-exactness against an independent reference**, not against the
   previous kernel: the node's own C++, which `bc3hash/` builds. Every card's
   self-test does that before it mines (the node's header vectors, random
   headers, a scan through the launch queue against the CPU's hit set), and
   `cargo test --release -- --ignored` on a GPU runs it at every window the
   card has the memory for, switching between them on one context
   (`cuda_bc3hash_every_window_passes_the_self_test`), and
   `cuda_bc3hash_queued_launches_match_the_cpu` on top. The lab's 20,000
   random inputs against the node's code run at window 16.
4. Check the OpenCL build too - under pocl it runs without a GPU (README). It
   is a different compiler with different rules, and it does not even share
   all of its arithmetic with the CUDA build (the C versions instead of inline
   PTX), so a CUDA-only test says nothing about it.
5. Measure, with the kernels alternated in one process: on a laptop the clock
   drifts more with heat than most changes are worth. `ptxas -arch=sm_89 -O3
   -v` reports registers and spills; the lab's `prof.cu` profiles the phases.

## Internal tools

The benchmark and the offline solver are for measuring the miner and for
one-off jobs, not for the people running it - `--solve` above all, which
mines a block template that is not the pool's. They are compiled in only with
the `internal-tools` feature:

```
cargo build --release --features internal-tools
```

**No release enables it**: `release.yml`, the Dockerfile and `build.yml`'s
package all build the default features. A default build has none of the
options below, not even hidden - passing one is clap's "unexpected argument"
error - and mines with the defaults they would otherwise override: the window
each card tunes for itself, and the CPU tables picked as `--bc3hash-mode`
describes. The code is
`src/internal.rs` (the options and the dispatch), `src/bench.rs` and
`src/solve.rs`; anything only they use is gated with the feature too, so a
default build has no dead code. `build.yml` runs `cargo test --release
--features internal-tools` so none of it rots.

| Option | Variable | What it does |
|--------|----------|--------------|
| `--benchmark` | | Measure instead of mining, see below |
| `--solve job.json` | | Mine one fixed block template offline, see below |
| `--bc3-window 16\|18\|19\|20` | `BC3_BC3HASH_WINDOW` | Table window of the GPU's BC3Hash kernel: 256 MiB, 0.9, 1.8 or 3.3 GiB of GPU memory. Fixes it for every card; without it each tunes its own |
| `--tune` | | Benchmark: leave the window to each card's tuning, as a release does. Measures afresh only with a new `BC3_TUNING_FILE` |
| `--bc3hash-mode fast\|compact` | `BC3_BC3HASH_MODE` | The CPU's BC3Hash tables: `fast` (~256 MiB of RAM, about 10 s to build) or `compact` (~7 MiB, 10-20 % slower); both find the same shares. Default `compact` when a GPU mines BC3Hash and the CPU only re-checks its hits, `fast` otherwise |
| `--seconds N` | | Benchmark: seconds a run (default 20 comparing windows, 540 with `--tune`, else 60) |
| `--difficulty D` | | Benchmark and solve: a fixed share difficulty instead of one from the measured rate |
| `--gpu-queue 1\|2` (hidden) | | Kernel launches each GPU keeps queued; 1 is the old one-at-a-time loop |

```
bc3-miner --benchmark                    # BC3Hash: every GPU, all four table windows
bc3-miner --benchmark --bc3-window 16    # BC3Hash at one window, 60 s
BC3_TUNING_FILE=/tmp/t.json bc3-miner --benchmark --tune   # the tuning, from scratch
bc3-miner --solve job.json
```

`--benchmark` mines without a pool, a network or an address. An in-process
job source hands random BC3Hash jobs to the normal workers - the same
dispatch, kernels and launch queue as real mining, every GPU hit re-checked
on the CPU - and every share is checked once more against the CPU reference,
the way the pool checks it. On GPUs with the kernel, and without
`--bc3-window`, it compares the four table windows. Each
window runs twice, once on the way up and once on the way back down, and the
two runs are averaged: a GPU's clock drifts as it warms up - on a laptop by
more than the windows differ - and a steady drift cancels out that way.
Before a window is measured, every card loads its kernel for it; the first
time on a machine the driver compiles it, which can take up to a minute and
is not counted. The summary ends with the fastest window for each card: on
an RTX 4090 window 20 (2.66 MH/s against 2.52 at 16, 60 s runs), on an RTX
3050 Ti laptop 19, the largest its 4 GiB hold.

Reading its output:

- **Rates** are per device, measured between the ends of whole kernel
  launches. With NVIDIA, each line also shows the first card's temperature
  and power.
- **busy** is the share of the time the GPU spent running kernels, by its
  own clock, over the wall time between launches. Close to 100 % (99.9 % on
  the cards measured) means the host keeps the card fed and the rate is the
  kernel's. Clearly lower means the card waited for the host between
  launches - a CPU too busy to queue the next launch and re-check hits, for
  instance. That rate is lost on the host side, and no kernel setting brings
  it back. Time a card spends drawing the desktop counts as busy.
- **shares ok/bad**: the last line is `RESULT: ok` (exit code 0). `RESULT:
  INVALID SHARES` (exit code 1) means this build or this GPU computes wrong
  hashes. `RESULT: nothing hashed` exits with 2.

`--solve` is for a block that has to be mined before any pool can hand it
out - the fork's first block was published this way. The job file holds
`height`, `hashPrevBlock`, `nVersion`, `nTime`, `nBits`, `scriptSigData` (hex,
put in the coinbase's scriptSig after the height) and `payouts` (a list of
`scriptPubKey` and `satoshis`). No pool, node or network is involved. It runs
on the same workers and kernels as pool mining, re-checks every GPU hit on
the CPU, and draws a random extranonce each run, so several machines - or a
restart - never search the same space. When a hash meets the block's target,
it prints the whole block and saves it as `bc3-block-<height>-<hash>.txt`,
ready for `submitblock` on a node that has the block before it.

## Testing

```
cargo test                              # CPU: consensus and node vectors, workers
cargo test --features internal-tools    # the same, plus the internal tools
cargo test --release -- --ignored       # GPU tests; need an actual GPU
```

The OpenCL tests need an OpenCL runtime but not a GPU - pocl provides a CPU
one, so they run in plain Docker. Both Docker recipes are in the README.

## Releasing

Tag; CI does the rest. `release.yml` runs the tests, builds Windows and Linux,
writes one `SHA256SUMS` over both, attaches a build provenance attestation, and
opens a **draft** release. Publishing stays a human click. Every release
artifact - the binaries, the GUI's sidecar, the container image - is built
with the default features: never with `internal-tools`.

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
- **Never hand-edit the PTX (`bc3hash_w*.ptx`).** Regenerate it from the
  source, and never edit `bc3hash.cl` here either - it comes from the lab.

## Commit messages

Subject line under ~72 characters, imperative, prefixed by area where it helps
(`GPU:`, `GUI:`, `CI:`, `Docs:`). Then a blank line and a body that explains the
reasoning: what was wrong, why this fix and not another, and what was measured.
Include the numbers. A reader six months from now needs the why, and the diff
already shows the what.
