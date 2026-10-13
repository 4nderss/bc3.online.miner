# bc3.online miner

GPU/CPU miner for BC3 (BitcoinIII): BC3Hash, the proof of work from block 64,360 on, and SHA3-256t before it. Connects to the [bc3.online](https://bc3.online) pool — PPLNS or solo.

- **Core:** Rust — CUDA and OpenCL GPU backends sharing one kernel source per proof of work, plus multi-threaded CPU mining
- **Proof of work:** picked per job from the block version, as the node picks it. BC3Hash runs on a GPU kernel of our own and on the CPU through the node's own C++
- **GPUs:** NVIDIA via CUDA; AMD and Intel (including integrated graphics) via OpenCL
- **Pool endpoints:** `bc3.online:3111` (PPLNS), `bc3.online:3112` (solo)
- **Protocol:** Stratum v1

## The hard fork at block 64,360: BC3Hash

At block 64,360 BC3 replaces SHA3-256t with **BC3Hash**, a proof of work of
its own, and changes how difficulty is set:

- **Version bit 11 (`0x800`) selects BC3Hash**, and it takes precedence over
  bit 12 (`0x1000`, SHA3-256t). The node sets it in every block template from
  the fork height on. The miner picks the hash per job from the version, the
  way the node's `GetHash` does, so there is no height rule in the miner that
  could disagree with the node.
- **Difficulty is ASERT** (aserti9-2d) instead of 2016-block epochs, and it
  adjusts every block. Block 64,360 has a fixed difficulty of 16; after it,
  each block's difficulty follows from how far the chain is ahead of or
  behind an absolute ten-minute schedule, with a half-life of two days.
- **Hashrates drop by about three orders of magnitude.** One BC3Hash costs
  about as much as 800 SHA3-256t hashes, so a card that did GH/s now does
  MH/s or kH/s. That is the algorithm, not a fault.

**bc3-miner 2.0.0 switches by itself.** It mines SHA3-256t until the pool
hands out the first BC3Hash job, and BC3Hash from then on - per job, without
a restart. An NVIDIA card compiles its BC3Hash kernel in the background
while it is still mining SHA3-256t, so at the switch it is ready within
seconds; an OpenCL card builds its kernel at the switch.

**1.x cannot mine after the fork: upgrade before block 64,360.** A 1.x miner
has no BC3Hash. After the fork the pool refuses it, with a message that
points to the [releases page](../../releases); the GUI of 1.2.0 and later
also shows a banner linking to the new release once it is published. Docker
users pull the new image - see *Quick start — Docker* below.

Right after the switch the pool starts every connection at a very low share
difficulty, and a fast card gets "too many shares per second" answers until
vardiff has raised it. That is expected. The miner prints the first such
reject at once and then one line every 10 s counting the repeats.

## Download

Release binaries are published as zip packages under [Releases](../../releases), with SHA256 checksums.

> **Note:** mining software is often flagged by antivirus heuristics (false positive). Verify the checksum of your download against the `SHA256SUMS` file published with it.

Releases also carry a build provenance attestation. A checksum only proves the
file matches what was uploaded; the attestation is a signed statement binding
the file to the commit and the workflow that built it, which is the part a
checksum cannot establish:

```
gh attestation verify bc3-miner-2.0.0-windows-x86_64.zip --repo 4nderss/bc3.online.miner
```

The Windows binaries need the Microsoft Visual C++ Redistributable (x64),
which most machines already have. 2.0.0 uses its `MSVCP140.dll` as well as
the `VCRUNTIME140.dll` earlier versions needed; if Windows says either is
missing, install the redistributable from Microsoft. On NVIDIA cards both
GPU kernels need driver R525 or newer.

## Quick start — GUI

Run **bc3-miner-gui.exe**. Paste your BC3 address, name your rig, pick PPLNS or
Solo, press *Start mining*. The window shows live hashrate with a sparkline,
accepted/rejected shares, estimated time to block, which backend is running,
and a full-screen celebration when you find a block. Settings are remembered
between runs.

If a newer release exists on GitHub, a banner says so and links to it. The
miner never updates itself: it holds your payout address, and a binary that
downloads and runs code is exactly the thing you should not trust with that.
The check asks GitHub and nothing else, runs in the window rather than in the
mining process, and stays silent if it fails.

The GUI runs the mining core (`bc3-miner.exe`) as a child process and reads its
JSON event stream — a GPU crash can never take the window down, and the core
stays independently testable.

## Quick start — command line

```
bc3-miner --user <your BC3 address>.<rigname>
```

Any Bitcoin-format address works: native segwit (bc1q...), taproot (bc1p...)
and the legacy formats (1... and 3...). Payouts go to it directly in the
block's coinbase.

Backend selection is automatic: CUDA if an NVIDIA GPU is present, otherwise OpenCL, otherwise CPU. Options:

| Flag | Description |
|------|-------------|
| `--mode pplns\|solo` | Payout mode. Picks the pool port: 3111 for `pplns` (default), 3112 for `solo` |
| `--pool host:port` | Pool address (default `bc3.online:3111`, solo `bc3.online:3112`). Overrides `--mode` |
| `--backend auto\|cuda\|opencl\|cpu` | Force a backend (default `auto`) |
| `--gpu-id N` | Use only the N:th detected GPU |
| `--threads N` | CPU threads, mining every job. Default: all cores in CPU mode. Next to a GPU the CPU does not mine unless you pass the flag: under `--backend auto` every core stands by and mines only what no GPU can (BC3Hash on a card whose kernel failed its self-test, for one), and with `--backend cuda` or `opencl` it stays out. `--threads 0` means "all cores", minus the GPU feeder threads when a GPU is running |
| `--intensity 1-100` | Percentage of full speed (default `100`). Below 100 the workers idle between passes, which lowers heat and power and keeps the machine usable while mining |
| `--stats-interval N` | Seconds between statistics lines (default `5`) |
| `--json` | Emit machine-readable JSON events on stdout (used by the GUI) |
| `--require-gpu` | Exit instead of falling back to CPU when no GPU is found (for rented GPU hosts, see Docker below) |
| `--probe` | List the detected hardware and exit. Shows CUDA and OpenCL devices separately, so you can tell which runtimes work on your machine |

Solo rewards are paid directly to your address in the block you find.

## Quick start — Docker

```
docker run -d --name bc3-miner --restart unless-stopped --gpus all \
  -e BC3_WALLET=bc1qyouraddresshere \
  -e BC3_WORKER=rig1 \
  ghcr.io/4nderss/bc3-miner
```

That is the whole thing for PPLNS. Set `BC3_MODE=solo` for solo. One image
covers NVIDIA, AMD/Intel and CPU: the miner opens the GPU runtimes with dlopen
rather than linking them, so it uses whatever the host exposes and falls back
to CPU when there is nothing.

Image tags follow the releases: `ghcr.io/4nderss/bc3-miner:2.0.0` (and
`:2.0`) is this version, and `:latest` - what the command above and
`docker-compose.yml` use - moves to a release when it is published. **The
fork needs 2.0.0 or later.** A container keeps running the image it was
created from, so pull the new one and recreate the container:

```
docker pull ghcr.io/4nderss/bc3-miner:latest
docker rm -f bc3-miner        # then the docker run above again
```

With compose: `docker compose pull && docker compose up -d`. Pin `:2.0.0`
instead of `:latest` if you would rather choose when to move.

| Variable | Default | Meaning |
|----------|---------|---------|
| `BC3_WALLET` | — | Your BC3 address. Required. |
| `BC3_WORKER` | — | Rig name, appended to the address. |
| `BC3_MODE` | `pplns` | `pplns` or `solo`. Picks the pool port. |
| `BC3_POOL` | from mode | `host:port`, overrides `BC3_MODE`. |
| `BC3_BACKEND` | `auto` | `auto`, `cuda`, `opencl` or `cpu`. |
| `BC3_THREADS` | none in GPU mode | CPU threads. |
| `BC3_INTENSITY` | `100` | 1–100. Below 100 the machine stays usable. |
| `BC3_GPU_ID` | all | Use only this one GPU, counting from 0. |
| `BC3_REQUIRE_GPU` | unset | Exit rather than fall back to CPU. |

Casing does not matter: `BC3_WALLET` and `bc3_wallet` both work. Environment
variables *are* case-sensitive on Linux and Docker passes them through
verbatim, so the miner promotes any casing of its own `BC3_*` names to the
canonical one before reading them.

**GPU access is a host-side decision** — no image can request a GPU by itself:

- **NVIDIA:** install the NVIDIA Container Toolkit and pass `--gpus all`. If
  your daemon already defaults to the nvidia runtime, the image sets
  `NVIDIA_VISIBLE_DEVICES` and works without the flag.
- **AMD or Intel:** pass `--device /dev/kfd --device /dev/dri`. The image ships
  the OpenCL loader; the driver comes from the host.
- **CPU only:** pass nothing. It just works, slowly.

Without GPU access the miner does not fail — it mines on CPU. Check
`docker logs bc3-miner` to see which backend it picked.

### On rented GPU hosts

Most GPU rental platforms only let you name an image and set environment
variables - there is no place to put `--gpus all`. That is fine: those hosts
run the NVIDIA container runtime as Docker's *default* runtime, and `--gpus
all` is only a wrapper that sets `NVIDIA_VISIBLE_DEVICES`. This image already
declares it, so the GPUs are exposed without any flag. Set
`NVIDIA_VISIBLE_DEVICES=0,1` yourself if you want only some of them.

One container uses **every** GPU it finds, each with its own worker and its own
slice of the nonce space. Running one container per card with `BC3_GPU_ID` is
worth it anyway on a rented machine: the pool then reports hashrate, last share
and found blocks per card, so a dead card is visible as a dead card rather than
as an unexplained dip.

Set `BC3_REQUIRE_GPU=1` there. Without it the miner falls back to CPU when no
device reaches the container, and you go on paying GPU prices for CPU hashrate
with nothing in the logs but a low number. With it the container exits and the
host's restart policy or alerting has something to notice.

`docker-compose.yml` in this repository has both patterns as a working example.

## Upgrading from 1.x

Replace the binaries, or pull the new image, **before block 64,360**. The
address, rig name, pool ports and every 1.x flag work as before, and the GUI
keeps its settings. What is new:

- BC3Hash needs memory for its tables: 256 MiB of GPU memory per card, and
  RAM for the CPU's - about 7 MiB when a GPU mines BC3Hash and the CPU only
  re-checks its hits, about 256 MiB when the CPU mines it.
- At startup the miner checks its BC3Hash build against the node's test
  vectors, and every GPU checks its kernel against the CPU before it mines.
  A card that fails mines no BC3Hash - under `--backend auto` the CPU takes
  over - and the log and the GUI say why.
- The first time a card needs its BC3Hash kernel, the driver compiles it -
  about 20 s on a fast CPU, up to a minute on a slow one - and caches the
  result. An NVIDIA card mining SHA3-256t does this in the background
  before the fork.

If you upgrade from 1.2.5 or older, read the next section too.

## Upgrading from 1.2.5 or older: check a pinned `--gpu-id` first

**If you pin a card with `--gpu-id` or `BC3_GPU_ID`, verify the number before
you upgrade.** Two things changed about how devices are numbered and selected.

The detected list is now de-duplicated. The same physical card can be exposed
by two OpenCL platforms — Mesa's rusticl beside ROCm, or two Intel runtimes —
and the miner used to treat those as two cards, start two workers on one GPU,
and report a full hashrate for each while they split one card's throughput
between them. They now collapse to one entry, so **positions after a removed
duplicate shift down**.

And an out-of-range id is now an error instead of a silent fall-through. It
used to produce an empty device list, which under the default `auto` backend
means "no GPU found — mining on CPU": a typo in `BC3_GPU_ID` looked exactly
like a card that had fallen off the bus, and on a host rented by the hour it
was billed at GPU prices the whole time. The miner now exits with a message
naming the valid range. That is deliberate, but note the consequence under
`Restart=always` or `restart: unless-stopped`: a stale id becomes a restart
loop rather than a quiet CPU run. Loud is the point; knowing about it first is
the courtesy.

Run `bc3-miner --probe` to see what each runtime exposes — but read it
carefully, because it is not the list `--gpu-id` indexes. The probe
deliberately shows **both** backends so you can tell whether each one works;
`--gpu-id` indexes only the backend actually in use, and under the default
`auto` that is CUDA whenever any CUDA device is found, and OpenCL only when
none is. On an NVIDIA machine the same card therefore appears twice in the
probe, once as `CUDA #n` and once as `OpenCL p.d` — and only the `CUDA #n`
numbering is the one `--gpu-id` means.

## Architecture

- **The proof of work is chosen per job**, from the block version, as the node's `CBlockHeader::GetHash` chooses it: bit 11 (`0x800`) is BC3Hash and wins over bit 12 (`0x1000`), SHA3-256t.
- The CPU builds coinbase/merkle root per extranonce2; the GPU grinds the 2³² nonce space in auto-tuned batches (~100 ms per launch for SHA3-256t, ~400 ms for BC3Hash). Every GPU hit is re-verified on the CPU against the consensus reference before submission.
- **The CUDA kernels are precompiled to PTX at build time** (see `build.rs`) and embedded in the binary. NVRTC ships with the CUDA *Toolkit*, not with the graphics driver, so runtime compilation would fail on end-user machines. The driver JITs the embedded PTX for whatever card is installed — the binary only needs `nvcuda.dll`.
- **The PTX is built with the CUDA 12.0 toolkit, and `build.rs` refuses anything newer.** A driver can only JIT the PTX ISA versions it knows: the embedded kernels, SHA3-256t and BC3Hash alike, declare ISA 8.0, which needs driver **R525 or newer**. Building it with a newer toolkit raises that requirement — ISA 8.8 from CUDA 12.9 needs R575 — and on an older driver `cuModuleLoadData` fails, the card is found but no backend opens, and the miner hashes nothing while looking connected. That shipped once; the version check in `build.rs` is there so it cannot ship again. If the CUDA load fails anyway, the miner now falls back to the same card through OpenCL, and exits with an error if no GPU worker survives and no CPU threads are running.
- **Both GPU runtimes are loaded dynamically** — CUDA via cudarc's fallback loader, OpenCL via `src/backend/cl_sys.rs`. Neither `nvcuda.dll` nor `OpenCL.dll` appears in the binary's import table, so one build runs everywhere: CUDA on NVIDIA, OpenCL on AMD/Intel, CPU if neither is present. Linking against `OpenCL.lib` instead would make the binary refuse to start on machines without an OpenCL runtime.

### BC3Hash

- **BC3Hash** of an 80-byte header is a final SHA3-256 over six tagged hashes of it: SHA3-384, SHA-512, RIPEMD-160, BLAKE2s, BLAKE2b, and BC3Mix of its SHA3-512. BC3Mix is where the work is: modular exponentiations with fixed exponents, table powers of fixed bases, 64 secp256k1 scalar multiplications from precomputed tables and 64 modular inversions, in four dependent stages.
- **On the CPU, BC3Hash is the node's own C++, not a port.** `bc3hash/` compiles the node's `src/crypto/bc3` and its secp256k1 (in `bc3hash/vendor`, at the commit in `vendor/SOURCE_COMMIT`) behind a small C shim. A port could differ from the node in one bit and no test of ours would notice; the node's own code checked against the node's own vectors cannot.
- **A compiler can build it wrong, so every start checks.** GCC 13 at -O3 turns this code into a program that runs and gets every hash wrong. The build works around that, and at startup the miner hashes four of the node's test vectors before it mines anything. A build that fails them never mines BC3Hash and says "this build computes BC3Hash incorrectly - report it", in the log and in the GUI.

### The BC3Hash GPU kernel

`src/kernels/bc3hash.cl` is a kernel of our own, built from the node's constants, with one source for CUDA and OpenCL like `sha3t.cl`:

- **32-bit limbs.** Products, squarings and reductions work on 32-bit limbs, which is what a GPU multiplies. Under CUDA they are inline PTX carry chains (`add.cc`/`addc`), 22 % faster on an RTX 3050 Ti than the same arithmetic in C; OpenCL compiles the C versions in the same file.
- **Inversions by safegcd** (Bernstein-Yang divsteps), ported from libsecp256k1's `modinv32`: about a quarter of the instructions of Fermat's 255 squarings. Scalar multiplications run four at a time and share one field inversion (Montgomery's trick).
- **Latency, not throughput, is the limit**: every step is a chain of dependent carries. The independent powers of a round run in pairs, so two chains interleave, and `__launch_bounds__` holds the kernel to 168 registers - three warps per scheduler instead of two. On an RTX 3050 Ti that was 30 % more hashes per clock.
- **Four table windows.** The curve tables live in GPU memory, in windows of 10, 12, 14 or 16 bits: 7, 22, 76 or 256 MiB. A bigger window means fewer point additions but more memory traffic, so the best one depends on the card's L2 cache. Releases mine with 16, the fastest on every card measured so far; the other three are there for the benchmark to compare (see *Internal tools*). The PTX is built once per window, and the host builds the matching tables at startup (about 3 s for window 16).
- **Checked bit-exact against the node.** In development: the node's reference vectors and 20,000 random BC3Mix/BC3Hash inputs at each of the four windows, and the PTX arithmetic against the C on inputs that reach every rare carry. On your machine, every card self-tests every window before it mines with it - the node's header vectors, random headers against the CPU reference, and a scan through the launch queue against the CPU's own hit set. A card that fails mines no BC3Hash, and under `--backend auto` the CPU takes over. Every GPU hit, of either proof of work, is re-checked on the CPU before it is submitted.

### SHA3-256t

- SHA3-256t = three sequential rounds of NIST SHA3-256 over the 80-byte block header. The header fits in a single SHA3-256 rate block, so each hash is exactly 3 keccak-f[1600] permutations.
- One shared kernel source (`src/kernels/sha3t.cl`) is compiled both to CUDA PTX (at build time) and by the OpenCL runtime. The keccak core is the same code for both; the two backends differ only in the intrinsics for the three-input XOR, the 32-bit rotate and the leading-zero count, and in whether the rounds are unrolled, all isolated in macros at the top of the file. Both are verified bit-exact against the same reference.
- The kernel keeps every 64-bit keccak lane **bit-interleaved**: one 32-bit word of the even bits and one of the odd bits. The host does the interleaving of the header and the target (`backend::kernel_lanes`, `backend::kernel_target`), the kernel interleaves the nonce, and only the most significant limb of the final hash leaves the GPU. See the note at the top of the kernel source for why.

## Performance

### BC3Hash

| Device | Backend | Hashrate | How measured |
|--------|---------|----------|--------------|
| NVIDIA RTX 4090 | CUDA | ~1.98 MH/s | bc3-miner 2.0.0 solving a real block template offline, on the operator's card |
| NVIDIA RTX 4050 Laptop GPU | CUDA | ~180 kH/s | reported by BC3's developer, with an earlier version of the kernel |
| NVIDIA RTX 3050 Ti Laptop GPU | CUDA, window 16 | ~165 kH/s | bc3-miner 2.0.0's benchmark (*Internal tools*); the card is power-capped at ~40 W |
| Intel Core i9-12900HK | CPU, 20 threads, fast tables | ~8 kH/s (~1.1 kH/s on one thread) | through the miner's worker loop, Linux, Clang build |

A laptop GPU's BC3Hash rate is set by its power limit as much as by the
kernel: the 3050 Ti above runs at its cap, where the 2.0.0 kernel - 30 %
more hashes per clock - runs at lower clocks than the one before it. A card
with power headroom gains more. On the laptop CPU the rate per thread falls
as threads are added: 1.1 kH/s on one thread, 3.9 on four, 6.3 on eight,
8.2 on all twenty.

### SHA3-256t (until block 64,359)

| Device | Backend | Hashrate | How measured |
|--------|---------|----------|--------------|
| NVIDIA RTX 3050 Ti Laptop GPU | CUDA, interleaved kernel (1.3.0) | ~165 MH/s | kernel benchmark, alternated A/B against the 1.2.x kernel: +4.6 % |
| NVIDIA RTX 4090 | CUDA, 1.2.x kernel | ~1.60 GH/s | in the miner, sustained |
| NVIDIA RTX 3050 Ti Laptop GPU | CUDA, 1.2.x kernel | ~158 MH/s | kernel benchmark |
| NVIDIA RTX 3050 Ti Laptop GPU + Intel Iris Xe | OpenCL, both devices | ~140–142 MH/s | in the miner, before the kernel work below |

The kernel is ALU-bound and at the hardware ceiling. SHA3-256t is three full
keccak-f[1600] permutations per nonce, and a GPU emulates its 64-bit
arithmetic with 32-bit ops, so throughput is set by how many integer
instructions the SM can issue - not by memory, occupancy or batch size. On the
3050 Ti the measured rate is within 0.1 % of 64 ALU instructions per clock per
SM at the clock sampled under load; the card is power-capped there, so cooling
and power limits move the hashrate more than anything else on the machine.

One keccak round compiles to ~168 SASS instructions on sm_86: 117 `LOP3` and
50 `SHF`. Every instruction is accounted for - 20 column parities, 50 theta,
50 chi, one or two iota and 47 rotates - and that is the floor for a machine
whose primitives are a three-input logic op and a funnel shift. Three things
get it there:

- **Bit-interleaved lanes.** Each 64-bit lane is one word of even bits and one
  of odd bits. A rotate by 2k is then two independent 32-bit rotates, and a
  rotate by 1 is a single one - the other word is a register rename. Keccak
  rotates by 1 six times per round (five in theta, one in rho); the word-split
  kernel this replaced spent two funnel shifts on each. Measured on the 3050
  Ti with the two kernels alternated in one process: 165.5 vs 158.2 MH/s,
  12 160 vs 12 592 SASS per hash, the same 64 registers.
- **Theta as three-input XOR.** Ampere and Ada compute an arbitrary function of
  three inputs in one `LOP3`. Writing theta's column sums and application that
  way, instead of chains of two-input XORs, folds `C[x-1]` and `ROTL(C[x+1],1)`
  into the same instruction.
- **Full unroll of the 24 rounds** (CUDA only). Removes the per-round overhead
  and lets ptxas fit the kernel in 64 registers, which is 32 warps per SM.
  Partial unrolling is *worse* than none.

Batches are auto-tuned toward ~100 ms of work per launch; on a 4090 a launch
that is too small pays full synchronisation overhead against very little work,
which shows up directly as lost GPU utilisation.

Measured and rejected, so nobody has to measure them again: moving the
rotates to the multiplier pipe (`IMAD.HI` and `IMAD.WIDE` are half rate on
Ampere and Ada, and five variants came out 22-33 % slower), two nonces per
thread with the chains interleaved, `__launch_bounds__`, block sizes from 128
to 1024, and pruning the last round by hand (ptxas already does it; 40
instructions of 12 000). The compiled kernel from `nvcc` 12.0, `nvcc` 12.4 and
the driver's own JIT are the same to within a handful of instructions.

OpenCL on the same NVIDIA card performs on par with CUDA. Adding the integrated
Iris Xe on top gained nothing measurable on this laptop: the discrete and
integrated GPUs share a thermal and power budget, so the driver throttles one to
feed the other (hashrate swung 92–159 MH/s with the GPU pinned at 75 °C before
settling back to ~140). On a desktop, where that coupling is weaker, combining
devices with `--backend opencl` may be worth measuring.

## Building

```
cargo build --release   # CUDA + OpenCL + BC3Hash + CPU (all default)
```

No GPU SDK is needed to build: neither runtime is linked, both are resolved at
run time. To leave one out — `--no-default-features --features cuda,bc3hash` or
`--no-default-features --features opencl,bc3hash`. Leave `bc3hash` in: without
it the miner cannot mine after the fork.

BC3Hash needs a C++20 compiler with `unsigned __int128`: GCC or Clang. The
build prefers `clang` when it finds one (about 10 % more hashrate than GCC 14)
and warns when it has to fall back. On Windows, `bc3hash/build.rs` uses
`clang-cl` from an LLVM install instead of MSVC's `cl.exe`, which cannot build
it. The node's sources are in `bc3hash/vendor` (commit in
`vendor/SOURCE_COMMIT`) and its test vectors in `bc3hash/vectors`;
`BC3HASH_VENDOR` and `BC3HASH_VECTORS` point the build and the tests
somewhere else.

### Internal tools

`--benchmark` (mining without a pool: every GPU, its table windows compared,
every share checked against the CPU) and `--solve` (mining one fixed block
template offline) are tools for development, compiled in only with a
feature that no release enables:

```
cargo build --release --features internal-tools
```

With them come `--bc3-window`, `--bc3hash-mode` and the benchmark's own
options. A release build has none of these, not even hidden. AGENTS.md
describes them.

### Tests

CPU tests (consensus vectors, the node's BC3Hash vectors through the miner's own path, and a Rust mirror of the SHA3-256t kernel, verified bit-exactly against the chain-verified reference):

```
cargo test
```

`cargo test` inside `bc3hash/` runs the node's reference vectors against its C++ build as well. `cargo test --features internal-tools` adds the tests of the internal tools.

GPU tests (compare the full GPU hit set against the CPU reference on random headers) are `#[ignore]`d by default and require an NVIDIA GPU:

```
docker build -t bc3-miner-cuda-dev -f docker/Dockerfile.cuda-dev .
docker run --rm --gpus all -v <repo>:/work -v bc3-cargo-registry:/root/.cargo/registry \
  -w /work bc3-miner-cuda-dev cargo test --release -- --ignored --nocapture
```

The OpenCL test compares the full GPU hit set against the CPU reference the same
way. It needs an OpenCL runtime but **not** a GPU — [pocl](https://portablecl.org/)
provides a CPU one, so it runs in plain Docker:

```
docker run --rm -v <repo>:/work -v bc3-cargo-registry:/usr/local/cargo/registry \
  -w /work rust:1-trixie bash -c \
  "apt-get update -qq && apt-get install -y -qq pocl-opencl-icd ocl-icd-libopencl1 && \
   cargo test --release opencl -- --ignored --test-threads=1"
```

## License

MIT. `bc3hash/vendor` holds the node's BC3Hash sources and its libsecp256k1,
each under its own MIT license (`COPYING` in each directory).
