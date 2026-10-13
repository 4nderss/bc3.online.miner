# bc3.online miner

GPU/CPU miner for BC3 (BitcoinIII): BC3Hash, its proof of work since block 64,360. Connects to the [bc3.online](https://bc3.online) pool — PPLNS or solo.

- **Core:** Rust — CUDA and OpenCL GPU backends sharing one kernel source, plus multi-threaded CPU mining
- **Proof of work:** BC3Hash, on a GPU kernel of our own and on the CPU through the node's own C++. Each job's block version must select it, as the node reads it; the miner mines nothing else
- **GPUs:** NVIDIA via CUDA; AMD and Intel (including integrated graphics) via OpenCL
- **Pool endpoints:** `bc3.online:3111` (PPLNS), `bc3.online:3112` (solo)
- **Protocol:** Stratum v1

## BC3Hash, since the hard fork at block 64,360

At block 64,360 BC3 replaced SHA3-256t with **BC3Hash**, a proof of work of
its own, and changed how difficulty is set:

- **Version bit 11 (`0x800`) selects BC3Hash**, and it takes precedence over
  bit 12 (`0x1000`, SHA3-256t). The node sets it in every block template from
  the fork height on. The miner reads the proof of work from each job's
  version, the way the node's `GetHash` does, so there is no height rule in
  the miner that could disagree with the node - and it mines BC3Hash only. A
  job without bit 11 is not mined at all: the miner idles on it and says why,
  in the log and in the GUI, rather than hash it into shares the pool would
  reject.
- **Difficulty is ASERT** (aserti9-2d) instead of 2016-block epochs, and it
  adjusts every block. Block 64,360 has a fixed difficulty of 16; after it,
  each block's difficulty follows from how far the chain is ahead of or
  behind an absolute ten-minute schedule, with a half-life of two days.
- **Hashrates are about three orders of magnitude below the old ones.** One
  BC3Hash costs about as much as 800 SHA3-256t hashes, so a card that did
  GH/s before the fork does MH/s or kH/s. That is the algorithm, not a fault.

2.0.0 mined SHA3-256t up to the fork and switched to BC3Hash by itself. The
chain is past the fork now, and the miner has no SHA3-256t any more.

**1.x cannot mine since the fork.** A 1.x miner has no BC3Hash, and the pool
refuses it with a message that points to the [releases page](../../releases);
the GUI of 1.2.0 and later also shows a banner linking to the new release.
Docker users pull the new image - see *Quick start — Docker* below.

When the pool starts a connection at a very low share difficulty, a fast card
gets "too many shares per second" answers until vardiff has raised it. Since
2.1.0 a miner with NVIDIA cards asks the pool for a starting difficulty that
suits them (`mining.suggest_difficulty`): an RTX 4090 starts at a few
shares a second rather than hundreds. A pool that knows the rig's recent
shares starts it where they say instead. Where it still happens - another device, another
pool - it is harmless, and the miner prints the first such reject at once and
then one line every 10 s counting the repeats.

## Download

Release binaries are published as zip packages under [Releases](../../releases), with SHA256 checksums.

> **Note:** mining software is often flagged by antivirus heuristics (false positive). Verify the checksum of your download against the `SHA256SUMS` file published with it.

Releases also carry a build provenance attestation. A checksum only proves the
file matches what was uploaded; the attestation is a signed statement binding
the file to the commit and the workflow that built it, which is the part a
checksum cannot establish:

```
gh attestation verify bc3-miner-2.1.0-windows-x86_64.zip --repo 4nderss/bc3.online.miner
```

The Windows binaries need the Microsoft Visual C++ Redistributable (x64),
which most machines already have. Since 2.0.0 the miner uses its
`MSVCP140.dll` as well as the `VCRUNTIME140.dll` 1.x needed; if Windows says either is
missing, install the redistributable from Microsoft. On NVIDIA cards the GPU
kernel needs driver R525 or newer; with an older one the miner falls back to
the card's OpenCL runtime.

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

Image tags follow the releases: `ghcr.io/4nderss/bc3-miner:2.1.0` (and
`:2.1`) is this version, and `:latest` - what the command above and
`docker-compose.yml` use - moves to a release when it is published. **The
fork needs 2.0.0 or later.** A container keeps running the image it was
created from, so pull the new one and recreate the container:

```
docker pull ghcr.io/4nderss/bc3-miner:latest
docker rm -f bc3-miner        # then the docker run above again
```

With compose: `docker compose pull && docker compose up -d`. Pin `:2.1.0`
instead of `:latest` if you would rather choose when to move.

An NVIDIA card spends its first minutes measuring which BC3Hash table window
suits it (see *The BC3Hash GPU kernel*), mining all along. The container
keeps the answer, so a restart goes straight to it; a new container - an
upgrade, say - measures again.

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

Replace the binaries, or pull the new image: 1.x cannot mine since block
64,360. The address, rig name, pool ports and every 1.x flag work as before,
and the GUI keeps its settings. What is new:

- BC3Hash needs memory for its tables: 256 MiB of GPU memory per card to
  start with - up to 3.3 GiB on an NVIDIA card that has the room and hashes
  faster with a bigger table window (see *The BC3Hash GPU kernel*) - and RAM
  for the CPU's: about 7 MiB when a GPU mines BC3Hash and the CPU only
  re-checks its hits, about 256 MiB when the CPU mines it.
- At startup the miner checks its BC3Hash build against the node's test
  vectors, and every GPU checks its kernel against the CPU before it mines.
  A card that fails mines no BC3Hash - under `--backend auto` the CPU takes
  over - and the log and the GUI say why.
- The first time a card needs its BC3Hash kernel, the driver compiles it -
  about 20 s on a fast CPU, up to a minute on a slow one - and caches the
  result.

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

- **The proof of work is read per job** from the block version, as the node's `CBlockHeader::GetHash` reads it: bit 11 (`0x800`) is BC3Hash and wins over bit 12 (`0x1000`, SHA3-256t). Only BC3Hash is mined: every worker sits out a job without bit 11, and the status says why.
- The CPU builds coinbase/merkle root per extranonce2; the GPU grinds the 2³² nonce space in auto-tuned batches (~400 ms per launch). Every GPU hit is re-verified on the CPU against the consensus reference before submission.
- **The CUDA kernel is precompiled to PTX at build time** (see `build.rs`), once per table window, and embedded in the binary. NVRTC ships with the CUDA *Toolkit*, not with the graphics driver, so runtime compilation would fail on end-user machines. The driver JITs the embedded PTX for whatever card is installed — the binary only needs `nvcuda.dll`.
- **The PTX is built with the CUDA 12.0 toolkit, and `build.rs` refuses anything newer.** A driver can only JIT the PTX ISA versions it knows: the embedded kernel declares ISA 8.0, which needs driver **R525 or newer**. Building it with a newer toolkit raises that requirement — ISA 8.8 from CUDA 12.9 needs R575 — and on an older driver `cuModuleLoadData` fails, the card is found, and the miner hashes nothing on it while looking connected. That shipped once; the version check in `build.rs` is there so it cannot ship again. With a driver older than R525 the CUDA backend does not open the card at all, and the miner falls back to the same card through OpenCL; it exits with an error if no GPU worker survives and no CPU threads are running.
- **Both GPU runtimes are loaded dynamically** — CUDA via cudarc's fallback loader, OpenCL via `src/backend/cl_sys.rs`. Neither `nvcuda.dll` nor `OpenCL.dll` appears in the binary's import table, so one build runs everywhere: CUDA on NVIDIA, OpenCL on AMD/Intel, CPU if neither is present. Linking against `OpenCL.lib` instead would make the binary refuse to start on machines without an OpenCL runtime.

### BC3Hash

- **BC3Hash** of an 80-byte header is a final SHA3-256 over six tagged hashes of it: SHA3-384, SHA-512, RIPEMD-160, BLAKE2s, BLAKE2b, and BC3Mix of its SHA3-512. BC3Mix is where the work is: modular exponentiations with fixed exponents, table powers of fixed bases, 64 secp256k1 scalar multiplications from precomputed tables and 64 modular inversions, in four dependent stages.
- **On the CPU, BC3Hash is the node's own C++, not a port.** `bc3hash/` compiles the node's `src/crypto/bc3` and its secp256k1 (in `bc3hash/vendor`, at the commit in `vendor/SOURCE_COMMIT`) behind a small C shim. A port could differ from the node in one bit and no test of ours would notice; the node's own code checked against the node's own vectors cannot.
- **A compiler can build it wrong, so every start checks.** GCC 13 at -O3 turns this code into a program that runs and gets every hash wrong. The build works around that, and at startup the miner hashes four of the node's test vectors before it mines anything. A build that fails them never mines BC3Hash and says "this build computes BC3Hash incorrectly - report it", in the log and in the GUI.

### The BC3Hash GPU kernel

`src/kernels/bc3hash.cl` is a kernel of our own, built from the node's constants, with one source for CUDA and OpenCL:

- **32-bit limbs, one instruction a product.** Products, squarings and reductions work on 32-bit limbs, which is what a GPU multiplies. Under CUDA each 32x32 product, together with the 64-bit addition into its accumulator and both carries, is a single `IMAD.WIDE.U32.X`: inline PTX `madc` chains, arranged by column parity so that ptxas can fuse them. That made the whole hash 23 % faster than 2.0.0's arithmetic on an RTX 3050 Ti; OpenCL compiles the C versions in the same file.
- **Inversions by safegcd** (Bernstein-Yang divsteps), ported from libsecp256k1's `modinv32`: about a quarter of the instructions of Fermat's 255 squarings. Scalar multiplications run four at a time and share one field inversion (Montgomery's trick).
- **Latency, not throughput, is the limit**: every step is a chain of dependent carries. The independent powers of a round run in pairs, so two chains interleave, and `__launch_bounds__` holds the kernel to 168 registers - three warps per scheduler instead of two. On an RTX 3050 Ti that was 30 % more hashes per clock.
- **Each card measures its own table window.** The curve tables live in GPU memory, in windows of 16, 18, 19 or 20 bits: 256 MiB, 0.9, 1.8 or 3.3 GiB. A bigger window means fewer point additions - 13 a scalar multiplication at 20 bits instead of 16 - but more memory traffic, so whether it pays depends on the card: on an RTX 4090 window 20 hashes 5.6 % faster than 16. Nobody can try every card, so every NVIDIA card starts at 16 and mines from the first second, builds the tables of the largest window it has the memory for meanwhile (and keeps a GiB free), and then measures: 16, the bigger window and 16 again, a minute each. It keeps the bigger window only if that hashes faster than the two measurements of 16 on average. The answer is remembered per card, driver and miner version - in `%LOCALAPPDATA%\bc3-miner\tuning.json` on Windows, `~/.cache/bc3-miner/tuning.json` on Linux - so the next start goes straight there; a new driver or version measures again, and so does deleting the file. When the two measurements of 16 disagree, something else moved the rate - in a laptop, the CPU taking the shared power budget - and the answer holds for that run only. The GUI shows the tables in use on the *Backend* card, and "tuning…" while it measures. OpenCL cards stay at 16: a switch there compiles the kernel while the card waits, and the tuning has not been tried on an AMD or Intel card yet.
- **Checked bit-exact against the node.** In development: the node's reference vectors and 20,000 random BC3Mix/BC3Hash inputs at window 16, and the PTX arithmetic against the C on inputs that reach every rare carry. On your machine, every card self-tests every window before it mines with it - the node's header vectors, random headers against the CPU reference, and a scan through the launch queue against the CPU's own hit set. A card that fails mines no BC3Hash, and under `--backend auto` the CPU takes over; a bigger window that fails leaves the card at 16. Every GPU hit is re-checked on the CPU before it is submitted.

## Performance

| Device | Backend | Hashrate | How measured |
|--------|---------|----------|--------------|
| NVIDIA RTX 4090 | CUDA, window 20 | ~2.66 MH/s (2.52 at window 16) | bc3-miner 2.1.0's benchmark (*Internal tools*), 60 s runs, on the operator's card at its 450 W limit; 2.0.0 did 1.82 |
| NVIDIA RTX 4050 Laptop GPU | CUDA | ~180 kH/s | reported by BC3's developer, with an earlier version of the kernel |
| NVIDIA RTX 3050 Ti Laptop GPU | CUDA, window 16 | ~215 kH/s | 2.1.0's benchmark; the card is power-capped at ~40 W; 2.0.0 did ~165 |
| Intel Core i9-12900HK | CPU, 20 threads, fast tables | ~8 kH/s (~1.1 kH/s on one thread) | through the miner's worker loop, Linux, Clang build |

Both cards above run at their power limit, so their rate follows the energy
a hash takes. 2.1.0's kernel does the same work in fewer instructions - a
quarter less energy a hash - and gains 30 % on the laptop and 38 % on the
RTX 4090 at the same window. On the laptop CPU the rate per thread falls as threads are
added: 1.1 kH/s on one thread, 3.9 on four, 6.3 on eight, 8.2 on all twenty.

## Building

```
cargo build --release   # CUDA + OpenCL + BC3Hash + CPU (all default)
```

No GPU SDK is needed to build: neither runtime is linked, both are resolved at
run time. To leave one out — `--no-default-features --features cuda` or
`--no-default-features --features opencl`; either brings `bc3hash` with it,
the one kernel both backends have. `--no-default-features --features bc3hash`
builds a miner for the CPU alone. Without `bc3hash` there is nothing to mine.

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

CPU tests (consensus vectors, the node's BC3Hash vectors through the miner's own path, the CPU and GPU worker pipelines, the pool protocol):

```
cargo test
```

`cargo test` inside `bc3hash/` runs the node's reference vectors against its C++ build as well. `cargo test --features internal-tools` adds the tests of the internal tools.

GPU tests (the BC3Hash kernel's self-test, and its hit sets through the launch queue against the CPU reference) are `#[ignore]`d by default and require an NVIDIA GPU:

```
docker build -t bc3-miner-cuda-dev -f docker/Dockerfile.cuda-dev .
docker run --rm --gpus all -v <repo>:/work -v bc3-cargo-registry:/root/.cargo/registry \
  -w /work bc3-miner-cuda-dev cargo test --release -- --ignored --nocapture
```

The OpenCL tests do the same through OpenCL. They need an OpenCL runtime but
**not** a GPU — [pocl](https://portablecl.org/) provides a CPU one, so they run
in plain Docker (building the kernel takes pocl minutes):

```
docker run --rm -v <repo>:/work -v bc3-cargo-registry:/usr/local/cargo/registry \
  -w /work rust:1-trixie bash -c \
  "apt-get update -qq && apt-get install -y -qq pocl-opencl-icd ocl-icd-libopencl1 && \
   cargo test --release opencl -- --ignored --test-threads=1"
```

## License

MIT. `bc3hash/vendor` holds the node's BC3Hash sources and its libsecp256k1,
each under its own MIT license (`COPYING` in each directory).
