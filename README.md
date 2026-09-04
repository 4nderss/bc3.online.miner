# bc3.online miner

GPU/CPU miner for BC3 (BitcoinIII, SHA3-256t). Connects to the [bc3.online](https://bc3.online) pool — PPLNS or solo.

- **Core:** Rust — CUDA and OpenCL GPU backends sharing one kernel source, plus multi-threaded CPU mining
- **GPUs:** NVIDIA via CUDA; AMD and Intel (including integrated graphics) via OpenCL
- **Pool endpoints:** `bc3.online:3111` (PPLNS), `bc3.online:3112` (solo)
- **Protocol:** Stratum v1

## Download

Release binaries are published as zip packages under [Releases](../../releases), with SHA256 checksums.

> **Note:** mining software is often flagged by antivirus heuristics (false positive). Verify the checksum of your download against the `SHA256SUMS` file published with it.

Releases also carry a build provenance attestation. A checksum only proves the
file matches what was uploaded; the attestation is a signed statement binding
the file to the commit and the workflow that built it, which is the part a
checksum cannot establish:

```
gh attestation verify bc3-miner-1.2.0-windows-x86_64.zip --repo 4nderss/bc3.online.miner
```

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
| `--pool host:port` | Pool address (default `bc3.online:3111`, solo `bc3.online:3112`) |
| `--backend auto\|cuda\|opencl\|cpu` | Force a backend (default `auto`) |
| `--gpu-id N` | Use only the N:th detected GPU |
| `--threads N` | CPU threads. Default: all cores in CPU mode, **none** in GPU mode — pass the flag to mine with both. `--threads 0` means "all cores", minus the GPU feeder threads when a GPU is running |
| `--intensity 1-100` | Percentage of full speed (default `100`). Below 100 the workers idle between passes, which lowers heat and power and keeps the machine usable while mining |
| `--stats-interval N` | Seconds between statistics lines (default `5`) |
| `--json` | Emit machine-readable JSON events on stdout (used by the GUI) |
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

## Upgrading: check a pinned `--gpu-id` first

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

- SHA3-256t = three sequential rounds of NIST SHA3-256 over the 80-byte block header. The header fits in a single SHA3-256 rate block, so each hash is exactly 3 keccak-f[1600] permutations.
- One shared kernel source (`src/kernels/sha3t.cl`) is compiled both to CUDA PTX (at build time) and by the OpenCL runtime. The keccak core is the same code for both; the two backends differ only in how the three-input XOR is expressed and in whether the rounds are unrolled, both isolated in macros at the top of the file. Both are verified bit-exact against the same reference.
- The CPU builds coinbase/merkle root per extranonce2; the GPU grinds the 2³² nonce space in auto-tuned batches (~100 ms per launch). Every GPU hit is re-verified on the CPU against the consensus reference before submission.
- **The CUDA kernel is precompiled to PTX at build time** (see `build.rs`) and embedded in the binary. NVRTC ships with the CUDA *Toolkit*, not with the graphics driver, so runtime compilation would fail on end-user machines. The driver JITs the embedded PTX for whatever card is installed — the binary only needs `nvcuda.dll`.
- **The PTX is built with the CUDA 12.0 toolkit, and `build.rs` refuses anything newer.** A driver can only JIT the PTX ISA versions it knows: the embedded kernel declares ISA 8.0, which needs driver **R525 or newer**. Building it with a newer toolkit raises that requirement — ISA 8.8 from CUDA 12.9 needs R575 — and on an older driver `cuModuleLoadData` fails, the card is found but no backend opens, and the miner hashes nothing while looking connected. That shipped once; the version check in `build.rs` is there so it cannot ship again. If the CUDA load fails anyway, the miner now falls back to the same card through OpenCL, and exits with an error if no GPU worker survives and no CPU threads are running.
- **Both GPU runtimes are loaded dynamically** — CUDA via cudarc's fallback loader, OpenCL via `src/backend/cl_sys.rs`. Neither `nvcuda.dll` nor `OpenCL.dll` appears in the binary's import table, so one build runs everywhere: CUDA on NVIDIA, OpenCL on AMD/Intel, CPU if neither is present. Linking against `OpenCL.lib` instead would make the binary refuse to start on machines without an OpenCL runtime.

## Performance

| Device | Backend | Hashrate | How measured |
|--------|---------|----------|--------------|
| NVIDIA RTX 4090 | CUDA | ~1.60 GH/s | in the miner, sustained |
| NVIDIA RTX 3050 Ti Laptop GPU | CUDA | ~163 MH/s | kernel benchmark |
| NVIDIA RTX 3050 Ti Laptop GPU + Intel Iris Xe | OpenCL, both devices | ~140–142 MH/s | in the miner, before the kernel work below |

The kernel is ALU-bound and close to the hardware ceiling. SHA3-256t is three
full keccak-f[1600] permutations per nonce, and a GPU emulates its 64-bit
arithmetic with 32-bit ops, so throughput is set by how many integer
instructions the SM can issue - not by memory, occupancy or batch size.

One keccak round compiles to 185 SASS instructions on sm_89: 122 `LOP3` and 58
`SHF`, plus loop overhead. That is the floor for this formulation, and every
instruction is accounted for - 70 three-input XORs, 50 chi, 2 iota, 29 rotations
at two funnel shifts each. Three things get it there:

- **Theta as three-input XOR.** Ampere and Ada compute an arbitrary function of
  three inputs in one `LOP3`. Writing theta's column sums and application that
  way, instead of chains of two-input XORs, folds `C[x-1]` and `ROTL(C[x+1],1)`
  into the same instruction and drops 14 instructions per round.
- **Full unroll of the 24 rounds** (CUDA only). Removes the per-round overhead
  and lets ptxas fit the kernel in 64 registers instead of 80, lifting occupancy
  from 25 to 32 warps per SM. Partial unrolling is *worse* than none.
- **Batches large enough to matter.** Nonces per launch are auto-tuned toward
  ~100 ms of work. On a 4090 a launch that is too small pays full
  synchronisation overhead against very little work, which shows up directly as
  lost GPU utilisation.

OpenCL on the same NVIDIA card performs on par with CUDA. Adding the integrated
Iris Xe on top gained nothing measurable on this laptop: the discrete and
integrated GPUs share a thermal and power budget, so the driver throttles one to
feed the other (hashrate swung 92–159 MH/s with the GPU pinned at 75 °C before
settling back to ~140). On a desktop, where that coupling is weaker, combining
devices with `--backend opencl` may be worth measuring.

## Building

```
cargo build --release   # CUDA + OpenCL + CPU (all default)
```

No GPU SDK is needed to build: neither runtime is linked, both are resolved at
run time. To leave one out — `--no-default-features --features cuda` or
`--features opencl`.

### Tests

CPU tests (consensus vectors + a Rust mirror of the kernel algorithm, verified bit-exactly against the chain-verified reference):

```
cargo test
```

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

MIT
