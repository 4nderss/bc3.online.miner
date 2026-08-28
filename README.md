# bc3.online miner

GPU/CPU miner for BC3 (BitcoinIII, SHA3-256t). Connects to the [bc3.online](https://bc3.online) pool — PPLNS or solo.

- **Core:** Rust — CUDA and OpenCL GPU backends sharing one kernel source, plus multi-threaded CPU mining
- **GPUs:** NVIDIA via CUDA; AMD and Intel (including integrated graphics) via OpenCL
- **Pool endpoints:** `bc3.online:3111` (PPLNS), `bc3.online:3112` (solo)
- **Protocol:** Stratum v1

## Download

Release binaries are published as zip packages under [Releases](../../releases), with SHA256 checksums.

> **Note:** mining software is often flagged by antivirus heuristics (false positive). Verify the checksum of your download against the release notes.

## Quick start — GUI

Run **bc3-miner-gui.exe**. Paste your BC3 address, name your rig, pick PPLNS or
Solo, press *Start mining*. The window shows live hashrate with a sparkline,
accepted/rejected shares, estimated time to block, which backend is running,
and a full-screen celebration when you find a block. Settings are remembered
between runs.

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

## Architecture

- SHA3-256t = three sequential rounds of NIST SHA3-256 over the 80-byte block header. The header fits in a single SHA3-256 rate block, so each hash is exactly 3 keccak-f[1600] permutations.
- One shared kernel source (`src/kernels/sha3t.cl`) is compiled both to CUDA PTX (at build time) and by the OpenCL runtime — the keccak core is byte-for-byte identical for both backends.
- The CPU builds coinbase/merkle root per extranonce2; the GPU grinds the 2³² nonce space in auto-tuned batches (~100 ms per launch). Every GPU hit is re-verified on the CPU against the consensus reference before submission.
- **The CUDA kernel is precompiled to PTX at build time** (see `build.rs`) and embedded in the binary. NVRTC ships with the CUDA *Toolkit*, not with the graphics driver, so runtime compilation would fail on end-user machines. The driver JITs the embedded PTX for whatever card is installed — the binary only needs `nvcuda.dll`.
- **Both GPU runtimes are loaded dynamically** — CUDA via cudarc's fallback loader, OpenCL via `src/backend/cl_sys.rs`. Neither `nvcuda.dll` nor `OpenCL.dll` appears in the binary's import table, so one build runs everywhere: CUDA on NVIDIA, OpenCL on AMD/Intel, CPU if neither is present. Linking against `OpenCL.lib` instead would make the binary refuse to start on machines without an OpenCL runtime.

## Performance

| Device | Backend | Hashrate |
|--------|---------|----------|
| NVIDIA RTX 3050 Ti Laptop GPU | CUDA | ~140 MH/s |
| NVIDIA RTX 3050 Ti Laptop GPU + Intel Iris Xe | OpenCL, both devices | ~140–142 MH/s |

Measured with the fully unrolled keccak-f[1600] kernel (release build, 2²⁴ nonces per launch). The kernel is ALU-bound: SHA3-256t is three full keccak permutations of 64-bit arithmetic per nonce, which GPUs emulate with 32-bit ops.

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
