# bc3.online miner

GPU/CPU miner for BC3 (BitcoinIII, SHA3-256t). Connects to the [bc3.online](https://bc3.online) pool — PPLNS or solo.

- **Core:** Rust — CUDA (NVRTC) and OpenCL GPU backends sharing one kernel source, plus multi-threaded CPU mining
- **Pool endpoints:** `bc3.online:3111` (PPLNS), `bc3.online:3112` (solo)
- **Protocol:** Stratum v1

## Download

Release binaries are published as zip packages under [Releases](../../releases), with SHA256 checksums.

> **Note:** mining software is often flagged by antivirus heuristics (false positive). Verify the checksum of your download against the release notes.

## Quick start

```
bc3-miner --user <your bc1... address>.<rigname>
```

Backend selection is automatic: CUDA if an NVIDIA GPU is present, otherwise OpenCL, otherwise CPU. Options:

| Flag | Description |
|------|-------------|
| `--pool host:port` | Pool address (default `bc3.online:3111`, solo `bc3.online:3112`) |
| `--backend auto\|cuda\|opencl\|cpu` | Force a backend (default `auto`) |
| `--gpu-id N` | Use only the N:th detected GPU |
| `--threads N` | CPU threads (default: all cores in CPU mode, none in GPU mode — set explicitly to mine with both) |

Solo rewards are paid directly to your address in the block you find.

## Architecture

- SHA3-256t = three sequential rounds of NIST SHA3-256 over the 80-byte block header. The header fits in a single SHA3-256 rate block, so each hash is exactly 3 keccak-f[1600] permutations.
- One shared kernel source (`src/kernels/sha3t.cl`) is compiled both by NVRTC (CUDA) and by the OpenCL runtime — the keccak core is byte-for-byte identical for both backends.
- The CPU builds coinbase/merkle root per extranonce2; the GPU grinds the 2³² nonce space in auto-tuned batches (~100 ms per launch). Every GPU hit is re-verified on the CPU against the consensus reference before submission.
- CUDA and NVRTC libraries are loaded dynamically at runtime — the same binary runs on machines without an NVIDIA driver and falls back gracefully.

## Performance

| Device | Backend | Hashrate |
|--------|---------|----------|
| NVIDIA RTX 3050 Ti Laptop GPU | CUDA | ~140 MH/s |

Measured with the fully unrolled keccak-f[1600] kernel (release build, 2²⁴ nonces per launch). The kernel is ALU-bound: SHA3-256t is three full keccak permutations of 64-bit arithmetic per nonce, which GPUs emulate with 32-bit ops.

## Building

```
cargo build --release                    # CUDA backend (default) + CPU
cargo build --release --features opencl  # + OpenCL backend
```

The OpenCL feature links against the system OpenCL loader (`OpenCL.dll` / `libOpenCL.so`).

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

The OpenCL runtime test (`opencl_matches_cpu_on_random_headers`) requires a machine with a native OpenCL GPU runtime (e.g. Windows with NVIDIA/AMD drivers) — it cannot run in WSL/Docker. Since both backends compile the same kernel source, bit-exactness verified via CUDA carries over to the OpenCL kernel.

## License

MIT
