# bc3.online miner

GPU/CPU miner for BC3 (BitcoinIII, SHA3-256t) with a clean Windows GUI. Connects to the [bc3.online](https://bc3.online) pool — PPLNS or solo.

- **Core:** Rust + OpenCL kernel (NVIDIA / AMD / Intel), optional CPU mining
- **GUI:** Tauri (WebView2) — live hashrate, block hits, estimated time-to-block for solo
- **Pool endpoints:** `bc3.online:3111` (PPLNS), `bc3.online:3112` (solo)
- **Protocol:** Stratum v1

## Download

Release binaries are published as zip packages under [Releases](../../releases), with SHA256 checksums.

> **Note:** mining software is often flagged by antivirus heuristics (false positive). Verify the checksum of your download against the release notes.

## Quick start

1. Download and unzip the latest release.
2. Run `bc3-miner.exe`, paste your BC3 wallet address (`bc1...`), name your rig, pick PPLNS or solo.
3. Start mining — solo rewards are paid directly to your address in the block you find.

## License

MIT
