#!/usr/bin/env bash
# Builds the WASM kernel for the web miner and copies it to the web repo.
#
# Runs in Docker (Smart App Control blocks local Rust builds on Windows):
#   docker run --rm -v C:/dev/bc3.online/miner:/work \
#     -v bc3-cargo-registry:/usr/local/cargo/registry \
#     -w /work/wasm rust:1-trixie ./build.sh
set -euo pipefail

TARGET=wasm32-unknown-unknown
OUT=target/$TARGET/release/bc3_miner_wasm.wasm

# The target is not in the base image and disappears with the container.
rustup target list --installed | grep -qx "$TARGET" || rustup target add "$TARGET"

echo "== Unit tests (host platform) =="
cargo test --lib

echo
echo "== Bygger $TARGET =="
cargo build --release --target "$TARGET"

SIZE=$(stat -c%s "$OUT")
echo
echo "OK: $OUT ($SIZE bytes)"
sha256sum "$OUT"

# Copy to the web repo if it is mounted alongside.
if [ -d /web/vendor ]; then
  cp "$OUT" /web/vendor/bc3-miner.wasm
  echo "Kopierad till /web/vendor/bc3-miner.wasm"
fi
