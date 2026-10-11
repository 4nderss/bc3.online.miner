#!/usr/bin/env bash
# Builds the WASM kernel for the web miner and copies it to the web repo.
#
# Runs in Docker (Smart App Control blocks local Rust builds on Windows):
#   docker run --rm -v C:/dev/bc3.online/miner:/work \
#     -v C:/dev/bc3.online/web:/web \
#     -v C:/dev/bc3.online/pool/crates/bc3hash:/bc3:ro \
#     -e BC3HASH_VENDOR=/bc3/vendor -e BC3HASH_VECTORS=/bc3/vectors \
#     -v bc3-cargo-registry:/usr/local/cargo/registry \
#     -w /work/wasm rust:1-trixie ./build.sh
#
# BC3Hash (the default `bc3hash` feature) is the node's C++, compiled for
# wasm32 by clang against WASI's libc/libc++ headers: BC3HASH_VENDOR is the
# directory with the node's src/crypto/bc3 and src/secp256k1 as bc3/ and
# secp256k1/ (see bc3hash/build.rs), BC3HASH_VECTORS its doc/bc3/vectors.
# Without the vectors the module is built but not shipped: the host tests
# check the host build of the C++, and only test/vectors.mjs checks the
# wasm32 build that actually runs in the browser.
set -euo pipefail

TARGET=wasm32-unknown-unknown
OUT=target/$TARGET/release/bc3_miner_wasm.wasm

# The target is not in the base image and disappears with the container.
rustup target list --installed | grep -qx "$TARGET" || rustup target add "$TARGET"

# clang for wasm32 and WASI's headers and libc++ (Debian trixie's packages;
# libc++-19 to match clang 19), and node for the vector harness.
if ! command -v clang >/dev/null || [ ! -e /usr/lib/wasm32-wasi/libc++.a ] || ! command -v node >/dev/null; then
  echo "== Installing clang, WASI libc/libc++ and node =="
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    clang lld wasi-libc libc++-19-dev-wasm32 libc++abi-19-dev-wasm32 nodejs >/dev/null
fi

: "${BC3HASH_VECTORS:?set BC3HASH_VECTORS to the vectors directory of the node (doc/bc3/vectors)}"

echo "== Unit tests (host platform) =="
cargo test --lib

echo
echo "== Building $TARGET =="
cargo build --release --target "$TARGET"

echo
echo "== The node's vectors against the wasm32 build =="
node test/vectors.mjs "$OUT" "$BC3HASH_VECTORS" 5

SIZE=$(stat -c%s "$OUT")
echo
echo "OK: $OUT ($SIZE bytes)"
sha256sum "$OUT"

# Copy to the web repo if it is mounted alongside.
if [ -d /web/vendor ]; then
  cp "$OUT" /web/vendor/bc3-miner.wasm
  echo "Copied to /web/vendor/bc3-miner.wasm"
fi
