#!/usr/bin/env bash
# Bygger WASM-kärnan för web-minern och kopierar den till webbrepot.
#
# Körs i Docker (Smart App Control blockerar lokala Rust-byggen på Windows):
#   docker run --rm -v C:/dev/bc3.online/miner:/work \
#     -v bc3-cargo-registry:/usr/local/cargo/registry \
#     -w /work/wasm rust:1-trixie ./build.sh
set -euo pipefail

TARGET=wasm32-unknown-unknown
OUT=target/$TARGET/release/bc3_miner_wasm.wasm

# Targetet ligger inte i basimagen och försvinner med containern.
rustup target list --installed | grep -qx "$TARGET" || rustup target add "$TARGET"

echo "== Enhetstester (värdplattformen) =="
cargo test --lib

echo
echo "== Bygger $TARGET =="
cargo build --release --target "$TARGET"

SIZE=$(stat -c%s "$OUT")
echo
echo "OK: $OUT ($SIZE bytes)"
sha256sum "$OUT"

# Kopiera till webbrepot om det är monterat bredvid.
if [ -d /web/vendor ]; then
  cp "$OUT" /web/vendor/bc3-miner.wasm
  echo "Kopierad till /web/vendor/bc3-miner.wasm"
fi
