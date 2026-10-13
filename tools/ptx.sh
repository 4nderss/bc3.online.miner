#!/bin/sh
# The PTX of the BC3Hash kernel, from src/kernels/bc3hash.cl: one per table
# window, and the curve table builder. The one list of what is built with
# which defines - CI regenerates the committed PTX with it and compares.
#
# Run in the CUDA 12.0 toolkit image, from the repository root, mounted at
# /work (nvcc derives the names of internal functions from the source's
# absolute path, and the committed PTX was built there):
#   docker run --rm -v "$PWD:/work" -w /work nvidia/cuda:12.0.1-devel-ubuntu22.04 sh tools/ptx.sh
#
# 12.0 because the toolkit decides the PTX ISA version, and a driver JITs
# only the versions it knows: ISA 8.0 loads on driver R525 and up. build.rs
# refuses anything above it. See AGENTS.md.
#
# Windows 16-20 are uniform: 256 / w digits of w bits, rounded up, the last
# one shorter. 22 and 24 are mixed schedules, digits of two widths that cover
# the 256 bits exactly - the fewest digits for the memory: 12 digits of 21
# and 22 bits in 8.6 GB, where 12 of 22 bits would be 12.9 GB, and 11 of 23
# and 24 bits in 30.1 GB. A window's id is its widest digit. BC3_BUILD_ONLY
# is the kernel that builds the curve tables on the card.
#
# Writes <out>/bc3hash_<name>.ptx; <out> defaults to src/kernels.
set -eu
OUT="${1:-src/kernels}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
mkdir -p "$OUT"
while read -r name defines; do
  "$NVCC" --ptx -arch compute_52 -O3 $defines -x cu src/kernels/bc3hash.cl -o "$OUT/bc3hash_$name.ptx"
  echo "  $OUT/bc3hash_$name.ptx: $defines"
done <<EOF
w16 -DBC3_WINDOW=16
w18 -DBC3_WINDOW=18
w19 -DBC3_WINDOW=19
w20 -DBC3_WINDOW=20
w22 -DBC3_WINDOW=21 -DBC3_WIDE=4
w24 -DBC3_WINDOW=23 -DBC3_WIDE=3
build -DBC3_BUILD_ONLY
EOF
