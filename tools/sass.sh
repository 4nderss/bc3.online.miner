#!/bin/sh
# Machine code (SASS) of the BC3Hash kernel for the GPU architectures miners
# have, from the committed PTX: what the driver's JIT would make of it, made
# once when the miner is built instead of on every card at every start.
#
# Run in the CUDA 12.9 toolkit image, from the repository root:
#   docker run --rm -v "$PWD:/work" -w /work nvidia/cuda:12.9.1-devel-ubuntu22.04 sh tools/sass.sh
#
# 12.9 because it is the newest toolkit whose machine code every CUDA 12
# driver loads (R525 and up, as the PTX needs), and it knows Blackwell
# (sm_120). Its ptxas makes the same code as the driver's JIT: on sm_86 the
# same 119,232 instructions, 168 registers and 3360 bytes of stack, 568 lines
# in a different order. The CUDA backend falls back to the PTX for any card
# whose architecture is not here, and for a driver that refuses the code.
#
# Writes src/kernels/sass/bc3hash_<name>_sm<arch>.cubin for every PTX - each
# table window's and the table builder's - and manifest.txt with the SHA-256
# of each PTX they were made from (line endings normalised: a Windows
# checkout has CRLF). build.rs embeds a kernel's machine code only while its
# PTX still matches - never code of an older kernel.
set -eu
ARCHS="${BC3_SASS_ARCHS:-61 75 86 89 120}"
OUT="${1:-src/kernels/sass}"
PTXAS="${PTXAS:-/usr/local/cuda/bin/ptxas}"
export PTXAS

mkdir -p "$OUT"
rm -f "$OUT"/*.cubin "$OUT"/manifest.txt

for ptx in src/kernels/bc3hash_*.ptx; do
  base=$(basename "$ptx" .ptx)
  for sm in $ARCHS; do
    printf '%s sm_%s %s/%s_sm%s.cubin\n' "$ptx" "$sm" "$OUT" "$base" "$sm"
  done
done | xargs -P "$(nproc)" -L 1 sh -c '"$PTXAS" -arch="$1" -O3 "$0" -o "$2" && echo "  $2"'

{
  echo "# $("$PTXAS" --version | tail -n 1)"
  echo "# archs: $ARCHS"
  for ptx in src/kernels/bc3hash_*.ptx; do
    printf '%s  %s\n' "$(tr -d '\r' < "$ptx" | sha256sum | cut -d' ' -f1)" "$(basename "$ptx")"
  done
} > "$OUT/manifest.txt"
cat "$OUT/manifest.txt"
