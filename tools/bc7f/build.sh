#!/usr/bin/env bash
# Builds bc7f_encode from source.
#   BASIS_DIR   = clone of https://github.com/BinomialLLC/basis_universal (Apache 2.0)
#   BC7ENC_DIR  = clone of https://github.com/richgel999/bc7enc_rdo (for lodepng)
# Example:
#   BASIS_DIR=~/src/basis_universal BC7ENC_DIR=~/src/bc7enc_rdo tools/bc7f/build.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
: "${BASIS_DIR:?set BASIS_DIR to a basis_universal clone}"
: "${BC7ENC_DIR:?set BC7ENC_DIR to a bc7enc_rdo clone}"
out="${OUT:-$here/bc7f_encode}"
${CXX:-g++} -O3 -std=c++17 -DNDEBUG \
  -DBASISD_SUPPORT_KTX2=0 -DBASISD_SUPPORT_KTX2_ZSTD=0 \
  -I"$BASIS_DIR/transcoder" -I"$BC7ENC_DIR" \
  "$here/bc7f_encode.cpp" "$BASIS_DIR/transcoder/basisu_transcoder.cpp" "$BC7ENC_DIR/lodepng.cpp" \
  -o "$out"
echo "built $out"
