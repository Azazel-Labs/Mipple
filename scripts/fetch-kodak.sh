#!/usr/bin/env bash
# Downloads the 24 Kodak Lossless True Color test images (768x512 PNG) into test-images/kodak.
# They are the de facto common corpus for texture-compression results (RAD and Geldreich both use them).
set -euo pipefail
dir="$(cd "$(dirname "$0")/.." && pwd)/test-images/kodak"
mkdir -p "$dir"
for i in $(seq -w 1 24); do
  f="$dir/kodim$i.png"
  [ -s "$f" ] && continue
  echo "fetching kodim$i.png"
  curl -fsSL -o "$f" "https://r0k.us/graphics/kodak/kodak/kodim$i.png"
done
echo "done: $(ls "$dir" | wc -l) images in $dir"
