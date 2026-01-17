#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <folder>"
  exit 1
fi

folder="${1%/}"
if [[ ! -d "$folder" ]]; then
  echo "Error: not a directory: $folder" >&2
  exit 1
fi

base="$(basename "$folder")"
dir="$(dirname "$folder")"
out="${base}.tar.zst"

tar -C "$dir" -cf - "$base" | zstd -T0 -19 --long=31 -o "$out"
echo "Wrote: $out"
