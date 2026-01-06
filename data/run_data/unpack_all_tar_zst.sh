#!/usr/bin/env bash
set -e

# Script for unpacking all *.tar.zst files in the current directory.
cd "$(dirname "$0")"
shopt -s nullglob

archives=( *.tar.zst )  # Get the list of all *.tar.zst files in the current directory.
if (( ${#archives[@]} == 0 )); then
  echo "No .tar.zst files found in $(pwd)"
  exit 0
fi

for f in "${archives[@]}"; do  # Extract each archive.
  echo "Extracting: $f"
  tar -I "zstd -d -T0" -xf "$f"
done

echo "Done."
