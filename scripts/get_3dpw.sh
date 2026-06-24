#!/usr/bin/env bash
# scripts/get_3dpw.sh — stage the 3DPW dataset onto the (persistent) volume.
#
# 3DPW (von Marcard et al., ECCV'18) is a credentialed-but-direct MPI download — the
# three zips are served over plain HTTPS with no auth. Lands under $DEST:
#   readme_and_demo/
#   sequenceFiles/{train,validation,test}/*.pkl   (SMPL GT + per-frame camera; ~265 MB)
#   imageFiles/<sequence>/image_*.jpg             (frames; ~4.3 GB)
# Idempotent: skips a zip that is already downloaded, unzips with -n (no overwrite).
#
# If you use 3DPW you agree to cite the ECCV'18 paper (see docs/pose-bakeoff-runbook.md).
#
# Usage (on the GPU pod):  DEST=/workspace/datasets/3dpw scripts/get_3dpw.sh
set -euo pipefail
DEST="${DEST:-/workspace/datasets/3dpw}"
BASE="https://virtualhumans.mpi-inf.mpg.de/3DPW"

command -v unzip >/dev/null || { apt-get update -qq && apt-get install -y -qq unzip; }
mkdir -p "$DEST"; cd "$DEST"
for z in readme_and_demo sequenceFiles imageFiles; do
  if [ ! -f "$z.zip" ]; then
    echo "== downloading $z.zip"
    curl -fSL --retry 3 -o "$z.zip" "$BASE/$z.zip"
  fi
  echo "== unzipping $z.zip"
  unzip -n -q "$z.zip"
done
echo "== staged -> $DEST"
find "$DEST" -maxdepth 2 -type d | sort
du -sh "$DEST"/* 2>/dev/null || true
