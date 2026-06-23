#!/usr/bin/env bash
# scripts/pod_real_e2e.sh — one real-model E2E pass on a GPU pod.
#
# Runs the SAME golden path as docs/cloud-dev.md §5 (real video -> RF-DETR detect ->
# ByteTrack -> assemble -> overlay render + smplx_npz/glTF export) but with the wired
# **real SMPLest-X** pose + **real WASB** ball backends injected by dotted path (ADR-0006)
# — i.e. the genuine single-cam -> world-SMPL-X path, end to end, on CUDA.
#
# Env (all have sensible pod defaults — override to point elsewhere):
#   PITCH3D_REPO=/workspace/fifa            repo root (PYTHONPATH=src is set from here)
#   PITCH3D_PY=/workspace/.venv/bin/python  interpreter with cuda torch + rfdetr + smplx
#   PITCH3D_CLIP=/workspace/clip.mp4        input broadcast clip
#   FRAMES=8                                --frames
#   OUT=out/run                             --out-dir (relative to repo)
# The SMPLest-X backend reads its own env (PITCH3D_SMPLESTX_REPO / _CKPT / _DEVICE;
# defaults /workspace/repos/SMPLest-X + smplest_x_h on cuda) — see the backend factory.
# The WASB ball backend likewise reads PITCH3D_WASB_REPO / _CKPT / _DATASET / _DEVICE
# (defaults /workspace/repos/WASB-SBDT + wasb_soccer_best.pth.tar on cuda).
set -euo pipefail

REPO="${PITCH3D_REPO:-/workspace/fifa}"
PY="${PITCH3D_PY:-/workspace/.venv/bin/python}"
CLIP="${PITCH3D_CLIP:-/workspace/clip.mp4}"
FRAMES="${FRAMES:-8}"
OUT="${OUT:-out/run}"

cd "$REPO"
echo "== pod real E2E :: frames=${FRAMES} out=${OUT} clip=${CLIP} =="
t0=$(date +%s)
PYTHONPATH=src "$PY" -m pitch3d \
  --clip "$CLIP" --frames "$FRAMES" \
  --detector rfdetr --tracker bytetrack --device cuda \
  --pose gvhmr --pose-backend pitch3d.adapters.models.smplestx_backend:make \
  --ball tracknet --ball-backend pitch3d.adapters.models.wasb_backend:make \
  --render overlay --export gltf --format smplx_npz --out-dir "$OUT"
echo "== done in $(( $(date +%s) - t0 ))s -> ${OUT} =="
ls -la "${OUT}/export/scene.smplx_npz" 2>/dev/null | head
