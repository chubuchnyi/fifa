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
#   FORMAT=smplx_npz                        --format (json carries the ball; smplx_npz = bodies)
#   STITCH=1                                --stitch    (continuity; off unless =1)
#   COHERENCE=1                             --coherence (gap-fill + smoothing; off unless =1)
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
FORMAT="${FORMAT:-smplx_npz}"

# Optional REAL calibration: set PNLCALIB_REPO (+ its weights) to swap the proxy field
# calibrator for the wired PnLCalib backend (ADR-0006 dotted path). Empty/absent -> proxy.
CALIB_ARGS=()
if [ -n "${PNLCALIB_REPO:-}" ] && [ -d "${PNLCALIB_REPO}" ]; then
  CALIB_ARGS=(--calibrator keypoints
              --calibrator-backend pitch3d.adapters.models.pnlcalib_backend:make)
  echo "== calibration: REAL PnLCalib (${PNLCALIB_REPO})"
else
  echo "== calibration: proxy (set PNLCALIB_REPO to a staged checkout to enable real PnLCalib)"
fi

# Optional continuity (Step 1-2) + temporal coherence (Step 3). Both off unless set to 1:
#   STITCH=1     re-link fragmented tracklets before POSE so occluded players don't
#                'appear from nowhere' (real ByteTrack fragments under occlusion).
#   COHERENCE=1  bridge short interior pose gaps (slerp/lerp) + auto temporal-smoothing.
COH_ARGS=()
if [ "${STITCH:-0}" = "1" ]; then COH_ARGS+=(--stitch); echo "== continuity: --stitch ON"; fi
if [ "${COHERENCE:-0}" = "1" ]; then COH_ARGS+=(--coherence); echo "== coherence: --coherence ON"; fi

cd "$REPO"
echo "== pod real E2E :: frames=${FRAMES} out=${OUT} format=${FORMAT} clip=${CLIP} =="
t0=$(date +%s)
PYTHONPATH=src "$PY" -m pitch3d \
  --clip "$CLIP" --frames "$FRAMES" \
  --detector rfdetr --tracker bytetrack --device cuda \
  "${CALIB_ARGS[@]}" "${COH_ARGS[@]}" \
  --pose gvhmr --pose-backend pitch3d.adapters.models.smplestx_backend:make \
  --ball tracknet --ball-backend pitch3d.adapters.models.wasb_backend:make \
  --render overlay --export gltf --format "$FORMAT" --out-dir "$OUT"
echo "== done in $(( $(date +%s) - t0 ))s -> ${OUT} =="
ls -la "${OUT}/export/scene.${FORMAT}" 2>/dev/null | head
