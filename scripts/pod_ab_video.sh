#!/usr/bin/env bash
# scripts/pod_ab_video.sh — ON THE POD: one variant (A=SMPLest-X | B=SAM 3D Body) of the pose
# A/B bake-off, full real E2E + multi-angle Blender video, into its OWN out dir.
#
# Thin wrapper over scripts/pod_make_video.sh that injects the per-variant pose backend and —
# for B — the native ABI env scripts/run_sam3dbody.sh sets (LD_LIBRARY_PATH=<torch>/lib +
# pymomentum-gpu==0.1.90.post0, else the MHR rig load segfaults on Blackwell/torch 2.8). A and B
# share IDENTICAL detect/track/calibrate/render knobs so they differ ONLY in the pose net.
#
# Usage (on the pod, from /workspace/fifa):
#   VARIANT=A            bash scripts/pod_ab_video.sh    # SMPLest-X  → out/anim_A
#   VARIANT=B            bash scripts/pod_ab_video.sh    # SAM 3D Body → out/anim_B
#   VARIANT=A MODE=smoke bash scripts/pod_ab_video.sh    # 4f · 1 cam · 8spp · 640x360 sanity+timing
#
# Env knobs (defaults in parens): VARIANT(A) MODE(full) OUT(out/anim_$VARIANT) FRAMES(60)
#   CAMERAS(broadcast,sideline,top,goal) SAMPLES(32) RES(1280x720) PREWARM(1). MODE=smoke
#   overrides the render to a fast single-camera pass (and suffixes OUT with _smoke). PREWARM=1
#   bulk-reads the venv+weights into page cache first (skips the ~20-min cold MooseFS import
#   stall; set PREWARM=0 to skip). Model/repo/weight paths default to the euro net-volume layout.
set -euo pipefail

VARIANT="${VARIANT:-A}"
MODE="${MODE:-full}"
REPO="${PITCH3D_REPO:-/workspace/fifa}"
PY="${PITCH3D_PY:-/workspace/.venv/bin/python}"
CLIP="${PITCH3D_CLIP:-/workspace/Colombia-1-0-Congo-DR1080p.mp4}"
cd "$REPO"

# Shared model/repo/weight env — identical for A and B (the upstream half is the same net-for-net).
export PITCH3D_REPO="$REPO" PITCH3D_PY="$PY" PITCH3D_CLIP="$CLIP"
export PITCH3D_SMPLESTX_REPO="${PITCH3D_SMPLESTX_REPO:-/workspace/repos/SMPLest-X}"
export PITCH3D_SMPLX_MODEL_PATH="${PITCH3D_SMPLX_MODEL_PATH:-/workspace/repos/SMPLest-X/human_models/human_model_files}"
export PITCH3D_SMPLX_MODELS="${PITCH3D_SMPLX_MODELS:-$PITCH3D_SMPLX_MODEL_PATH}"
export PITCH3D_WASB_REPO="${PITCH3D_WASB_REPO:-/workspace/repos/WASB-SBDT}"
export PITCH3D_WASB_CKPT="${PITCH3D_WASB_CKPT:-/workspace/weights/wasb/wasb_soccer_best.pth.tar}"
export PITCH3D_WASB_DATASET="${PITCH3D_WASB_DATASET:-soccer}"
export PNLCALIB_REPO="${PNLCALIB_REPO:-/workspace/repos/PnLCalib}"
export PNLCALIB_WEIGHTS_KP="${PNLCALIB_WEIGHTS_KP:-/workspace/weights/pnlcalib/SV_kp}"
export PNLCALIB_WEIGHTS_LINES="${PNLCALIB_WEIGHTS_LINES:-/workspace/weights/pnlcalib/SV_lines}"
# Measured appearance (stadium/crowd/body texture/floodlight colour) samples the SAME clip.
export PITCH3D_STADIUM_VIDEO="${PITCH3D_STADIUM_VIDEO:-$CLIP}"

case "$VARIANT" in
  A)
    export POSE_BACKEND="pitch3d.adapters.models.smplestx_backend:make"
    ;;
  B)
    # Native ABI env (mirror run_sam3dbody.sh): the pymomentum solver links libtorch.so at
    # import, so <torch>/lib must be on the loader path for this process AND its children
    # (pod_real_e2e imports the backend inside).
    TL="$("$PY" -c 'import torch,os;print(os.path.dirname(torch.__file__)+"/lib")')"
    export LD_LIBRARY_PATH="$TL:${LD_LIBRARY_PATH:-}"
    if [ "${PYM_PIN:-1}" = "1" ]; then
      have="$("$PY" -c 'import importlib.metadata as m;print(m.version("pymomentum-gpu"))' 2>/dev/null || echo none)"
      if [ "$have" != "0.1.90.post0" ]; then
        echo "== pin pymomentum-gpu==0.1.90.post0 (was: $have) — newer wheels segfault on Blackwell/torch2.8"
        "$PY" -m pip install -q --force-reinstall --no-deps "pymomentum-gpu==0.1.90.post0"
      fi
    fi
    export POSE_BACKEND="pitch3d.adapters.models.sam3dbody_backend:make"
    ;;
  *) echo "pod_ab_video: VARIANT must be A or B (got '$VARIANT')" >&2; exit 2;;
esac

# ── page-cache prewarm — dodge the MooseFS network-volume latency stall ──
# The venv + model weights live on the euro MooseFS FUSE mount. A COLD run
# otherwise starves the GPU for ~20 min: Python imports + torch.load pay a
# per-file FUSE round-trip tax (~150 KB/s EFFECTIVE) even though bulk bandwidth
# is fine (measured 417 MB/s cold, 6.1 GB/s warm — the kernel page cache DOES
# persist across processes here). Bulk-sequential reads pull the trees into
# cache so the real run loads from RAM. B needs sam-3d-body(2.0G)+MHR(664M) too.
# Override: PREWARM=0 to skip · PREWARM_JOBS to tune parallelism.
if [ "${PREWARM:-1}" = "1" ]; then
  PY_VENV="$(dirname "$(dirname "$PY")")"
  PW_TREES=( "$PY_VENV" "$REPO/src" "$PITCH3D_SMPLESTX_REPO" \
             "$PNLCALIB_REPO" "$PITCH3D_WASB_REPO" "$(dirname "$PITCH3D_SMPLX_MODEL_PATH")" )
  [ "$VARIANT" = "B" ] && PW_TREES+=( "${PITCH3D_SAM3D_REPO:-/workspace/repos/sam-3d-body}" \
                                      "${PITCH3D_MHR_REPO:-/workspace/repos/MHR}" )
  echo "== prewarm page cache (PREWARM=1): ${PW_TREES[*]}"
  _pw_t=$(date +%s)
  find "${PW_TREES[@]}" -type f -print0 2>/dev/null \
    | xargs -0 -P"${PREWARM_JOBS:-48}" -n8 cat > /dev/null 2>&1 || true
  echo "== prewarm done in $(( $(date +%s) - _pw_t ))s"
fi

OUT="${OUT:-out/anim_$VARIANT}"
if [ "$MODE" = "smoke" ]; then
  FRAMES="${FRAMES:-4}"; CAMERAS="${CAMERAS:-broadcast}"; SAMPLES="${SAMPLES:-8}"; RES="${RES:-640x360}"; OUT="${OUT}_smoke"
else
  FRAMES="${FRAMES:-60}"; CAMERAS="${CAMERAS:-broadcast,sideline,top,goal}"; SAMPLES="${SAMPLES:-32}"; RES="${RES:-1280x720}"
fi
RES_X="${RES%x*}"; RES_Y="${RES#*x}"

echo "== pod_ab_video VARIANT=$VARIANT MODE=$MODE pose=$POSE_BACKEND OUT=$OUT frames=$FRAMES cams=$CAMERAS ${RES_X}x${RES_Y} @${SAMPLES}spp $(date) =="
FRAMES="$FRAMES" OUT="$OUT" ANIM_CAMERAS="$CAMERAS" ANIM_SAMPLES="$SAMPLES" \
  ANIM_RES_X="$RES_X" ANIM_RES_Y="$RES_Y" ANIM_DEVICE="${ANIM_DEVICE:-gpu}" \
  STITCH="${STITCH:-1}" COHERENCE="${COHERENCE:-1}" PHYSICS="${PHYSICS:-1}" DEMO_EDITS="${DEMO_EDITS:-0}" \
  bash scripts/pod_make_video.sh
echo "== pod_ab_video VARIANT=$VARIANT MODE=$MODE OUT=$OUT DONE $(date) =="
