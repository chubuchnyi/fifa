#!/usr/bin/env bash
# scripts/pod_cluster_ab.sh — ON THE POD: player-cluster A/B (v2 lever, 2026-07-03).
#
# Diagnosis (out/clusters/control_vs_output_f10.png): in tight player clusters the night-
# graded control keeps clean separable geometry but grade3 all but erased kit colour — every
# player is the same dark-teal mannequin, so Wan gets shape without identity and hallucinates
# white/orange shirts. Fix: control_kit_inject.py pushes H/S toward each team's kit colour
# inside the team-mask AOV (V kept for shading/limb boundaries), then the SAME 720p v2v runs
# on the injected control. CS (VACE conditioning_scale, default 1.0) is a second knob.
#
# Needs a finished pod_finish_batch.sh run on the volume (frames_night + mask + frames).
# Env: OUT=out/anim_finish  ANIM_CAMERAS=sideline  ALPHA=0.8  ERODE=3  CS=1.0  TAG=kitinj
#      WIDTH=1280 HEIGHT=720 FLOW=5.0 RES=720 BATCH=17  INJECT_ONLY=1 → stop after step 1
set -euo pipefail
cd /workspace/fifa

BATCH_OUT="${OUT:-out/anim_finish}"
CAM="${ANIM_CAMERAS:-sideline}"
PY="${PITCH3D_PY:-/workspace/.venv/bin/python}"
CS="${CS:-1.0}"
TAG="${TAG:-kitinj}"
RES="${RES:-720}"

NIGHT="$BATCH_OUT/mesh/frames_night/$CAM"
MASKS="$BATCH_OUT/mesh/mask/$CAM"
test -d "$NIGHT" || { echo "cluster_ab: missing $NIGHT (run pod_finish_batch.sh first)" >&2; exit 1; }
test -d "$MASKS" || { echo "cluster_ab: missing $MASKS (run pod_finish_batch.sh first)" >&2; exit 1; }

# 1) kit-colour injection into the control frames
INJ="$BATCH_OUT/mesh/frames_nightkit/$CAM"
"$PY" scripts/control_kit_inject.py --frames "$NIGHT" --masks "$MASKS" --out "$INJ" \
  --alpha "${ALPHA:-0.8}" --erode "${ERODE:-3}"
if [ "${INJECT_ONLY:-0}" = "1" ]; then
  echo "CLUSTER_AB_INJECT_ONLY ok: $INJ"
  exit 0
fi

# 2) v2v 720p on the injected control — same seed/steps/prompt as the finish chain
V2V="$(realpath -m "$BATCH_OUT")/v2v/${CAM}_rgbnight_${TAG}.mp4"
FRAMES="$INJ" OUT="$V2V" bash scripts/pod_v2v.sh --control rgb \
  --width "${WIDTH:-1280}" --height "${HEIGHT:-720}" --flow-shift "${FLOW:-5.0}" \
  --conditioning-scale "$CS"

# 3) SeedVR2 restore (input already >= RES: detail pass, not an upscale)
INPUTS="$V2V" RES="$RES" BATCH="${BATCH:-17}" bash scripts/pod_seedvr2.sh
V2V_SR="${V2V%.mp4}_${RES}p.mp4"
test -f "$V2V_SR" || { echo "cluster_ab: missing $V2V_SR" >&2; exit 1; }

# 4) hue-pin, auto target from this run's beauty render
PINNED="${V2V_SR%.mp4}_pinned.mp4"
"$PY" scripts/hue_pin.py --video "$V2V_SR" --mask-dir "$MASKS" \
  --out "$PINNED" --channel g --dilate 15 \
  --target-from-frames "$BATCH_OUT/mesh/frames/$CAM"

echo "CLUSTER_AB_OK"
echo "  injected ctrl : $INJ"
echo "  v2v ${TAG}    : $V2V"
echo "  seedvr2 ${RES}p : $V2V_SR"
echo "  FINAL pinned  : $PINNED"
