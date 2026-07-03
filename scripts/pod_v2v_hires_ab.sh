#!/usr/bin/env bash
# scripts/pod_v2v_hires_ab.sh — ON THE POD: player-crispness A/B (v2 lever, 2026-07-03).
#
# Diagnosis (crisp_zoom_*_f30.png): distant players are clean in the 720p beauty render but
# turn to mush in Wan-VACE at 832x480 — ~15-25 px tall there, i.e. 2-3 latent px after the
# VAE's 8x spatial compression; SeedVR2 then only sharpens the mush. This cell re-runs the
# SAME night-graded frames through v2v at higher resolution (default 1280x720, flow-shift 5.0
# per Wan's 720p recipe), SeedVR2-restores 1:1, and hue-pins — so the ONLY changed variable
# vs the finish run is the v2v resolution.
#
# Fallback if 720p OOMs or breaks structure: WIDTH=1104 HEIGHT=624 TAG=hi624 (multiple of 16).
# Env: OUT=out/anim_finish  ANIM_CAMERAS=sideline  WIDTH=1280 HEIGHT=720 FLOW=5.0
#      RES=720 (SeedVR2 target)  BATCH=17  TAG=hi720
set -euo pipefail
cd /workspace/fifa

BATCH_OUT="${OUT:-out/anim_finish}"
CAM="${ANIM_CAMERAS:-sideline}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FLOW="${FLOW:-5.0}"
RES="${RES:-720}"
TAG="${TAG:-hi${HEIGHT}}"
PY="${PITCH3D_PY:-/workspace/.venv/bin/python}"

NIGHT="$BATCH_OUT/mesh/frames_night/$CAM"
test -d "$NIGHT" || { echo "hires_ab: missing $NIGHT (run pod_finish_batch.sh first)" >&2; exit 1; }
V2V="$(realpath -m "$BATCH_OUT")/v2v/${CAM}_rgbnight_${TAG}.mp4"

# 1) v2v at the higher resolution — same frames, same control, same seed
FRAMES="$NIGHT" OUT="$V2V" bash scripts/pod_v2v.sh --control rgb \
  --width "$WIDTH" --height "$HEIGHT" --flow-shift "$FLOW"

# 2) SeedVR2 restore (input already >= RES: detail pass, not an upscale)
INPUTS="$V2V" RES="$RES" BATCH="${BATCH:-17}" bash scripts/pod_seedvr2.sh
V2V_SR="${V2V%.mp4}_${RES}p.mp4"
test -f "$V2V_SR" || { echo "hires_ab: missing $V2V_SR" >&2; exit 1; }

# 3) hue-pin (masks + auto target from the finish run's artifacts on the volume)
PINNED="${V2V_SR%.mp4}_pinned.mp4"
"$PY" scripts/hue_pin.py --video "$V2V_SR" --mask-dir "$BATCH_OUT/mesh/mask/$CAM" \
  --out "$PINNED" --channel g --dilate 15 \
  --target-from-frames "$BATCH_OUT/mesh/frames/$CAM"

echo "AB_HIRES_OK"
echo "  v2v ${WIDTH}x${HEIGHT} : $V2V"
echo "  seedvr2 ${RES}p       : $V2V_SR"
echo "  FINAL pinned          : $PINNED"
