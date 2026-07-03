#!/usr/bin/env bash
# scripts/pod_finish_batch.sh — ON THE POD: the full v2 finishing chain, one command (STATUS §0).
#
#   1) recon → kit-boost export (crowd QUILT default) → beauty render   (pod_make_video.sh;
#      PHYSICS=1 and DEMO_EDITS=0 default inside it)
#   2) night-grade (grade3 — the eye-picked recipe)
#   3) rgb-control Wan-VACE over the night-graded frames (the winning variant, STATUS §6)
#   4) SeedVR2 720p upscale
#   5) team-mask AOV pass (same frame list as the beauty pass; 1 sample — cheap)
#   6) hue-pin: undo the v2v/SeedVR2 kit-hue drift — target auto-measured from THIS run's
#      beauty render (TARGET_HUE env = manual override)
#
# Finishing is per-camera (v2v eats one frame dir): ANIM_CAMERAS must be ONE camera.
# Env: OUT=out/anim_finish  ANIM_CAMERAS=sideline  REUSE_SCENE=0  TARGET_HUE=  DILATE=15
set -euo pipefail
cd /workspace/fifa
. scripts/video_defaults.sh

PY="${PITCH3D_PY:-/workspace/.venv/bin/python}"
CAM="${ANIM_CAMERAS:-sideline}"
case "$CAM" in *,*) echo "finish_batch: ANIM_CAMERAS must be a single camera" >&2; exit 1;; esac
BATCH_OUT="${OUT:-out/anim_finish}"
echo "== BATCH finish start $(date -u +%H:%M:%SZ) :: cam=$CAM out=$BATCH_OUT =="

# 1) recon + kit-boosted QUILT export + beauty render
PITCH3D_CLIP="${PITCH3D_CLIP:-/workspace/Colombia-1-0-Congo-DR1080p.mp4}" \
  REUSE_SCENE="${REUSE_SCENE:-0}" ANIM_CAMERAS="$CAM" OUT="$BATCH_OUT" \
  bash scripts/pod_make_video.sh

# 2) night-grade (grade3)
GRADE='eq=brightness=-0.28:contrast=1.12:gamma=0.75:saturation=0.9,colorbalance=bs=0.12:bm=0.06'
SRC="$BATCH_OUT/mesh/frames/$CAM"
DST="$BATCH_OUT/mesh/frames_night/$CAM"
mkdir -p "$DST"
for f in "$SRC"/frame_*.png; do
  ffmpeg -y -v error -i "$f" -vf "$GRADE" "$DST/$(basename "$f")"
done
echo "== night grade done: $(ls "$DST" | wc -l) frames =="

# 3) v2v: rgb control over the night-graded frames.
# V2V must be ABSOLUTE: pod_seedvr2.sh cd's into its own repo before testing INPUTS.
V2V="$(realpath -m "$BATCH_OUT")/v2v/${CAM}_rgbnight.mp4"
FRAMES="$DST" OUT="$V2V" bash scripts/pod_v2v.sh --control rgb

# 4) SeedVR2 720p
INPUTS="$V2V" bash scripts/pod_seedvr2.sh
V2V720="${V2V%.mp4}_720p.mp4"
test -f "$V2V720" || { echo "finish_batch: missing $V2V720" >&2; exit 1; }

# 5) team-mask AOV pass — same frame list as the beauty pass (masks are resized to the
# video by hue_pin, so 832x480 is plenty)
BMODE="$(PITCH3D_PY="$PY" bash scripts/pod_ensure_blender.sh)"
case "$BMODE" in
  BLENDER_MODE=module)   RENDER=("$PY" scripts/blender_animate.py);;
  BLENDER_MODE=binary:*) RENDER=("${BMODE#BLENDER_MODE=binary:}" --background --python scripts/blender_animate.py --);;
  *) echo "finish_batch: blender unavailable ($BMODE)" >&2; exit 1;;
esac
"${RENDER[@]}" \
  --in "$BATCH_OUT/mesh" --out "$BATCH_OUT/mesh/mask" --team-mask 1 --cameras "$CAM" \
  --device "${ANIM_DEVICE:-$VIDEO_DEVICE_DEFAULT}" --res-x 832 --res-y 480 \
  --frame-step "${ANIM_STEP:-$VIDEO_STEP_DEFAULT}"

# 6) hue-pin the finished clip back to THIS run's render hue (or TARGET_HUE)
PINNED="${V2V720%.mp4}_pinned.mp4"
PIN=(--video "$V2V720" --mask-dir "$BATCH_OUT/mesh/mask/$CAM" --out "$PINNED"
     --channel g --dilate "${DILATE:-15}")
if [ -n "${TARGET_HUE:-}" ]; then
  PIN+=(--target-hue "$TARGET_HUE")
else
  PIN+=(--target-from-frames "$SRC")
fi
"$PY" scripts/hue_pin.py "${PIN[@]}"

echo "BATCH_FINISH_OK"
echo "  beauty video : $BATCH_OUT/video/$CAM.mp4"
echo "  v2v 480p     : $V2V"
echo "  seedvr2 720p : $V2V720"
echo "  FINAL pinned : $PINNED"
