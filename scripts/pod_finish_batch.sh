#!/usr/bin/env bash
# scripts/pod_finish_batch.sh — ON THE POD: the full v2 finishing chain, one command (STATUS §0).
#
#   1) recon → kit-boost export (crowd QUILT default) → beauty render   (pod_make_video.sh;
#      PHYSICS=1 and DEMO_EDITS=0 default inside it)
#   2) night-grade (grade3 — the eye-picked recipe)
#   3) team-mask AOV pass (same frame list as the beauty pass; 1 sample — cheap)
#   4) kit-colour re-injection into the control (cluster fix, A/B 2026-07-03): grade3 erases
#      kit colour in tight clusters → Wan gets shape without identity and hallucinates kits;
#      KIT_INJECT=0 reverts to the plain night frames
#   5) rgb-control Wan-VACE over the injected night frames (the winning variant, STATUS §6)
#   6) SeedVR2 720p upscale
#   7) hue-pin team B: undo the v2v/SeedVR2 kit-hue drift — target auto-measured from THIS
#      run's beauty render (TARGET_HUE env = manual override)
#   8) hue-pin team A (yellow band 5-80, sat-min 0.35 keeps faces out) — PIN_A=0 skips
#
# Finishing is per-camera (v2v eats one frame dir): ANIM_CAMERAS must be ONE camera.
# Env: OUT=out/anim_finish  ANIM_CAMERAS=sideline  REUSE_SCENE=0  TAIL_ONLY=0  TARGET_HUE=  DILATE=15
#      V2V_WIDTH=1280 V2V_HEIGHT=720 V2V_FLOW=5.0  (832x480/3.0 = the old fast-draft cell)
#      V2V_PROMPT= (override the measured default text prompt in pod_v2v_finish.py)
#      KIT_INJECT=1  ALPHA=0.8  ERODE=3  CS=1.0  PIN_A=1
#      TEAM_A_HSV/TEAM_B_HSV: unset = validated "65 0.85"/"185 0.95"; set-empty = auto-measure
set -euo pipefail
cd /workspace/fifa
. scripts/video_defaults.sh

PY="${PITCH3D_PY:-/workspace/.venv/bin/python}"
CAM="${ANIM_CAMERAS:-sideline}"
case "$CAM" in *,*) echo "finish_batch: ANIM_CAMERAS must be a single camera" >&2; exit 1;; esac
BATCH_OUT="${OUT:-out/anim_finish}"
echo "== BATCH finish start $(date -u +%H:%M:%SZ) :: cam=$CAM out=$BATCH_OUT =="

# Step-output paths, shared by both branches below.
SRC="$BATCH_OUT/mesh/frames/$CAM"
DST="$BATCH_OUT/mesh/frames_night/$CAM"
MASKS="$BATCH_OUT/mesh/mask/$CAM"
if [ "${KIT_INJECT:-1}" = "1" ]; then
  CTRL="$BATCH_OUT/mesh/frames_nightkit/$CAM"
else
  CTRL="$DST"
fi

if [ "${TAIL_ONLY:-0}" = "1" ]; then
  # Generative-stage iteration (prompt/CS/seed sweeps): reuse steps 1-4 outputs from a
  # previous run in the same $OUT — only v2v -> SeedVR2 -> pins rerun (~15 min, not ~55).
  for d in "$SRC" "$MASKS" "$CTRL"; do
    if [ ! -d "$d" ]; then echo "finish_batch: TAIL_ONLY=1 but missing $d" >&2; exit 1; fi
  done
  echo "== TAIL_ONLY: reusing control $CTRL ($(ls "$CTRL" | wc -l) frames) =="
else

# 1) recon + kit-boosted QUILT export + beauty render
PITCH3D_CLIP="${PITCH3D_CLIP:-/workspace/Colombia-1-0-Congo-DR1080p.mp4}" \
  REUSE_SCENE="${REUSE_SCENE:-0}" ANIM_CAMERAS="$CAM" OUT="$BATCH_OUT" \
  bash scripts/pod_make_video.sh

# 2) night-grade (grade3)
GRADE='eq=brightness=-0.28:contrast=1.12:gamma=0.75:saturation=0.9,colorbalance=bs=0.12:bm=0.06'
mkdir -p "$DST"
for f in "$SRC"/frame_*.png; do
  ffmpeg -y -v error -i "$f" -vf "$GRADE" "$DST/$(basename "$f")"
done
echo "== night grade done: $(ls "$DST" | wc -l) frames =="

# 3) team-mask AOV pass — feeds the kit injection AND the hue pins (masks are resized to
# the consumer's resolution, so 832x480 is plenty)
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

# 4) kit-colour re-injection into the control frames (KIT_INJECT=0 → plain night frames)
if [ "${KIT_INJECT:-1}" = "1" ]; then
  INJ_FLAGS=(--alpha "${ALPHA:-0.8}" --erode "${ERODE:-3}")
  A_HSV="${TEAM_A_HSV-65 0.85}"
  B_HSV="${TEAM_B_HSV-185 0.95}"
  if [ -n "$A_HSV" ]; then INJ_FLAGS+=(--team-a-hsv $A_HSV); fi
  if [ -n "$B_HSV" ]; then INJ_FLAGS+=(--team-b-hsv $B_HSV); fi
  "$PY" scripts/control_kit_inject.py --frames "$DST" --masks "$MASKS" --out "$CTRL" \
    "${INJ_FLAGS[@]}"
fi

fi  # TAIL_ONLY

# 5) v2v: rgb control over the (kit-injected) night frames.
# V2V must be ABSOLUTE: pod_seedvr2.sh cd's into its own repo before testing INPUTS.
# 1280x720 default (A/B 2026-07-03): at 832x480 a distant player is 2-3 latent px after the
# 8x VAE and Wan repaints him as mush; 720p resolves limbs/poses (~10 min vs ~3 min per clip).
V2V="$(realpath -m "$BATCH_OUT")/v2v/${CAM}_rgbnight.mp4"
V2V_ARGS=(--control rgb
  --width "${V2V_WIDTH:-1280}" --height "${V2V_HEIGHT:-720}" --flow-shift "${V2V_FLOW:-5.0}"
  --conditioning-scale "${CS:-1.0}")
if [ -n "${V2V_PROMPT:-}" ]; then V2V_ARGS+=(--prompt "$V2V_PROMPT"); fi
FRAMES="$CTRL" OUT="$V2V" bash scripts/pod_v2v.sh "${V2V_ARGS[@]}"

# 6) SeedVR2 720p
INPUTS="$V2V" bash scripts/pod_seedvr2.sh
V2V720="${V2V%.mp4}_720p.mp4"
test -f "$V2V720" || { echo "finish_batch: missing $V2V720" >&2; exit 1; }

# 7) hue-pin team B back to THIS run's render hue (or TARGET_HUE)
PINNED="${V2V720%.mp4}_pinned.mp4"
PIN=(--video "$V2V720" --mask-dir "$MASKS" --out "$PINNED"
     --channel g --dilate "${DILATE:-15}")
if [ -n "${TARGET_HUE:-}" ]; then
  PIN+=(--target-hue "$TARGET_HUE")
else
  PIN+=(--target-from-frames "$SRC")
fi
"$PY" scripts/hue_pin.py "${PIN[@]}"

# 8) hue-pin team A (validated 2026-07-03: 48.0→70.6, faces untouched at sat-min 0.35)
FINAL="$PINNED"
if [ "${PIN_A:-1}" = "1" ]; then
  FINAL="${V2V720%.mp4}_pinned2.mp4"
  "$PY" scripts/hue_pin.py --video "$PINNED" --mask-dir "$MASKS" --out "$FINAL" \
    --channel r --dilate "${DILATE:-15}" --hue-band 5 80 --sat-min 0.35 \
    --target-from-frames "$SRC"
fi

echo "BATCH_FINISH_OK"
echo "  beauty video : $BATCH_OUT/video/$CAM.mp4"
echo "  v2v          : $V2V"
echo "  seedvr2 720p : $V2V720"
echo "  FINAL pinned : $FINAL"
