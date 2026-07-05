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
#   9) grass pin: grass-band H+S back to the clip's measured tone (prompt wording can't land
#      it — three A/Bs 2026-07-05, see grass_pin.py); kit masks excluded — GRASS_PIN=0 skips
#  10) stands tone pin: warm band (15-80) in the top-third ROI back to the clip crowd's
#      darker+yellower tone (H+8.5 S x1.22 V x0.70 measured 2026-07-05) — STANDS_PIN=0 skips;
#      STANDS_ROI is tuned for the SIDELINE framing, override it for other cameras
#  11) boards white pin: LED whites (desaturated gate, S<=0.35 V>=0.35 in the boards ROI)
#      back to the clip's glow — t15 measured Vmed .64 vs clip .93-.96, letters (V<.35) stay
#      dark, kits are saturated i.e. outside the gate — BOARDS_PIN=0 skips; BOARDS_ROI is
#      tuned for the SIDELINE framing (boards y .43-.47)
#  12) panel-row V pin: the fascia panel band (crowd->panels->hedge sandwich) rendered ~1.5x
#      hot (t16 Vmed .42 vs clip .27-.31) — V-ONLY pin (--val-only: the zone mixes gold
#      panels + green hedge + crowd bottom, matching its MEDIAN hue would repaint the
#      minority materials) — PANELS_PIN=0 skips; PANELS_ROI tuned for the SIDELINE framing
#  13) stands red-scatter pin: scattered dark-red fans at SCREEN scale (quilt-space red dies
#      in the render, Wan re-adds only ~0.8% vs clip 3.6% — t19 measured twice) —
#      STANDS_RED=0 skips; STANDS_RED_ROI matches the stands pin's SIDELINE framing
#
# Finishing is per-camera (v2v eats one frame dir): ANIM_CAMERAS must be ONE camera.
# Env: OUT=out/anim_finish  ANIM_CAMERAS=sideline  REUSE_SCENE=0  TAIL_ONLY=0  TARGET_HUE=  DILATE=15
#      V2V_WIDTH=1280 V2V_HEIGHT=720 V2V_FLOW=5.0  (832x480/3.0 = the old fast-draft cell)
#      V2V_PROMPT= / V2V_NEGATIVE= (override the measured default prompts in pod_v2v_finish.py)
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
if [ -n "${V2V_NEGATIVE:-}" ]; then V2V_ARGS+=(--negative "$V2V_NEGATIVE"); fi
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

# 9) grass pin (targets auto-measured from the clip-derived night reference frame)
if [ "${GRASS_PIN:-1}" = "1" ]; then
  GFINAL="${V2V720%.mp4}_pinned3.mp4"
  GPIN=(--video "$FINAL" --mask-dir "$MASKS" --out "$GFINAL" --dilate "${DILATE:-15}")
  if [ -n "${GRASS_TARGET_HUE:-}" ] && [ -n "${GRASS_TARGET_SAT:-}" ]; then
    GPIN+=(--target-hue "$GRASS_TARGET_HUE" --target-sat "$GRASS_TARGET_SAT")
  else
    GPIN+=(--target-from-image "${GRASS_REF:-out/v2v/ref_night.png}")
  fi
  "$PY" scripts/grass_pin.py "${GPIN[@]}"
  FINAL="$GFINAL"
fi

# 10) stands tone pin (see header; target auto-measured from the clip-derived reference)
if [ "${STANDS_PIN:-1}" = "1" ]; then
  SFINAL="${V2V720%.mp4}_pinned4.mp4"
  SPIN=(--video "$FINAL" --mask-dir "$MASKS" --out "$SFINAL" --dilate "${DILATE:-15}"
        --roi ${STANDS_ROI:-0.08 0.32 0.02 0.98} --hue-band 15 80 --sat-min 0.15
        --val-min 0.05 --pin-val)
  if [ -n "${STANDS_TARGET_HUE:-}" ]; then
    SPIN+=(--target-hue "$STANDS_TARGET_HUE" --target-sat "${STANDS_TARGET_SAT:?}"
           --target-val "${STANDS_TARGET_VAL:?}")
  else
    SPIN+=(--target-from-image "${STANDS_REF:-out/v2v/ref_night.png}"
           --target-roi ${STANDS_TARGET_ROI:-0.05 0.26 0.05 0.95})
  fi
  if [ -n "${STANDS_XFLAT_BINS:-}" ]; then SPIN+=(--flatten-val-x "$STANDS_XFLAT_BINS"); fi
  "$PY" scripts/grass_pin.py "${SPIN[@]}"
  FINAL="$SFINAL"
fi

# 11) boards white pin (see header; LED whites back to the clip's glow, letters untouched)
if [ "${BOARDS_PIN:-1}" = "1" ]; then
  BFINAL="${V2V720%.mp4}_pinned5.mp4"
  BPIN=(--video "$FINAL" --mask-dir "$MASKS" --out "$BFINAL" --dilate "${DILATE:-15}"
        --roi ${BOARDS_ROI:-0.42 0.48 0.0 1.0} --hue-band 0 360 --sat-min 0
        --sat-max "${BOARDS_SAT_MAX:-0.35}" --val-min 0.35 --pin-val)
  if [ -n "${BOARDS_TARGET_VAL:-}" ]; then
    BPIN+=(--target-hue "${BOARDS_TARGET_HUE:-0}" --target-sat "${BOARDS_TARGET_SAT:-0.07}"
           --target-val "$BOARDS_TARGET_VAL")
  else
    BPIN+=(--target-from-image "${BOARDS_REF:-out/v2v/ref_night.png}"
           --target-roi ${BOARDS_TARGET_ROI:-0.417 0.450 0.05 0.95})
  fi
  "$PY" scripts/grass_pin.py "${BPIN[@]}"
  FINAL="$BFINAL"
fi

# 12) panel-row V pin (see header; mixed-material zone -> V only)
if [ "${PANELS_PIN:-1}" = "1" ]; then
  PFINAL="${V2V720%.mp4}_pinned6.mp4"
  PPIN=(--video "$FINAL" --mask-dir "$MASKS" --out "$PFINAL" --dilate "${DILATE:-15}"
        --roi ${PANELS_ROI:-0.335 0.385 0.0 1.0} --hue-band 0 360 --sat-min 0
        --val-min 0.05 --val-only)
  if [ -n "${PANELS_TARGET_VAL:-}" ]; then
    PPIN+=(--target-val "$PANELS_TARGET_VAL")
  else
    PPIN+=(--target-from-image "${PANELS_REF:-out/v2v/ref_night.png}"
           --target-roi ${PANELS_TARGET_ROI:-0.286 0.320 0.05 0.95})
  fi
  "$PY" scripts/grass_pin.py "${PPIN[@]}"
  FINAL="$PFINAL"
fi

# 13) stands red-scatter pin: the clip's 3.6% scattered dark-red fans re-enter at SCREEN
#     scale — texture-space red dies upstream (render minification+denoise -> beauty 0.001,
#     Wan re-adds only ~0.8%; measured twice, t19). Runs last: the tone pins (10-12) would
#     re-amber the specks. STANDS_RED=0 skips.
if [ "${STANDS_RED:-1}" = "1" ]; then
  RFINAL="${V2V720%.mp4}_pinned7.mp4"
  RPIN=(--video "$FINAL" --out "$RFINAL"
        --roi ${STANDS_RED_ROI:-0.08 0.32 0.0 1.0}
        --target "${STANDS_RED_TARGET:-0.036}"
        --seed "${STANDS_RED_SEED:-0}")
  [ -n "${STANDS_RED_FRAC:-}" ] && RPIN+=(--frac "$STANDS_RED_FRAC")
  "$PY" scripts/stands_red_pin.py "${RPIN[@]}"
  FINAL="$RFINAL"
fi

echo "BATCH_FINISH_OK"
echo "  beauty video : $BATCH_OUT/video/$CAM.mp4"
echo "  v2v          : $V2V"
echo "  seedvr2 720p : $V2V720"
echo "  FINAL pinned : $FINAL"
