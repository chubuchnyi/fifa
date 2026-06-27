#!/usr/bin/env bash
# scripts/pod_make_video.sh — ON THE POD: one broadcast clip → multi-angle 3D animation mp4s.
#
# The whole heavy half runs here on the GPU box; the local driver (scripts/demo_video.sh) only
# stages the clip, calls this, then pulls out/anim/video/*.mp4 back to the machine. Chain:
#   1) real reconstruction (RF-DETR · ByteTrack · SMPLest-X-H · WASB) → canonical scene JSON
#      (json is the only export that carries the ball) — reuses scripts/pod_real_e2e.sh.
#   2) anim_export.py: forward EVERY frame through SMPL-X + resolve the ball → anim_subject_*.npz.
#   3) blender_animate.py: render N fixed cameras × all frames with Cycles (GPU, CPU fallback).
#   4) ffmpeg: stitch each camera's PNG sequence into out/anim/video/<camera>.mp4.
#
# Env (sensible pod defaults; override to taste):
#   PITCH3D_REPO=/workspace/fifa   PITCH3D_PY=/workspace/.venv/bin/python   PITCH3D_CLIP=/workspace/clip.mp4
#   FRAMES=60   OUT=out/anim
#   ANIM_DEVICE=gpu  ANIM_RES_X=1280  ANIM_RES_Y=720  ANIM_SAMPLES=32  ANIM_FPS=25
#   ANIM_STEP=1  ANIM_CAMERAS=broadcast,sideline,top,goal
#   REUSE_SCENE=0   set 1 to skip reconstruction when $OUT/export/scene.json already exists (cheap re-render)
#   PITCH3D_SMPLX_MODELS=/workspace/repos/SMPLest-X/human_models/human_model_files  (smplx pkg models dir)
set -euo pipefail

REPO="${PITCH3D_REPO:-/workspace/fifa}"
PY="${PITCH3D_PY:-/workspace/.venv/bin/python}"
CLIP="${PITCH3D_CLIP:-/workspace/clip.mp4}"
FRAMES="${FRAMES:-60}"
OUT="${OUT:-out/anim}"
ANIM_DEVICE="${ANIM_DEVICE:-gpu}"
ANIM_RES_X="${ANIM_RES_X:-1280}"
ANIM_RES_Y="${ANIM_RES_Y:-720}"
ANIM_SAMPLES="${ANIM_SAMPLES:-32}"
ANIM_FPS="${ANIM_FPS:-25}"
ANIM_STEP="${ANIM_STEP:-1}"
ANIM_CAMERAS="${ANIM_CAMERAS:-broadcast,sideline,top,goal}"
SMPLX_MODELS="${PITCH3D_SMPLX_MODELS:-/workspace/repos/SMPLest-X/human_models/human_model_files}"

cd "$REPO"
echo "== pod make video :: frames=${FRAMES} cams=${ANIM_CAMERAS} ${ANIM_RES_X}x${ANIM_RES_Y} dev=${ANIM_DEVICE} out=${OUT} =="

# 1) real reconstruction → canonical JSON (carries subjects' motion AND the ball)
SCENE_JSON="$OUT/export/scene.json"
if [ "${REUSE_SCENE:-0}" = 1 ] && [ -f "$SCENE_JSON" ]; then
  echo "== reuse scene: $SCENE_JSON exists, skipping reconstruction (REUSE_SCENE=1) =="
else
  # Forward continuity (stitch, ON by default — #202) + temporal coherence (opt-in) into the
  # reconstruction so the animated bodies inherit the re-linked tracklets + gap-fill. pod_real_e2e.sh
  # stitches unless STITCH=0; coherence stays off unless COHERENCE=1. demo_video.sh sets both.
  FORMAT=json OUT="$OUT" FRAMES="$FRAMES" \
    STITCH="${STITCH:-1}" COHERENCE="${COHERENCE:-0}" \
    PITCH3D_REPO="$REPO" PITCH3D_PY="$PY" PITCH3D_CLIP="$CLIP" \
    bash scripts/pod_real_e2e.sh
fi
test -f "$SCENE_JSON" || { echo "pod_make_video: missing $SCENE_JSON" >&2; exit 1; }

# 2) forward all frames through SMPL-X + resolve the ball (venv: torch+smplx)
echo "== anim export: $SCENE_JSON → $OUT/mesh =="
PITCH3D_SMPLX_MODELS="$SMPLX_MODELS" PITCH3D_SCENE_JSON="$SCENE_JSON" PITCH3D_ANIM_OUT="$OUT/mesh" \
  PYTHONPATH=src "$PY" scripts/anim_export.py

# 3) ensure Blender (bpy module, or a tarball binary) + ffmpeg, then render the cameras
BMODE="$(PITCH3D_PY="$PY" bash scripts/pod_ensure_blender.sh)"
case "$BMODE" in
  BLENDER_MODE=module)    RENDER=("$PY" scripts/blender_animate.py);;
  BLENDER_MODE=binary:*)  RENDER=("${BMODE#BLENDER_MODE=binary:}" --background --python scripts/blender_animate.py --);;
  *) echo "pod_make_video: blender unavailable ($BMODE)" >&2; exit 1;;
esac
echo "== blender animate ($BMODE) → $OUT/mesh/frames =="
"${RENDER[@]}" \
  --in "$OUT/mesh" --out "$OUT/mesh/frames" --device "$ANIM_DEVICE" \
  --res-x "$ANIM_RES_X" --res-y "$ANIM_RES_Y" --samples "$ANIM_SAMPLES" \
  --fps "$ANIM_FPS" --frame-step "$ANIM_STEP" --cameras "$ANIM_CAMERAS"

# 4) stitch each camera's PNG sequence into one mp4
mkdir -p "$OUT/video"
shopt -s nullglob
for camdir in "$OUT"/mesh/frames/*/; do
  cam="$(basename "$camdir")"
  pngs=("$camdir"frame_*.png)
  [ "${#pngs[@]}" -gt 0 ] || { echo "pod_make_video: no frames for camera $cam" >&2; continue; }
  ffmpeg -y -framerate "$ANIM_FPS" -pattern_type glob -i "$camdir/frame_*.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUT/video/$cam.mp4" </dev/null
  echo "VIDEO_OK $cam (${#pngs[@]}f) -> $OUT/video/$cam.mp4"
done
ls -la "$OUT/video"
echo "POD_MAKE_VIDEO_OK -> $OUT/video"
