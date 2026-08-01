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
#   PITCH3D_REPO=/workspace/fifa   PITCH3D_PY=/workspace/.venv/bin/python   PITCH3D_CLIP=<required>
#   FRAMES=60   OUT=out/anim
#   ANIM_DEVICE=gpu  ANIM_RES_X=1280  ANIM_RES_Y=720  ANIM_SAMPLES=32  ANIM_FPS=25
#   ANIM_STEP=1  ANIM_CAMERAS=broadcast,sideline,top,goal
#   REUSE_SCENE=0   set 1 to skip reconstruction when $OUT/export/scene.json already exists (cheap re-render)
#   PITCH3D_SMPLX_MODELS=/workspace/repos/SMPLest-X/human_models/human_model_files  (smplx pkg models dir)
set -euo pipefail

# Shared knob defaults (FRAMES/STITCH/COHERENCE/ANIM_*) — single source of truth with demo_video.sh.
. "$(dirname "${BASH_SOURCE[0]}")/video_defaults.sh"

REPO="${PITCH3D_REPO:-/workspace/fifa}"
PY="${PITCH3D_PY:-/workspace/.venv/bin/python}"
# Required, not defaulted — this wrapper is how the 2026-07-03 wrong-clip run happened
# (see pod_real_e2e.sh, same knob). Name the footage you mean.
CLIP="${PITCH3D_CLIP:-}"
[ -n "$CLIP" ] || { echo "pod_make_video.sh: set PITCH3D_CLIP to the clip you mean to render." >&2; exit 2; }
FRAMES="${FRAMES:-$VIDEO_FRAMES_DEFAULT}"
OUT="${OUT:-out/anim}"
ANIM_DEVICE="${ANIM_DEVICE:-$VIDEO_DEVICE_DEFAULT}"
ANIM_RES_X="${ANIM_RES_X:-$VIDEO_RES_X_DEFAULT}"
ANIM_RES_Y="${ANIM_RES_Y:-$VIDEO_RES_Y_DEFAULT}"
ANIM_SAMPLES="${ANIM_SAMPLES:-$VIDEO_SAMPLES_DEFAULT}"
ANIM_FPS="${ANIM_FPS:-$VIDEO_FPS_DEFAULT}"
ANIM_STEP="${ANIM_STEP:-$VIDEO_STEP_DEFAULT}"
ANIM_CAMERAS="${ANIM_CAMERAS:-$VIDEO_CAMERAS_DEFAULT}"
SMPLX_MODELS="${PITCH3D_SMPLX_MODELS:-/workspace/repos/SMPLest-X/human_models/human_model_files}"

cd "$REPO"
echo "== pod make video :: frames=${FRAMES} cams=${ANIM_CAMERAS} ${ANIM_RES_X}x${ANIM_RES_Y} dev=${ANIM_DEVICE} out=${OUT} =="

# 1) real reconstruction → canonical JSON (carries subjects' motion AND the ball)
SCENE_JSON="$OUT/export/scene.json"
if [ "${REUSE_SCENE:-0}" = 1 ] && [ -f "$SCENE_JSON" ]; then
  echo "== reuse scene: $SCENE_JSON exists, skipping reconstruction (REUSE_SCENE=1) =="
else
  # Forward continuity (stitch — #202) + temporal coherence (gap-fill + zero-phase smoothing) into
  # the reconstruction so the animated bodies inherit re-linked tracklets and don't render raw
  # jittery poses. BOTH default ON from video_defaults.sh — a direct pod run must match demo_video.sh.
  FORMAT=json OUT="$OUT" FRAMES="$FRAMES" \
    STITCH="${STITCH:-$VIDEO_STITCH_DEFAULT}" COHERENCE="${COHERENCE:-$VIDEO_COHERENCE_DEFAULT}" \
    PHYSICS="${PHYSICS:-$VIDEO_PHYSICS_DEFAULT}" \
    CAMERA_CARRY="${CAMERA_CARRY:-$VIDEO_CAMERA_CARRY_DEFAULT}" \
    DEMO_EDITS="${DEMO_EDITS:-0}" \
    PITCH3D_REPO="$REPO" PITCH3D_PY="$PY" PITCH3D_CLIP="$CLIP" \
    bash scripts/pod_real_e2e.sh
fi
test -f "$SCENE_JSON" || { echo "pod_make_video: missing $SCENE_JSON" >&2; exit 1; }

# 2) forward all frames through SMPL-X + resolve the ball (venv: torch+smplx).
# The staged clip doubles as the MEASURED-appearance source (stadium crowd, per-vertex body
# texture, floodlight colour): without it anim_export silently skips all three, so default it to
# the clip we just reconstructed from.
# Kit-zone colour overrides (kit-zones lever, 2026-07-04 §6): the per-vertex sampler carries no
# kit chroma at broadcast distance (~65% of samples are grass, the rest line/LED white), so the
# leg-zone colours for THIS clip are measured off its closeup frames (f30 leg bands, f275 tile
# classes): Colombia (A) white shorts + red socks; dark skin tone (both squads). Congo (B) must
# be overridden TOO (azure, f275 tile class): on the full pod scene (23 subjects) the pooled
# post-grass samples pass the 40 floor, so the polluted line-white median (batch #2 measured
# B shorts 0.765,0.702,0.71) WINS over the shorts→shirt azure fallback that a sparse local
# scene falls into.
echo "== anim export: $SCENE_JSON → $OUT/mesh =="
PITCH3D_SMPLX_MODELS="$SMPLX_MODELS" PITCH3D_SCENE_JSON="$SCENE_JSON" PITCH3D_ANIM_OUT="$OUT/mesh" \
  PITCH3D_STADIUM_VIDEO="${PITCH3D_STADIUM_VIDEO:-$CLIP}" \
  PITCH3D_SHORTS_RGB_A="${PITCH3D_SHORTS_RGB_A:-0.85,0.88,0.82}" \
  PITCH3D_SOCKS_RGB_A="${PITCH3D_SOCKS_RGB_A:-0.71,0.14,0.31}" \
  PITCH3D_SHORTS_RGB_B="${PITCH3D_SHORTS_RGB_B:-0.189,0.52,0.688}" \
  PITCH3D_SOCKS_RGB_B="${PITCH3D_SOCKS_RGB_B:-0.189,0.52,0.688}" \
  PITCH3D_SKIN_RGB="${PITCH3D_SKIN_RGB:-0.32,0.26,0.20}" \
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
