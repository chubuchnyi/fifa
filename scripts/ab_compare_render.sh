#!/usr/bin/env bash
# scripts/ab_compare_render.sh — put two finishing runs side by side so the eye can judge them.
#
# The project's rule is that the user's eye is ground truth, and an eye cannot compare two videos
# it has to watch one after the other. This stacks them into one file, labelled, plus a few stills
# from the BEAUTY render — which is where a geometry change (#129's one fitted camera) shows
# cleanly, before the generative tail adds its own frame-to-frame variation on top.
#
#   bash scripts/ab_compare_render.sh out/n129_off out/n129_on sideline out/cmp_129
#
# Args: LEFT_DIR RIGHT_DIR CAMERA OUT_DIR   (labels are taken from the dir basenames)
set -euo pipefail

LEFT="${1:?left run dir}"
RIGHT="${2:?right run dir}"
CAM="${3:-sideline}"
OUT="${4:-out/ab_compare}"
STILLS="${STILLS:-0 15 30 45 59}"

mkdir -p "$OUT"
L_NAME="$(basename "$LEFT")"
R_NAME="$(basename "$RIGHT")"

# 1) stills from the beauty render — the geometry comparison, uncontaminated by the generative tail
for f in $STILLS; do
  LF="$(printf '%s/mesh/frames/%s/frame_%04d.png' "$LEFT" "$CAM" "$f")"
  RF="$(printf '%s/mesh/frames/%s/frame_%04d.png' "$RIGHT" "$CAM" "$f")"
  if [ -f "$LF" ] && [ -f "$RF" ]; then
    ffmpeg -y -v error -i "$LF" -i "$RF" -filter_complex \
      "[0:v]drawtext=text='$L_NAME':x=12:y=12:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6[l];\
       [1:v]drawtext=text='$R_NAME':x=12:y=12:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6[r];\
       [l][r]hstack=inputs=2" "$OUT/beauty_f$(printf '%04d' "$f").png"
    echo "  still frame $f -> $OUT/beauty_f$(printf '%04d' "$f").png"
  else
    echo "  still frame $f: missing ($LF / $RF)" >&2
  fi
done

# 2) side-by-side of whatever each arm produced last. Both finals, one timeline.
pick_final() {
  # The chain names its output by how far the pins got, so take the most advanced that exists.
  local d="$1"
  for suffix in _pinned8 _pinned7 _pinned6b _pinned6 _pinned5 _pinned4 _pinned3b _pinned3 _pinned2 _pinned ''; do
    local c
    c="$(ls "$d"/v2v/*_720p${suffix}.mp4 2>/dev/null | head -1)" || true
    [ -n "$c" ] && { echo "$c"; return 0; }
  done
  ls "$d"/video/*.mp4 2>/dev/null | head -1
}

LV="$(pick_final "$LEFT")"
RV="$(pick_final "$RIGHT")"
echo "left  video: ${LV:-NONE}"
echo "right video: ${RV:-NONE}"
if [ -n "${LV:-}" ] && [ -n "${RV:-}" ]; then
  ffmpeg -y -v error -i "$LV" -i "$RV" -filter_complex \
    "[0:v]scale=960:-2,drawtext=text='$L_NAME':x=12:y=12:fontsize=32:fontcolor=white:box=1:boxcolor=black@0.6[l];\
     [1:v]scale=960:-2,drawtext=text='$R_NAME':x=12:y=12:fontsize=32:fontcolor=white:box=1:boxcolor=black@0.6[r];\
     [l][r]hstack=inputs=2" -c:v libx264 -crf 18 -pix_fmt yuv420p "$OUT/side_by_side.mp4"
  echo "  -> $OUT/side_by_side.mp4"
else
  echo "  no pair of finals to stack" >&2
fi
echo "AB_COMPARE_OK $OUT"
