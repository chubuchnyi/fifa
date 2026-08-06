#!/usr/bin/env bash
# scripts/view_cue_ab.sh — open the #133 cue A/B in the pitch-3D viewer, overlaid.
#
# The primary scene is the run WITHOUT the mask cue (team colours); the overlay is the run WITH it
# (magenta). Toggle "overlay B" in the toolbar, and "trails" to see where a track broke.
#
#   bash scripts/view_cue_ab.sh          # -> http://localhost:8000/world
#   PORT=8123 bash scripts/view_cue_ab.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"
A="${A:-$PWD/out/cue/scene_off.json}"
B="${B:-$PWD/out/cue/scene_on.json}"

for f in "$A" "$B"; do
  [ -f "$f" ] || { echo "missing scene: $f" >&2; exit 1; }
done

echo "primary (no cue) : $A"
echo "overlay (McByte) : $B"
echo "open             : http://localhost:$PORT/world   — tick 'overlay B'"
echo
# Both scenes build their own SMPL-X FK cache, so the first frame takes a few seconds each.
POSEANNOT_SCENE_JSON="$A" POSEANNOT_SCENE_JSON_B="$B" \
  exec .venv/bin/uvicorn poseannot.app:app --host 127.0.0.1 --port "$PORT"
