#!/usr/bin/env bash
# scripts/pod_carry_ab.sh — ON THE POD: the R2 camera-propagation A/B (#94, judged by #60).
#
# Renders the SAME episode twice, differing in exactly one knob: --camera-carry.
#   out/carry_on   CAMERA_CARRY=8  each frame's homography re-fitted from its +-8 neighbours
#   out/carry_off  CAMERA_CARRY=0  today's behaviour — every frame solved independently
#
# Why this needs eyes rather than a number: R2 is a TRADE. It removes 92 % of the scene swim
# (0.119 -> 0.011 m frame-to-frame slide) and gives up ~0.004 m of paint accuracy. No metric we
# have can settle "stable-but-slightly-offset vs accurate-but-jittery" — the swim metric is
# circular (a 10 m-displaced anchor carried the same way scores 0.0000 m) and the paint metric
# is blind to temporal consistency. So both videos get rendered and a human picks.
#
# WHAT TO ACTUALLY LOOK FOR (#107): NOT a drifting camera. The render's camera is synthetic and
# frozen — AppController replaces the solved CameraTrack with a tiled standard_viewpoints(BROADCAST)
# pose just before export, so the exported camera is bit-identical on every frame of BOTH sides.
# The calibration reaches the picture through SUBJECT PLACEMENT (world positions are projected
# through the per-frame homography). So the defect reads as players sliding/jittering against a
# pitch that never moves, and the question for the eye is which side's players sit more steadily.
#
# Everything downstream of these two runs is measured LOCALLY and costs no GPU time: #106 reads
# the carry_off scene.json (per-frame confidences describe the per-frame solve, so the control
# side is the one whose confidence-vs-error correlation is meaningful), and #103 reads team_id
# out of the same file plus the contact sheets emitted here. Pull, then stop the pod.
#
# Usage (on the pod, from /workspace/fifa):
#   bash scripts/pod_carry_ab.sh                 # full: 60f, 2 cams, 1280x720 @32spp
#   MODE=smoke bash scripts/pod_carry_ab.sh      # 4f, 1 cam, 640x360 @8spp — timing sanity
#   CAMERAS=broadcast,sideline,top,goal bash scripts/pod_carry_ab.sh
#
# Env: MODE(full) FRAMES(60) CAMERAS(broadcast,sideline) SAMPLES(32) RES(1280x720) SHEETS(1).
# Two cameras by default, not four: swim is a whole-scene slide and reads just as clearly from
# one novel angle beside the broadcast reference, at half the render cost.
set -euo pipefail

REPO="${PITCH3D_REPO:-/workspace/fifa}"
PY="${PITCH3D_PY:-/workspace/.venv/bin/python}"
CLIP="${PITCH3D_CLIP:-/workspace/Colombia-1-0-Congo-DR1080p.mp4}"
MODE="${MODE:-full}"
cd "$REPO"

if [ "$MODE" = "smoke" ]; then
  FRAMES="${FRAMES:-4}"; CAMERAS="${CAMERAS:-broadcast}"; SAMPLES="${SAMPLES:-8}"; RES="${RES:-640x360}"
  SUF="_smoke"
else
  FRAMES="${FRAMES:-60}"; CAMERAS="${CAMERAS:-broadcast,sideline}"; SAMPLES="${SAMPLES:-32}"
  RES="${RES:-1280x720}"; SUF=""
fi

echo "== CARRY_AB START $(date -u +%FT%TZ) mode=$MODE frames=$FRAMES cams=$CAMERAS $RES @${SAMPLES}spp =="

# The ON side runs first and pays the page-cache prewarm; the OFF side reuses the warm cache.
_warm=1
for side in on off; do
  [ "$side" = on ] && carry="${CARRY_N:-8}" || carry=0
  out="out/carry_${side}${SUF}"
  echo ""
  echo "== CARRY_AB SIDE=$side CAMERA_CARRY=$carry OUT=$out $(date -u +%FT%TZ) =="
  VARIANT=A MODE=full PREWARM="$_warm" \
    CAMERA_CARRY="$carry" OUT="$out" \
    FRAMES="$FRAMES" CAMERAS="$CAMERAS" SAMPLES="$SAMPLES" RES="$RES" \
    PITCH3D_REPO="$REPO" PITCH3D_PY="$PY" PITCH3D_CLIP="$CLIP" \
    bash scripts/pod_ab_video.sh
  _warm=0
  echo "== CARRY_AB SIDE=$side DONE $(date -u +%FT%TZ) =="
done

# #103: torso contact sheets so the shirt numbers can be READ by a human/LLM off-pod. Only the
# sheets are produced here — pinning a number nobody actually read would be inventing data (R-6).
if [ "${SHEETS:-1}" = "1" ]; then
  scene="out/carry_on${SUF}/export/scene.json"
  echo ""
  echo "== CARRY_AB sheets (#103) from $scene =="
  PYTHONPATH=src "$PY" scripts/jersey_numbers.py sheets \
    --scene "$scene" --clip "$CLIP" --out "out/carry_on${SUF}/backs" || echo "sheets FAILED (non-fatal)"
fi

echo ""
echo "CARRY_AB_FINISH_OK $(date -u +%FT%TZ)"
