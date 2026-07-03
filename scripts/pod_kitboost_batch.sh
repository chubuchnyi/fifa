#!/usr/bin/env bash
# Kit-boost validation batch (priority 2a+2b): fresh recon (DEMO_EDITS=0 e2e check) +
# boosted export + sideline render, then rgb-control v2v over night-graded vs ungraded frames.
set -euo pipefail
cd /workspace/fifa
echo "== BATCH kitboost start $(date -u +%H:%M:%SZ) =="

# 1) fresh recon + kit-boosted export + sideline render (DEMO_EDITS defaults 0 in pod_make_video)
PITCH3D_CLIP=/workspace/Colombia-1-0-Congo-DR1080p.mp4 REUSE_SCENE=0 ANIM_CAMERAS=sideline \
  OUT=out/anim_kitboost bash scripts/pod_make_video.sh

# 2) night-grade the rendered frames (grade3, picked locally against ref_night)
GRADE='eq=brightness=-0.28:contrast=1.12:gamma=0.75:saturation=0.9,colorbalance=bs=0.12:bm=0.06'
SRC=out/anim_kitboost/mesh/frames/sideline
DST=out/anim_kitboost/mesh/frames_night/sideline
mkdir -p "$DST"
for f in "$SRC"/frame_*.png; do
  ffmpeg -y -v error -i "$f" -vf "$GRADE" "$DST/$(basename "$f")"
done
echo "== night grade done: $(ls "$DST" | wc -l) frames =="

# 3) v2v A: rgb control over NIGHT-GRADED boosted frames
FRAMES="$DST" OUT=out/v2v/sideline_rgbnight.mp4 bash scripts/pod_v2v.sh --control rgb

# 4) v2v B: rgb control over UNGRADED boosted frames (control cell)
FRAMES="$SRC" OUT=out/v2v/sideline_rgbboost.mp4 bash scripts/pod_v2v.sh --control rgb

echo "BATCH_KITBOOST_OK"
