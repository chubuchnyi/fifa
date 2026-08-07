#!/usr/bin/env bash
# scripts/view_handover_ab.sh — the W3/W4 handover merge, side by side, for the eye to judge.
#
# A = the scene the user judged track-by-track on 2026-08-07 (24 subjects).
# B = the same scene with the handover merge applied (21 subjects).
#
# Deliberately the *same* scene rather than a fresh run: every verdict in
# docs/findings/track-labels-2026-08-07.json maps onto these track ids, so "did t3 stop being two
# people" is answerable directly instead of through a new id space.
#
# What to look at, in order:
#   1. t3, t10, t15 — each was two bodies (3+66, 10+77, 15+25) and is now one. Do they read as
#      one player crossing, or does the seam teleport?
#   2. **t10 specifically.** Its merge is the one the pass flags: the rebuilt t10 passes within
#      0.05 m of t5 on 13 frames. The video pixels say t77 wears YELLOW on all 3 of its measured
#      frames while t10 wears BLUE on all 46 — so this merge may have taken t5's man. The eye
#      called 10+77 one player on 2026-08-07; this is the measurement that disagrees.
#   3. t20 and t71 survive unmerged, as the assignment intends.
#
#   bash scripts/view_handover_ab.sh          # -> http://localhost:8000/world, tick "overlay B"
#   PORT=8123 bash scripts/view_handover_ab.sh
#
# Regenerate B after changing the pass:
#   PYTHONPATH=src .venv/bin/python -c "
#   from pitch3d.core.scene.serialization import load_scene, save_scene
#   from pitch3d.core.orchestration.handover import merge_handovers, HandoverConfig
#   s,r = merge_handovers(load_scene('out/cue/scene_off.json'), HandoverConfig(enabled=True))
#   save_scene(s,'out/cue/scene_off_handover.json'); print(r.merges, r.suspect)"
set -euo pipefail
cd "$(dirname "$0")/.."
exec env A="$PWD/out/cue/scene_off.json" B="$PWD/out/cue/scene_off_handover.json" \
  bash scripts/view_cue_ab.sh
