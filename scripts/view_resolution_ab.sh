#!/usr/bin/env bash
# scripts/view_resolution_ab.sh — compare detector resolutions in the 3D viewer.
#
# Three scenes, same clip, same 60 frames, same code, one run each. Only the named thing differs:
#
#   out/res_ab/res560.json           detector input square 560 (the old default)
#   out/res_ab/res896.json           detector input square 896
#   out/res_ab/res896_handover.json  896 plus the #135 handover merge
#
# Deliberately NOT compared against out/cue/scene_off.json. That scene predates the #137 team fix,
# the handover pass and the kit-reader fix, so a difference against it would mix the resolution
# change with a month of other changes.
#
# Default pair is 560 (primary) vs 896 (overlay). Pick another pair with A= and B=:
#
#   bash scripts/view_resolution_ab.sh                                  # 560 vs 896
#   B=$PWD/out/res_ab/res896_handover.json bash scripts/view_resolution_ab.sh   # 560 vs 896+merge
#   A=$PWD/out/res_ab/res896.json B=$PWD/out/res_ab/res896_handover.json \
#       bash scripts/view_resolution_ab.sh                              # what the merge did
#
# Then open http://localhost:8000/world and tick "overlay B".
#
# Measured on these three scenes (scripts/track_quality.py, 60 frames):
#
#   scene              subjects  measured  interpolated  imputed  imputed %
#   res560                   22      1052            52      216      16.4
#   res896                   24      1089            42      309      21.5
#   res896_handover          22      1085            48      187      14.2
#
# `imputed` frames are the ones to look at: a subject on an imputed frame has frozen limbs and a
# coasting root, which is what reads as a sliding mannequin.
set -euo pipefail
cd "$(dirname "$0")/.."
A="${A:-$PWD/out/res_ab/res560.json}"
B="${B:-$PWD/out/res_ab/res896.json}"
exec env A="$A" B="$B" bash scripts/view_cue_ab.sh
