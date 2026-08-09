#!/usr/bin/env bash
# scripts/view_extend_ab.sh — how far a lost subject may be claimed present, for the eye to judge.
#
# A = out/fan_auto        edge reach UNBOUNDED (the shipped default since #102)
# B = out/ab_ext12        edge reach capped at 12 frames past the last measurement
#
# Same clip, same detections, same calibration, one knob apart: CoherenceConfig.max_extend_frames.
#
# Why this A/B exists. `extend_to_span` runs every subject to the full clip so a tracker-lost
# player is reconstructed rather than blinked out (R-6). Nothing bounded how long that claim
# survives. Measured 2026-08-09:
#
#   out/res_ab236/f236_res896.json   38 subjects, all 236 frames, median 37 % measured, worst 2 %
#                                    47.9 % of subject-frames > 12 frames from ANY measurement
#                                    11.9 % > 120 frames; worst 228 frames held from one edge
#   out/fan_auto/scene_fan_auto.json 32 subjects, all 120 frames, 41.4 % > 12 frames
#
# 12 is `max_fill_gap` — the gap this same pass refuses to bridge *between two* real observations.
# An edge has no observation on the far side at all.
#
# What to look at, in order:
#   1. The players standing still and doing nothing. In A many of them are held postures, not
#      people. In B they should end. Does the pitch read as emptier-but-honest, or as broken?
#   2. The moment a subject disappears in B. Is it where you would agree the evidence ran out,
#      or does a player you can clearly see vanish?
#   3. Anyone who survives in B but jumps or slides at their last frame — that is the coast
#      still running, and it means the cap is right but 12 is the wrong number.
#
# The verdict decides the DEFAULT. Until you judge it, `max_extend_frames` stays None
# (unbounded), because switching it changes what every scene on disk contains.
#
#   bash scripts/view_extend_ab.sh            # -> http://localhost:8000/world, tick "overlay B"
#   PORT=8123 bash scripts/view_extend_ab.sh
set -euo pipefail
cd "$(dirname "$0")/.."
exec env A="$PWD/out/fan_auto/scene_fan_auto.json" B="$PWD/out/ab_ext12/scene_ab_ext12.json" \
  bash scripts/view_cue_ab.sh
