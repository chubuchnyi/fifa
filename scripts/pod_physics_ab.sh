#!/usr/bin/env bash
# T1b/T1c A/B: are the joint + orientation ceilings worth enforcing, judged by eye?
#
# Measured off the 2026-08-05 pod scene, the exported poses violate the joint ceiling 118 times
# and the orientation ceiling 11 times (worst 2212 and 4514 deg/s against 600 and 720), because
# config/physics.yaml shipped those gates disabled under a profile that calls itself "no future
# gates". Enabling them via the `safe_new` profile takes both counts to 0 while keeping 97.8% of
# root and 98.5% of body-joint angular travel, and moves players 0.0000 m
# (scripts/pose_gate_ab.py, scripts/bench_subject_steadiness.py).
#
# The numbers say the clamp lands on jitter, not on real motion. This renders both so the eye can
# say the same thing — or not. Only PHYSICS_PROFILE differs; #129's rigid camera is on in both.
#
#   bash scripts/pod_physics_ab.sh          # -> out/phys_default, out/phys_safe_new
set -euo pipefail
cd /workspace/fifa
export PITCH3D_CLIP="${PITCH3D_CLIP:-/workspace/Colombia-1-0-Congo-DR1080p.mp4}"

for prof in default safe_new; do
  echo "########## PROFILE=$prof  $(date -u +%H:%M:%SZ)"
  PHYSICS_PROFILE="$prof" RIGID_CAMERA=1 \
  ANIM_CAMERAS="${ANIM_CAMERAS:-sideline}" OUT="out/phys_$prof" REUSE_SCENE=0 \
    bash scripts/pod_finish_batch.sh 2>&1 | sed "s/^/[$prof] /"
  echo "########## PROFILE=$prof exit=${PIPESTATUS[0]} $(date -u +%H:%M:%SZ)"
done
echo "AB_PHYSICS_ALL_DONE"
