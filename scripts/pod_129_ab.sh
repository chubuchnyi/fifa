#!/usr/bin/env bash
# #129 A/B: the one fitted camera in the chain, vs the chain without it. Same clip, same
# frames, same camera framing -- only RIGID_CAMERA differs.
cd /workspace/fifa
export PITCH3D_CLIP=/workspace/Colombia-1-0-Congo-DR1080p.mp4
for arm in on off; do
  echo "########## ARM=$arm  RIGID_CAMERA=$([ $arm = on ] && echo 1 || echo 0)  $(date -u +%H:%M:%SZ)"
  RIGID_CAMERA=$([ "$arm" = on ] && echo 1 || echo 0) \
  ANIM_CAMERAS=sideline OUT="out/n129_$arm" REUSE_SCENE=0 \
    bash scripts/pod_finish_batch.sh 2>&1 | sed "s/^/[$arm] /"
  echo "########## ARM=$arm exit=${PIPESTATUS[0]} $(date -u +%H:%M:%SZ)"
done
echo "AB_129_ALL_DONE"
