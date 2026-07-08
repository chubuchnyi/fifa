#!/usr/bin/env bash
# scripts/run_sam3dbody.sh — variant B (SAM 3D Body) real E2E on a GPU pod.
#
# Same golden path as scripts/pod_real_e2e.sh but forces the SAM 3D Body pose backend
# (MHR -> SMPL-X via Meta's converter) instead of the default SMPLest-X. Use it to produce
# the B-side of the A/B pose bake-off (scene.json + overlays) on IDENTICAL detect/track/
# calibrate upstream, so A and B differ ONLY in the pose net.
#
# Prereqs on the pod (see src/pitch3d/adapters/models/sam3dbody_backend.py docstring):
#   * repos: PITCH3D_SAM3D_REPO (sam-3d-body, gated ckpt downloaded), PITCH3D_MHR_REPO (MHR,
#     `python -m mhr.download_assets` run so assets/*.fbx + tools/mhr_smpl_conversion/assets/*).
#   * pip: braceexpand roma yacs iopath einops omegaconf termcolor + smplx.
#   * NATIVE ABI PIN (Blackwell / torch 2.8.0+cu128): pymomentum-gpu==0.1.90.post0.
#     Newer wheels (>=0.1.97) segfault in Character.load_fbx (libtorch ABI mismatch).
#     This script re-asserts the pin unless PYM_PIN=0.
#
# Env (overridable): FRAMES (default 48), OUT (default out/B_sam3dbody), PITCH3D_CLIP.
set -euo pipefail

REPO="${PITCH3D_REPO:-/workspace/fifa}"
PY="${PITCH3D_PY:-/workspace/.venv/bin/python}"
cd "$REPO"

# The pymomentum solver extension links libtorch.so at import — put <torch>/lib on the
# loader path for THIS process and everything it spawns (the pipeline imports it inside).
TL="$("$PY" -c 'import torch,os;print(os.path.dirname(torch.__file__)+"/lib")')"
export LD_LIBRARY_PATH="$TL:${LD_LIBRARY_PATH:-}"

# Re-assert the working pymomentum-gpu wheel (idempotent, ~instant if already correct).
if [ "${PYM_PIN:-1}" = "1" ]; then
  have="$("$PY" -c 'import importlib.metadata as m;print(m.version("pymomentum-gpu"))' 2>/dev/null || echo none)"
  if [ "$have" != "0.1.90.post0" ]; then
    echo "== pinning pymomentum-gpu==0.1.90.post0 (was: $have) — newer wheels segfault on Blackwell/torch2.8"
    "$PY" -m pip install --quiet --force-reinstall --no-deps "pymomentum-gpu==0.1.90.post0"
  else
    echo "== pymomentum-gpu already pinned at 0.1.90.post0"
  fi
fi

export POSE_BACKEND="pitch3d.adapters.models.sam3dbody_backend:make"
export FRAMES="${FRAMES:-48}"
export OUT="${OUT:-out/B_sam3dbody}"
export FORMAT="${FORMAT:-json}"

echo "== run_sam3dbody (variant B) FRAMES=$FRAMES OUT=$OUT LD=$TL $(date) =="
bash scripts/pod_real_e2e.sh
echo "== run_sam3dbody done rc=$? $(date) =="
