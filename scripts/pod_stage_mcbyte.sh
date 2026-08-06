#!/usr/bin/env bash
# scripts/pod_stage_mcbyte.sh — ON THE POD: everything #133's mask cue needs, in one command.
#
# The point is that the pod is billed by the hour and this is all download-and-verify, no GPU. Run
# it once on a fresh pod (or after the volume is wiped) and the mask work is ready to start.
#
# Stages, all idempotent:
#   1. backends/McByte      — the checkout (gitignored here, so it does not come with git pull).
#                             We use its vendored Cutie and its tracker only as the reference for
#                             the cue; the association itself is ours.
#   2. models/cutie/*.pth   — propagation weights, verified by SIZE and md5. A truncated download
#                             of cutie-base-mega still exists, still weighs a plausible 109 MB, and
#                             fails only when torch reads the zip — which cost a debugging round
#                             locally. Check the length, not the path.
#   3. weights/ symlinks    — where CutiePropagator looks, pointing back at models/ (the repo's
#                             one-place-for-weights rule).
#   4. python deps          — hydra-core, omegaconf; plus lap/cython_bbox/loguru if the reference
#                             tracker is to be importable.
set -euo pipefail
VOL="${PITCH3D_VOL:-/workspace}"
REPO="${PITCH3D_REPO:-$VOL/fifa}"
PY="${PITCH3D_PY:-$VOL/.venv/bin/python}"
CUTIE_URL=https://github.com/hkchengrex/Cutie/releases/download/v1.0
cd "$REPO"

echo "== 1/4 McByte checkout =="
if [ -d backends/McByte/.git ]; then
  echo "   present"
else
  mkdir -p backends
  git clone --depth 1 https://github.com/tstanczyk95/McByte.git backends/McByte
fi

echo "== 2/4 Cutie weights =="
mkdir -p "$VOL/models/cutie"
stage_weight() {  # name expected_md5 expected_bytes
  local f="$VOL/models/cutie/$1"
  if [ -f "$f" ] && [ "$(stat -c%s "$f")" = "$3" ] && [ "$(md5sum "$f" | cut -d' ' -f1)" = "$2" ]; then
    echo "   $1 ok"; return
  fi
  echo "   fetching $1 ..."
  curl -sL -C - -o "$f" "$CUTIE_URL/$1"
  local sz md5
  sz=$(stat -c%s "$f"); md5=$(md5sum "$f" | cut -d' ' -f1)
  # Both checks, because they fail differently: a wrong size is a truncated transfer, a wrong
  # md5 with the right size is a different file than the one this code was written against.
  [ "$sz" = "$3" ] || { echo "   $1: got $sz bytes, expected $3" >&2; exit 1; }
  [ "$md5" = "$2" ] || { echo "   $1: md5 $md5, expected $2" >&2; exit 1; }
  echo "   $1 ok ($sz bytes)"
}
stage_weight cutie-base-mega.pth       a6071de6136982e396851903ab4c083a 140443788
stage_weight coco_lvis_h18_itermask.pth 6fb97de7ea32f4856f2e63d146a09f31 40700707

echo "== 3/4 weights symlinks =="
W="$REPO/backends/McByte/mask_propagation/Cutie/weights"
mkdir -p "$W"
for f in cutie-base-mega.pth coco_lvis_h18_itermask.pth; do
  ln -sf "$VOL/models/cutie/$f" "$W/$f"
done
ls -l "$W" | sed 's/^/   /'

echo "== 4/4 python deps =="
"$PY" -m pip install -q hydra-core omegaconf lap cython_bbox loguru 2>&1 | tail -2 || true
"$PY" - <<'EOF'
import importlib.util  # find_spec lives in the submodule; plain `import importlib` has no .util
missing = [m for m in ("hydra", "omegaconf", "torch", "transformers")
           if not importlib.util.find_spec(m)]
print("   missing:", missing or "none")
EOF

echo "== smoke: does the propagator build and run one step? =="
cd "$REPO" && PITCH3D_DEVICE="${PITCH3D_DEVICE:-cuda}" "$PY" - <<'EOF'
import sys; sys.path.insert(0, "src")
import numpy as np
from pitch3d.adapters.models.mask_propagation import make
p = make()
img = np.zeros((256, 256, 3), dtype=np.uint8); img[60:200, 80:160] = 200
lab = np.zeros((256, 256), dtype=np.int64); lab[60:200, 80:160] = 1
m = p.seed(img, lab, [1])
print("   seed ok, ids", sorted(np.unique(m).tolist()))
print("   step ok, ids", sorted(np.unique(p.step(img)).tolist()))
EOF
echo "MCBYTE_STAGE_OK"
