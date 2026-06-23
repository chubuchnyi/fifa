#!/usr/bin/env bash
# scripts/stage_wasb_weight.sh — stage the WASB ball-tracking backend onto the persistent disk.
#
# What it does (idempotent): clone nttcom/WASB-SBDT (MIT) into $WS/repos, install gdown, and pull
# the *soccer* WASB checkpoint from the repo's Google-Drive model zoo into $WS/weights/wasb/. These
# are exactly the two paths pitch3d's WASB backend reads by default
# (src/pitch3d/adapters/models/wasb_backend.py): the checkout that provides
# `src/{detectors,trackers,dataloaders,configs}` and the `*.pth.tar` weight.
#
# The Google-Drive file id is the "WASB (Ours) / Soccer" cell of the repo's MODEL_ZOO.md
# (verified 2026-06-23 against github.com/nttcom/WASB-SBDT) — NOT invented. If the id ever moves,
# re-read MODEL_ZOO.md and update GDRIVE_ID below.
#
# After staging, wire it at the composition root (config flows through the env vars this script
# honours, so staging and runtime agree on the paths):
#
#   PYTHONPATH=src python -m pitch3d --clip clip.mp4 --frames 6 --device cuda \
#     --ball tracknet --ball-backend pitch3d.adapters.models.wasb_backend:make \
#     --detector rfdetr --tracker bytetrack --render overlay --export gltf
#
# Overridable via env (defaults match the backend's and runpod-agent-setup.md's $WS layout):
#   WS=/workspace                  the persistent disk root
#   PITCH3D_WASB_REPO=$WS/repos/WASB-SBDT                          checkout dir (must contain src/)
#   PITCH3D_WASB_CKPT=$WS/weights/wasb/wasb_soccer_best.pth.tar    soccer weight destination
set -euo pipefail

WS="${WS:-/workspace}"
REPO_DIR="${PITCH3D_WASB_REPO:-$WS/repos/WASB-SBDT}"
CKPT="${PITCH3D_WASB_CKPT:-$WS/weights/wasb/wasb_soccer_best.pth.tar}"
GDRIVE_ID="1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l"   # MODEL_ZOO.md → WASB (Ours) → Soccer

echo "== stage WASB ball backend =="
echo "repo:    $REPO_DIR"
echo "weight:  $CKPT"
echo

# 1) WASB checkout (provides src/{detectors,trackers,dataloaders,configs}). Clone is idempotent.
if [ -d "$REPO_DIR/src" ]; then
  echo "[1/4] checkout present — skipping clone"
else
  echo "[1/4] cloning nttcom/WASB-SBDT -> $REPO_DIR"
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone https://github.com/nttcom/WASB-SBDT.git "$REPO_DIR"
fi

# 2) gdown — the model zoo lives on Google Drive; pip into whatever venv is active.
echo "[2/4] ensuring gdown is installed"
python -m pip install -q -U gdown

# 3) soccer weight. Skip if a non-trivial file is already there (a quota/HTML error page is tiny).
mkdir -p "$(dirname "$CKPT")"
if [ -f "$CKPT" ] && [ "$(stat -c%s "$CKPT")" -gt 1000000 ]; then
  echo "[3/4] weight present ($(stat -c%s "$CKPT") bytes) — skipping download"
else
  echo "[3/4] downloading soccer checkpoint (Drive id $GDRIVE_ID)"
  python -m gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O "$CKPT"
fi

echo
SZ="$(stat -c%s "$CKPT" 2>/dev/null || echo 0)"
if [ "$SZ" -le 1000000 ]; then
  echo "ERROR: $CKPT is only ${SZ} bytes — Drive likely returned a quota/confirm page, not the weight."
  echo "       Retry later, or use the repo's bulk setup script (MODEL_ZOO.md links a GET_STARTED.md)."
  exit 1
fi

# 4) NumPy 2.0 compat. WASB (2023) uses the removed `np.Inf` alias in its trackers/evaluator;
#    the pod ships NumPy >= 2, where that alias raises AttributeError. Rewrite it to `np.inf` in
#    the checkout. Idempotent: a second run finds nothing left to change.
echo "[4/4] patching np.Inf -> np.inf for NumPy 2.0 (WASB trackers/evaluator)"
_np_inf_files="$(grep -rlE '\bnp\.Inf\b' "$REPO_DIR/src" 2>/dev/null || true)"
if [ -n "$_np_inf_files" ]; then
  echo "$_np_inf_files" | xargs sed -i 's/\bnp\.Inf\b/np.inf/g'
  echo "      patched $(echo "$_np_inf_files" | wc -l) file(s)"
else
  echo "      none found (already patched)"
fi

echo "OK: staged $CKPT (${SZ} bytes)."
echo
echo "Next — first validate the ball backend in isolation (committed GPU smoke):"
echo "  PYTHONPATH=src python scripts/smoke_wasb_gpu.py"
echo
echo "Then drive the full real ball path (needs CUDA; the WASB detector asserts cuda):"
echo "  PITCH3D_WASB_REPO=$REPO_DIR PITCH3D_WASB_CKPT=$CKPT \\"
echo "  PYTHONPATH=src python -m pitch3d --clip clip.mp4 --frames 6 --device cuda \\"
echo "    --ball tracknet --ball-backend pitch3d.adapters.models.wasb_backend:make \\"
echo "    --detector rfdetr --tracker bytetrack --render overlay --export gltf"
