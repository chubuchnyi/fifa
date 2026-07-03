#!/usr/bin/env bash
# scripts/pod_seedvr2.sh — ON THE POD: SeedVR2 upscale of v2v-finished clips (priority-2c).
#
# Uses the standalone CLI from numz/ComfyUI-SeedVR2_VideoUpscaler (no ComfyUI needed;
# requirements are light — no flash-attn/apex). Weights auto-download from HF into
# MODEL_DIR on the persistent volume. Defaults upscale the WINNING v2v variant
# (kit-boost + night-grade + rgb control, STATUS §6 2026-07-03) 480p → 720p.
#
# Env (pod defaults):
#   INPUTS   space-separated clip list   (default: sideline_rgbnight.mp4 sideline_rgbboost.mp4)
#   RES=720  BATCH=33 (4n+1 for temporal consistency)  DIT=seedvr2_ema_3b_fp16.safetensors
#   MODEL_DIR=/workspace/models/SEEDVR2  VENV=/workspace/venvs/seedvr2
set -euo pipefail

FIFA=/workspace/fifa
REPO_DIR=/workspace/repos/seedvr2_videoupscaler
VENV="${VENV:-/workspace/venvs/seedvr2}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/SEEDVR2}"
INPUTS="${INPUTS:-$FIFA/out/v2v/sideline_rgbnight.mp4 $FIFA/out/v2v/sideline_rgbboost.mp4}"
RES="${RES:-720}"
BATCH="${BATCH:-33}"
DIT="${DIT:-seedvr2_ema_3b_fp16.safetensors}"
export HF_HOME="${HF_HOME:-/workspace/hf}"

echo "== seedvr2 :: res=$RES batch=$BATCH dit=$DIT =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

[ -d "$REPO_DIR" ] || git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git "$REPO_DIR"

if [ ! -x "$VENV/bin/python" ]; then
  echo "== venv: creating $VENV =="
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
  "$VENV/bin/pip" install -q -r "$REPO_DIR/requirements.txt"
fi
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

mkdir -p "$MODEL_DIR"
cd "$REPO_DIR"
for IN in $INPUTS; do
  test -f "$IN" || { echo "seedvr2: missing input $IN" >&2; exit 1; }
  OUT="${IN%.mp4}_${RES}p.mp4"
  echo "== upscaling: $IN → $OUT =="
  "$VENV/bin/python" inference_cli.py "$IN" \
    --output "$OUT" --model_dir "$MODEL_DIR" --dit_model "$DIT" \
    --resolution "$RES" --batch_size "$BATCH" --video_backend ffmpeg
  echo "SEEDVR2_OK -> $OUT"
done
echo "BATCH_SEEDVR2_OK"
