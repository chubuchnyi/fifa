#!/usr/bin/env bash
# Structure-locked v2v finishing spike on the pod (research priority 1).
#
#   FRAMES=out/anim_adr11/mesh/frames/broadcast OUT=out/v2v/broadcast_vace.mp4 \
#     bash scripts/pod_v2v.sh [extra pod_v2v_finish.py flags]
#
# Idempotent env: separate venv (torch pin differs from the pipeline venv) and
# HF weights cached on the volume, both survive pod stop/start.
set -euo pipefail
cd /workspace/fifa

VENV=${VENV:-/workspace/venvs/genfinish}
export HF_HOME=${HF_HOME:-/workspace/hf}
FRAMES=${FRAMES:-out/anim_adr11/mesh/frames/broadcast}
REF=${REF:-out/v2v/ref_night.png}
OUT=${OUT:-out/v2v/broadcast_vace.mp4}
SRC_CLIP=${SRC_CLIP:-/workspace/Colombia-1-0-Congo-DR1080p.mp4}

echo "== pod v2v :: frames=$FRAMES out=$OUT venv=$VENV hf=$HF_HOME =="

if [ ! -x "$VENV/bin/python" ]; then
  echo "== creating venv $VENV"
  mkdir -p "$(dirname "$VENV")"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -U pip
fi
if ! "$VENV/bin/python" -c "import diffusers, transformers, torch" 2>/dev/null; then
  echo "== installing deps (torch + diffusers stack)"
  "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu124
  "$VENV/bin/pip" install -q diffusers transformers accelerate ftfy sentencepiece protobuf "imageio[ffmpeg]" opencv-python-headless
fi
"$VENV/bin/python" - <<'PY'
import diffusers, torch, transformers
print(f"== env: torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"diffusers {diffusers.__version__} transformers {transformers.__version__}")
PY

mkdir -p "$(dirname "$REF")" "$(dirname "$OUT")"
if [ ! -f "$REF" ]; then
  echo "== extracting night reference frame from $SRC_CLIP"
  ffmpeg -y -v error -i "$SRC_CLIP" -vf "select=eq(n\,0)" -vframes 1 "$REF"
fi

"$VENV/bin/python" scripts/pod_v2v_finish.py \
  --frames-dir "$FRAMES" --ref-image "$REF" --out "$OUT" "$@"

echo "POD_V2V_OK -> $OUT"
