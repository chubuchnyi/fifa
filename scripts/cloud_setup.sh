#!/usr/bin/env bash
# scripts/cloud_setup.sh — provision a GPU cloud box for pitch3d MVP development.
#
# What it does: detect the GPU/driver, create a venv, install a CUDA build of the
# pinned torch BEFORE the extras (so pip can't pull the CPU wheel over it), install
# the package with the heavy reals + dev tooling, then verify torch sees the GPU.
#
# What it does NOT do: make the still-unwired backends work. GVHMR (`--pose gvhmr`),
# TrackNet (`--ball tracknet`) and the keypoint calibrator (`--calibrator keypoints`)
# remain stubs — their upstreams aren't pip-installable, so the extras ship only the
# substrate (torch/smplx/chumpy). A GPU box unblocks *wiring* them and runs the real
# detection+tracking path (RF-DETR + ByteTrack) on `--device cuda`. See docs/cloud-dev.md.
#
# Overridable via env:
#   PITCH3D_CUDA=cu124   torch 2.6.0 wheels exist for cu118 | cu124 | cu126
#   PITCH3D_EXTRAS=...   pip extras to install (bpy is Blender-provided — never here)
#   PITCH3D_VENV=.venv   venv directory
#   PITCH3D_PYTHON=python3
set -euo pipefail

CUDA="${PITCH3D_CUDA:-cu124}"
TORCH_VERSION="2.6.0"   # MUST equal the `torch==` pin in pyproject (hmr/ball/env/...)
EXTRAS="${PITCH3D_EXTRAS:-cv,hmr,ball,export,mcp,dev}"
VENV="${PITCH3D_VENV:-.venv}"
PY="${PITCH3D_PYTHON:-python3}"

cd "$(dirname "$0")/.."   # repo root, regardless of where this is invoked from

echo "== pitch3d cloud setup =="
echo "repo:     $(pwd)"
echo "python:   $("$PY" --version 2>&1)"
echo "cuda tag: ${CUDA}    torch: ${TORCH_VERSION}    extras: [${EXTRAS}]"
echo

# 1) GPU + driver visible? (informational — a CPU box is still a valid target)
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi not found — no NVIDIA driver visible. The CUDA torch wheel"
  echo "         will still install, but torch.cuda.is_available() will be False."
fi
echo

# 2) ffmpeg/ffprobe — system dep for the real video ingestor (FFmpegIngestor).
if ! command -v ffprobe >/dev/null 2>&1; then
  echo "WARNING: ffprobe not found — real --clip ingest needs it. Install it with:"
  echo "         sudo apt-get update && sudo apt-get install -y ffmpeg"
  echo
fi

# 3) venv
if [ ! -d "${VENV}" ]; then
  "$PY" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
. "${VENV}/bin/activate"
python -m pip install -U pip wheel

# 4) CUDA torch FIRST, pinned, from the pytorch index. Installing it before the
#    extras means each extra's `torch==2.6.0` constraint is already satisfied, so
#    pip leaves this GPU wheel in place instead of resolving the default CPU one.
python -m pip install "torch==${TORCH_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${CUDA}"

# 5) the package + heavy reals + dev tooling (editable). bpy stays out on purpose.
python -m pip install -e ".[${EXTRAS}]"

# 6) verify the GPU is actually reachable from torch
echo
echo "== verify =="
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
ok = torch.cuda.is_available()
print("cuda available:", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0))
else:
    print("NOTE: torch cannot see a GPU — check the driver and that PITCH3D_CUDA",
          "matches it (nvidia-smi top-right shows the max CUDA version).")
PY

echo
echo "Done. Next:"
echo "  PYTHONPATH=src python -m pytest                 # core suite, GPU-free"
echo "  then run the --device cuda golden path in docs/cloud-dev.md"
