#!/usr/bin/env bash
# scripts/local_setup.sh — one command to bring up the LOCAL (CPU) pitch3d environment.
#
# `just setup` installs only the `[dev]` extra (numpy) — enough for the tests and the
# fake-adapter dry-run, but NOT for the SMPL-X mesh demos (they need torch + smplx +
# matplotlib). This script is the full local bring-up: a CPU venv that can run the core
# suite, the dry-run, AND the Blender / matplotlib skinned-mesh demos (docs/blender-demo.md).
#
# This box has no NVIDIA GPU, so torch is installed from the CPU wheel index (small, no CUDA).
# For a GPU cloud box use scripts/cloud_setup.sh instead (CUDA torch + the real adapters).
#
# Overridable via env:
#   PITCH3D_VENV=.venv               venv directory
#   PITCH3D_PYTHON=python3
#   PITCH3D_TORCH=2.6.0              torch version (matches the pyproject pin)
#   PITCH3D_LOCAL_EXTRAS=export,demo,dev   pip extras (demo-ready default)
#   PITCH3D_SKIP_TORCH=0             =1 to skip the CPU torch install (tests/dry-run only;
#                                    pair with PITCH3D_LOCAL_EXTRAS=dev)
#
# Note: this does NOT install the `hmr` extra. `hmr` pins chumpy==0.70, whose legacy
# setup.py `import pip`s and fails to build under PEP 517 isolation — and the SMPL-X mesh
# demos don't need it (smplx 0.1.28 needs only numpy+torch; nothing here imports chumpy).
# So we install `smplx` directly instead. The full `hmr` extra is the GPU/cloud path
# (scripts/cloud_setup.sh), not this local demo path.
#   PITCH3D_BLENDER=<path>           Blender binary for the demo (else auto-detected)
#   PITCH3D_SMPLX_MODELS=SMPL-X/models   dir holding smplx/SMPLX_NEUTRAL.npz
set -euo pipefail

VENV="${PITCH3D_VENV:-.venv}"
PY="${PITCH3D_PYTHON:-python3}"
TORCH_VERSION="${PITCH3D_TORCH:-2.6.0}"
EXTRAS="${PITCH3D_LOCAL_EXTRAS:-export,demo,dev}"
SMPLX_PIN="smplx==0.1.28"   # == the pin in pyproject's [hmr] extra; installed sans chumpy (see header)
SKIP_TORCH="${PITCH3D_SKIP_TORCH:-0}"
SMPLX_MODELS="${PITCH3D_SMPLX_MODELS:-SMPL-X/models}"

cd "$(dirname "$0")/.."   # repo root, regardless of where this is invoked from

echo "== pitch3d local setup (CPU) =="
echo "repo:    $(pwd)"
echo "python:  $("$PY" --version 2>&1)"
echo "torch:   ${TORCH_VERSION} (cpu)    extras: [${EXTRAS}]    skip-torch: ${SKIP_TORCH}"
echo

# 1) ffmpeg/ffprobe — only needed for real --clip ingest; the demo + dry-run don't need it.
if ! command -v ffprobe >/dev/null 2>&1; then
  echo "NOTE: ffprobe not found — only required for real --clip ingest. Install with:"
  echo "      sudo apt-get update && sudo apt-get install -y ffmpeg"
  echo
fi

# 2) venv
if [ ! -d "${VENV}" ]; then
  "$PY" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
. "${VENV}/bin/activate"
python -m pip install -U pip wheel

# 3) CPU torch FIRST (from the cpu wheel index), so the torch==2.6.0-pinned extras are
#    already satisfied and pip won't pull the large default CUDA build over it.
if [ "${SKIP_TORCH}" = "1" ]; then
  echo "PITCH3D_SKIP_TORCH=1 — not installing torch (tests/dry-run only)."
else
  python -m pip install "torch==${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/cpu"
fi

# 4) the package + the chosen extras (editable). bpy is Blender-provided — never here.
python -m pip install -e ".[${EXTRAS}]"

# 4b) smplx for the mesh demos — installed directly (not via [hmr]) to avoid chumpy's
#     broken build under PEP 517 isolation. Skipped if torch is absent (SKIP_TORCH).
if [ "${SKIP_TORCH}" != "1" ]; then
  python -m pip install "${SMPLX_PIN}"
fi

# 5) verify the import surface the demos need
echo
echo "== verify =="
python - <<'PY'
import importlib
from importlib import metadata
for m in ("numpy", "torch", "smplx", "matplotlib"):
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", None) or metadata.version(m)
        print(f"  {m}: {ver}")
    except Exception as e:  # noqa: BLE001
        print(f"  {m}: NOT importable ({e})")
PY

# 6) point at the assets the Blender demo needs (informational — no download here)
echo
echo "== demo assets =="
BLENDER=""
for cand in "${PITCH3D_BLENDER:-}" \
            "/home/chubuchnyi/Downloads/blender-5.1.2-linux-x64/blender" \
            "$(command -v blender 2>/dev/null || true)"; do
  if [ -n "${cand}" ] && [ -x "${cand}" ]; then BLENDER="${cand}"; break; fi
done
if [ -n "${BLENDER}" ]; then
  echo "  Blender:    ${BLENDER}"
else
  echo "  Blender:    NOT found — set PITCH3D_BLENDER, or install Blender (docs/blender-demo.md)."
fi
if [ -f "${SMPLX_MODELS}/smplx/SMPLX_NEUTRAL.npz" ]; then
  echo "  SMPL-X:     ${SMPLX_MODELS}/smplx/SMPLX_NEUTRAL.npz"
else
  echo "  SMPL-X:     MISSING under ${SMPLX_MODELS} (MPI-gated, manual — docs/blender-demo.md)."
fi

echo
echo "Done. Next:"
echo "  python -m pytest                                      # core suite (GPU-free)"
echo "  PYTHONPATH=src python -m pitch3d --out-dir out/dryrun  # fake-adapter dry-run"
echo "  # SMPL-X skinned-mesh demo (out/cuda is a fake-pose export -> CANONICAL_UP=1):"
echo "  PITCH3D_CANONICAL_UP=1 .venv/bin/python scripts/smplx_export_meshes.py"
if [ -n "${BLENDER}" ]; then
  echo "  ${BLENDER} --background --python scripts/blender_render_meshes.py -- \\"
  echo "      --in out/cuda/mesh --out out/cuda/mesh/blender_scene.png   # -> docs/blender-demo.md"
fi
