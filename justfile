# pitch3d — task runner. Install `just` (https://github.com/casey/just) or run the
# underlying commands directly (shown in README).

# Default: list available recipes
default:
    @just --list

# Minimal local venv: the package + dev tooling only ([dev] = numpy/pytest/mypy/ruff).
# Enough for `just test` and `just dryrun`. For the SMPL-X mesh demos use `setup-local`.
setup:
    python3 -m venv .venv
    .venv/bin/pip install -U pip
    .venv/bin/pip install -e ".[dev]"

# Full LOCAL (CPU) env: CPU torch + smplx + export/demo/dev so the SMPL-X Blender/matplotlib
# demos run too (docs/blender-demo.md). Tune via env (see scripts/local_setup.sh header).
setup-local:
    ./scripts/local_setup.sh

# Provision a rented GPU box: CUDA torch + real adapters + dev, then verify the GPU.
# Override the CUDA build / extras via env (see docs/cloud-dev.md): PITCH3D_CUDA=cu118 ...
cloud-setup:
    ./scripts/cloud_setup.sh

# Same, for a Blackwell (sm_120) box on the RunPod cu128 image: reuse the image's
# torch 2.8.0+cu128 (held by scripts/constraints-cu128.txt) — see docs/runpod-runbook.md §2.
cloud-setup-blackwell:
    PITCH3D_REUSE_SYSTEM_TORCH=1 ./scripts/cloud_setup.sh

# Run the core test suite (no GPU, no Blender required).
test:
    python3 -m pytest

# Run the end-to-end dry-run pipeline on fake adapters.
dryrun:
    PYTHONPATH=src python3 -m pitch3d --out-dir out/dryrun

# Static checks (best-effort; requires dev extras).
lint:
    python3 -m ruff check src tests
    python3 -m mypy

# Remove caches and dry-run outputs.
clean:
    rm -rf out .cache .pytest_cache .mypy_cache .ruff_cache
