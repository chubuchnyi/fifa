# pitch3d — task runner. Install `just` (https://github.com/casey/just) or run the
# underlying commands directly (shown in README).

# Default: list available recipes
default:
    @just --list

# Create a local virtualenv and install the package with dev + all extras (no bpy).
setup:
    python3 -m venv .venv
    .venv/bin/pip install -U pip
    .venv/bin/pip install -e ".[dev]"

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
