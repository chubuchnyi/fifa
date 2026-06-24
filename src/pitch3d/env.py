"""Zero-dependency ``.env`` loader for machine-specific paths and keys.

Absolute paths (Blender binary, SMPL-X model dir, on-pod repo/weight locations) and access
details (RunPod SSH key) differ per machine, so they must never be hard-coded in tracked
source. They live in a gitignored ``.env`` at the repo root; ``.env.example`` documents the
full set. Both shell scripts (``set -a; . .env; set +a``) and Python read the same file.

Contract: :func:`load_env` is **non-destructive** — a variable already present in the real
environment (e.g. exported by ``scripts/demo.sh`` or passed inline) always wins over ``.env``.
``~`` is expanded so ``POD_SSH_KEY=~/.ssh/id`` works from either shell or Python.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_env(start: Path | None = None) -> Path | None:
    """Return the nearest ``.env`` walking up from ``start`` (default: CWD), or ``None``."""
    here = Path(start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        candidate = d / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env(path: str | Path | None = None) -> Path | None:
    """Populate ``os.environ`` from ``.env`` without overriding existing variables.

    Returns the file that was loaded (so callers can log it), or ``None`` if none was found.
    Lines that are blank, comments (``#``), or lack ``=`` are skipped; surrounding quotes and a
    leading ``~`` in the value are resolved.
    """
    env_file = Path(path) if path is not None else find_env()
    if env_file is None or not env_file.is_file():
        return None
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        if value.startswith("~"):
            value = os.path.expanduser(value)
        os.environ.setdefault(key, value)
    return env_file


def env(key: str, default: str | None = None) -> str | None:
    """Read ``key`` from the environment, loading ``.env`` lazily on first miss."""
    if key not in os.environ:
        load_env()
    return os.environ.get(key, default)
