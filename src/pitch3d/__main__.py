"""``python -m pitch3d`` → the CLI dry-run (full golden path on fakes)."""

from __future__ import annotations

from .app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
