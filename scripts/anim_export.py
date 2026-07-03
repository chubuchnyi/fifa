"""Thin shim — the anim exporter lives in the package now: :mod:`pitch3d.app.anim_export`.

Kept so every existing wrapper keeps working unchanged (`pod_make_video.sh`, `demo_video.sh`,
docs all call `python scripts/anim_export.py`). Env variables still act as defaults; the CLI
flags of the package module override them (see `--help`).
"""

from pitch3d.app.anim_export import main

if __name__ == "__main__":
    raise SystemExit(main())
