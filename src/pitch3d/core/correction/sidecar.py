"""Reading the annotator's ``edits.json`` sidecar from outside ``poseannot`` (#128).

A scene carries the corrections the *pipeline* made. The ones the *operator* made live beside it
in a sidecar, because ``poseannot`` never rewrites a scene file — it appends rows to
``edits.json`` and folds them in at load time (``poseannot/scene_state.py``). That is a good
design for the annotator and an invisible cliff for everything else: an exporter handed
``scene.json`` sees the pipeline's work and none of the operator's.

So the loader lives here, where the export can reach it, and ``poseannot.edits`` re-exports it.
The merge order is fixed — sidecar rows go *after* the scene's own, exactly as the annotator
stacks them — because with corrections the order is the meaning, not a detail.
"""

from __future__ import annotations

import json
from pathlib import Path

from pitch3d.core.scene.serialization import from_json


def load_corrections(path: str | Path) -> list:
    """The corrections persisted in ``path``, or ``[]`` if it is absent or empty."""
    p = Path(path)
    if not p.exists():
        return []
    raw = p.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    data = json.loads(raw)
    return list(from_json(json.dumps(data.get("corrections", []))))


def sidecar_for(scene_path: str | Path) -> Path | None:
    """The annotator's sidecar for ``scene_path``, if one exists.

    Two names, both minted by ``poseannot/clips.py``: per-scene when a clip dir holds more than
    one scene, plain ``edits.json`` otherwise. Returns ``None`` rather than a missing path so a
    caller cannot accidentally report "applied nothing" as "applied an empty file".
    """
    p = Path(scene_path)
    for cand in (p.with_name(f"{p.stem}_edits.json"), p.with_name("edits.json")):
        if cand.exists():
            return cand
    return None


def merge_sidecar(scene, path: str | Path):
    """``scene`` with ``path``'s corrections appended — or ``scene`` itself if there are none."""
    from dataclasses import replace

    persisted = load_corrections(path)
    if not persisted:
        return scene
    return replace(scene, corrections=[*scene.corrections, *persisted])
