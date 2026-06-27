"""Scene versioning — named snapshots + rollback over the in-memory edit session (M3-6).

Corrections are the sole edit path (ADR-0002) and the proposal is never mutated, so the mutable
state worth checkpointing is the whole :class:`Scene` (its correction stack plus any assembled
assets). A :class:`Snapshot` is a **deep, independent copy** of a Scene tagged with a content
fingerprint (ADR-0004, content-addressed): identical scene content → identical fingerprint, so a
redundant snapshot is detectable and a rollback to the current state is a provable no-op.

This is pure core (no GPU/Blender) and is the same checkpoint primitive the LLM agent uses to
bracket a risky edit and roll back when the attention list gets worse (ADR-0008/0010, A-10).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass

from .scene import Scene
from .serialization import encode


def scene_fingerprint(scene: Scene) -> str:
    """Content-addressed SHA-256 over the scene's canonical JSON (ADR-0004).

    Reuses the serialization codec so the fingerprint covers exactly what a saved scene would —
    correction stack included — and is stable across deep copies (equal content → equal digest).
    """
    canonical = json.dumps(encode(scene), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    """An immutable, deep-copied checkpoint of a Scene at a point in the edit session."""

    name: str
    scene_id: str
    fingerprint: str
    scene: Scene  # an independent deep copy — never the live object
    note: str | None = None
    created_at: str | None = None


class SnapshotStore:
    """Named snapshots per scene (last-write-wins on name); pure in-memory, no I/O."""

    def __init__(self) -> None:
        self._by_scene: dict[str, dict[str, Snapshot]] = {}

    def take(
        self, scene: Scene, name: str, *, note: str | None = None, created_at: str | None = None
    ) -> Snapshot:
        """Deep-copy ``scene`` under ``name`` (replacing a same-named snapshot)."""
        if not name:
            raise ValueError("snapshot name must be non-empty")
        snap = Snapshot(
            name=name,
            scene_id=scene.id,
            fingerprint=scene_fingerprint(scene),
            scene=copy.deepcopy(scene),
            note=note,
            created_at=created_at,
        )
        self._by_scene.setdefault(scene.id, {})[name] = snap
        return snap

    def names(self, scene_id: str) -> list[str]:
        return list(self._by_scene.get(scene_id, {}))

    def list(self, scene_id: str) -> list[Snapshot]:
        return list(self._by_scene.get(scene_id, {}).values())

    def get(self, scene_id: str, name: str) -> Snapshot:
        snaps = self._by_scene.get(scene_id, {})
        if name not in snaps:
            raise KeyError(f"no snapshot {name!r} for scene {scene_id!r}")
        return snaps[name]

    def restore(self, scene_id: str, name: str) -> Scene:
        """Return a fresh deep copy of the named snapshot's scene (the caller installs it).

        The stored snapshot stays pristine even if the caller mutates (or further edits) the
        returned scene — both ``take`` and ``restore`` copy, so a snapshot is a true checkpoint.
        """
        return copy.deepcopy(self.get(scene_id, name).scene)


__all__ = ["Snapshot", "SnapshotStore", "scene_fingerprint"]
