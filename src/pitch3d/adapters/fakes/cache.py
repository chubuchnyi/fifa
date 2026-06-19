"""Content-addressable caches — in-memory and on-disk (ADR-0004, NFR-4).

Both back :class:`~pitch3d.core.ports.cache.Cache`, so the key derivation (pure, in core)
is shared and an artifact written by one run is found by another. ``MemoryCache`` is a dict
for tests; ``DiskCache`` pickles artifacts under a root dir so a real session survives a
restart. Pickle is fine here — artifacts are our own dataclasses + numpy arrays; a real
deployment would swap a typed/safer codec behind the same port.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pitch3d.core.ports.cache import Cache


@dataclass
class MemoryCache(Cache):
    """Process-local dict cache."""

    _store: dict[str, Any] = field(default_factory=dict, repr=False)

    def has(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str) -> Any | None:
        return self._store.get(key)

    def put(self, key: str, artifact: Any) -> None:
        self._store[key] = artifact


@dataclass
class DiskCache(Cache):
    """Pickle-on-disk cache keyed by filename; an in-memory index avoids re-reads."""

    root: Path = field(default_factory=lambda: Path("out/cache"))
    _index: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.pkl"

    def has(self, key: str) -> bool:
        return key in self._index or self._path(key).exists()

    def get(self, key: str) -> Any | None:
        if key in self._index:
            return self._index[key]
        path = self._path(key)
        if not path.exists():
            return None
        with path.open("rb") as fh:
            artifact = pickle.load(fh)
        self._index[key] = artifact
        return artifact

    def put(self, key: str, artifact: Any) -> None:
        self._index[key] = artifact
        with self._path(key).open("wb") as fh:
            pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
