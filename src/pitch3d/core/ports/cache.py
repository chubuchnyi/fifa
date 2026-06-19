"""Cache — content-addressable artifact store (NFR-4, ADR-0004).

Each stage's output is keyed by ``hash(stage + input_hash + params + model_version)``.
Re-running with unchanged inputs is a hit, so expensive generative passes (avatars,
ViewSynthesizer) are never recomputed needlessly. The core defines the contract and the
key derivation; concrete stores (memory, disk) are adapters.
"""

from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from typing import Any

from .base import Port


def content_key(stage: str, input_hash: str, params: dict, model_version: str) -> str:
    """Deterministic cache key from a stage's full identity.

    Pure and adapter-independent so the key is identical across cache backends and
    across processes (a cache written by one run is found by another). ``params`` is
    canonicalized (sorted keys) before hashing.
    """
    blob = json.dumps(
        {"stage": stage, "input": input_hash, "params": params, "model": model_version},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"{stage}-{digest[:16]}"


class Cache(Port):
    """A content-addressable artifact store."""

    def key_for(self, stage: str, input_hash: str, params: dict, model_version: str) -> str:
        """Return the deterministic key for a stage result (see :func:`content_key`)."""
        return content_key(stage, input_hash, params, model_version)

    @abstractmethod
    def has(self, key: str) -> bool:
        """Whether an artifact exists for ``key``."""
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return the cached artifact for ``key`` or ``None`` on miss."""
        raise NotImplementedError

    @abstractmethod
    def put(self, key: str, artifact: Any) -> None:
        """Store ``artifact`` under ``key``."""
        raise NotImplementedError
