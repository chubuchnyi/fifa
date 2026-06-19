"""Port foundations: the marker base and the provenance-aware model base.

Every adapter (real model, ViewSynthesizer backend, Blender, renderer, exporter, queue,
cache) implements one of these ABCs. The core depends only on the ABCs — never on a
concrete adapter, never on ``bpy`` or any ML/render library (ADR-0001).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..scene.provenance import ModelInfo


class Port(ABC):
    """Marker base for all hexagonal ports (documentation + isinstance grouping)."""


class ModelProvider(Port):
    """Base for anything that behaves like a swappable model (NFR-6).

    Provides provenance so the scene's run log and the UX (UX-7) can show which
    model/version/cost produced each artifact.
    """

    @abstractmethod
    def info(self) -> ModelInfo:
        """Return identity + cost metadata for this model/adapter."""
        raise NotImplementedError
