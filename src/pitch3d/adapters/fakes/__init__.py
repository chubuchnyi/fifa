"""Fake adapters — deterministic, dependency-free implementations of the ports.

They satisfy the same ABCs as the real adapters so the whole pipeline (and the LLM feedback
loop) runs in tests and the CLI dry-run with no GPU / Blender / models. More fakes land with
Task 6; the observer is here because it backs the visual-feedback seam (ADR-0008).
"""

from __future__ import annotations

from .observer import FakeSceneObserver

__all__ = ["FakeSceneObserver"]
