"""Fake adapters — deterministic, dependency-free implementations of the ports.

They satisfy the same ABCs as the real adapters so the whole pipeline (reconstruction →
edit → resolve → render → export) and the LLM feedback loop run in tests and the CLI dry-run
with no GPU / Blender / models / network. Everything here uses only numpy + the stdlib and
writes inspectable placeholder artifacts under ``out/`` (ADR-0001, ADR-0008).
"""

from __future__ import annotations

from .cache import DiskCache, MemoryCache
from .export import FakeExporter
from .jobs import InProcessJobQueue, InProcessWorker
from .motion_prior import FakeMotionPrior
from .observer import FakeSceneObserver
from .perception import (
    FakeBallTracker,
    FakeDetector,
    FakeFieldCalibrator,
    FakeTracker,
)
from .pose import FakePoseEstimator
from .reconstruction import FakeAvatarBuilder, FakeEnvReconstructor
from .render import FakeRenderPass
from .viewsynth import FakeViewSynthesizer

__all__ = [
    "DiskCache",
    "FakeAvatarBuilder",
    "FakeBallTracker",
    "FakeDetector",
    "FakeEnvReconstructor",
    "FakeExporter",
    "FakeFieldCalibrator",
    "FakeMotionPrior",
    "FakePoseEstimator",
    "FakeRenderPass",
    "FakeSceneObserver",
    "FakeTracker",
    "FakeViewSynthesizer",
    "InProcessJobQueue",
    "InProcessWorker",
    "MemoryCache",
]
