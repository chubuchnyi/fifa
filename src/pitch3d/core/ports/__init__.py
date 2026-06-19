"""Port contracts (ABCs) — the only surface the core exposes to infrastructure.

Adapters implement these; the core depends on nothing concrete (ADR-0001).
"""

from __future__ import annotations

from .base import ModelProvider, Port
from .cache import Cache, content_key
from .export import Exporter, ExportFormat, ExportResult
from .io import ClipRef, CropRef
from .jobs import JobHandle, JobQueue, JobState, Worker
from .perception import (
    BallTracker,
    Detection,
    Detections,
    Detector,
    FieldCalibrator,
    FrameDetections,
    Tracker,
    Tracklet,
    Tracks,
)
from .pose import PoseEstimator
from .reconstruction import AvatarBuilder, EnvReconstructor
from .render import RenderPass, RenderQuality, RenderResult
from .view_synthesizer import ViewSynthesizer

__all__ = [
    "AvatarBuilder",
    "BallTracker",
    "Cache",
    "ClipRef",
    "CropRef",
    "Detection",
    "Detections",
    "Detector",
    "EnvReconstructor",
    "ExportFormat",
    "ExportResult",
    "Exporter",
    "FieldCalibrator",
    "FrameDetections",
    "JobHandle",
    "JobQueue",
    "JobState",
    "ModelProvider",
    "Port",
    "PoseEstimator",
    "RenderPass",
    "RenderQuality",
    "RenderResult",
    "Tracker",
    "Tracklet",
    "Tracks",
    "ViewSynthesizer",
    "Worker",
    "content_key",
]
