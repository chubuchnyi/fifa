"""Pose bake-off evaluation: synthetic GT, MPJPE metrics, and the scoring harness.

Pure / numpy-only by design (no Blender, no GPU, no external asset), so the bake-off
harness can be built and tested before WorldPose frames are obtained. See
``docs/pose-bakeoff-runbook.md`` for the procedure this implements.
"""

from __future__ import annotations

from .bodymodel import (
    CANONICAL_SKELETON,
    JOINT_NAMES,
    JointModel,
    PlaceholderJointModel,
)
from .harness import evaluate, place_under_gt_camera, run_backend
from .metrics import mpjpe_global, mpjpe_local
from .synthetic import SyntheticScene, generate_scene

__all__ = [
    "CANONICAL_SKELETON",
    "JOINT_NAMES",
    "JointModel",
    "PlaceholderJointModel",
    "SyntheticScene",
    "generate_scene",
    "mpjpe_global",
    "mpjpe_local",
    "evaluate",
    "place_under_gt_camera",
    "run_backend",
]
