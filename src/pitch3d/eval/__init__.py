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
    SmplxJointModel,
)
from .calib_metrics import (
    evaluate_calibration,
    frame_metrics,
    frame_pixel_errors,
    frame_world_errors,
)
from .dataset import PoseEvalScene, evaluate_dataset
from .datasets_soccernet import (
    CalibFrameGT,
    CalibLineGT,
    as_clip,
    load_calib_annotation,
    load_calib_dir,
    pitch_plane_lines,
    synthetic_calib_frames,
)
from .harness import (
    evaluate,
    place_under_gt_camera,
    run_backend,
    run_backend_grounded,
    run_conditions,
)
from .metrics import mpjpe_global, mpjpe_local
from .synthetic import CAMERA_VIEWS, CameraView, SyntheticScene, generate_scene

__all__ = [
    "CANONICAL_SKELETON",
    "JOINT_NAMES",
    "JointModel",
    "PlaceholderJointModel",
    "SmplxJointModel",
    "CAMERA_VIEWS",
    "CameraView",
    "SyntheticScene",
    "PoseEvalScene",
    "generate_scene",
    "mpjpe_global",
    "mpjpe_local",
    "evaluate",
    "evaluate_dataset",
    "place_under_gt_camera",
    "run_backend",
    "run_backend_grounded",
    "run_conditions",
    # calibration bake-off (B1: SoccerNet pitch-line GT)
    "pitch_plane_lines",
    "CalibLineGT",
    "CalibFrameGT",
    "load_calib_annotation",
    "load_calib_dir",
    "as_clip",
    "synthetic_calib_frames",
    "frame_world_errors",
    "frame_pixel_errors",
    "frame_metrics",
    "evaluate_calibration",
]
