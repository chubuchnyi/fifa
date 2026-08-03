"""Orchestration: stage DAG, cache-keyed job execution, and the ball 2D→3D lift."""

from __future__ import annotations

from .assemble import assemble_scene, resolve_scene
from .ball_lift import ballistic_z, lift_ball_to_3d
from .continuity import (
    StitchConfig,
    StitchReport,
    stitch_tracks,
    stitch_tracks_with_report,
)
from .pipeline import ReconstructionPipeline, ReconstructionResult, describe_calibration_solve
from .stages import RECON_ORDER, Stage, StageRun, clip_hash, run_cached

__all__ = [
    "RECON_ORDER",
    "ReconstructionPipeline",
    "ReconstructionResult",
    "Stage",
    "StageRun",
    "StitchConfig",
    "StitchReport",
    "assemble_scene",
    "ballistic_z",
    "clip_hash",
    "describe_calibration_solve",
    "lift_ball_to_3d",
    "resolve_scene",
    "run_cached",
    "stitch_tracks",
    "stitch_tracks_with_report",
]
