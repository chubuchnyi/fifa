"""Correction engine: honest rotation math + the four propagation modes (FR-21, FR-22).

``resolved = proposal ⊕ corrections``, computed on demand, never mutating the proposal.
"""

from __future__ import annotations

from .coherence import (
    CoherenceConfig,
    CoherenceReport,
    add_temporal_coherence,
    coherence_corrections,
    fill_motion_gaps,
    fill_pose_gaps,
)
from .engine import (
    apply_offset_rotation,
    apply_offset_vector,
    interp_rotation,
    interp_vector,
    make_keyframes,
    make_offset,
    make_refit,
    make_smoothing,
    preview_subject_motion,
    resolve_ball,
    resolve_subject_motion,
    smooth_rotation,
    smooth_vector,
)
from .rotations import (
    average_quats,
    axis_angle_to_matrix,
    axis_angle_to_quat,
    compose_axis_angle,
    matrix_to_axis_angle,
    matrix_to_quat,
    quat_mul,
    quat_to_axis_angle,
    slerp_axis_angle,
    slerp_quat,
)

__all__ = [
    "CoherenceConfig",
    "CoherenceReport",
    "add_temporal_coherence",
    "apply_offset_rotation",
    "apply_offset_vector",
    "average_quats",
    "coherence_corrections",
    "fill_motion_gaps",
    "fill_pose_gaps",
    "axis_angle_to_matrix",
    "axis_angle_to_quat",
    "compose_axis_angle",
    "interp_rotation",
    "interp_vector",
    "make_keyframes",
    "make_offset",
    "make_refit",
    "make_smoothing",
    "matrix_to_axis_angle",
    "matrix_to_quat",
    "preview_subject_motion",
    "quat_mul",
    "quat_to_axis_angle",
    "resolve_ball",
    "resolve_subject_motion",
    "slerp_axis_angle",
    "slerp_quat",
    "smooth_rotation",
    "smooth_vector",
]
