"""Correction engine: honest rotation math + the four propagation modes (FR-21, FR-22).

``resolved = proposal ⊕ corrections``, computed on demand, never mutating the proposal.
"""

from __future__ import annotations

from .anchor import (
    DEFAULT_MAX_RESIDUAL_M,
    AnchorReport,
    anchor_residuals,
    blend_to_anchor,
    validate_against_anchor,
)
from .coherence import (
    CoherenceConfig,
    CoherenceReport,
    add_temporal_coherence,
    coherence_corrections,
    extend_pose_to_span,
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
from .collision import (
    CollisionReport,
    collision_gate,
)
from .contact_probe import (
    ContactProbeReport,
    ContactRun,
    FootPositionProvider,
    SubjectContactReport,
    contact_probe,
)
from .foot_floor import (
    FootFloorReport,
    SubjectFootReport,
    foot_floor_gate,
)
from .facing_align import FacingAlignReport, facing_align_gate
from .foot_plant import (
    FootPlantReport,
    SubjectPlantReport,
    foot_plant_gate,
)
from .gravity_probe import GravityReport, gravity_probe
from .gravity_project import GravityProjectReport, gravity_project_gate
from .inertia_probe import InertiaReport, inertia_probe
from .inertia_smooth import InertiaSmoothReport, inertia_smooth_gate
from .interpen_probe import InterpenReport, interpen_probe
from .momentum_probe import MomentumProbeReport, momentum_probe
from .momentum_smooth import MomentumSmoothReport, momentum_smooth_gate
from .pose_motion_probe import PoseMotionReport, pose_motion_probe
from .pose_motion_sync import PoseMotionSyncReport, pose_motion_sync_gate
from .stride_probe import StrideProbeReport, stride_probe
from .body_scale_probe import BodyScaleReport, body_scale_probe
from .contact_lock import ContactLockReport, contact_lock_gate
from .ball_contact_probe import BallContactReport, ball_contact_probe
from .joint_kinematics import (
    JointKinematicReport,
    JointViolation,
    joint_kinematic_gate,
)
from .orientation import (
    OrientationReport,
    OrientationViolation,
    orientation_gate,
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
    "DEFAULT_MAX_RESIDUAL_M",
    "AnchorReport",
    "CoherenceConfig",
    "CoherenceReport",
    "add_temporal_coherence",
    "anchor_residuals",
    "apply_offset_rotation",
    "apply_offset_vector",
    "average_quats",
    "blend_to_anchor",
    "validate_against_anchor",
    "coherence_corrections",
    "extend_pose_to_span",
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
    "CollisionReport",
    "collision_gate",
    "ContactProbeReport",
    "ContactRun",
    "FootPositionProvider",
    "SubjectContactReport",
    "contact_probe",
    "FootFloorReport",
    "SubjectFootReport",
    "foot_floor_gate",
    "FootPlantReport",
    "SubjectPlantReport",
    "foot_plant_gate",
    "FacingAlignReport", "facing_align_gate",
    "GravityReport", "gravity_probe",
    "GravityProjectReport", "gravity_project_gate",
    "InertiaReport", "inertia_probe",
    "InertiaSmoothReport", "inertia_smooth_gate",
    "InterpenReport", "interpen_probe",
    "MomentumProbeReport", "momentum_probe",
    "MomentumSmoothReport", "momentum_smooth_gate",
    "PoseMotionReport", "pose_motion_probe",
    "PoseMotionSyncReport", "pose_motion_sync_gate",
    "StrideProbeReport", "stride_probe",
    "BodyScaleReport", "body_scale_probe",
    "ContactLockReport", "contact_lock_gate",
    "BallContactReport", "ball_contact_probe",
    "JointKinematicReport",
    "JointViolation",
    "joint_kinematic_gate",
    "OrientationReport",
    "OrientationViolation",
    "orientation_gate",
]
