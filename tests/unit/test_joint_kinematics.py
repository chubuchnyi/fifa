"""Joint kinematic gate (T1b): per-joint slerp clamp + honest reporting."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.config.gates import JointKinematicConfig
from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.correction.joint_kinematics import (
    _interval_angle_deg,
    joint_kinematic_gate,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, body_pose: np.ndarray) -> Subject:
    """Build a subject whose body_pose is the given (T, K, 3) array."""
    T = body_pose.shape[0]
    frames = np.arange(T, dtype=int)
    transl = np.zeros((T, 3))
    global_orient = np.zeros((T, 3))
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=global_orient, body_pose=body_pose,
            transl=transl,
        ),
    )
    return Subject(track_id=track_id, proposal=motion)


def _scene(*subjects: Subject) -> Scene:
    return Scene(
        id="test-scene", episode_id="test-ep", source_id="test-clip",
        subjects=list(subjects), corrections=[],
    )


def test_interval_angle_matches_true_group_angle():
    """|angle_deg(R_x(90°) → R_y(90°))| == 120°, not the componentwise 127°."""
    a = np.array([np.pi / 2, 0, 0])
    b = np.array([0, np.pi / 2, 0])
    ang = _interval_angle_deg(a, b)
    assert 119.9 < ang < 120.1, ang


def test_interval_angle_zero_when_identical():
    a = np.array([0.3, -0.1, 0.2])
    assert _interval_angle_deg(a, a) < 1e-6


def test_disabled_gate_measures_but_emits_no_corrections():
    """Disabled → measure-only. Violations counted but no corrections added."""
    fps = 30.0
    T = 6
    # joint 0 rotates 60° per frame around Z = 1800°/s (over 600 default)
    body = np.zeros((T, 3, 3))
    body[:, 0, 2] = np.radians(60) * np.arange(T)
    s = _subject(1, body)
    new_scene, report = joint_kinematic_gate(
        _scene(s), JointKinematicConfig(enabled=False, max_omega_dps=600.0), fps=fps,
    )
    assert report.intervals_over_limit == 5
    assert report.intervals_clamped == 0
    assert report.corrections_added == 0
    assert new_scene.corrections == []
    assert 1790 < report.max_rate_before_dps < 1810


def test_enabled_gate_clamps_over_limit_intervals():
    """Enabled → per-joint slerp clamp; rate stays under the ceiling after resolve."""
    fps = 30.0
    T = 6
    body = np.zeros((T, 3, 3))
    body[:, 0, 2] = np.radians(60) * np.arange(T)   # 1800°/s
    s = _subject(1, body)
    new_scene, report = joint_kinematic_gate(
        _scene(s), JointKinematicConfig(enabled=True, max_omega_dps=600.0), fps=fps,
    )
    assert report.corrections_added == 1
    assert report.subjects_corrected == 1
    assert report.joints_corrected == 1
    resolved = resolve_subject_motion(s.proposal, new_scene.corrections_for(1))
    aa = np.asarray(resolved.pose.body_pose)[:, 0, :]
    # every clamped interval sits AT (or under) the 600 dps limit
    for i in range(1, T):
        angle = _interval_angle_deg(aa[i - 1], aa[i])
        assert angle * fps <= 600.0 + 1e-6, f"interval {i} still {angle * fps}°/s"


def test_untouched_joints_never_emit_corrections():
    """Joints that never violate emit nothing — the correction count stays tight."""
    fps = 30.0
    T = 6
    K = 5
    body = np.zeros((T, K, 3))
    body[:, 2, 2] = np.radians(60) * np.arange(T)  # only joint 2 exceeds
    s = _subject(1, body)
    _, report = joint_kinematic_gate(
        _scene(s), JointKinematicConfig(enabled=True, max_omega_dps=600.0), fps=fps,
    )
    assert report.joints_corrected == 1


def test_within_limits_zero_corrections():
    """A 500°/s rotation stays untouched at limit=600°/s."""
    fps = 30.0
    T = 6
    body = np.zeros((T, 3, 3))
    body[:, 0, 2] = np.radians(15) * np.arange(T)  # 450°/s < 600
    s = _subject(1, body)
    _, report = joint_kinematic_gate(
        _scene(s), JointKinematicConfig(enabled=True, max_omega_dps=600.0), fps=fps,
    )
    assert report.intervals_over_limit == 0
    assert report.corrections_added == 0


def test_gate_is_idempotent():
    fps = 30.0
    T = 6
    body = np.zeros((T, 3, 3))
    body[:, 0, 2] = np.radians(60) * np.arange(T)
    s = _subject(1, body)
    cfg = JointKinematicConfig(enabled=True, max_omega_dps=600.0)
    once, _ = joint_kinematic_gate(_scene(s), cfg, fps=fps)
    twice, twice_report = joint_kinematic_gate(once, cfg, fps=fps)
    # After the first pass the resolved rates already sit at the limit; the
    # second call should not emit anything new.
    assert twice_report.corrections_added == 0


def test_report_records_violations_with_track_and_joint():
    fps = 30.0
    T = 4
    body = np.zeros((T, 2, 3))
    body[:, 1, 2] = np.radians(60) * np.arange(T)
    s = _subject(track_id=7, body_pose=body)
    _, report = joint_kinematic_gate(
        _scene(s), JointKinematicConfig(enabled=True, max_omega_dps=600.0), fps=fps,
    )
    assert report.violations
    for v in report.violations:
        assert v.track_id == 7
        assert v.joint_index == 1
        assert v.rate_dps > 600.0
        assert v.clamped_dps <= 600.0 + 1e-6


def test_none_config_uses_defaults_and_is_measure_only():
    fps = 30.0
    T = 4
    body = np.zeros((T, 2, 3))
    body[:, 0, 2] = np.radians(60) * np.arange(T)
    s = _subject(1, body)
    new_scene, report = joint_kinematic_gate(_scene(s), None, fps=fps)
    # default has enabled=False
    assert report.corrections_added == 0
    assert new_scene.corrections == []


def test_empty_scene_returns_empty_report():
    _, report = joint_kinematic_gate(
        _scene(), JointKinematicConfig(enabled=True), fps=30.0,
    )
    assert report.n_subjects == 0
    assert report.corrections_added == 0


def test_bad_fps_raises():
    s = _subject(1, np.zeros((5, 2, 3)))
    with pytest.raises(ValueError, match="fps"):
        joint_kinematic_gate(_scene(s), JointKinematicConfig(), fps=0.0)
