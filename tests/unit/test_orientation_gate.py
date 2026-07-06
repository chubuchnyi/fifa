"""Root-orientation gate (T1c) — mirrors joint gate semantics on global_orient."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.config.gates import OrientationConfig
from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.correction.joint_kinematics import _interval_angle_deg
from pitch3d.core.correction.orientation import orientation_gate
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, orient: np.ndarray) -> Subject:
    T = orient.shape[0]
    frames = np.arange(T, dtype=int)
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=orient,
            body_pose=np.zeros((T, 21, 3)), transl=np.zeros((T, 3)),
        ),
    )
    return Subject(track_id=track_id, proposal=motion)


def _scene(*subjects: Subject) -> Scene:
    return Scene(
        id="s", episode_id="e", source_id="c",
        subjects=list(subjects), corrections=[],
    )


def test_disabled_measures_only():
    """A 1800°/s rotation flags 5 violations but emits nothing when disabled."""
    T = 6
    orient = np.zeros((T, 3))
    orient[:, 2] = np.radians(60) * np.arange(T)   # 1800°/s
    s = _subject(1, orient)
    _, report = orientation_gate(
        _scene(s), OrientationConfig(enabled=False, max_turn_rate_dps=720.0), fps=30.0,
    )
    assert report.intervals_over_limit == 5
    assert report.corrections_added == 0
    assert 1790 < report.max_rate_before_dps < 1810


def test_enabled_clamps_below_max_dps():
    """Enabled → every interval rate <= max_turn_rate_dps after resolve."""
    T = 6
    orient = np.zeros((T, 3))
    orient[:, 2] = np.radians(60) * np.arange(T)
    s = _subject(1, orient)
    new_scene, report = orientation_gate(
        _scene(s), OrientationConfig(enabled=True, max_turn_rate_dps=720.0), fps=30.0,
    )
    assert report.corrections_added == 1
    assert report.subjects_corrected == 1
    resolved = resolve_subject_motion(s.proposal, new_scene.corrections_for(1))
    got = np.asarray(resolved.pose.global_orient)
    for i in range(1, T):
        rate = _interval_angle_deg(got[i - 1], got[i]) * 30.0
        assert rate <= 720.0 + 1e-6, f"i={i}: {rate}°/s over 720°/s"


def test_within_limits_no_correction():
    """A 500°/s spin stays untouched at limit=720°/s."""
    T = 6
    orient = np.zeros((T, 3))
    orient[:, 2] = np.radians(15) * np.arange(T)  # 450°/s
    s = _subject(1, orient)
    _, report = orientation_gate(
        _scene(s), OrientationConfig(enabled=True, max_turn_rate_dps=720.0), fps=30.0,
    )
    assert report.corrections_added == 0
    assert report.intervals_over_limit == 0


def test_idempotent():
    T = 6
    orient = np.zeros((T, 3))
    orient[:, 2] = np.radians(60) * np.arange(T)
    s = _subject(1, orient)
    cfg = OrientationConfig(enabled=True, max_turn_rate_dps=720.0)
    once, _ = orientation_gate(_scene(s), cfg, fps=30.0)
    twice, twice_report = orientation_gate(once, cfg, fps=30.0)
    assert twice_report.corrections_added == 0


def test_bad_fps_raises():
    s = _subject(1, np.zeros((5, 3)))
    with pytest.raises(ValueError, match="fps"):
        orientation_gate(_scene(s), OrientationConfig(enabled=True), fps=0.0)


def test_empty_scene():
    _, report = orientation_gate(
        _scene(), OrientationConfig(enabled=True), fps=30.0,
    )
    assert report.corrections_added == 0


def test_none_config_defaults_to_disabled():
    T = 4
    orient = np.zeros((T, 3))
    orient[:, 2] = np.radians(60) * np.arange(T)
    s = _subject(1, orient)
    new_scene, report = orientation_gate(_scene(s), None, fps=30.0)
    assert report.corrections_added == 0
    assert new_scene.corrections == []
