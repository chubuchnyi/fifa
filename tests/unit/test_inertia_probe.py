"""Inertia probe — torso angular acceleration measurement."""

from __future__ import annotations

import numpy as np

from pitch3d.core.correction.inertia_probe import (
    InertiaConfig,
    inertia_probe,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, global_orient: np.ndarray) -> Subject:
    T = global_orient.shape[0]
    frames = np.arange(T, dtype=int)
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=global_orient,
            body_pose=np.zeros((T, 21, 3)), transl=np.zeros((T, 3)),
        ),
    )
    return Subject(track_id=track_id, proposal=motion)


def _scene(*subjects: Subject) -> Scene:
    return Scene(
        id="s", episode_id="e", source_id="c",
        subjects=list(subjects), corrections=[],
    )


def test_disabled_returns_empty():
    T = 10
    orient = np.zeros((T, 3))
    r = inertia_probe(_scene(_subject(1, orient)),
                      InertiaConfig(enabled=False), fps=30)
    assert r.subjects_flagged == 0
    assert r.subjects == []


def test_smooth_rotation_no_flag():
    """Constant angular velocity → α = 0."""
    T = 20
    orient = np.zeros((T, 3))
    orient[:, 2] = np.linspace(0, np.pi / 2, T)
    r = inertia_probe(_scene(_subject(1, orient)),
                      InertiaConfig(enabled=True), fps=30)
    assert r.subjects[0].alpha_max_rad_s2 < 1.0
    assert r.subjects_flagged == 0


def test_snap_rotation_flagged():
    """Yaw jumps by π/2 in a single frame → huge α."""
    T = 20
    orient = np.zeros((T, 3))
    orient[10:, 2] = np.pi / 2   # step at frame 10
    r = inertia_probe(_scene(_subject(1, orient)),
                      InertiaConfig(enabled=True, max_alpha_rad_s2=15), fps=30)
    assert r.subjects_flagged == 1
    assert r.subjects[0].alpha_viol >= 1


def test_wrap_pi_handled_correctly():
    """A yaw crossing +π wrapped to -π must not read as a giant α.

    Simulates a smoothly rotating body whose principal-value yaw jumps
    from ~+π to ~-π at the boundary — physically the same angular
    velocity, just a different representation.
    """
    T = 20
    # smooth continuous yaw 3.0 → 4.5 (~90°/s at fps=30), stored as
    # principal-value (wraps at π)
    smooth_yaw = np.linspace(3.0, 4.5, T)
    principal_yaw = np.mod(smooth_yaw + np.pi, 2 * np.pi) - np.pi
    orient = np.zeros((T, 3))
    orient[:, 2] = principal_yaw
    r = inertia_probe(_scene(_subject(1, orient)),
                      InertiaConfig(enabled=True, max_alpha_rad_s2=15), fps=30)
    # Continuous angular velocity → α should be tiny (< 1 rad/s²)
    assert r.subjects[0].alpha_max_rad_s2 < 5.0


def test_bad_fps_returns_empty():
    s = _subject(1, np.zeros((5, 3)))
    r = inertia_probe(_scene(s), InertiaConfig(enabled=True), fps=0)
    assert r.subjects == []
