"""inertia_smooth — yaw low-pass to bound α."""

from __future__ import annotations

import numpy as np

from pitch3d.core.correction.inertia_smooth import (
    InertiaSmoothConfig,
    _moving_average,
    inertia_smooth_gate,
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


def test_moving_average_preserves_flat():
    x = np.ones(10)
    assert np.allclose(_moving_average(x, 3), 1.0)


def test_moving_average_smooths_step():
    x = np.array([0.0] * 5 + [1.0] * 5)
    y = _moving_average(x, 3)
    assert y[4] < 1.0 and y[5] > 0.0


def test_disabled_passthrough():
    T = 10
    orient = np.zeros((T, 3))
    scene, report = inertia_smooth_gate(
        _scene(_subject(1, orient)), InertiaSmoothConfig(enabled=False), fps=30,
    )
    assert report.corrections_added == 0


def test_smooth_within_ceiling_no_correction():
    """Constant α = 0 → subject already feasible, no correction emitted."""
    T = 20
    orient = np.zeros((T, 3))
    orient[:, 2] = np.linspace(0, np.pi / 4, T)
    _, report = inertia_smooth_gate(
        _scene(_subject(1, orient)),
        InertiaSmoothConfig(enabled=True, max_alpha_rad_s2=15), fps=30,
    )
    assert report.corrections_added == 0


def test_snap_yaw_reduced_after_smooth():
    """A step in yaw at frame 10 has huge α before → falls after low-pass."""
    T = 20
    orient = np.zeros((T, 3))
    orient[10:, 2] = np.pi / 2
    scene, report = inertia_smooth_gate(
        _scene(_subject(1, orient)),
        InertiaSmoothConfig(enabled=True, smooth_window=5, max_alpha_rad_s2=15), fps=30,
    )
    assert report.corrections_added == 1
    assert report.max_alpha_after_rad_s2 < report.max_alpha_before_rad_s2


def test_bad_fps_passthrough():
    _, report = inertia_smooth_gate(
        _scene(_subject(1, np.zeros((5, 3)))),
        InertiaSmoothConfig(enabled=True), fps=0,
    )
    assert report.corrections_added == 0


def test_empty_scene():
    _, report = inertia_smooth_gate(
        _scene(), InertiaSmoothConfig(enabled=True), fps=30,
    )
    assert report.corrections_added == 0
