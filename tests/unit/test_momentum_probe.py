"""CoM / momentum probe — detects "chatty" pelvis motion."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.correction.momentum_probe import (
    MomentumProbeConfig,
    _windowed_std,
    momentum_probe,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, transl: np.ndarray) -> Subject:
    T = transl.shape[0]
    frames = np.arange(T, dtype=int)
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=np.zeros((T, 3)),
            body_pose=np.zeros((T, 21, 3)), transl=transl,
        ),
    )
    return Subject(track_id=track_id, proposal=motion)


def _scene(*subjects: Subject) -> Scene:
    return Scene(
        id="s", episode_id="e", source_id="c",
        subjects=list(subjects), corrections=[],
    )


def test_windowed_std_zero_on_constant():
    x = np.ones(10)
    assert np.allclose(_windowed_std(x, 5), 0.0)


def test_windowed_std_short_window_returns_zero_only_for_window_1():
    """Window ≥ 2 gives non-zero std on a noisy series."""
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert np.all(_windowed_std(x, 3) > 0.0)
    assert np.allclose(_windowed_std(x, 1), 0.0)


def test_disabled_returns_empty_report():
    s = _subject(1, np.zeros((5, 3)))
    r = momentum_probe(_scene(s), MomentumProbeConfig(enabled=False), fps=30)
    assert r.subjects_chatty == 0
    assert r.subjects == []


def test_smooth_motion_not_flagged():
    """A subject on a straight-line path is NOT chatty."""
    T = 30
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0.0, 3.0, T)   # 0.1 m/step, smooth
    s = _subject(1, transl)
    r = momentum_probe(_scene(s), MomentumProbeConfig(enabled=True), fps=30)
    assert r.subjects_chatty == 0
    assert r.subjects[0].is_chatty is False


def test_chatty_motion_flagged():
    """High-frequency jitter on a smooth trend → large jerk → chatty."""
    T = 30
    rng = np.random.default_rng(0)
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0.0, 3.0, T) + 0.05 * rng.standard_normal(T)
    s = _subject(1, transl)
    r = momentum_probe(_scene(s), MomentumProbeConfig(enabled=True,
                                                    jerk_threshold_mps3=100.0),
                       fps=30)
    assert r.subjects_chatty == 1
    assert r.max_jerk_mps3 > 100.0


def test_accel_max_reported():
    T = 10
    transl = np.zeros((T, 3))
    transl[:, 0] = [0, 0, 0, 0, 10, 10, 10, 10, 10, 10]  # accel spike
    s = _subject(1, transl)
    r = momentum_probe(_scene(s), MomentumProbeConfig(enabled=True), fps=30)
    assert r.max_accel_mps2 > 100.0    # 10m / (1/30)^2 giant jerk


def test_bad_fps_raises():
    s = _subject(1, np.zeros((5, 3)))
    with pytest.raises(ValueError, match="fps"):
        momentum_probe(_scene(s), MomentumProbeConfig(enabled=True), fps=0)


def test_short_track_produces_only_bookkeeping():
    s = _subject(1, np.zeros((2, 3)))
    r = momentum_probe(_scene(s), MomentumProbeConfig(enabled=True), fps=30)
    assert r.subjects[0].n_frames == 2
    assert r.subjects[0].accel_max_mps2 == 0.0
