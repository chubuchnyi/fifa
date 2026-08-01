"""step 4b — momentum_smooth low-pass corrects HF chatter."""

from __future__ import annotations

import numpy as np

from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.correction.momentum_smooth import (
    MomentumSmoothConfig,
    _moving_average,
    momentum_smooth_gate,
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


def _provider(feats_by_id: dict[int, np.ndarray]):
    def p(subject):
        return feats_by_id.get(int(subject.track_id))
    return p


def test_moving_average_smoothes_step():
    x = np.zeros((10, 1))
    x[5:, 0] = 1.0
    y = _moving_average(x, 5)
    assert y[5, 0] < 1.0 and y[4, 0] > 0.0  # smoothed at the step


def test_moving_average_preserves_constant():
    x = np.tile([[2.0, 3.0, 4.0]], (10, 1))
    assert np.allclose(_moving_average(x, 5), x)


def test_moving_average_window_1_is_identity():
    x = np.arange(30, dtype=float).reshape(10, 3)
    assert np.allclose(_moving_average(x, 1), x)


def test_disabled_gate_passthrough():
    s = _subject(1, np.zeros((5, 3)))
    scene, report = momentum_smooth_gate(
        _scene(s), MomentumSmoothConfig(enabled=False), fps=30,
    )
    assert report.corrections_added == 0
    assert scene.corrections == []


def test_smooth_kills_hf_chatter():
    """A noisy trajectory sees jerk drop after the low-pass pass."""
    T = 30
    rng = np.random.default_rng(0)
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0, 3, T) + 0.05 * rng.standard_normal(T)
    s = _subject(1, transl)
    scene, report = momentum_smooth_gate(
        _scene(s), MomentumSmoothConfig(enabled=True, smooth_window=5,
                                       preserve_contact=False), fps=30,
    )
    assert report.corrections_added == 1
    assert report.max_jerk_after < report.max_jerk_before


def test_preserve_contact_leaves_stance_frames_untouched():
    """When contact_provider is wired and a stance run is detected, those
    frames stay at their pre-smooth values (foot lock preserved)."""
    T = 20
    rng = np.random.default_rng(0)
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0, 2, T) + 0.05 * rng.standard_normal(T)
    s = _subject(1, transl)
    feats = np.zeros((T, 3))
    feats[:10, 2] = 0.0     # stance in the first 10 frames
    feats[:10, 0] = transl[:10, 0]
    feats[10:, 2] = 0.5     # airborne after
    feats[10:, 0] = transl[10:, 0]
    scene, _ = momentum_smooth_gate(
        _scene(s), MomentumSmoothConfig(enabled=True, smooth_window=5,
                                       preserve_contact=True), fps=30,
        foot_position_provider=_provider({1: feats}),
    )
    resolved = resolve_subject_motion(s.proposal, scene.corrections_for(1))
    got = np.asarray(resolved.pose.transl)
    # stance frames identical to originals; swing frames smoothed
    assert np.allclose(got[:10], transl[:10])
    # swing part should have shifted (smoothing applied)
    assert not np.allclose(got[10:], transl[10:])


def test_below_threshold_shift_no_emit():
    """A tiny smooth adjustment is skipped."""
    s = _subject(1, np.zeros((10, 3)))
    _, report = momentum_smooth_gate(
        _scene(s), MomentumSmoothConfig(enabled=True, smooth_window=3,
                                       preserve_contact=False,
                                       min_correction_m=0.5), fps=30,
    )
    assert report.corrections_added == 0


def test_empty_scene():
    scene, report = momentum_smooth_gate(
        _scene(), MomentumSmoothConfig(enabled=True), fps=30,
    )
    assert report.n_subjects == 0
    assert report.corrections_added == 0
