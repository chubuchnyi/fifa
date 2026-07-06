"""Jerk clamp — iterative low-pass until peak jerk under threshold."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.correction.jerk_clamp import (
    JerkClampConfig,
    jerk_clamp_gate,
)
from pitch3d.core.correction.momentum_smooth import _peak_jerk
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, transl: np.ndarray) -> Subject:
    T = transl.shape[0]
    frames = np.arange(T, dtype=int)
    return Subject(track_id=track_id, proposal=SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=np.zeros((T, 3)),
            body_pose=np.zeros((T, 21, 3)), transl=transl,
        ),
    ))


def _scene(*subjects):
    return Scene(id="s", episode_id="e", source_id="c",
                subjects=list(subjects), corrections=[])


def test_disabled_passthrough():
    _, report = jerk_clamp_gate(
        _scene(_subject(1, np.zeros((5, 3)))),
        JerkClampConfig(enabled=False), fps=30,
    )
    assert report.corrections_added == 0


def test_smooth_track_no_correction():
    T = 30
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0, 3, T)
    _, report = jerk_clamp_gate(
        _scene(_subject(1, transl)),
        JerkClampConfig(enabled=True, max_jerk_mps3=100), fps=30,
    )
    assert report.corrections_added == 0


def test_jerky_track_clamped():
    """A very jerky track drops below the ceiling after enough passes."""
    T = 60
    rng = np.random.default_rng(0)
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0, 3, T) + 0.1 * rng.standard_normal(T)
    scene, report = jerk_clamp_gate(
        _scene(_subject(1, transl)),
        JerkClampConfig(enabled=True, max_jerk_mps3=100.0,
                       smooth_window=5, max_passes=10), fps=30,
    )
    assert report.corrections_added == 1
    assert report.max_jerk_after_mps3 < report.max_jerk_before_mps3


def test_max_passes_respected():
    """Even with insane input, gate stops after max_passes iterations."""
    T = 20
    rng = np.random.default_rng(0)
    transl = np.zeros((T, 3))
    transl[:, 0] = 10.0 * rng.standard_normal(T)
    _, report = jerk_clamp_gate(
        _scene(_subject(1, transl)),
        JerkClampConfig(enabled=True, max_jerk_mps3=0.001, max_passes=3), fps=30,
    )
    assert report.total_passes_used <= 3


def test_bad_fps_passthrough():
    _, report = jerk_clamp_gate(
        _scene(_subject(1, np.zeros((5, 3)))),
        JerkClampConfig(enabled=True), fps=0,
    )
    assert report.corrections_added == 0


def test_empty_scene():
    _, report = jerk_clamp_gate(
        _scene(), JerkClampConfig(enabled=True), fps=30,
    )
    assert report.corrections_added == 0
