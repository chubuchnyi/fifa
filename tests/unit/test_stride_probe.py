"""Stride cadence probe — knee-swing frequency vs root speed."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.correction.stride_probe import (
    StrideProbeConfig,
    _zero_crossings,
    stride_probe,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, transl: np.ndarray, body_pose: np.ndarray) -> Subject:
    T = transl.shape[0]
    frames = np.arange(T, dtype=int)
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=np.zeros((T, 3)),
            body_pose=body_pose, transl=transl,
        ),
    )
    return Subject(track_id=track_id, proposal=motion)


def _scene(*subjects):
    return Scene(id="s", episode_id="e", source_id="c",
                subjects=list(subjects), corrections=[])


def test_zero_crossings_basic():
    x = np.array([1, -1, 1, -1, 1])
    assert _zero_crossings(x) == 4
    x = np.ones(5)
    assert _zero_crossings(x) == 0


def test_disabled_returns_empty():
    r = stride_probe(_scene(_subject(1, np.zeros((5, 3)), np.zeros((5, 21, 3)))),
                    StrideProbeConfig(enabled=False), fps=30)
    assert r.subjects == []


def test_still_subject_not_flagged():
    T = 30
    r = stride_probe(
        _scene(_subject(1, np.zeros((T, 3)), np.zeros((T, 21, 3)))),
        StrideProbeConfig(enabled=True), fps=30,
    )
    assert r.subjects_off == 0


def test_walking_pose_matches_speed_no_flag():
    """Walking at 2 m/s, knee swings at ~1.4 Hz → ratio ≈ 1."""
    T = 60
    fps = 30
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0, 4, T)   # 2 m/s
    body = np.zeros((T, 21, 3))
    body[:, 4, 0] = 0.5 * np.sin(2 * np.pi * 1.4 * np.arange(T) / fps)
    r = stride_probe(
        _scene(_subject(1, transl, body)),
        StrideProbeConfig(enabled=True), fps=fps,
    )
    assert not r.subjects[0].is_off


def test_moving_without_knee_swing_flagged():
    """Root moves at 2 m/s but knees are still → cadence 0, off."""
    T = 60
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0, 4, T)
    body = np.zeros((T, 21, 3))
    r = stride_probe(
        _scene(_subject(1, transl, body)),
        StrideProbeConfig(enabled=True), fps=30,
    )
    assert r.subjects_off == 1
    assert r.subjects[0].cadence_hz < 0.1


def test_wildly_fast_knee_flagged():
    """Slow-moving root but knees swinging at 10 Hz → ratio 10× → off."""
    T = 60
    fps = 30
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0, 3, T)   # 1.5 m/s
    body = np.zeros((T, 21, 3))
    body[:, 4, 0] = 0.5 * np.sin(2 * np.pi * 10 * np.arange(T) / fps)
    r = stride_probe(
        _scene(_subject(1, transl, body)),
        StrideProbeConfig(enabled=True), fps=fps,
    )
    assert r.subjects_off == 1


def test_short_track_ignored():
    T = 3
    r = stride_probe(
        _scene(_subject(1, np.zeros((T, 3)), np.zeros((T, 21, 3)))),
        StrideProbeConfig(enabled=True), fps=30,
    )
    assert r.subjects_off == 0
