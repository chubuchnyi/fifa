"""Pose-motion consistency probe — 'standing pose that walks' detection."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.correction.pose_motion_probe import (
    PoseMotionConfig,
    pose_motion_probe,
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


def _scene(*subjects: Subject) -> Scene:
    return Scene(
        id="s", episode_id="e", source_id="c",
        subjects=list(subjects), corrections=[],
    )


def test_disabled_returns_empty():
    T = 10
    s = _subject(1, np.zeros((T, 3)), np.zeros((T, 21, 3)))
    r = pose_motion_probe(_scene(s), PoseMotionConfig(enabled=False), fps=30)
    assert r.n_subjects == 1
    assert r.subjects_desynced == 0
    assert r.subjects == []


def test_moving_root_with_still_pose_flags_desync():
    """Root translates at ~3 m/s but joints don't move → desync."""
    T = 30
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0.0, 3.0, T)  # ~3 m/s at fps=30
    body = np.zeros((T, 21, 3))              # never moves
    s = _subject(1, transl, body)
    r = pose_motion_probe(_scene(s), PoseMotionConfig(enabled=True,
                                                     velocity_threshold_mps=2.0,
                                                     desync_fraction_threshold=0.3),
                          fps=30)
    assert r.subjects_desynced == 1
    assert r.subjects[0].is_desynced
    assert r.subjects[0].desync_fraction > 0.5


def test_moving_root_with_moving_pose_no_desync():
    """Both root AND joints move → consistent, not desync."""
    T = 30
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0.0, 3.0, T)
    body = np.zeros((T, 21, 3))
    # busy walk cycle — several joints animating consistently, large amplitude
    for j in (4, 5, 6, 7):
        body[:, j, 0] = 0.5 * np.sin(np.linspace(0, 6 * np.pi, T) + 0.5 * j)
    s = _subject(1, transl, body)
    r = pose_motion_probe(_scene(s), PoseMotionConfig(enabled=True,
                                                     velocity_threshold_mps=2.0),
                          fps=30)
    assert r.subjects_desynced == 0


def test_still_root_with_still_pose_no_desync():
    """Both static → no motion at all, no desync flag (needs moving frames)."""
    T = 10
    s = _subject(1, np.zeros((T, 3)), np.zeros((T, 21, 3)))
    r = pose_motion_probe(_scene(s), PoseMotionConfig(enabled=True), fps=30)
    assert r.subjects_desynced == 0
    assert r.subjects[0].moving_frames == 0


def test_bad_fps_raises():
    s = _subject(1, np.zeros((5, 3)), np.zeros((5, 21, 3)))
    with pytest.raises(ValueError, match="fps"):
        pose_motion_probe(_scene(s), PoseMotionConfig(enabled=True), fps=0)


def test_short_track_produces_only_bookkeeping():
    s = _subject(1, np.zeros((1, 3)), np.zeros((1, 21, 3)))
    r = pose_motion_probe(_scene(s), PoseMotionConfig(enabled=True), fps=30)
    assert r.subjects[0].n_frames == 1
    assert r.subjects_desynced == 0
