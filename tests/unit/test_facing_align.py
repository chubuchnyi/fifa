"""Facing-align gate — rotate body yaw to match motion direction."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.correction.facing_align import (
    FACING_INFERRED_CONF,
    FacingAlignConfig,
    facing_align_gate,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, transl: np.ndarray, global_orient: np.ndarray) -> Subject:
    T = transl.shape[0]
    frames = np.arange(T, dtype=int)
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=global_orient,
            body_pose=np.zeros((T, 21, 3)), transl=transl,
        ),
    )
    return Subject(track_id=track_id, proposal=motion)


def _scene(*subjects: Subject) -> Scene:
    return Scene(
        id="s", episode_id="e", source_id="c",
        subjects=list(subjects), corrections=[],
    )


def test_disabled_passthrough():
    T = 10
    s = _subject(1, np.zeros((T, 3)), np.zeros((T, 3)))
    scene, report = facing_align_gate(
        _scene(s), FacingAlignConfig(enabled=False), fps=30,
    )
    assert report.corrections_added == 0


def test_moving_forward_facing_backward_gets_corrected():
    """Subject moves +X with yaw=π (facing -X) → gate rotates yaw to 0."""
    T = 30
    transl = np.zeros((T, 3)); transl[:, 0] = np.linspace(0, 3, T)
    orient = np.zeros((T, 3)); orient[:, 2] = np.pi   # facing -X
    s = _subject(1, transl, orient)
    scene, report = facing_align_gate(
        _scene(s), FacingAlignConfig(enabled=True), fps=30,
    )
    assert report.corrections_added == 1
    assert report.subjects_corrected == 1
    resolved = resolve_subject_motion(s.proposal, scene.corrections_for(1))
    got = np.asarray(resolved.pose.global_orient)
    # yaw should be close to 0 (facing +X) on the corrected frames
    later_yaws = got[10:, 2]
    assert np.all(np.abs(later_yaws) < 0.1) or np.all(
        np.abs(np.abs(later_yaws) - 2 * np.pi) < 0.1
    )


def test_moving_within_tolerance_not_corrected():
    """Yaw within tolerance → gate leaves it alone."""
    T = 30
    transl = np.zeros((T, 3)); transl[:, 0] = np.linspace(0, 3, T)
    orient = np.zeros((T, 3)); orient[:, 2] = 0.5   # small yaw, within tol
    s = _subject(1, transl, orient)
    _, report = facing_align_gate(
        _scene(s), FacingAlignConfig(enabled=True), fps=30,
    )
    assert report.corrections_added == 0


def test_still_root_not_corrected():
    T = 10
    s = _subject(1, np.zeros((T, 3)), np.zeros((T, 3)))
    _, report = facing_align_gate(
        _scene(s), FacingAlignConfig(enabled=True), fps=30,
    )
    assert report.corrections_added == 0


def test_confidence_marked_on_corrected_frames():
    T = 30
    transl = np.zeros((T, 3)); transl[:, 0] = np.linspace(0, 3, T)
    orient = np.zeros((T, 3)); orient[:, 2] = np.pi
    s = _subject(1, transl, orient)
    scene, _ = facing_align_gate(
        _scene(s), FacingAlignConfig(enabled=True), fps=30,
    )
    conf = np.asarray(scene.confidence.subject_frame_conf[1])
    assert (conf == FACING_INFERRED_CONF).any()


def test_bad_fps_ignored():
    """fps<=0 → no-op passthrough (contract: gate doesn't crash)."""
    s = _subject(1, np.zeros((5, 3)), np.zeros((5, 3)))
    _, report = facing_align_gate(
        _scene(s), FacingAlignConfig(enabled=True), fps=0,
    )
    assert report.corrections_added == 0


def test_empty_scene():
    _, report = facing_align_gate(
        _scene(), FacingAlignConfig(enabled=True), fps=30,
    )
    assert report.corrections_added == 0
