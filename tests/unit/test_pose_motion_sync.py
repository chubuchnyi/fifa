"""Pose-motion sync — synth walk cycle on desynced frames."""

from __future__ import annotations

import numpy as np

from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.correction.pose_motion_sync import (
    JOINT_KNEE_L,
    JOINT_KNEE_R,
    PATCHED_CONF,
    PoseMotionSyncConfig,
    pose_motion_sync_gate,
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


def test_disabled_passthrough():
    T = 10
    s = _subject(1, np.zeros((T, 3)), np.zeros((T, 21, 3)))
    scene, report = pose_motion_sync_gate(
        _scene(s), PoseMotionSyncConfig(enabled=False), fps=30,
    )
    assert report.corrections_added == 0
    assert scene.corrections == []


def test_syncs_desynced_subject():
    """Root moves without pose animation → gate synthesizes knee/hip swing."""
    T = 30
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0.0, 3.0, T)
    body = np.zeros((T, 21, 3))
    s = _subject(1, transl, body)
    scene, report = pose_motion_sync_gate(
        _scene(s), PoseMotionSyncConfig(enabled=True), fps=30,
    )
    assert report.corrections_added >= 2   # at least one knee + one hip pair
    assert report.subjects_patched == 1
    resolved = resolve_subject_motion(s.proposal, scene.corrections_for(1))
    got_pose = np.asarray(resolved.pose.body_pose)
    # knees actually rotate now
    assert np.abs(got_pose[:, JOINT_KNEE_L, 0]).max() > 0.05
    assert np.abs(got_pose[:, JOINT_KNEE_R, 0]).max() > 0.05


def test_confidence_map_marked_on_patched_frames():
    T = 30
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0.0, 3.0, T)
    body = np.zeros((T, 21, 3))
    s = _subject(1, transl, body)
    scene, _ = pose_motion_sync_gate(
        _scene(s), PoseMotionSyncConfig(enabled=True), fps=30,
    )
    conf = scene.confidence.subject_frame_conf[1]
    # at least some frames should be at PATCHED_CONF
    assert (np.asarray(conf) == PATCHED_CONF).any()


def test_still_root_no_corrections():
    """A static subject has nothing to sync — always safe passthrough."""
    T = 10
    s = _subject(1, np.zeros((T, 3)), np.zeros((T, 21, 3)))
    _, report = pose_motion_sync_gate(
        _scene(s), PoseMotionSyncConfig(enabled=True), fps=30,
    )
    assert report.corrections_added == 0


def test_still_root_no_sync():
    """Static subject → nothing to sync."""
    T = 10
    s = _subject(1, np.zeros((T, 3)), np.zeros((T, 21, 3)))
    scene, report = pose_motion_sync_gate(
        _scene(s), PoseMotionSyncConfig(enabled=True), fps=30,
    )
    assert report.corrections_added == 0


def test_amplitude_scales_with_speed():
    """A faster subject gets a bigger knee amplitude."""
    T = 30
    transl_slow = np.zeros((T, 3))
    transl_slow[:, 0] = np.linspace(0.0, 3.0, T)     # ~3 m/s at fps=30
    transl_fast = np.zeros((T, 3))
    transl_fast[:, 0] = np.linspace(0.0, 6.0, T)     # ~6 m/s
    s_slow = _subject(1, transl_slow, np.zeros((T, 21, 3)))
    s_fast = _subject(2, transl_fast, np.zeros((T, 21, 3)))
    scene, report = pose_motion_sync_gate(
        _scene(s_slow, s_fast), PoseMotionSyncConfig(enabled=True), fps=30,
    )
    fast_report = next(r for r in report.subjects if r.track_id == 2)
    slow_report = next(r for r in report.subjects if r.track_id == 1)
    assert fast_report.max_amplitude_rad >= slow_report.max_amplitude_rad
