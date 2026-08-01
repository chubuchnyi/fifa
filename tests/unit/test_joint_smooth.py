"""Joint smooth — per-joint MA low-pass on body_pose."""

from __future__ import annotations

import numpy as np

from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.correction.joint_smooth import (
    JointSmoothConfig,
    _moving_average_axisangle,
    joint_smooth_gate,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, body: np.ndarray) -> Subject:
    T = body.shape[0]
    frames = np.arange(T, dtype=int)
    return Subject(track_id=track_id, proposal=SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=np.zeros((T, 3)),
            body_pose=body, transl=np.zeros((T, 3)),
        ),
    ))


def _scene(*subjects):
    return Scene(id="s", episode_id="e", source_id="c",
                subjects=list(subjects), corrections=[])


def test_ma_axisangle_preserves_flat():
    x = np.ones((10, 3, 3))
    out = _moving_average_axisangle(x, 3)
    assert np.allclose(out, 1.0)


def test_ma_axisangle_smooths_step():
    x = np.zeros((10, 1, 3))
    x[5:, 0, 0] = 1.0
    out = _moving_average_axisangle(x, 3)
    assert out[5, 0, 0] < 1.0
    assert out[4, 0, 0] > 0.0


def test_disabled_passthrough():
    body = np.zeros((5, 21, 3))
    scene, report = joint_smooth_gate(
        _scene(_subject(1, body)), JointSmoothConfig(enabled=False), fps=30,
    )
    assert report.corrections_added == 0


def test_flat_pose_no_correction():
    body = np.zeros((10, 21, 3))
    _, report = joint_smooth_gate(
        _scene(_subject(1, body)), JointSmoothConfig(enabled=True), fps=30,
    )
    assert report.corrections_added == 0


def test_jittery_joint_smoothed():
    T = 20
    rng = np.random.default_rng(0)
    body = np.zeros((T, 21, 3))
    body[:, 5, 0] = 0.5 * rng.standard_normal(T)
    scene, report = joint_smooth_gate(
        _scene(_subject(1, body)),
        JointSmoothConfig(enabled=True, smooth_window=5,
                         min_correction_rad=1e-3), fps=30,
    )
    # jittery joint 5 emits at least one correction
    assert report.corrections_added >= 1
    resolved = resolve_subject_motion(
        _subject(1, body).proposal, scene.corrections_for(1),
    )
    got = np.asarray(resolved.pose.body_pose)
    # smoothed joint has lower std than raw
    assert got[:, 5, 0].std() < body[:, 5, 0].std()


def test_flat_joints_not_emitted():
    """Only jittery joints exceed min_correction_rad; flat ones are silent."""
    T = 20
    rng = np.random.default_rng(0)
    body = np.zeros((T, 21, 3))
    body[:, 5, 0] = 0.5 * rng.standard_normal(T)
    _, report = joint_smooth_gate(
        _scene(_subject(1, body)),
        JointSmoothConfig(enabled=True, min_correction_rad=1e-3), fps=30,
    )
    # jittery joint 5 emits — the other 20 flat joints do not
    assert 1 <= report.corrections_added <= 5


def test_bad_fps_passthrough():
    body = np.zeros((5, 21, 3))
    _, report = joint_smooth_gate(
        _scene(_subject(1, body)), JointSmoothConfig(enabled=True), fps=0,
    )
    assert report.corrections_added == 0
