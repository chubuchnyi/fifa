"""Angular-momentum probe — spine/limb co-variation."""

from __future__ import annotations

import numpy as np

from pitch3d.core.correction.angular_momentum_probe import (
    JOINT_LIMBS,
    JOINT_SPINE,
    AngularMomentumConfig,
    angular_momentum_probe,
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


def test_disabled_returns_empty():
    body = np.zeros((10, 21, 3))
    r = angular_momentum_probe(_scene(_subject(1, body)),
                              AngularMomentumConfig(enabled=False), fps=30)
    assert r.subjects == []


def test_static_pose_not_flagged():
    body = np.zeros((10, 21, 3))
    r = angular_momentum_probe(_scene(_subject(1, body)),
                              AngularMomentumConfig(enabled=True), fps=30)
    assert r.subjects_uncoordinated == 0


def test_coordinated_motion_not_flagged():
    """Spine + limbs animate in lockstep → high correlation, not flagged."""
    T = 30
    body = np.zeros((T, 21, 3))
    ts = np.linspace(0, 4 * np.pi, T)
    signal = 0.3 * np.sin(ts)
    for j in JOINT_SPINE:
        body[:, j, 0] = signal
    for j in JOINT_LIMBS:
        body[:, j, 0] = signal
    r = angular_momentum_probe(_scene(_subject(1, body)),
                              AngularMomentumConfig(enabled=True), fps=30)
    assert r.subjects_uncoordinated == 0


def test_uncoordinated_motion_flagged():
    """Spine active but arms independent → low correlation, flagged."""
    T = 30
    body = np.zeros((T, 21, 3))
    ts = np.linspace(0, 4 * np.pi, T)
    for j in JOINT_SPINE:
        body[:, j, 0] = 0.5 * np.sin(ts)          # coherent spine
    rng = np.random.default_rng(0)
    for j in JOINT_LIMBS:
        body[:, j, 0] = 0.5 * rng.standard_normal(T)   # random limbs
    r = angular_momentum_probe(_scene(_subject(1, body)),
                              AngularMomentumConfig(enabled=True), fps=30)
    assert r.subjects_uncoordinated == 1


def test_bad_fps_returns_empty():
    body = np.zeros((5, 21, 3))
    r = angular_momentum_probe(_scene(_subject(1, body)),
                              AngularMomentumConfig(enabled=True), fps=0)
    assert r.subjects == []


def test_short_track():
    body = np.zeros((2, 21, 3))
    r = angular_momentum_probe(_scene(_subject(1, body)),
                              AngularMomentumConfig(enabled=True), fps=30)
    assert r.subjects_uncoordinated == 0
