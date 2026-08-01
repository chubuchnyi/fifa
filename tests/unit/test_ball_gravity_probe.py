"""Ball gravity probe — airborne ball obeys gravity."""

from __future__ import annotations

import numpy as np

from pitch3d.core.correction.ball_gravity_probe import (
    BallGravityConfig,
    ball_gravity_probe,
)
from pitch3d.core.scene.motion import BallTrack
from pitch3d.core.scene.scene import Scene


def _scene_with_ball(positions: np.ndarray) -> Scene:
    frames = np.arange(positions.shape[0], dtype=int)
    ball = BallTrack(
        frames=frames, positions_3d=positions,
        height_confidence=np.ones(positions.shape[0]),
    )
    return Scene(id="s", episode_id="e", source_id="c",
                subjects=[], corrections=[], ball=ball)


def test_disabled_returns_empty():
    T = 5
    pos = np.zeros((T, 3))
    r = ball_gravity_probe(_scene_with_ball(pos),
                          BallGravityConfig(enabled=False), fps=30)
    assert r.n_frames == 0


def test_ballistic_z_no_violation():
    T = 20
    dt = 1.0 / 30.0
    z = 5.0 * dt * np.arange(T) + 0.5 * (-9.81) * (dt * np.arange(T)) ** 2 + 1.0
    pos = np.zeros((T, 3))
    pos[:, 2] = z
    r = ball_gravity_probe(_scene_with_ball(pos),
                          BallGravityConfig(enabled=True, tolerance_mps2=1.0),
                          fps=30)
    assert not r.is_violating
    assert -10.0 < r.mean_vertical_accel_mps2 < -9.0


def test_hovering_ball_flagged():
    """Constant Z (0 accel) → deviation ~ 9.81, flagged."""
    T = 20
    pos = np.ones((T, 3))
    pos[:, 2] = 2.0
    r = ball_gravity_probe(_scene_with_ball(pos),
                          BallGravityConfig(enabled=True), fps=30)
    assert r.is_violating
    assert r.max_deviation_mps2 > 5.0


def test_no_ball_returns_empty():
    scene = Scene(id="s", episode_id="e", source_id="c",
                 subjects=[], corrections=[], ball=None)
    r = ball_gravity_probe(scene, BallGravityConfig(enabled=True), fps=30)
    assert r.n_frames == 0


def test_short_track():
    pos = np.zeros((2, 3))
    r = ball_gravity_probe(_scene_with_ball(pos),
                          BallGravityConfig(enabled=True), fps=30)
    assert r.n_frames == 2
    assert r.mean_vertical_accel_mps2 == 0.0
