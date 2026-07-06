"""Ball gravity project — airborne ball Z rewritten to parabola."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.correction.ball_gravity_project import (
    BallGravityProjectConfig,
    ball_gravity_project_gate,
)
from pitch3d.core.correction.engine import resolve_ball
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


def test_disabled_passthrough():
    pos = np.zeros((10, 3))
    pos[:, 2] = 2.0  # ball hovering
    scene, report = ball_gravity_project_gate(
        _scene_with_ball(pos), BallGravityProjectConfig(enabled=False), fps=30,
    )
    assert not report.correction_added


def test_grounded_no_projection():
    """Ball at Z=0 throughout → no airborne runs → no correction."""
    pos = np.zeros((20, 3))
    scene, report = ball_gravity_project_gate(
        _scene_with_ball(pos), BallGravityProjectConfig(enabled=True), fps=30,
    )
    assert not report.correction_added


def test_airborne_ball_projected_to_parabola():
    """Airborne ball with linear Z gets a parabolic arc."""
    T = 20
    pos = np.zeros((T, 3))
    pos[:, 2] = np.linspace(2.0, 4.0, T)   # linearly rising = 0 accel
    scene, report = ball_gravity_project_gate(
        _scene_with_ball(pos),
        BallGravityProjectConfig(enabled=True, airborne_z_threshold_m=0.1), fps=30,
    )
    assert report.correction_added
    assert report.runs_projected == 1
    ball_after = resolve_ball(scene.ball, scene.corrections_for(None))
    z = np.asarray(ball_after.positions_3d)[:, 2]
    # Endpoints preserved
    assert z[0] == pytest.approx(2.0, abs=1e-6)
    assert z[-1] == pytest.approx(4.0, abs=1e-6)
    # Interior no longer linear — check for curvature
    linear = np.linspace(2.0, 4.0, T)
    dev = np.abs(z - linear).max()
    assert dev > 0.1


def test_no_ball_returns_empty():
    scene = Scene(id="s", episode_id="e", source_id="c",
                 subjects=[], corrections=[], ball=None)
    scene_after, report = ball_gravity_project_gate(
        scene, BallGravityProjectConfig(enabled=True), fps=30,
    )
    assert not report.correction_added


def test_short_track():
    pos = np.ones((2, 3)) * 2.0
    scene_after, report = ball_gravity_project_gate(
        _scene_with_ball(pos), BallGravityProjectConfig(enabled=True), fps=30,
    )
    assert not report.correction_added
