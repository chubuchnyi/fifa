"""Viewpoint camera math — orthonormality, the degenerate TOP up-vector, view selection."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.agent.viewpoints import look_at, standard_viewpoints
from pitch3d.core.correction.rotations import axis_angle_to_matrix, quat_to_axis_angle
from pitch3d.core.ports.observation import Viewpoint
from pitch3d.core.scene.scene import Scene
from pitch3d.core.scene.subject import Subject


def _R(q) -> np.ndarray:
    return axis_angle_to_matrix(quat_to_axis_angle(q))


def test_look_at_is_orthonormal_rotation():
    q, _ = look_at(np.array([5.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0]))
    R = _R(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-7)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_look_at_top_view_degenerate_up_is_finite():
    # straight-down camera: up is parallel to the view axis → must swap, not produce NaNs
    q, t = look_at(np.array([0.0, 0.0, 30.0]), np.zeros(3), up=np.array([0.0, 0.0, 1.0]))
    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))
    np.testing.assert_allclose(_R(q) @ _R(q).T, np.eye(3), atol=1e-7)


def test_standard_viewpoints_default_and_orbit(make_motion):
    scene = Scene(
        id="s", episode_id="e", source_id="x",
        subjects=[Subject(track_id=0, proposal=make_motion([0, 1, 2]))],
    )
    default = [v.viewpoint for v in standard_viewpoints(scene)]
    assert default == [Viewpoint.FRONT, Viewpoint.LEFT, Viewpoint.TOP, Viewpoint.BROADCAST]

    picked = standard_viewpoints(scene, which=[Viewpoint.TOP], n_orbit=3)
    assert picked[0].viewpoint == Viewpoint.TOP
    assert sum(v.viewpoint == Viewpoint.ORBIT for v in picked) == 3
