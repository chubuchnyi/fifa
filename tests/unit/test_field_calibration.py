"""FieldCalibration projection — image↔world must round-trip (the #206 anchor primitive)."""

from __future__ import annotations

import numpy as np

from pitch3d.core.scene.field import FieldCalibration


def _calib(h: np.ndarray) -> FieldCalibration:
    return FieldCalibration(
        homographies=h[None], frames=np.array([0]), confidence=np.ones(1)
    )


def test_world_to_image_inverts_image_to_world():
    # A non-trivial perspective homography (not just a scale): world_to_image must be its
    # exact inverse, since contact detection projects a known foot world point into the image
    # to compare against the ball's 2D detection.
    h = np.array([[0.05, 0.002, -16.0], [0.001, 0.05, -9.0], [1e-4, 2e-4, 1.0]])
    cal = _calib(h)
    world = np.array([[0.0, 0.0], [20.0, -10.0], [-33.0, 25.0]])
    back = cal.image_to_world(0, cal.world_to_image(0, world))
    np.testing.assert_allclose(back, world, atol=1e-9)


def test_image_to_world_round_trips_through_world_to_image():
    h = np.array([[0.05, 0.0, -16.0], [0.0, 0.05, -9.0], [0.0, 0.0, 1.0]])
    cal = _calib(h)
    uv = np.array([[640.0, 360.0], [100.0, 900.0], [1850.0, 80.0]])
    back = cal.world_to_image(0, cal.image_to_world(0, uv))
    np.testing.assert_allclose(back, uv, atol=1e-9)
