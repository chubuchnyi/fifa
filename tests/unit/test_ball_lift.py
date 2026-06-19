"""Ball 2D→3D lift — ballistics, ground projection, the apex confidence dip (R-4)."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.orchestration.ball_lift import ballistic_z, lift_ball_to_3d
from pitch3d.core.scene.field import FieldCalibration
from pitch3d.core.scene.motion import Ball2DTrack
from pitch3d.core.scene.units import GRAVITY


def _identity_calib(frames) -> FieldCalibration:
    n = len(frames)
    return FieldCalibration(
        homographies=np.tile(np.eye(3), (n, 1, 1)),
        frames=np.asarray(frames, dtype=int),
        confidence=np.ones(n),
    )


def _ball2d(frames, confidence=1.0) -> Ball2DTrack:
    n = len(frames)
    return Ball2DTrack(
        frames=np.asarray(frames),
        positions_2d=np.column_stack([np.arange(n), np.zeros(n)]),
        confidence=np.full(n, confidence),
    )


def test_ballistic_z_endpoints_and_peak():
    z = ballistic_z(np.array([0.0, 0.5, 1.0]), flight_s=1.0)
    assert z[0] == pytest.approx(0.0)
    assert z[2] == pytest.approx(0.0)
    assert z[1] == pytest.approx(GRAVITY * 1.0**2 / 8.0)
    assert np.all(ballistic_z(np.array([0.1, 0.2]), flight_s=0.0) == 0.0)


def test_all_ground_is_flat_and_confident():
    frames = np.arange(5)
    bt = lift_ball_to_3d(_ball2d(frames), _identity_calib(frames), on_ground=np.ones(5, bool))
    np.testing.assert_allclose(bt.positions_3d[:, 2], 0.0)
    np.testing.assert_allclose(bt.height_confidence, 1.0)


def test_bracketed_arc_has_apex_and_confidence_dip():
    frames = np.arange(5)
    og = np.array([True, False, False, False, True])
    bt = lift_ball_to_3d(_ball2d(frames), _identity_calib(frames), on_ground=og, fps=25.0)
    assert bt.positions_3d[0, 2] == pytest.approx(0.0)
    assert bt.positions_3d[4, 2] == pytest.approx(0.0)
    assert bt.positions_3d[2, 2] > bt.positions_3d[1, 2]  # apex is the highest interior frame
    assert bt.height_confidence[2] < 1.0                  # mono ambiguity surfaced at apex
    assert bt.height_confidence[2] < bt.height_confidence[0]


def test_no_contact_anywhere_is_low_confidence():
    frames = np.arange(4)
    bt = lift_ball_to_3d(
        _ball2d(frames), _identity_calib(frames), on_ground=np.zeros(4, bool),
        airborne_confidence=0.5,
    )
    np.testing.assert_allclose(bt.height_confidence, 0.25)  # 0.5 * 0.5 fallback
