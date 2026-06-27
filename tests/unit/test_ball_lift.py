"""Ball 2D→3D lift — ballistics, ground projection, the apex confidence dip (R-4)."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.orchestration.ball_lift import (
    _stale_mask,
    ballistic_z,
    detect_ball_contacts,
    lift_ball_to_3d,
)
from pitch3d.core.scene.field import FieldCalibration
from pitch3d.core.scene.motion import (
    Ball2DTrack,
    PoseSequence,
    SmplxShape,
    SubjectMotion,
)
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


# --- contact-anchoring: the #206 fix (ball pinned to the players who play it) ---------------
def _ball2d_xy(frames, positions_2d) -> Ball2DTrack:
    return Ball2DTrack(
        frames=np.asarray(frames),
        positions_2d=np.asarray(positions_2d, dtype=float),
        confidence=np.ones(len(frames)),
    )


def _motion(frames, foot_xy) -> SubjectMotion:
    """A subject standing still with its root (foot) at a constant world ``foot_xy``."""
    pose = PoseSequence.rest(np.asarray(frames))
    pose.transl[:, 0] = foot_xy[0]
    pose.transl[:, 1] = foot_xy[1]
    return SubjectMotion(shape=SmplxShape(betas=np.zeros(10)), pose=pose)


def test_stale_mask_flags_exact_repeats():
    pos = np.array([[1, 1], [1, 1], [2, 2], [2, 2], [2, 2], [3, 3]], dtype=float)
    np.testing.assert_array_equal(
        _stale_mask(pos), [False, True, False, True, True, False]
    )


def test_contact_anchoring_finds_two_player_pass():
    # Identity calibration ⇒ image px == world m, so contacts are trivial to place.
    frames = np.arange(11)
    cal = _identity_calib(frames)
    motions = {1: _motion(frames, (0.0, 0.0)), 2: _motion(frames, (10.0, 0.0))}
    # Ball sits on player 1 at f0, on player 2 at f10, flies (arcing in v) in between.
    pos2d = [(i, 0.0 if i in (0, 10) else 5.0) for i in range(11)]
    pos2d[0], pos2d[10] = (0.0, 0.0), (10.0, 0.0)
    ball2d = _ball2d_xy(frames, pos2d)

    anchors = detect_ball_contacts(ball2d, cal, motions, fps=25.0)
    assert [(a[0], a[2]) for a in anchors] == [(0, 1), (10, 2)]  # kicker → receiver

    bt = lift_ball_to_3d(ball2d, cal, motions=motions, fps=25.0)
    np.testing.assert_allclose(bt.positions_3d[0], [0.0, 0.0, 0.0])   # at the kicker's foot
    np.testing.assert_allclose(bt.positions_3d[10], [10.0, 0.0, 0.0])  # at the receiver's foot
    np.testing.assert_allclose(bt.positions_3d[5, :2], [5.0, 0.0])     # XY interpolated
    assert bt.positions_3d[5, 2] > 0.0                                 # airborne between
    assert bool(bt.on_ground[0]) and bool(bt.on_ground[10])
    assert not bt.on_ground[1:10].any()
    assert bt.height_confidence[0] == pytest.approx(1.0)
    assert bt.height_confidence[5] < 1.0
    # Every frame stays on a 105×68 pitch — the whole point of #206.
    assert (np.abs(bt.positions_3d[:, 0]) <= 52.5).all()
    assert (np.abs(bt.positions_3d[:, 1]) <= 34.0).all()


def test_contact_anchoring_rejects_physically_impossible_contact():
    # A third player C lines up with a high airborne ball in image but is 60 m away in world:
    # reaching it would need ~300 m/s, so it must be dropped (the depth-ambiguity artefact).
    frames = np.arange(11)
    cal = _identity_calib(frames)
    motions = {
        1: _motion(frames, (0.0, 0.0)),
        2: _motion(frames, (10.0, 0.0)),
        3: _motion(frames, (5.0, 60.0)),
    }
    pos2d = [(0.0, 0.0)] + [(i, 2.0) for i in range(1, 10)] + [(10.0, 0.0)]
    pos2d[5] = (5.0, 60.0)  # ball image lands exactly on C here
    ball2d = _ball2d_xy(frames, pos2d)

    anchors = detect_ball_contacts(ball2d, cal, motions, fps=25.0)
    tids = {a[2] for a in anchors}
    assert tids == {1, 2}  # the impossible C (tid 3) is rejected; the real pass survives


def test_no_contact_falls_back_to_ground_projection():
    # Players nowhere near the ball ⇒ no anchors ⇒ the honest mono ground projection.
    frames = np.arange(5)
    cal = _identity_calib(frames)
    far = {1: _motion(frames, (1000.0, 1000.0))}
    ball2d = _ball2d(frames)  # positions (i, 0)
    bt = lift_ball_to_3d(ball2d, cal, motions=far, fps=25.0)
    np.testing.assert_allclose(bt.positions_3d[:, :2], ball2d.positions_2d)  # == image (identity)
    np.testing.assert_allclose(bt.positions_3d[:, 2], 0.0)


def test_motions_none_matches_legacy_mono_path():
    frames = np.arange(5)
    cal = _identity_calib(frames)
    og = np.array([True, False, False, False, True])
    a = lift_ball_to_3d(_ball2d(frames), cal, on_ground=og, fps=25.0)
    b = lift_ball_to_3d(_ball2d(frames), cal, on_ground=og, motions=None, fps=25.0)
    np.testing.assert_allclose(a.positions_3d, b.positions_3d)
    np.testing.assert_allclose(a.height_confidence, b.height_confidence)
