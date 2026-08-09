"""A singular homography must not kill a run — it is an unsolved frame (2026-08-09).

Found by moving off the clip everything had been measured on. The portrait fan clip
(`14604731_1080_1920_30fps.mp4`) produced at least one singular homography from PnLCalib, and
`FieldCalibration.world_to_image` inverted it unconditionally:

    numpy.linalg.LinAlgError: Singular matrix
      ball_lift._nearest_player_per_frame -> calibration.world_to_image -> np.linalg.inv

A 236-frame GPU run died in the ball lift with **no scene written at all**. Not a bad scene — no
scene.

The repo already has the concept for "this frame was not solved": `confidence` 0, honoured by
`solved_mask`. A singular homography is the strongest possible case of it — the plane maps onto a
line — so it is marked once at construction and every consumer that reads `solved_mask` skips it
for free.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.scene.field import MIN_SOLVED_CONFIDENCE, FieldCalibration

#: Rank 2: rows 2 and 3 are multiples of row 1's span, so the plane collapses onto a line.
SINGULAR = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]])


def _calib(mats, conf=None) -> FieldCalibration:
    mats = np.asarray(mats, dtype=float).reshape(-1, 3, 3)
    n = mats.shape[0]
    return FieldCalibration(
        homographies=mats,
        frames=np.arange(n),
        confidence=np.full(n, 0.8) if conf is None else np.asarray(conf, dtype=float),
    )


def test_a_singular_homography_is_marked_unsolved_at_construction():
    c = _calib([np.eye(3), SINGULAR])
    assert c.confidence[0] == pytest.approx(0.8)
    assert c.confidence[1] == 0.0, "a singular homography is not a low-quality solve, it is none"
    assert not c.solved_mask(np.array([1]), MIN_SOLVED_CONFIDENCE)[0]


def test_world_to_image_returns_nan_instead_of_raising():
    """The whole point. A caller can skip NaN; it cannot skip a LinAlgError four frames deep."""
    c = _calib([np.eye(3), SINGULAR])
    out = c.world_to_image(1, np.array([[10.0, 5.0]]))
    assert out.shape == (1, 2)
    assert np.isnan(out).all()


def test_a_good_frame_is_untouched():
    """A guard that broke the working path would be worse than the crash."""
    c = _calib([np.eye(3), SINGULAR])
    np.testing.assert_allclose(c.world_to_image(0, np.array([[10.0, 5.0]])), [[10.0, 5.0]])
    assert c.confidence[0] == pytest.approx(0.8)


def test_non_finite_homographies_count_too():
    """NaN or inf comes out of a diverged solve and inverts to garbage rather than raising."""
    bad = np.full((3, 3), np.nan)
    c = _calib([np.eye(3), bad])
    assert c.confidence[1] == 0.0
    assert np.isnan(c.world_to_image(1, np.array([[1.0, 1.0]]))).all()


def test_degenerate_frames_lists_them():
    c = _calib([np.eye(3), SINGULAR, np.eye(3), np.full((3, 3), np.inf)])
    np.testing.assert_array_equal(c.degenerate_frames, [1, 3])


def test_a_point_on_the_horizon_is_nan_not_an_overflow():
    """w == 0 is a world point the camera cannot see; dividing by it used to produce inf."""
    # a homography whose inverse sends (1, 0) to the plane at infinity
    h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    c = _calib([h])
    out = c.world_to_image(0, np.array([[0.0, 5.0]]))
    assert np.isnan(out).all() or np.isfinite(out).all()


def test_an_all_good_calibration_keeps_every_confidence():
    conf = np.array([0.9, 0.4, 0.02])
    c = _calib([np.eye(3)] * 3, conf)
    np.testing.assert_allclose(c.confidence, conf)
    assert c.degenerate_frames.size == 0
