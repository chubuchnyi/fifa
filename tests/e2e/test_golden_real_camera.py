"""Golden test over the one real measurement this repo actually ships.

Every other test is fakes-backed by design (``tests/conftest.py`` says so), which is why a
green suite has never been evidence that anything works on real footage.
``calib/Colombia-1-0-Congo-DR1080p.npz`` is the exception: a camera measured off the real
broadcast clip by ``scripts/fit_rigid_camera.py`` — ONE focal, ONE optical centre, one
rotation per frame — and at 7 kB it is small enough to live in git, so unlike the clip
itself this test runs everywhere, CI included.

What it pins is not the file. A static file cannot regress. It pins what the *code* derives
from it: ``plane_camera.camera_from_calibration`` has to recover a real pinhole from these
homographies, and ``projection.project_world_points`` has to reproduce the framing a human
operator actually shot. Break the focal search, the pose decomposition or the world-frame
convention and these numbers move.

This is the pair of bugs the numbers below defend against:

* **#61 / #119** — the camera used to be re-solved as 60 free homographies, so it drifted
  frame to frame and no single pinhole existed. ``test_it_is_one_camera_not_sixty`` is that
  property stated as a number.
* **#118 / #120** — the world frame was mirrored, in which no camera can exist at all
  (``det(M·R) = −1``). Injecting a mirror back into ``_decompose`` is caught here, though by
  the two framing tests rather than by the determinant one — see below.

Every figure was measured on 2026-08-01 with this code, not copied out of a docstring.

Mutation-checked the same day, because a golden test nobody has tried to break is a guess.
Injected into ``core/scene/plane_camera.py``, one at a time:

===========================================  =========================================
mutation                                     verdict
===========================================  =========================================
planar sign ambiguity resolved backwards     caught (pan)
focal search corrupted (the #61 class)       caught (pan)
mirrored world frame (the #118/#120 class)   caught (framing + pan)
``det(rot) < 0`` reflection guard removed    **not caught** — see below
===========================================  =========================================

That last row is not a hole to plug. The guard is reached on all 60 frames and taken on none:
this clip's ``K⁻¹H`` is already proper, so the branch cannot change an answer here. It defends
inputs this measurement does not contain, and no assertion over this file can exercise it.
Recorded rather than papered over, so the next reader does not assume it is covered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pitch3d.core.scene.field import FieldCalibration
from pitch3d.core.scene.plane_camera import REALIZABLE_PX, camera_from_calibration
from pitch3d.core.scene.projection import (
    camera_center,
    project_world_points,
    quat_to_rotation_matrix,
)

CALIB = Path(__file__).resolve().parents[2] / "calib" / "Colombia-1-0-Congo-DR1080p.npz"

#: The broadcast frame the homographies were solved in.
WIDTH, HEIGHT = 1920, 1080

#: FIFA pitch landmarks in metres, origin at the centre spot, in the right-handed Z-up world.
CENTRE_SPOT = np.array([[0.0, 0.0, 0.0]])
LEFT_PENALTY_SPOT = np.array([[-41.5, 0.0, 0.0]])


@pytest.fixture(scope="module")
def measured_calibration() -> FieldCalibration:
    """The real solved camera, as the pipeline's own calibration type."""
    if not CALIB.exists():
        pytest.skip(f"missing the committed measurement {CALIB}")
    blob = np.load(CALIB)
    world_to_image = np.asarray(blob["world_to_image"], dtype=float)
    return FieldCalibration(
        # FieldCalibration stores image->world; the fit is recorded the other way round.
        homographies=np.linalg.inv(world_to_image),
        frames=np.asarray(blob["frames"], dtype=int),
        confidence=np.ones(len(blob["frames"]), dtype=float),
    )


@pytest.fixture(scope="module")
def fit(measured_calibration):
    return camera_from_calibration(measured_calibration, width=WIDTH, height=HEIGHT)


def test_the_measurement_is_still_the_clip_we_think_it_is():
    """Guards the fixture. If someone refits the camera, the rest of this file is about a
    different clip and should be re-measured rather than nudged until it passes."""
    if not CALIB.exists():
        pytest.skip(f"missing the committed measurement {CALIB}")
    blob = np.load(CALIB)
    assert set(blob.files) == {"focal", "centre", "rvecs", "frames", "world_to_image"}
    assert blob["world_to_image"].shape == (60, 3, 3)
    assert np.array_equal(blob["frames"], np.arange(60))


def test_a_real_pinhole_comes_back_out(fit):
    """The homographies are camera-realizable — the thing PnLCalib's were not (#125).

    Measured 0.0000 px. The threshold is the production one, so this asserts the same
    verdict ``app/controller.py`` acts on rather than a bar invented for the test.
    """
    assert fit.camera is not None
    assert fit.realizable
    assert fit.reprojection_px < REALIZABLE_PX
    assert fit.reprojection_px == pytest.approx(0.0, abs=1e-3)


def test_the_focal_is_recovered_from_the_homographies_alone(fit):
    """``camera_from_calibration`` never sees ``blob["focal"]`` — it searches for the focal
    that makes ``K⁻¹H`` orthonormal. Measured 4169.32 px, matching the stored fit."""
    stored_focal = float(np.load(CALIB)["focal"])
    assert fit.focal_px == pytest.approx(stored_focal, abs=1.0)
    assert fit.focal_px == pytest.approx(4169.32, abs=1.0)

    # ~26deg across 1920 px: a long broadcast lens. A focal off by the 3.9x of #61 lands
    # far outside this band, which is the point of checking it as an angle.
    fov_deg = 2.0 * np.degrees(np.arctan(WIDTH / (2.0 * fit.focal_px)))
    assert 20.0 < fov_deg < 35.0


def test_it_is_one_camera_not_sixty(fit):
    """#119 in a number: one optical centre for the whole clip.

    Sixty independently-solved homographies put the centre somewhere different every frame.
    Measured drift here is 7e-14 m — floating point, not motion.
    """
    centres = np.stack([camera_center(fit.camera, f) for f in range(60)])
    drift = centres.max(axis=0) - centres.min(axis=0)
    assert drift.max() < 1e-6, f"the camera moves between frames: {drift} m"


def test_the_camera_is_where_a_broadcast_camera_would_be(fit):
    """Measured (-2.29, -70.13, 17.22) m: off to the side of the halfway line, 70 m back
    and 17 m up. This is the assertion that fails if the world frame flips or the scale is
    wrong — both of which have happened (#61 shipped a camera 3.9x too small)."""
    centre = camera_center(fit.camera, 0)
    np.testing.assert_allclose(centre, [-2.292, -70.134, 17.220], atol=1e-2)

    # Stated as physics rather than as the three numbers above, so the intent survives a refit:
    assert abs(centre[1]) > 34.0, "camera should sit beyond the touchline, not on the pitch"
    assert 5.0 < centre[2] < 40.0, "camera should be on a gantry, not underground or in orbit"


def test_every_frame_is_a_proper_rotation(fit):
    """``det(R) = +1`` everywhere, never −1 — no reflection reaches the exported camera.

    A cheap invariant, and honestly a weak one: the mutation run above mirrored the world and
    this still passed, because ``_decompose`` snaps to the nearest *rotation* and so launders a
    mirror into a proper matrix pointing the wrong way. The framing tests are what catch that.
    Kept because it is the one assertion that would survive a rewrite of the pose decomposition.
    """
    dets = np.array(
        [np.linalg.det(quat_to_rotation_matrix(fit.camera.rotation_quat[f])) for f in range(60)]
    )
    np.testing.assert_allclose(dets, 1.0, atol=1e-9)


def test_the_recovered_camera_reproduces_the_framing_that_was_shot(fit):
    """The operator held the left penalty area in frame for the whole clip and never showed
    the centre spot. Measured: 60/60 and 0/60.

    This is the end-to-end check — homographies through focal search, pose decomposition and
    projection, compared against what is actually visible in the video.
    """
    penalty_seen = sum(bool(project_world_points(fit.camera, f, LEFT_PENALTY_SPOT)[1][0])
                       for f in range(60))
    centre_seen = sum(bool(project_world_points(fit.camera, f, CENTRE_SPOT)[1][0])
                      for f in range(60))
    assert penalty_seen == 60
    assert centre_seen == 0


def test_the_camera_pans_across_the_clip(fit):
    """It is a live broadcast camera, not a locked-off one: the left penalty spot slides from
    u=173 px to u=633 px over the 60 frames. A camera track that silently froze would keep
    every other assertion here green."""
    u_first = project_world_points(fit.camera, 0, LEFT_PENALTY_SPOT)[0][0][0]
    u_last = project_world_points(fit.camera, 59, LEFT_PENALTY_SPOT)[0][0][0]
    assert u_first == pytest.approx(173.2, abs=5.0)
    assert u_last == pytest.approx(633.0, abs=5.0)
    assert u_last - u_first > 100.0, "the camera should pan right across the clip"
