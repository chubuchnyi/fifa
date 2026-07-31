"""Drawing the things that stand UP off the pitch (#116) — goal frames and corner flags.

Every other overlay in this project lives on ``Z = 0``, where the solved homography is an exact
map of the lawn and the focal length never enters the arithmetic. These do not, which is the
entire point of drawing them: a wrong focal is *invisible* on the markings and obvious on a
goalpost, so the goal frame is the only instrument on the pitch that can measure one.

That makes these functions untestable against our own overlay — the reference has to come from
outside. It does: ``pitch3d.eval.synthetic`` builds a broadcast camera with a known pose and a
known ``fx = fy = 1400``, principal point exactly at the image centre, so it satisfies the
assumptions and its pinhole projection is an independent ground truth for all three of "what is
the focal", "where is the camera" and "where does this point land".
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.scene.pitch import (
    CORNER_FLAG_HEIGHT,
    GOAL_FRAME_HEIGHT,
    GOAL_INNER_WIDTH,
    pitch_upright_polylines,
)
from pitch3d.core.scene.units import FieldDimensions
from pitch3d.eval.synthetic import CAMERA_VIEWS, generate_scene

pytest.importorskip("scipy")

from poseannot.camera import (  # noqa: E402  (after importorskip, as elsewhere in this suite)
    camera_centre,
    focal_from_homography,
    lift_homography,
    project_world,
    world_to_image,
)

#: Ground truth: fx = fy = 1400 px, principal point at (640, 360) = the exact image centre, camera
#: standing 15 m up and 50 m back from the halfway line. Everything below is checked against it.
TRUE_FOCAL = 1400.0
TRUE_CENTRE = np.array([0.0, -50.0, 15.0])


def _truth():
    scene = generate_scene(n_subjects=1, n_frames=2, seed=0, camera=CAMERA_VIEWS["main_sideline"])
    w2i = world_to_image(scene.field_calibration(), 0)
    return scene, w2i, scene.intrinsics.width, scene.intrinsics.height


#: A ground point, the same point 2 m up, and a goalpost base + its crossbar height.
SAMPLES = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [52.5, -3.66, 0.0], [52.5, -3.66, 2.44]])


# --- reading the focal and the camera pose back out of one ground homography ---


def test_focal_is_recovered_from_a_homography_whose_focal_we_know():
    scene, w2i, wid, hgt = _truth()
    del scene
    assert focal_from_homography(w2i, wid, hgt) == pytest.approx(TRUE_FOCAL, rel=1e-6)


def test_camera_centre_is_recovered_in_world_metres():
    # The readout the user reads before trusting a hand-set focal, so it has to be right in
    # metres and not merely plausible: a broadcast rig is ~15-25 m up and well past the touchline.
    scene, w2i, wid, hgt = _truth()
    del scene
    np.testing.assert_allclose(camera_centre(w2i, TRUE_FOCAL, wid, hgt), TRUE_CENTRE, atol=1e-9)


def test_off_plane_points_land_where_the_real_pinhole_camera_puts_them():
    # The claim that matters. `scene.project` runs the full X_c = R·X_w + t pinhole; `project_world`
    # rebuilds the same geometry from the flattened ground homography plus a focal. Agreement on
    # points at Z = 2.0 and Z = 2.44 m is what says the crossbar is drawn where it really is.
    scene, w2i, wid, hgt = _truth()
    np.testing.assert_allclose(
        project_world(SAMPLES, w2i, TRUE_FOCAL, wid, hgt), scene.project(SAMPLES), atol=1e-6
    )


# --- the signs, which are self-consistent when wrong (see test_golden_projection_sign.py) ---


def test_lifting_a_point_off_the_grass_moves_it_up_the_image():
    # The defect this replaced: taking the algebraic decomposition branch drew every goalpost
    # buried — crossbar *below* its own base — while the pitch markings stayed perfect, because
    # nothing on the plane touches the sign of the Z term.
    scene, w2i, wid, hgt = _truth()
    del scene
    base, top = project_world(SAMPLES[2:4], w2i, TRUE_FOCAL, wid, hgt)
    assert top[1] < base[1]


def test_the_sign_of_the_homography_itself_changes_nothing():
    # A homography is a projective object: H and -H are the same map, and a solver may hand back
    # either. Deriving the drawn geometry from H's raw sign therefore has a 50% chance of putting
    # every upright behind the camera, where it is culled — the overlay does not glitch, it
    # silently empties.
    scene, w2i, wid, hgt = _truth()
    del scene
    np.testing.assert_allclose(
        project_world(SAMPLES, w2i, TRUE_FOCAL, wid, hgt),
        project_world(SAMPLES, -w2i, TRUE_FOCAL, wid, hgt),
        atol=1e-9,
    )


def test_points_behind_the_camera_are_dropped_rather_than_mirrored():
    # Behind-camera points come back through the origin as a plausible-looking pixel, which draws
    # a phantom goal in the middle of the crowd. NaN so the caller can cull the polyline.
    scene, w2i, wid, hgt = _truth()
    del scene
    behind = np.array([[0.0, -80.0, 0.0]])  # camera is at y = -50 looking toward +Y
    assert np.isnan(project_world(behind, w2i, TRUE_FOCAL, wid, hgt)).all()


# --- what makes the goal frame an instrument: the focal moves it and nothing else ---


def test_the_focal_moves_the_crossbar_and_leaves_the_markings_untouched():
    # The premise of the whole feature, and the reason the control is worth exposing to the user.
    # If a wrong focal also moved the lines, the lines would already have measured it (#114) and
    # there would be nothing to hand-tune.
    scene, w2i, wid, hgt = _truth()
    del scene
    low = project_world(SAMPLES, w2i, 900.0, wid, hgt)
    high = project_world(SAMPLES, w2i, 2200.0, wid, hgt)

    np.testing.assert_allclose(low[0], high[0], atol=1e-9)  # centre spot, on the plane
    np.testing.assert_allclose(low[2], high[2], atol=1e-9)  # goalpost base, on the plane
    assert np.linalg.norm(low[3] - high[3]) > 20.0          # crossbar, 2.44 m up: 32 px apart


def test_lift_reports_a_ground_map_and_an_up_direction_that_compose():
    # `lift_homography` is the seam the two sign decisions live behind, so pin its contract:
    # `ground @ (X, Y, 1) + Z * up` must be exactly what `project_world` divides through.
    scene, w2i, wid, hgt = _truth()
    del scene
    ground, up = lift_homography(w2i, TRUE_FOCAL, wid, hgt)
    p = np.column_stack([SAMPLES[:, :2], np.ones(len(SAMPLES))]) @ ground.T + SAMPLES[:, 2:3] * up
    assert (p[:, 2] > 0).all()  # all four are in front of the camera, honestly signed
    np.testing.assert_allclose(
        p[:, :2] / p[:, 2, None], project_world(SAMPLES, w2i, TRUE_FOCAL, wid, hgt), atol=1e-9
    )


# --- the geometry being drawn ---


def test_uprights_are_two_goal_frames_and_four_corner_flags():
    polys = pitch_upright_polylines(FieldDimensions(length=105.0, width=68.0))
    goals = [p for p in polys if len(p) == 4]
    flags = [p for p in polys if len(p) == 2]
    assert len(goals) == 2 and len(flags) == 4

    for goal in goals:
        # Traced base → up → across → down, so it draws as one connected frame.
        np.testing.assert_allclose(goal[:, 2], [0.0, GOAL_FRAME_HEIGHT, GOAL_FRAME_HEIGHT, 0.0])
        assert abs(goal[0, 0]) == pytest.approx(52.5)          # standing on a goal line
        np.testing.assert_allclose(goal[:, 0], goal[0, 0])     # and staying in that plane
        assert goal[:, 1].max() - goal[:, 1].min() == pytest.approx(GOAL_INNER_WIDTH)

    corners = {(round(p[0, 0], 3), round(p[0, 1], 3)) for p in flags}
    assert corners == {(52.5, 34.0), (52.5, -34.0), (-52.5, 34.0), (-52.5, -34.0)}
    for flag in flags:
        np.testing.assert_allclose(flag[:, 2], [0.0, CORNER_FLAG_HEIGHT])


def test_uprights_ride_the_requested_ground_plane():
    # The pitch plane is not always Z = 0 (`field.plane_z`); a goal left at 0 would float or sink.
    polys = pitch_upright_polylines(plane_z=0.4)
    assert min(float(p[:, 2].min()) for p in polys) == pytest.approx(0.4)
