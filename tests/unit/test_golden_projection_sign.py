"""R6 (#98): golden tests for the projection round-trip and the **sign** of our transforms.

A sign error does not produce garbage. It produces a scene that is mirrored, or upside down,
and otherwise perfectly self-consistent — every mutual-inverse test still passes and the only
detector is a human eye on a finished render. This project has paid for that lesson three
times: the overlay mirror (#50), the overlay rebuild (#64), and the 180° camera roll that left
the solved ``CameraTrack`` correct *only* on an upside-down frame.

So these are deliberately not round-trips against ourselves. ``test_field_calibration.py`` already
proves ``image_to_world`` and ``world_to_image`` invert each other, and
:func:`test_a_mirror_still_round_trips_which_is_why_round_trips_are_not_enough` shows exactly how
little that buys. Each test here either anchors to a ground truth built independently of the code
under test, or asserts an invariant (winding, which way is up) that a mirror breaks and
self-consistency cannot repair.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.correction.rotations import matrix_to_quat
from pitch3d.core.scene.camera import CameraTrack
from pitch3d.core.scene.field import FieldCalibration
from pitch3d.core.scene.projection import (
    project_world_points,
    quat_to_rotation_matrix,
)
from pitch3d.eval.synthetic import CAMERA_VIEWS, generate_scene

#: The 180° roll about the optical axis. `poseannot.camera` applies this to un-flip a solved
#: camera; applying it to a *correct* camera manufactures the exact defect that gate exists for.
ROLL_180 = np.diag([-1.0, -1.0, 1.0])


def _ground_truth():
    """A synthetic broadcast camera whose geometry is known independently of the code tested."""
    return generate_scene(n_subjects=1, n_frames=2, seed=0, camera=CAMERA_VIEWS["main_sideline"])


def _camera_track(scene, rot, transl, *, aligned=False) -> CameraTrack:
    n = scene.n_frames
    return CameraTrack(
        intrinsics=scene.intrinsics,
        frames=np.arange(n),
        rotation_quat=np.tile(matrix_to_quat(rot), (n, 1)),
        translation=np.tile(transl, (n, 1)),
        raw_frame_aligned=aligned,
    )


def _signed_area(p: np.ndarray) -> float:
    """Twice-signed area of a triangle; its **sign** is the winding, which a mirror flips."""
    a, b, c = p[0], p[1], p[2]
    return float((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def _mirrored_calibration(scene) -> FieldCalibration:
    """The GT calibration with the image convention mirrored (u → W-u) — the #50 defect."""
    mirror = np.array(
        [[-1.0, 0.0, float(scene.intrinsics.width)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    good = scene.field_calibration()
    return FieldCalibration(
        homographies=(good.homographies[0] @ mirror)[None],
        frames=np.array([0]),
        confidence=np.ones(1),
    )


# --- direction: which way does H actually go? ---------------------------------------------


def test_the_homography_maps_pixels_to_world_and_not_the_reverse():
    # `FieldCalibration.homographies` is image→world. Store the inverse by mistake and *both*
    # existing round-trip tests still pass — they only check the two methods undo each other.
    # This one cannot be satisfied by a self-consistent pair: the pixels come from the GT
    # camera's own pinhole projection, so only the correct direction lands back on the point.
    gt = _ground_truth()
    world = np.array([[0.0, 0.0, 0.0], [20.0, -10.0, 0.0], [-33.0, 25.0, 0.0]])
    pixels = gt.project(world)

    back = gt.field_calibration().image_to_world(0, pixels)
    np.testing.assert_allclose(back, world[:, :2], atol=1e-9)

    # ...and the two directions are genuinely different, so the assertion above has teeth.
    swapped = gt.field_calibration().image_to_world(0, world[:, :2])
    assert not np.allclose(swapped, pixels, atol=1.0)


def test_the_pinhole_projector_and_the_homography_anchor_place_a_point_identically():
    # Two independent implementations of the same geometry: `projection.py` runs X_c = R·X_w + t
    # through the intrinsics, while `field.world_to_image` inverts the homography. Players are
    # placed by one and overlaid by the other, so a convention drift between them puts bodies in
    # the wrong place on the pitch while each path stays internally coherent.
    gt = _ground_truth()
    world = np.array([[0.0, 0.0, 0.0], [20.0, -10.0, 0.0], [-33.0, 25.0, 0.0]])

    uv_pinhole, _ = project_world_points(_camera_track(gt, gt.rotation, gt.translation), 0, world)
    uv_homography = gt.field_calibration().world_to_image(0, world[:, :2])
    np.testing.assert_allclose(uv_pinhole, uv_homography, atol=1e-6)


# --- sign: a mirror is self-consistent, which is the whole problem -------------------------


def test_a_mirror_still_round_trips_which_is_why_round_trips_are_not_enough():
    # The justification for this whole file. Mirror the image convention (u → W-u), the kind of
    # error behind #50, and the round-trip closes to 1e-9 — while the same homography puts a
    # known world point tens of metres from where it belongs.
    gt = _ground_truth()
    bad = _mirrored_calibration(gt)

    world = np.array([[20.0, -10.0], [-33.0, 25.0]])
    closes = bad.image_to_world(0, bad.world_to_image(0, world))
    np.testing.assert_allclose(closes, world, atol=1e-9)

    pixels = gt.project(np.column_stack([world, np.zeros(2)]))
    error = np.linalg.norm(bad.image_to_world(0, pixels) - world, axis=1)
    assert error.min() > 10.0  # measured 40 m and 66 m


def test_projection_flips_the_winding_of_a_triangle_on_the_pitch():
    # Winding is the invariant a mirror cannot fake. A triangle counter-clockwise when viewed
    # from world +Z projects **clockwise** in pixels, because image v points down while world Z
    # points up — one flip, not two. If a future estimator returns a mirrored-but-consistent
    # camera, this sign is what catches it.
    gt = _ground_truth()
    triangle = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    assert _signed_area(triangle[:, :2]) > 0  # counter-clockwise seen from above

    uv, _ = project_world_points(_camera_track(gt, gt.rotation, gt.translation), 0, triangle)
    assert _signed_area(uv) < 0

    # The same pitch triangle through a mirrored calibration comes back wound the other way.
    assert _signed_area(_mirrored_calibration(gt).world_to_image(0, triangle[:, :2])) > 0


def test_raising_a_point_in_world_z_moves_it_up_the_image():
    # World is Z-up; image v is down. Pins that single negation — the one the 180°-roll bug got
    # wrong, which is why every body rendered head-down while the camera solve looked fine.
    gt = _ground_truth()
    camera = _camera_track(gt, gt.rotation, gt.translation)
    uv, _ = project_world_points(camera, 0, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]))
    on_pitch, head_height = uv

    assert head_height[1] < on_pitch[1]
    np.testing.assert_allclose(head_height[0], on_pitch[0], atol=1e-9)  # straight up, no drift


def test_the_two_quaternion_implementations_in_the_tree_agree():
    # `projection.quat_to_rotation_matrix` (hand-rolled, (w,x,y,z)) and `rotations.matrix_to_quat`
    # (Shepperd) are separate implementations of one convention, and scipy — which poseannot uses
    # via `np.roll(q, -1)` — is a third. Disagree on component order and you get a camera that is
    # rotated or mirrored but perfectly self-consistent.
    gt = _ground_truth()
    quat = matrix_to_quat(gt.rotation)
    np.testing.assert_allclose(quat_to_rotation_matrix(quat), gt.rotation, atol=1e-12)

    scipy_transform = pytest.importorskip("scipy.spatial.transform")
    scipy_rot = scipy_transform.Rotation.from_quat(np.roll(quat, -1)).as_matrix()
    np.testing.assert_allclose(scipy_rot, gt.rotation, atol=1e-12)


# --- the 180° roll gate, which until now was untested production code ----------------------


def test_the_roll_gate_fires_on_the_camera_geometry_that_produced_it():
    # Reproduces the real defect: PnLCalib's solve is self-consistent only on the frame turned
    # upside down, so `D @ R` (D = diag(-1,-1,1)) is what the pipeline actually hands us. The
    # gate `-R[1,2] < 0` must fire, and since D is its own inverse the fix must recover the true
    # camera exactly — not merely something that looks better.
    pytest.importorskip("scipy")
    from poseannot.camera import frame_projector

    gt = _ground_truth()
    assert not bool(-gt.rotation[1, 2] < 0)  # the correct camera must NOT trip the gate

    flipped = _camera_track(gt, ROLL_180 @ gt.rotation, ROLL_180 @ gt.translation)
    projected = frame_projector(flipped, 0)

    assert projected.frame_flipped
    np.testing.assert_allclose(projected.R, gt.rotation, atol=1e-9)
    np.testing.assert_allclose(projected.t, gt.translation, atol=1e-9)


def test_the_uncorrected_roll_puts_heads_below_feet():
    # The failure the gate prevents, asserted directly. Without the correction a standing body
    # projects head-down — and at ~20 px tall that is invisible to the eye, which is why #50 was
    # first "fixed" with an X-only mirror that left every body inverted.
    pytest.importorskip("scipy")
    from poseannot.camera import frame_projector, project_points

    gt = _ground_truth()
    foot, head = np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.8]])
    rot, transl = ROLL_180 @ gt.rotation, ROLL_180 @ gt.translation

    # `raw_frame_aligned` bypasses the gate — here it exposes the raw defect, and in production
    # it is the documented escape hatch for a camera rebuilt from the raw-frame homography.
    uncorrected = frame_projector(_camera_track(gt, rot, transl, aligned=True), 0)
    assert not uncorrected.frame_flipped
    assert project_points(head, uncorrected)[0, 1] > project_points(foot, uncorrected)[0, 1]

    corrected = frame_projector(_camera_track(gt, rot, transl), 0)
    assert project_points(head, corrected)[0, 1] < project_points(foot, corrected)[0, 1]
