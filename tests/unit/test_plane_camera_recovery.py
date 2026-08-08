"""The render camera must be the clip's camera, or admit it is not (#107).

``AppController`` used to overwrite the solved camera with a synthetic broadcast pose one line
before export, so every artifact carried a camera that had never seen the clip. The fix is not
"always decompose the calibration" — for a calibration of *free* per-frame homographies there is
no camera to decompose, at any focal. So the contract has two halves and both need pinning: it
recovers the camera exactly when one exists, and it **refuses** when one does not.

The refusal is the half that protects #61. A wrong-but-plausible camera is precisely what shipped
for months — two cameras in one scene, 12686 px apart, each looking fine to the consumer that read
it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.scene.camera import CameraSource
from pitch3d.core.scene.plane_camera import REALIZABLE_PX, camera_from_calibration
from pitch3d.core.scene.projection import quat_to_rotation_matrix
from pitch3d.eval.synthetic import CAMERA_VIEWS, generate_scene

#: Corners on purpose — two cameras that disagree in focal still agree near the principal point.
LANDMARKS = np.array(
    [[0.0, 0.0], [52.5, 34.0], [-52.5, -34.0], [52.5, -34.0], [-52.5, 34.0], [0.0, 34.0]]
)


def _truth():
    """A known camera (fx = fy = 1400, 15 m up, 50 m back) and the calibration it produces."""
    scene = generate_scene(n_subjects=1, n_frames=3, seed=0, camera=CAMERA_VIEWS["main_sideline"])
    return scene, scene.field_calibration()


def _centres(camera):
    return np.array([
        -quat_to_rotation_matrix(camera.rotation_quat[i]).T @ camera.translation[i]
        for i in range(camera.n_frames)
    ])


def test_a_real_camera_is_recovered_from_its_own_calibration():
    scene, cal = _truth()
    fit = camera_from_calibration(
        cal, width=scene.intrinsics.width, height=scene.intrinsics.height
    )
    assert fit.realizable
    # The focal was never passed in — it is read back out of the homographies alone.
    assert fit.focal_px == pytest.approx(float(scene.intrinsics.fx), rel=1e-6)
    assert fit.reprojection_px < 1e-6
    truth_centre = -scene.rotation.T @ scene.translation
    np.testing.assert_allclose(_centres(fit.camera), np.tile(truth_centre, (3, 1)), atol=1e-6)


def test_the_recovered_camera_projects_where_the_ground_truth_pinhole_does():
    # Anchored, not merely self-consistent: `scene.project` is the full X_c = R·X_w + t of a
    # camera we specified, independent of anything this module computed.
    scene, cal = _truth()
    fit = camera_from_calibration(
        cal, width=scene.intrinsics.width, height=scene.intrinsics.height
    )
    ground = np.column_stack([LANDMARKS, np.zeros(len(LANDMARKS))])
    rot = quat_to_rotation_matrix(fit.camera.rotation_quat[0])
    p = (ground @ rot.T + fit.camera.translation[0]) @ fit.camera.intrinsics.matrix().T
    np.testing.assert_allclose(p[:, :2] / p[:, 2, None], scene.project(ground), atol=1e-6)


def test_free_per_frame_homographies_are_refused_rather_than_decomposed():
    # The #107 finding in miniature. Perturb each frame's homography independently — the way a
    # per-frame solve does — and no single K can explain them any more. Nothing may be returned.
    scene, cal = _truth()
    rng = np.random.default_rng(0)
    h = np.asarray(cal.homographies, dtype=float).copy()
    h[:, :2, :] *= 1.0 + rng.normal(scale=0.05, size=(len(h), 2, 3))
    cal.homographies = h

    fit = camera_from_calibration(
        cal, width=scene.intrinsics.width, height=scene.intrinsics.height
    )
    assert not fit.realizable
    assert fit.camera is None
    assert fit.reprojection_px > REALIZABLE_PX


def test_a_supplied_focal_that_the_homographies_reject_is_not_quietly_accepted():
    # The manual override may not become a back door around the honesty check: overriding to a
    # 3.6x-short focal is the #61 defect exactly, so it has to fail the same way.
    scene, cal = _truth()
    fit = camera_from_calibration(
        cal, width=scene.intrinsics.width, height=scene.intrinsics.height,
        focal_px=float(scene.intrinsics.fx) / 3.6,
    )
    assert not fit.realizable
    assert fit.camera is None


def test_the_pipeline_keeps_the_camera_it_solved(reconstructed):
    # The #107 regression itself: for years `run_reconstruction` ended with an unconditional
    # `scene.camera = self._static_camera(scene)`, so a solved camera never survived to export.
    # Assert it through the invariant that actually matters (#61) — the scene's camera and the
    # scene's calibration must be the same camera, which is exactly what nothing checked.
    app, scene_id = reconstructed
    scene = app.get_scene(scene_id)
    fit = app.camera_fit(scene_id)
    assert fit is not None and fit.camera is not None, "the fakes' calibration IS a camera"
    # Substance, not object identity: since #140 the controller wraps the solved camera in a
    # `replace(...)` to record HOW it was obtained, so `is` no longer holds. What must hold is
    # that the scene carries the solved camera's numbers and says it is a solve — which is a
    # stronger statement than identity, because it also fails if a fallback sneaks in with the
    # right pose.
    assert scene.camera.source is CameraSource.PLANE_FIT, "scene claims a camera it did not solve"
    assert scene.camera.is_measured
    np.testing.assert_allclose(scene.camera.rotation_quat, fit.camera.rotation_quat)
    np.testing.assert_allclose(scene.camera.translation, fit.camera.translation)
    assert scene.camera.intrinsics.fx == fit.camera.intrinsics.fx
    assert scene.camera.fit_reprojection_px == pytest.approx(fit.reprojection_px)

    h_i2w = np.asarray(scene.field.calibration.homographies[0], dtype=float)
    rot = quat_to_rotation_matrix(scene.camera.rotation_quat[0])
    world = np.column_stack([LANDMARKS, np.zeros(len(LANDMARKS))])
    p = (world @ rot.T + scene.camera.translation[0]) @ scene.camera.intrinsics.matrix().T
    q = np.column_stack([LANDMARKS, np.ones(len(LANDMARKS))]) @ np.linalg.inv(h_i2w).T
    np.testing.assert_allclose(p[:, :2] / p[:, 2, None], q[:, :2] / q[:, 2, None], atol=1e-6)


def test_the_probe_would_notice_a_focal_that_is_only_slightly_wrong():
    # Guards the tests above from passing on a probe too blunt to see anything. 2% of focal is far
    # finer than any defect this module exists to catch.
    scene, cal = _truth()
    fit = camera_from_calibration(
        cal, width=scene.intrinsics.width, height=scene.intrinsics.height,
        focal_px=float(scene.intrinsics.fx) * 1.02,
    )
    assert not fit.realizable
