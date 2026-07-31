"""A scene carries two descriptions of one camera, and nothing checked they agreed (#61, #119).

``Scene.camera`` is a pinhole ``K [R | t]``; ``Scene.field.calibration`` is a per-frame
image↔world-plane homography. They describe the *same* physical camera, so on the pitch plane
they must produce the same pixel. In ``out/carry_off/export/scene.json`` they did not: six pitch
landmarks came out **12686 px** apart, because the stored intrinsics were 1158 px in 1920-space
against a measured 4169. That is the whole of #61's "players ~3× too small" — not a scale bug in
the overlay, two different cameras in one file.

Nothing caught it for months, and the reason is worth stating: every consumer uses *one* of the
two. The pitch overlay reads the homography and looks right (1.4 px against real paint, #114);
the players read the camera and look plausible on their own. The defect is only visible when you
ask the two to agree, which is what this file does.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from pitch3d.eval.synthetic import CAMERA_VIEWS, generate_scene

pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[2]

#: Landmarks spread to the far corners on purpose. Two cameras that disagree in focal still agree
#: near the principal point, so a centre-huddled probe reads 0 px on a scene that is 12686 px out.
LANDMARKS = np.array(
    [[0.0, 0.0], [52.5, 34.0], [-52.5, -34.0], [52.5, -34.0], [-52.5, 34.0], [0.0, 34.0]]
)


def _script(name: str):
    """Import a ``scripts/`` module — that directory is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fitter(width: int, height: int):
    """``fit_rigid_camera``, retargeted from the 1920×1080 clip to the synthetic pixel space.

    The two scripts are only correct *together*: the fit writes ``world_to_image`` via its
    ``plane_h``, the applier rebuilds a camera from ``focal``/``centre``/``rvecs``, and the scene
    is broken unless those two round-trip. Reproducing ``plane_h`` here would test the copy.
    """
    mod = _script("fit_rigid_camera")
    mod.WIDTH, mod.HEIGHT = width, height
    return mod


def _truth():
    """A known camera: fx = fy = 1400, 15 m up and 50 m back, principal point at image centre."""
    scene = generate_scene(n_subjects=1, n_frames=3, seed=0, camera=CAMERA_VIEWS["main_sideline"])
    return scene, scene.field_calibration()


def _through_homography(h_iw, xy):
    p = np.column_stack([xy, np.ones(len(xy))]) @ np.linalg.inv(h_iw).T
    return p[:, :2] / p[:, 2, None]


def _through_camera(track, row, xy):
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_quat(np.roll(track.rotation_quat[row], -1)).as_matrix()
    k = track.intrinsics.matrix()
    c = np.column_stack([xy, np.zeros(len(xy))]) @ rot.T + track.translation[row]
    p = c @ k.T
    return p[:, :2] / p[:, 2, None]


def test_the_fit_and_the_applier_describe_the_same_camera():
    # The #61 assertion, across the seam it really has to hold on: `plane_h` (what the fit writes
    # into `world_to_image`, and therefore what becomes the FieldCalibration) against the
    # CameraTrack `camera_track` rebuilds from the same focal, centre and rotations.
    scene, _ = _truth()
    fit = _fitter(scene.intrinsics.width, scene.intrinsics.height)
    applier = _script("apply_rigid_camera")
    focal = float(scene.intrinsics.fx)
    centre = -scene.rotation.T @ scene.translation
    rvecs = np.tile(_rotvec(scene.rotation), (3, 1))

    w2i = np.stack([fit.plane_h(focal, r, centre) for r in rvecs])
    track = applier.camera_track(
        {"focal": focal, "centre": centre, "rvecs": rvecs, "frames": [0, 1, 2]},
        scene.intrinsics.width,
        scene.intrinsics.height,
    )
    for row in range(3):
        np.testing.assert_allclose(
            _through_camera(track, row, LANDMARKS),
            _through_homography(np.linalg.inv(w2i[row]), LANDMARKS),
            atol=1e-6,
        )


def test_the_fitted_camera_agrees_with_the_ground_truth_pinhole():
    # And the seam is anchored, not merely self-consistent: `scene.project` runs the full
    # X_c = R·X_w + t of a camera whose focal and pose we know, independently of both scripts.
    scene, _ = _truth()
    applier = _script("apply_rigid_camera")
    centre = -scene.rotation.T @ scene.translation
    track = applier.camera_track(
        {"focal": float(scene.intrinsics.fx), "centre": centre,
         "rvecs": _rotvec(scene.rotation)[None], "frames": [0]},
        scene.intrinsics.width,
        scene.intrinsics.height,
    )
    np.testing.assert_allclose(
        _through_camera(track, 0, LANDMARKS),
        scene.project(np.column_stack([LANDMARKS, np.zeros(len(LANDMARKS))])),
        atol=1e-6,
    )


def test_the_probe_can_actually_see_a_wrong_focal():
    # Guards the two above from passing vacuously. A camera at the 3.6×-short focal that #61
    # really carried must be caught, and caught by a wide margin.
    scene, _ = _truth()
    applier = _script("apply_rigid_camera")
    centre = -scene.rotation.T @ scene.translation
    truth = scene.project(np.column_stack([LANDMARKS, np.zeros(len(LANDMARKS))]))
    wrong = applier.camera_track(
        {"focal": scene.intrinsics.fx / 3.6, "centre": centre,
         "rvecs": _rotvec(scene.rotation)[None], "frames": [0]},
        scene.intrinsics.width,
        scene.intrinsics.height,
    )
    assert np.max(np.linalg.norm(_through_camera(wrong, 0, LANDMARKS) - truth, axis=1)) > 500.0


def test_the_world_mirror_is_improper_so_no_camera_exists_in_it():
    # Why #118's flip and #119's camera are one fact seen twice: writing a CameraTrack *forces*
    # the scene right-handed, because a mirrored world admits no valid rotation at all.
    scene, _ = _truth()
    mod = _script("apply_rigid_camera")
    assert np.linalg.det(mod.M) == pytest.approx(-1.0)
    assert np.linalg.det(mod.M @ scene.rotation) == pytest.approx(-1.0)
    # The conjugation used on `global_orient` is the way out: still a rotation, det +1.
    assert np.linalg.det(mod.M @ scene.rotation @ mod.M) == pytest.approx(1.0)


def test_mirroring_a_scene_twice_returns_it_unchanged():
    # The applier is the only writer of the flip, so its one algebraic promise is worth pinning:
    # a user who mirrors a scene they should not have can get back by mirroring again.
    from scipy.spatial.transform import Rotation

    mod = _script("apply_rigid_camera")
    rng = np.random.default_rng(0)
    transl = rng.normal(size=(5, 3))
    orient = Rotation.random(5, random_state=1).as_rotvec()

    def flip(t, o):
        rot = Rotation.from_rotvec(o).as_matrix()
        return t @ mod.M, Rotation.from_matrix(mod.M @ rot @ mod.M).as_rotvec()

    once = flip(transl, orient)
    twice = flip(*once)
    np.testing.assert_allclose(twice[0], transl, atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_rotvec(twice[1]).as_matrix(),
        Rotation.from_rotvec(orient).as_matrix(),
        atol=1e-12,
    )


def _rotvec(rot):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rot).as_rotvec()
