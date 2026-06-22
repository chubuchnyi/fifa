"""Synthetic broadcast-soccer GT fixture (pitch3d.eval.synthetic).

Verifies the generator is deterministic and internally consistent: stored 2D joints equal
a re-projection of the world joints, the emitted GT homography inverts that projection on
the ground plane, feet sit on the pitch, and bboxes bound the subjects. These are the
fixtures the bake-off harness scores against, so their geometry must be exact.
"""

from __future__ import annotations

import numpy as np

from pitch3d.eval.synthetic import (
    CAMERA_VIEWS,
    CANONICAL_SKELETON,
    JOINT_NAMES,
    CameraView,
    _visibility_mask,
    generate_scene,
)


def test_generate_scene_is_deterministic():
    a = generate_scene(seed=7)
    b = generate_scene(seed=7)
    assert np.array_equal(a.joints_world, b.joints_world)
    assert np.array_equal(a.joints_image, b.joints_image)


def test_different_seed_differs():
    a = generate_scene(seed=1)
    b = generate_scene(seed=2)
    assert not np.allclose(a.joints_world, b.joints_world)


def test_shapes():
    n, t = 4, 6
    s = generate_scene(n_subjects=n, n_frames=t)
    j = len(CANONICAL_SKELETON)
    assert s.joints_world.shape == (t, n, j, 3)
    assert s.joints_image.shape == (t, n, j, 2)
    assert s.boxes_xyxy.shape == (t, n, 4)
    assert len(JOINT_NAMES) == j


def test_projection_matches_stored_image_joints():
    s = generate_scene(seed=3)
    assert np.allclose(s.project(s.joints_world), s.joints_image, atol=1e-9)


def test_homography_recovers_ground_xy():
    s = generate_scene(seed=0)
    cal = s.field_calibration()
    xy = np.array([[0.0, 0.0], [20.0, -10.0], [-15.0, 8.0], [40.0, 25.0]])
    ground = np.column_stack([xy, np.zeros(len(xy))])
    uv = s.project(ground)
    recovered = cal.image_to_world(int(s.frames[0]), uv)
    assert np.allclose(recovered, xy, atol=1e-6)


def test_higher_world_point_projects_higher_in_image():
    s = generate_scene(seed=0)
    # y is image-down, so a point higher in world (larger Z) gets a smaller v.
    assert s.project(np.array([0.0, 0.0, 2.0]))[1] < s.project(np.array([0.0, 0.0, 0.0]))[1]


def test_feet_sit_near_the_ground_plane():
    s = generate_scene(seed=0)
    ankles = [JOINT_NAMES.index("l_ankle"), JOINT_NAMES.index("r_ankle")]
    foot_z = s.joints_world[:, :, ankles, 2]
    assert foot_z.min() > -0.05
    assert foot_z.max() < 0.15


def test_boxes_contain_subject_centroids():
    s = generate_scene(seed=0)
    cu = s.joints_image[..., 0].mean(-1)
    cv = s.joints_image[..., 1].mean(-1)
    assert np.all(s.boxes_xyxy[..., 0] <= cu) and np.all(cu <= s.boxes_xyxy[..., 2])
    assert np.all(s.boxes_xyxy[..., 1] <= cv) and np.all(cv <= s.boxes_xyxy[..., 3])


# --- camera views -----------------------------------------------------------------------------


def test_default_camera_is_main_sideline():
    # The default path must stay byte-identical to passing the named default explicitly.
    a = generate_scene(seed=0)
    b = generate_scene(seed=0, camera=CAMERA_VIEWS["main_sideline"])
    assert np.array_equal(a.joints_world, b.joints_world)
    assert np.array_equal(a.joints_image, b.joints_image)


def test_camera_views_set_intrinsics_and_valid_rotation():
    for view in CAMERA_VIEWS.values():
        s = generate_scene(seed=0, camera=view)
        assert s.intrinsics.fx == view.fx and s.intrinsics.fy == view.fy
        # World→camera rotation must be a proper orthonormal rotation (rows are camera axes).
        assert np.allclose(s.rotation @ s.rotation.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(s.rotation), 1.0, atol=1e-9)


def test_different_camera_changes_projection():
    a = generate_scene(seed=0, camera=CAMERA_VIEWS["main_sideline"])
    b = generate_scene(seed=0, camera=CAMERA_VIEWS["behind_goal"])
    # Same world subjects (same seed) — the camera only relocates the eye, not the players, so the
    # world layout is invariant — but the extrinsics and pixels must change.
    assert np.allclose(a.joints_world, b.joints_world)
    assert not np.allclose(a.rotation, b.rotation)
    assert not np.allclose(a.joints_image, b.joints_image)


def test_camera_view_is_frozen():
    v = CameraView(eye=(1.0, 2.0, 3.0))
    assert v.target == (0.0, 0.0, 0.0) and v.fx == 1400.0
    try:
        v.eye = (0.0, 0.0, 0.0)  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("CameraView should be frozen/immutable")


# --- visibility / occlusion -------------------------------------------------------------------


def test_visibility_mask_occludes_farther_subject():
    # subject 0 (depth 10) is nearer than subject 1 (depth 20); both subject-1 joints land inside
    # subject-0's box → subject 1's joints are occluded, subject 0's stay visible.
    joints_image = np.array([[[[50.0, 50.0], [50.0, 60.0]], [[50.0, 50.0], [50.0, 60.0]]]])
    cam_depth = np.array([[10.0, 20.0]])
    boxes = np.array([[[0.0, 0.0, 100.0, 100.0], [200.0, 200.0, 300.0, 300.0]]])
    mask = _visibility_mask(joints_image, cam_depth, boxes, width=400, height=400)
    assert mask.shape == (1, 2, 2) and mask.dtype == bool
    assert mask[0, 0].all()       # nearer subject: never occluded by the farther one
    assert not mask[0, 1].any()   # farther subject: both joints inside the nearer box


def test_visibility_mask_no_self_occlusion():
    # A subject's own joints sitting inside its own box must not be flagged occluded.
    joints_image = np.array([[[[50.0, 50.0]]]])
    cam_depth = np.array([[10.0]])
    boxes = np.array([[[0.0, 0.0, 100.0, 100.0]]])
    assert _visibility_mask(joints_image, cam_depth, boxes, 400, 400).all()


def test_visibility_mask_flags_out_of_frame():
    joints_image = np.array([[[[-5.0, 50.0]]]])  # u < 0 → out of frame
    cam_depth = np.array([[10.0]])
    boxes = np.array([[[-100.0, 0.0, 100.0, 100.0]]])
    assert not _visibility_mask(joints_image, cam_depth, boxes, 400, 400).any()


def test_start_xy_override_positions_roots():
    start = np.array([[3.0, -4.0], [-2.0, 5.0]])
    s = generate_scene(n_subjects=2, n_frames=1, start_xy=start, seed=0)
    assert np.allclose(s.root_world[0, :, :2], start)


def test_stacked_subjects_trigger_occlusion():
    # Two subjects nearly collinear with the camera → the farther one's joints fall in the nearer
    # one's box and are occluded. n_frames=1 pins the layout (no rng velocity drift).
    start = np.array([[0.0, -8.0], [0.0, -7.6]])
    s = generate_scene(n_subjects=2, n_frames=1, start_xy=start, seed=0)
    depth = s.world_to_camera(s.root_world)[..., 2]  # (1, 2) forward depth
    far = int(np.argmax(depth[0]))
    assert not s.visibility[0, far].all()  # at least one joint of the farther subject occluded
    assert s.visibility.shape == s.joints_world.shape[:-1]
