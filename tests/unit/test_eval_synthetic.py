"""Synthetic broadcast-soccer GT fixture (pitch3d.eval.synthetic).

Verifies the generator is deterministic and internally consistent: stored 2D joints equal
a re-projection of the world joints, the emitted GT homography inverts that projection on
the ground plane, feet sit on the pitch, and bboxes bound the subjects. These are the
fixtures the bake-off harness scores against, so their geometry must be exact.
"""

from __future__ import annotations

import numpy as np

from pitch3d.eval.synthetic import CANONICAL_SKELETON, JOINT_NAMES, generate_scene


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
