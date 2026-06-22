"""MPJPE metrics + condition-A placement (pitch3d.eval.metrics / .harness).

The metric's defining property is that a pure world translation is fully charged to GLOBAL
and invisible to LOCAL (root-relative). We also round-trip a 'perfect' camera-space
prediction through the GT camera back to world (condition A) and confirm it scores ~0.
"""

from __future__ import annotations

import numpy as np

from pitch3d.eval.harness import evaluate, place_under_gt_camera
from pitch3d.eval.metrics import mpjpe_global, mpjpe_local
from pitch3d.eval.synthetic import generate_scene


def test_translation_is_global_only():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(4, 15, 3))
    pred = gt + np.array([1.0, 0.0, 0.0])
    assert abs(mpjpe_global(pred, gt) - 1.0) < 1e-9
    assert mpjpe_local(pred, gt) < 1e-9


def test_perfect_prediction_is_zero():
    rng = np.random.default_rng(1)
    gt = rng.normal(size=(3, 5, 16, 3))
    assert mpjpe_global(gt, gt) == 0.0
    assert mpjpe_local(gt, gt) == 0.0


def test_local_invariant_to_global_shift():
    rng = np.random.default_rng(2)
    gt = rng.normal(size=(2, 16, 3))
    pred = rng.normal(size=(2, 16, 3))
    shift = np.array([3.0, -2.0, 1.0])
    assert abs(mpjpe_local(pred + shift, gt) - mpjpe_local(pred, gt)) < 1e-9
    assert mpjpe_global(pred + shift, gt) > mpjpe_global(pred, gt)


def test_place_under_gt_camera_round_trip():
    s = generate_scene(seed=4)
    cam = s.world_to_camera(s.joints_world)  # a 'perfect' camera-space prediction
    placed = place_under_gt_camera(s, cam)
    assert np.allclose(placed, s.joints_world, atol=1e-9)


def test_evaluate_grid_on_perfect_prediction():
    s = generate_scene(seed=5)
    placed = place_under_gt_camera(s, s.world_to_camera(s.joints_world))
    grid = evaluate(placed, s.joints_world)
    assert set(grid) == {"global_mpjpe_m", "local_mpjpe_m"}
    assert grid["global_mpjpe_m"] < 1e-9
    assert grid["local_mpjpe_m"] < 1e-9


def test_zero_pose_baseline_is_finite_and_positive():
    s = generate_scene(seed=6)
    j = s.joints_world.shape[2]
    pred = np.repeat(s.root_world[:, :, None, :], j, axis=2)  # all joints collapsed to pelvis
    grid = evaluate(pred, s.joints_world)
    assert np.isfinite(grid["global_mpjpe_m"])
    assert grid["local_mpjpe_m"] > 0.0
