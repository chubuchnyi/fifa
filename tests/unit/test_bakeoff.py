"""Bake-off driver + condition B (grounded placement).

Condition A isolates the pose net (the oracle → 0); condition B grounds the root from the bbox
foot point through the (perfect GT) homography, so its only extra cost is grounding — the A→B gap
is **Global-MPJPE only** (Local is root-relative, identical to A). Also smoke-tests the CLI driver's
candidate × condition grid + table on synthetic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

from pitch3d.eval.backends import GtOracleBackend, ZeroPoseBackend
from pitch3d.eval.harness import run_backend, run_backend_grounded, run_conditions
from pitch3d.eval.synthetic import CAMERA_VIEWS, generate_scene


def test_oracle_condition_a_reconstructs_gt():
    s = generate_scene(seed=4)
    grid = run_backend(s, GtOracleBackend(s))
    assert grid["global_mpjpe_m"] < 1e-9
    assert grid["local_mpjpe_m"] < 1e-9


def test_condition_b_local_equals_a_global_is_grounding_floor():
    # B differs from A only in the root source → Local identical (root-relative), Global > A.
    s = generate_scene(seed=4)
    calib = s.field_calibration()
    a = run_backend(s, GtOracleBackend(s))
    b = run_backend_grounded(s, GtOracleBackend(s), calib)
    assert abs(b["local_mpjpe_m"] - a["local_mpjpe_m"]) < 1e-9
    assert b["local_mpjpe_m"] < 1e-9
    assert np.isfinite(b["global_mpjpe_m"])
    assert b["global_mpjpe_m"] > a["global_mpjpe_m"] + 1e-4  # foot-point grounding adds finite cost


def test_run_conditions_skips_b_without_calibration():
    s = generate_scene(seed=4)
    conds = run_conditions(s, GtOracleBackend(s))
    assert conds["B"] is None
    assert conds["A"] is not None
    assert conds["A"]["global_mpjpe_m"] < 1e-9


def test_zero_pose_floor_is_finite_and_positive():
    s = generate_scene(seed=4)
    grid = run_backend(s, ZeroPoseBackend(s))
    assert np.isfinite(grid["global_mpjpe_m"])
    assert grid["local_mpjpe_m"] > 0.0


def test_oracle_scores_zero_from_every_camera_view():
    # A perfect backend must reconstruct the GT from any broadcast viewpoint — this is what the
    # camera sweep buys: it hardens condition-A *placement*, independent of the pose net.
    for view in CAMERA_VIEWS.values():
        s = generate_scene(seed=4, camera=view)
        grid = run_backend(s, GtOracleBackend(s))
        assert grid["global_mpjpe_m"] < 1e-9
        assert grid["local_mpjpe_m"] < 1e-9


def _occluded_scene():
    # Two subjects stacked on the camera ray (n_frames=1 pins the layout) → real occlusion.
    return generate_scene(
        n_subjects=2, n_frames=1, seed=0, start_xy=np.array([[0.0, -8.0], [0.0, -7.6]])
    )


def test_visible_only_changes_score_under_occlusion():
    s = _occluded_scene()
    assert not s.visibility.all()  # the fixture must actually occlude something
    full = run_backend(s, ZeroPoseBackend(s))
    masked = run_backend(s, ZeroPoseBackend(s), visible_only=True)
    assert np.isfinite(masked["global_mpjpe_m"])
    assert not np.isclose(full["global_mpjpe_m"], masked["global_mpjpe_m"])


def test_run_conditions_threads_visible_only():
    s = _occluded_scene()
    calib = s.field_calibration()
    conds = run_conditions(s, ZeroPoseBackend(s), calibration=calib, visible_only=True)
    assert conds["A"]["global_mpjpe_m"] == run_backend(
        s, ZeroPoseBackend(s), visible_only=True
    )["global_mpjpe_m"]
    assert conds["B"]["global_mpjpe_m"] == run_backend_grounded(
        s, ZeroPoseBackend(s), calib, visible_only=True
    )["global_mpjpe_m"]


def _load_driver() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_bakeoff.py"
    spec = importlib.util.spec_from_file_location("run_bakeoff", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_driver_grid_and_table_on_synthetic():
    drv = _load_driver()
    s = generate_scene(seed=2, n_subjects=2, n_frames=4)
    grid = drv.run_bakeoff(s)
    assert set(grid) == {"gt-oracle", "zero-pose"}
    assert grid["gt-oracle"]["A"]["global_mpjpe_m"] < 1e-9
    assert grid["gt-oracle"]["B"]["global_mpjpe_m"] > 0.0
    table = drv.format_table(grid)
    assert "gt-oracle" in table and "zero-pose" in table and "Global" in table


def test_driver_main_runs(capsys):
    drv = _load_driver()
    assert drv.main(["--seed", "1", "--subjects", "2", "--frames", "3"]) == 0
    out = capsys.readouterr().out
    assert "pose bake-off" in out
    assert "gt-oracle" in out
