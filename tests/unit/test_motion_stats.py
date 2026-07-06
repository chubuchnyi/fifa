"""T0 probe: per-category stat helpers + config wiring in ``scripts/motion_stats.py``.

The point of these tests is to pin the SEMANTICS the operator reads off the
probe. If a rate helper silently starts returning componentwise diffs instead
of group-metric angular velocity, a real scene will look fine while the numbers
are lying.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from motion_stats import (  # noqa: E402
    _axis_angle_rates,
    foot_z_stats,
    joint_stats,
    orient_stats,
    xy_stats,
)


def test_axis_angle_rates_group_metric_matches_true_angle():
    """Two axis-angle rotations far apart must give the true group angle, not the
    componentwise diff. Componentwise on ``[π/2, 0, 0]`` → ``[0, π/2, 0]`` gives
    |Δ| = π/√2 (falsely 127°); the correct group delta is 120°."""
    rot = np.array([[np.pi / 2, 0, 0], [0, np.pi / 2, 0]])
    dt = np.array([1.0])
    rate = _axis_angle_rates(rot, dt)
    assert rate.shape == (1,)
    # Correct group answer for 90° X then 90° Y composition is 120° (2·arccos(1/2)).
    assert 119.9 < rate[0] < 120.1, rate[0]
    # Wrong componentwise answer would be degrees(pi/sqrt(2)) ≈ 127.3° — reject that.
    assert not (127.0 < rate[0] < 128.0), rate[0]


def test_axis_angle_rates_zero_when_identical():
    rot = np.tile(np.array([[0.3, -0.4, 0.1]]), (5, 1))
    rate = _axis_angle_rates(rot, np.full(4, 1.0))
    assert np.all(rate < 1e-6)


def test_xy_stats_flags_speed_and_accel_violations_and_teleports():
    fps = 30.0
    # 6 frames: constant 15 m/s (over 10.5 limit) + one big teleport step
    frames = np.arange(6, dtype=int)
    xy = np.zeros((6, 3))
    xy[:, 0] = [0.0, 0.5, 1.0, 1.5, 20.0, 20.5]  # step of 18.5 m at f3→f4
    st = xy_stats(frames, xy, fps=fps, turn_min_speed=2.0,
                  max_speed=10.5, max_accel=8.0, teleport_factor=2.0)
    # step 18.5m over 1/30s = 555 m/s — well over 10.5 AND over 2×10.5=21 → teleport
    assert st["sp_max"] > 500
    assert st["viol_sp"] >= 1
    assert st["viol_ac"] >= 1
    assert st["teleport_intervals"] >= 1


def test_foot_z_stats_detects_helicopter_plateau():
    """Constant Z at pelvis_height with zero variance = the user's helicopter symptom.
    z_max - z_min ≈ 0 is the diagnostic."""
    transl = np.zeros((10, 3))
    transl[:, 2] = 0.92
    st = foot_z_stats(np.arange(10), transl, floor_m=0.0, hover_m=0.30)
    assert st["z_min"] == pytest.approx(0.92)
    assert st["z_max"] == pytest.approx(0.92)
    assert st["z_max"] - st["z_min"] < 1e-9  # THE symptom
    assert st["hover_frac"] == 0.0            # not "over the extra hover margin"
    assert st["below_floor_frac"] == 0.0


def test_foot_z_stats_flags_below_floor_and_hover_over_margin():
    transl = np.zeros((10, 3))
    transl[:, 2] = np.array([0.90, 0.91, 0.92, 0.92, 0.92, 0.92, 1.30, 1.30, 1.30, 1.30])
    st = foot_z_stats(np.arange(10), transl, floor_m=0.0, hover_m=0.30)
    # >0.30 above pelvis_height (0.92 + 0.30 = 1.22) → hover
    assert st["hover_frac"] == pytest.approx(0.4)
    # dip a couple frames below the floor
    transl[:3, 2] = -0.05
    st = foot_z_stats(np.arange(10), transl, floor_m=0.0, hover_m=0.30)
    assert st["below_floor_frac"] == pytest.approx(0.3)


def test_orient_stats_measures_true_rate():
    """Rotate 60° per frame around Z at 30 fps → 1800°/s."""
    fps = 30.0
    frames = np.arange(6, dtype=int)
    # 6 rotations, 60° step around Z axis
    theta = np.radians(60) * np.arange(6)
    orient = np.zeros((6, 3))
    orient[:, 2] = theta
    st = orient_stats(frames, orient, fps=fps, flag_dps=720.0)
    assert 1790 < st["orient_max_dps"] < 1810, st["orient_max_dps"]
    assert st["orient_viol"] == 5   # every interval over 720


def test_orient_stats_zero_when_static():
    fps = 30.0
    frames = np.arange(6, dtype=int)
    orient = np.tile(np.array([[0.1, 0.2, 0.3]]), (6, 1))
    st = orient_stats(frames, orient, fps=fps, flag_dps=720.0)
    assert st["orient_max_dps"] < 1e-6
    assert st["orient_viol"] == 0


def test_joint_stats_finds_hottest_joint():
    """K=3 joints: joint 1 rotates fast; joints 0 & 2 stay still.
    The probe must surface joint 1 as the hottest."""
    fps = 30.0
    T, K = 6, 3
    body = np.zeros((T, K, 3))
    body[:, 1, 2] = np.radians(60) * np.arange(T)  # joint 1 rotates 60° per frame
    st = joint_stats(np.arange(T), body, fps=fps, flag_dps=600.0)
    assert st["hottest_joint_idx"] == 1
    assert 1790 < st["hottest_joint_max_dps"] < 1810
    assert st["joint_viol_samples"] == 5


def test_joint_stats_empty_body_pose_is_safe():
    st = joint_stats(np.arange(3), np.zeros((3, 0, 3)), fps=30.0, flag_dps=600.0)
    # no joints → no rates → helpers return zeros without crashing
    assert st.get("joint_max_dps", 0.0) == 0.0


def test_xy_stats_short_track_returns_only_n():
    st = xy_stats(np.arange(2), np.zeros((2, 3)), fps=30.0, turn_min_speed=2.0,
                  max_speed=10.5, max_accel=8.0, teleport_factor=2.0)
    assert "sp_max" not in st  # too short to compute
    assert st["n"] == 2
