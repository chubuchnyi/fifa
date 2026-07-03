"""The virtual operator must aim fixed mounts at the action and keep it in frame."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.scene.cameras import (
    BOWL_APRON,
    BOWL_RISE,
    BOWL_ROWS,
    BOWL_RUN,
    OperatorRig,
    action_track,
    plan_virtual_cameras,
    project_normalized,
    smooth_moving_average,
)
from pitch3d.core.scene.units import FieldDimensions


@pytest.fixture()
def cluster():
    """A jittery 8-player cluster drifting through the -x half, with tracking gaps + a ball."""
    rng = np.random.default_rng(7)
    n_frames, n_sub = 60, 8
    t = np.linspace(0.0, 1.0, n_frames)
    centre = np.stack([-30.0 + 18.0 * t, -12.0 + 20.0 * t, np.zeros(n_frames)], axis=1)
    offsets = rng.uniform(-8.0, 8.0, (1, n_sub, 2)) + rng.normal(0.0, 0.35, (n_frames, n_sub, 2))
    roots = np.repeat(centre[:, None, :], n_sub, axis=1)
    roots[..., :2] += offsets
    roots[..., 2] = 0.95
    roots[10:14, 2] = np.nan  # a subject drops out for a few frames
    ball = centre.copy()
    ball[:, 2] = 0.2 + 0.4 * np.abs(np.sin(8.0 * t))
    frames = np.arange(n_frames)
    return roots, ball, frames


def test_smoother_is_zero_phase_on_a_constant(cluster):
    const = np.full((30, 3), 4.5)
    assert np.allclose(smooth_moving_average(const, 9), const)


def test_mounts_are_fixed_and_inside_the_bowl(cluster):
    roots, ball, frames = cluster
    dims = FieldDimensions()
    tracks = plan_virtual_cameras(roots, ball, frames)
    bowl_outer_x = dims.length / 2.0 + BOWL_APRON + BOWL_ROWS * BOWL_RUN
    bowl_outer_y = dims.width / 2.0 + BOWL_APRON + BOWL_ROWS * BOWL_RUN
    bowl_top = BOWL_ROWS * BOWL_RISE
    for name in ("broadcast", "sideline", "goal"):
        pos = tracks[name].position
        assert pos.shape == (3,), f"{name} mount must be a single fixed point"
        assert abs(pos[0]) < bowl_outer_x and abs(pos[1]) < bowl_outer_y, name
        assert 0.0 < pos[2] <= bowl_top, f"{name} sits above the bowl rim"


def test_look_tracks_the_action_and_holds_through_gaps(cluster):
    roots, ball, frames = cluster
    centroid, _ = action_track(roots, ball)
    tracks = plan_virtual_cameras(roots, ball, frames)
    look = tracks["broadcast"].look_at
    assert not np.isnan(look).any()
    assert np.linalg.norm(look[:, :2] - centroid[:, :2], axis=1).max() < 1e-6


def test_zoom_is_smooth(cluster):
    roots, ball, frames = cluster
    fov = plan_virtual_cameras(roots, ball, frames)["broadcast"].fov_x_deg
    assert np.abs(np.diff(fov)).max() < 1.0, "zoom must not pump frame-to-frame"
    assert fov.min() >= 8.0 and fov.max() <= 63.0


def test_action_fits_in_frame_for_all_tracking_cameras(cluster):
    roots, ball, frames = cluster
    rig = OperatorRig()
    tracks = plan_virtual_cameras(roots, ball, frames, rig)
    heads = roots.copy()
    heads[..., 2] = np.nan_to_num(heads[..., 2], nan=0.95) + 0.9
    for name in ("broadcast", "sideline", "goal"):
        track = tracks[name]
        for row in range(0, len(frames), 7):
            pts = np.concatenate([roots[row], heads[row], ball[None, row]], axis=0)
            pts = pts[~np.isnan(pts[:, 2])]
            uv = project_normalized(track, row, pts, rig.aspect)
            assert np.nanmax(np.abs(uv)) <= 1.0, f"{name} crops the action at row {row}"


def test_far_straggler_does_not_blow_up_the_zoom(cluster):
    roots, ball, frames = cluster
    with_keeper = np.concatenate([roots, np.full((len(frames), 1, 3), np.nan)], axis=1)
    with_keeper[:, -1] = (50.0, 0.0, 0.95)  # a goalkeeper 60+ m from the action
    fov_tight = plan_virtual_cameras(roots, ball, frames)["broadcast"].fov_x_deg
    fov_keeper = plan_virtual_cameras(with_keeper, ball, frames)["broadcast"].fov_x_deg
    assert np.abs(fov_keeper - fov_tight).max() < 6.0, "one straggler must not zoom the shot out"


def test_goal_camera_sits_behind_the_action_half(cluster):
    roots, ball, frames = cluster
    tracks = plan_virtual_cameras(roots, ball, frames)
    assert tracks["goal"].position[0] < 0.0, "action lives in the -x half"


def test_top_frames_the_whole_pitch(cluster):
    roots, ball, frames = cluster
    rig = OperatorRig()
    track = plan_virtual_cameras(roots, ball, frames, rig)["top"]
    dims = FieldDimensions()
    corners = np.array([
        [sx * dims.length / 2.0, sy * dims.width / 2.0, 0.0]
        for sx in (-1, 1) for sy in (-1, 1)
    ])
    uv = project_normalized(track, 0, corners, rig.aspect)
    assert np.abs(uv).max() <= 1.0


def test_all_absent_frames_raise(cluster):
    roots, _, frames = cluster
    empty = np.full_like(roots, np.nan)
    with pytest.raises(ValueError):
        action_track(empty, None)
