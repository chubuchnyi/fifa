"""Bounded limited-orbit camera — the prescribed seam-A trajectory (M2-5, R-14/R-15).

These pin the invariants the ViewSynthesizer relies on: the track is *prescribed* (not a
measurement), it sweeps a genuine orbit (constant distance to the action centroid), the azimuth
arc is *bounded* by the request and *hard-capped* at 45° (seam A is a moderate re-aim, never a
free-viewpoint orbit), and it crosses the source framing at the midpoint.
"""

from __future__ import annotations

import numpy as np

from pitch3d.core.agent import action_centroid, bounded_orbit_camera
from pitch3d.core.correction.rotations import axis_angle_to_matrix, quat_to_axis_angle
from pitch3d.core.scene.scene import Scene

_BASE_AZIMUTH = -np.pi / 2.0  # BROADCAST sits behind the action (−Y) ⇒ atan2(−r, 0)


def _scene() -> Scene:
    return Scene(id="s", episode_id="e", source_id="src")


def _eye(cam, k: int) -> np.ndarray:
    """Recover the world-space camera centre from world→camera ``(quat, t)`` (eye = −Rᵀ t)."""
    rot = axis_angle_to_matrix(quat_to_axis_angle(cam.rotation_quat[k][None, :]))[0]
    return -rot.T @ cam.translation[k]


def _azimuth_deviations(cam, target: np.ndarray) -> np.ndarray:
    devs = []
    for k in range(cam.frames.shape[0]):
        e = _eye(cam, k)
        ang = np.arctan2(e[1] - target[1], e[0] - target[0])
        devs.append((ang - _BASE_AZIMUTH + np.pi) % (2 * np.pi) - np.pi)  # wrap to (−π, π]
    return np.asarray(devs)


def test_track_is_prescribed_not_estimated():
    cam = bounded_orbit_camera(_scene(), np.arange(8))
    assert cam.estimated is False


def test_shapes_follow_frames_and_are_finite():
    frames = np.arange(6)
    cam = bounded_orbit_camera(_scene(), frames)
    np.testing.assert_array_equal(cam.frames, frames)
    assert cam.rotation_quat.shape == (6, 4)
    assert cam.translation.shape == (6, 3)
    assert np.isfinite(cam.rotation_quat).all() and np.isfinite(cam.translation).all()


def test_is_an_orbit_constant_distance_to_target():
    scene = _scene()
    cam = bounded_orbit_camera(scene, np.arange(10), max_deviation_deg=20.0)
    target = action_centroid(scene)
    dists = [np.linalg.norm(_eye(cam, k) - target) for k in range(10)]
    np.testing.assert_allclose(dists, dists[0], rtol=1e-6)  # radius preserved ⇒ orbit, not dolly


def test_azimuth_arc_is_bounded_by_the_request():
    scene = _scene()
    cam = bounded_orbit_camera(scene, np.arange(9), max_deviation_deg=15.0)
    devs = _azimuth_deviations(cam, action_centroid(scene))
    assert np.max(np.abs(devs)) <= np.radians(15.0) + 1e-9


def test_sweep_is_hard_capped_at_45_degrees():
    # R-15: even an absurd request stays a moderate re-aim, never free-viewpoint.
    scene = _scene()
    cam = bounded_orbit_camera(scene, np.arange(9), max_deviation_deg=90.0)
    devs = _azimuth_deviations(cam, action_centroid(scene))
    assert np.max(np.abs(devs)) <= np.radians(45.0) + 1e-9


def test_midpoint_crosses_the_source_framing():
    scene = _scene()
    cam = bounded_orbit_camera(scene, np.arange(7), max_deviation_deg=20.0)  # odd ⇒ exact centre
    devs = _azimuth_deviations(cam, action_centroid(scene))
    assert abs(devs[3]) < 1e-6


def test_empty_frames_yield_an_empty_track():
    cam = bounded_orbit_camera(_scene(), np.empty(0, dtype=int))
    assert cam.frames.shape == (0,)
    assert cam.rotation_quat.shape == (0, 4)
    assert cam.estimated is False
