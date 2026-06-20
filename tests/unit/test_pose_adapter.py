"""PoseEstimator adapters — shared port contract + GVHMR grounding/refit (FR-8, M1).

The real GVHMR adapter is exercised with an *injected* stub backend, so its pure half
(homography root-grounding, articulation pass-through, geometric refit, root smoothing) is
verified with **no torch, no GPU** — the same AC-7 discipline the fakes follow.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from pitch3d.adapters.fakes import FakePoseEstimator
from pitch3d.adapters.models.pose import (
    GVHMRBackend,
    GVHMRPoseEstimator,
    HMRBackend,
    RawBodyMotion,
    _smooth_path,
)
from pitch3d.core.ports.io import ClipRef
from pitch3d.core.ports.perception import Tracklet, Tracks
from pitch3d.core.ports.pose import PoseEstimator
from pitch3d.core.scene.field import FieldCalibration
from pitch3d.core.scene.motion import SubjectMotion

_J = 21  # SMPL-X body joints


def _clip(frames=(0, 1, 2), width=1280, height=720) -> ClipRef:
    return ClipRef(
        source_id="s", uri="x", frames=np.array(frames), width=width, height=height, fps=25.0
    )


def _calibration(frames=(0, 1, 2)) -> FieldCalibration:
    """Identity homography per frame → ``image_to_world`` is a no-op (easy grounding asserts)."""
    t = len(frames)
    return FieldCalibration(
        homographies=np.stack([np.eye(3)] * t),
        frames=np.array(frames),
        confidence=np.ones(t),
    )


def _boxes(offset: float) -> np.ndarray:
    return np.array(
        [[10 + offset, 20, 30 + offset, 60], [12 + offset, 22, 32 + offset, 62],
         [14 + offset, 24, 34 + offset, 64]], dtype=float,
    )


def _tracks() -> Tracks:
    """Player (id 0), referee (id 1), and a ball (id 2) the estimator must skip."""
    return Tracks(
        tracklets=[
            Tracklet(track_id=0, frames=(0, 1, 2), bboxes_xyxy=_boxes(0), cls="player"),
            Tracklet(track_id=1, frames=(0, 1, 2), bboxes_xyxy=_boxes(100), cls="referee"),
            Tracklet(track_id=2, frames=(0, 1, 2), bboxes_xyxy=_boxes(5), cls="ball"),
        ]
    )


def _raw(track_id: int, frames=(0, 1, 2)) -> RawBodyMotion:
    """Canned camera-space articulation with distinct, checkable per-row values."""
    frames = np.asarray(frames)
    t = frames.shape[0]
    return RawBodyMotion(
        track_id=track_id,
        frames=frames,
        global_orient=(np.arange(t * 3, dtype=float).reshape(t, 3) + track_id) * 0.01,
        body_pose=np.arange(t * _J * 3, dtype=float).reshape(t, _J, 3),
        betas=np.full(10, float(track_id)),
    )


class _StubHMRBackend:
    """Returns canned per-subject articulation — stands in for the HMR network."""

    def __init__(self, raws: dict[int, RawBodyMotion]):
        self.raws = raws

    def estimate_bodies(self, clip: ClipRef, tracks: Tracks) -> dict[int, RawBodyMotion]:
        return self.raws


def _gvhmr_with_stub(**kw) -> GVHMRPoseEstimator:
    return GVHMRPoseEstimator(backend=_StubHMRBackend({0: _raw(0), 1: _raw(1)}), **kw)


# --- pure helpers --------------------------------------------------------------
def test_smooth_path_box_averages_window():
    sm = _smooth_path(np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 0.0]]), 3)
    assert sm[1, 0] == pytest.approx(1.0)   # mean(0, 3, 0)
    assert sm[0, 0] == pytest.approx(1.5)   # endpoint clamps to mean(0, 3)


# --- adapter behaviour ---------------------------------------------------------
@pytest.mark.parametrize(
    "make", [FakePoseEstimator, _gvhmr_with_stub], ids=["fake", "gvhmr"]
)
def test_pose_estimator_port_contract(make):
    pse = make()
    assert isinstance(pse, PoseEstimator)
    out = pse.estimate(_clip(), _tracks(), _calibration())
    assert set(out.keys()) == {0, 1}  # ball (id 2) is skipped — it has its own BallTracker
    for motion in out.values():
        assert isinstance(motion, SubjectMotion)
        assert motion.pose.frames.tolist() == [0, 1, 2]
        assert motion.pose.transl.shape == (3, 3)
        assert motion.pose.global_orient.shape == (3, 3)
        assert motion.pose.body_pose.shape[0] == 3

    before = float(out[0].pose.transl[0, 2])
    refined = pse.refit(_clip(), out[0], {"root_z_nudge": 0.1}, np.array([0]))
    assert refined is not out[0]
    assert out[0].pose.transl[0, 2] == before               # input untouched
    assert refined.pose.transl[0, 2] == pytest.approx(before + 0.1)


def test_estimate_grounds_root_and_preserves_articulation():
    out = _gvhmr_with_stub().estimate(_clip(), _tracks(), _calibration())
    # Articulation is passed through untouched (the network owns it, FR-8).
    np.testing.assert_array_equal(out[0].pose.global_orient, _raw(0).global_orient)
    np.testing.assert_array_equal(out[0].pose.body_pose, _raw(0).body_pose)
    np.testing.assert_array_equal(out[1].shape.betas, np.full(10, 1.0))  # per-subject shape
    # Root is grounded: foot point (bbox x-mid, bottom y) → world via identity homography.
    np.testing.assert_allclose(out[0].pose.transl[:, :2], [[20, 60], [22, 62], [24, 64]])
    np.testing.assert_allclose(out[0].pose.transl[:, 2], 0.92)  # pelvis-height Z anchor (R-4)


def test_refit_applies_constraints_on_selected_frames_only():
    motion = _gvhmr_with_stub().estimate(_clip(), _tracks(), _calibration())[0]
    z_before = motion.pose.transl[:, 2].copy()
    refined = _gvhmr_with_stub().refit(_clip(), motion, {"root_z_nudge": 0.5}, np.array([1]))
    assert refined is not motion
    np.testing.assert_array_equal(motion.pose.transl[:, 2], z_before)  # original unmutated
    assert refined.pose.transl[1, 2] == pytest.approx(z_before[1] + 0.5)  # only frame 1 nudged
    np.testing.assert_array_equal(refined.pose.transl[[0, 2], 2], z_before[[0, 2]])


def test_refit_honours_lock_floor_and_relax():
    pse = _gvhmr_with_stub()
    motion = pse.estimate(_clip(), _tracks(), _calibration())[0]
    refined = pse.refit(
        _clip(), motion,
        {"root_xy": [5.0, 6.0], "foot_floor": 1.5, "relax_to_rest": 0.0},
        np.array([0, 1, 2]),
    )
    np.testing.assert_allclose(refined.pose.transl[:, :2], [[5.0, 6.0]] * 3)  # XY locked
    np.testing.assert_allclose(refined.pose.transl[:, 2], 1.5)               # floor raised Z
    np.testing.assert_array_equal(refined.pose.body_pose, 0.0)                # relaxed to rest


def test_estimate_raises_when_backend_frames_miss_tracklet():
    short = GVHMRPoseEstimator(backend=_StubHMRBackend({0: _raw(0, frames=(0, 1)), 1: _raw(1)}))
    with pytest.raises(ValueError, match="do not cover"):
        short.estimate(_clip(), _tracks(), _calibration())


def test_provenance():
    info = GVHMRPoseEstimator(pelvis_height_m=1.0, device="cpu").info()
    assert info.name == "GVHMR"
    assert info.backend.value == "local"
    assert "non-commercial" in info.license
    assert info.params["pelvis_height_m"] == 1.0 and info.params["device"] == "cpu"


def test_backends_satisfy_protocol():
    assert isinstance(_StubHMRBackend({}), HMRBackend)
    assert isinstance(GVHMRBackend(), HMRBackend)  # structural: has estimate_bodies


def test_raw_body_motion_rejects_ragged():
    with pytest.raises(ValueError, match="ragged"):
        RawBodyMotion(
            track_id=0, frames=[0, 1, 2], global_orient=np.zeros((2, 3)),
            body_pose=np.zeros((3, _J, 3)), betas=np.zeros(10),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is not None, reason="hmr extra installed"
)
def test_default_backend_without_extra_is_actionable():
    # No backend injected and the `hmr` extra absent → a clear, install-pointing error.
    with pytest.raises(RuntimeError, match=r"pitch3d\[hmr\]"):
        GVHMRPoseEstimator().estimate(_clip(), _tracks(), _calibration())
