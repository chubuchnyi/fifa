"""FieldCalibrator adapters — shared port contract + DLT/scoring/smoothing (FR-7, M1).

The real pitch-keypoint adapter is exercised with an *injected* stub backend, so its pure half
(normalized-DLT homography, reprojection scoring, last-good carry, temporal smoothing) is
verified with **no model, no cv2, no GPU** — the same AC-7 discipline the fakes follow.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from pitch3d.adapters.fakes import FakeFieldCalibrator
from pitch3d.adapters.models.calibration import (
    FrameKeypoints,
    KeypointBackend,
    KeypointFieldCalibrator,
    PitchKeypointBackend,
    _apply_homography,
    _confidence_from_error,
    _temporal_smooth,
    reprojection_error,
    solve_homography,
)
from pitch3d.core.ports.io import ClipRef
from pitch3d.core.ports.perception import FieldCalibrator
from pitch3d.core.scene.field import FieldCalibration

# Ground-truth image→world homography (mild perspective) and a non-degenerate landmark set.
_H_GT = np.array(
    [[0.050, 0.002, -30.0], [0.001, -0.050, 18.0], [2e-4, 1e-4, 1.0]], dtype=float
)
_WORLD = np.array(
    [[-30.0, -15.0], [30.0, -15.0], [30.0, 15.0], [-30.0, 15.0],
     [0.0, 0.0], [12.0, -6.0], [-8.0, 9.0]], dtype=float
)
#: Image points that map back to _WORLD under _H_GT (so solve should recover _H_GT).
_IMAGE = _apply_homography(np.linalg.inv(_H_GT), _WORLD)


def _clip(frames=(0, 1, 2), width=1280, height=720) -> ClipRef:
    return ClipRef(
        source_id="s", uri="x", frames=np.array(frames), width=width, height=height, fps=25.0
    )


class _StubKeypointBackend:
    """Returns canned (image↔world) landmark matches per frame — stands in for the model."""

    def __init__(self, per: dict[int, tuple]):
        self.per = per

    def detect_keypoints(self, clip: ClipRef) -> list[FrameKeypoints]:
        out = []
        for f in clip.frames.tolist():
            image_uv, world_xy = self.per[int(f)]
            out.append(FrameKeypoints(frame=int(f), image_uv=image_uv, world_xy=world_xy))
        return out


def _keypoints_calibrator(frames=(0, 1, 2), smooth_window=1) -> KeypointFieldCalibrator:
    per = {int(f): (_IMAGE, _WORLD) for f in frames}
    return KeypointFieldCalibrator(backend=_StubKeypointBackend(per), smooth_window=smooth_window)


# --- pure homography maths -----------------------------------------------------
def test_solve_homography_round_trips_known_H():
    h = solve_homography(_IMAGE, _WORLD)
    np.testing.assert_allclose(h, _H_GT, atol=1e-6)
    assert reprojection_error(h, _IMAGE, _WORLD) < 1e-6


def test_reprojection_error_grows_with_noise():
    h = solve_homography(_IMAGE, _WORLD)
    rng = np.random.default_rng(0)
    noisy = _IMAGE + rng.normal(scale=3.0, size=_IMAGE.shape)
    assert reprojection_error(h, noisy, _WORLD) > reprojection_error(h, _IMAGE, _WORLD)


def test_solve_homography_needs_four_points():
    with pytest.raises(ValueError, match="≥4"):
        solve_homography(_IMAGE[:3], _WORLD[:3])


def test_confidence_decreases_with_error():
    assert _confidence_from_error(0.0, 0.5) == 1.0
    assert _confidence_from_error(0.5, 0.5) == pytest.approx(0.5)
    assert _confidence_from_error(2.0, 0.5) < _confidence_from_error(0.5, 0.5)


def test_temporal_smoothing_box_averages_window():
    a, b, c = (np.eye(3) for _ in range(3))
    a[0, 2], b[0, 2], c[0, 2] = 0.0, 3.0, 0.0
    sm = _temporal_smooth(np.stack([a, b, c]), 3)
    assert sm[1][0, 2] == pytest.approx(1.0)   # mean(0, 3, 0)
    assert sm[0][0, 2] == pytest.approx(1.5)   # endpoint clamps to mean(0, 3)


# --- adapter behaviour ---------------------------------------------------------
@pytest.mark.parametrize(
    "make", [FakeFieldCalibrator, _keypoints_calibrator], ids=["fake", "keypoints"]
)
def test_calibrator_port_contract(make):
    cal = make()
    assert isinstance(cal, FieldCalibrator)
    clip = _clip()
    result = cal.calibrate(clip)
    assert isinstance(result, FieldCalibration)
    assert result.homographies.shape == (clip.n_frames, 3, 3)
    assert result.frames.tolist() == clip.frames.tolist()
    assert result.confidence.shape == (clip.n_frames,)
    assert np.all((result.confidence >= 0.0) & (result.confidence <= 1.0))


def test_keypoints_recovers_world_points_and_scores_high():
    result = _keypoints_calibrator(frames=(0,)).calibrate(_clip(frames=(0,)))
    np.testing.assert_allclose(result.image_to_world(0, _IMAGE), _WORLD, atol=1e-6)
    assert result.confidence[0] > 0.99  # exact fit → near-1 confidence


def test_under_detected_frame_carries_last_good_at_zero_confidence():
    per = {0: (_IMAGE, _WORLD), 1: (_IMAGE[:2], _WORLD[:2])}  # frame 1 has only 2 landmarks
    cal = KeypointFieldCalibrator(backend=_StubKeypointBackend(per))
    result = cal.calibrate(_clip(frames=(0, 1)))
    np.testing.assert_allclose(result.homographies[1], result.homographies[0], atol=1e-12)
    assert result.confidence[0] > 0.99 and result.confidence[1] == 0.0


def test_first_frame_under_detected_falls_back_to_identity():
    per = {0: (_IMAGE[:2], _WORLD[:2])}
    cal = KeypointFieldCalibrator(backend=_StubKeypointBackend(per))
    result = cal.calibrate(_clip(frames=(0,)))
    np.testing.assert_allclose(result.homographies[0], np.eye(3))
    assert result.confidence[0] == 0.0


def test_keypoint_provenance():
    info = KeypointFieldCalibrator(smooth_window=5).info()
    assert info.name == "PitchKeypoints+DLT"
    assert info.backend.value == "local"
    assert info.params["smooth_window"] == 5


def test_backends_satisfy_protocol():
    assert isinstance(_StubKeypointBackend({}), KeypointBackend)
    assert isinstance(PitchKeypointBackend(), KeypointBackend)  # structural: has detect_keypoints


def test_frame_keypoints_rejects_ragged():
    with pytest.raises(ValueError, match="ragged"):
        FrameKeypoints(frame=0, image_uv=np.zeros((3, 2)), world_xy=np.zeros((2, 2)))


@pytest.mark.skipif(
    importlib.util.find_spec("cv2") is not None, reason="cv extra installed"
)
def test_default_backend_without_extra_is_actionable():
    # No backend injected and the `cv` extra absent → a clear, install-pointing error.
    with pytest.raises(RuntimeError, match=r"pitch3d\[cv\]"):
        KeypointFieldCalibrator().calibrate(_clip(frames=(0,)))
