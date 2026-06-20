"""BallTracker adapters — shared port contract + TrackNet threshold/gap-fill (FR-9, M1).

The real TrackNet adapter is exercised with an *injected* stub backend, so its pure half
(score threshold, linear gap interpolation, honest zero-confidence fills, smoothing) is
verified with **no torch, no GPU** — the same AC-7 discipline the fakes follow.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from pitch3d.adapters.fakes import FakeBallTracker
from pitch3d.adapters.models.ball import (
    BallDetectionBackend,
    RawBallDetections,
    TrackNetBackend,
    TrackNetBallTracker,
    _interpolate_track,
    _smooth_xy,
)
from pitch3d.core.ports.io import ClipRef
from pitch3d.core.ports.perception import BallTracker
from pitch3d.core.scene.motion import Ball2DTrack


def _clip(frames=(0, 1, 2, 3, 4), width=1280, height=720) -> ClipRef:
    return ClipRef(
        source_id="s", uri="x", frames=np.array(frames), width=width, height=height, fps=25.0
    )


class _StubBallBackend:
    """Returns a canned per-frame ball track — stands in for the heatmap network."""

    def __init__(self, raw: RawBallDetections):
        self.raw = raw

    def detect_ball(self, clip: ClipRef) -> RawBallDetections:
        return self.raw


def _raw(points, scores, frames=(0, 1, 2, 3, 4)) -> RawBallDetections:
    return RawBallDetections(frames=np.array(frames), points_xy=points, scores=scores)


def _tracknet_with_stub(**kw) -> TrackNetBallTracker:
    pts = np.array([[0, 0], [10, 10], [20, 20], [30, 30], [40, 40]], dtype=float)
    return TrackNetBallTracker(backend=_StubBallBackend(_raw(pts, np.full(5, 0.9))), **kw)


# --- pure helpers --------------------------------------------------------------
def test_smooth_xy_box_averages_window():
    sm = _smooth_xy(np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 0.0]]), 3)
    assert sm[1, 0] == pytest.approx(1.0)   # mean(0, 3, 0)
    assert sm[0, 0] == pytest.approx(1.5)   # endpoint clamps to mean(0, 3)


def test_interpolate_fills_gaps_and_zeroes_their_confidence():
    frames = np.array([0, 1, 2, 3, 4])
    pts = np.array([[0, 0], [99, 99], [99, 99], [99, 99], [40, 40]], dtype=float)
    scores = np.array([0.9, 0.0, 0.0, 0.0, 0.9])  # only endpoints detected
    filled, conf = _interpolate_track(frames, pts, scores, threshold=0.5)
    np.testing.assert_allclose(filled, [[0, 0], [10, 10], [20, 20], [30, 30], [40, 40]])
    np.testing.assert_array_equal(conf, [0.9, 0.0, 0.0, 0.0, 0.9])


def test_interpolate_carries_endpoints_for_leading_and_trailing_gaps():
    frames = np.array([0, 1, 2, 3, 4])
    pts = np.array([[9, 9], [9, 9], [10, 10], [9, 9], [9, 9]], dtype=float)
    scores = np.array([0.1, 0.1, 0.8, 0.1, 0.1])  # only frame 2 detected
    filled, conf = _interpolate_track(frames, pts, scores, threshold=0.5)
    np.testing.assert_allclose(filled, [[10, 10]] * 5)             # carry the lone detection
    np.testing.assert_array_equal(conf, [0.0, 0.0, 0.8, 0.0, 0.0])


def test_interpolate_all_missed_returns_zero_confidence():
    frames = np.array([0, 1, 2])
    pts = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    filled, conf = _interpolate_track(frames, pts, np.zeros(3), threshold=0.5)
    np.testing.assert_array_equal(filled, pts)   # nothing to interpolate from → unchanged
    np.testing.assert_array_equal(conf, [0.0, 0.0, 0.0])


# --- adapter behaviour ---------------------------------------------------------
@pytest.mark.parametrize(
    "make", [FakeBallTracker, _tracknet_with_stub], ids=["fake", "tracknet"]
)
def test_ball_tracker_port_contract(make):
    blt = make()
    assert isinstance(blt, BallTracker)
    clip = _clip()
    track = blt.track_ball(clip)
    assert isinstance(track, Ball2DTrack)
    assert track.frames.tolist() == clip.frames.tolist()
    assert track.positions_2d.shape == (clip.n_frames, 2)
    assert track.confidence.shape == (clip.n_frames,)
    assert np.all((track.confidence >= 0.0) & (track.confidence <= 1.0))


def test_tracknet_threshold_drops_low_peaks_and_interpolates():
    pts = np.array([[0, 0], [99, 99], [20, 20], [99, 99], [40, 40]], dtype=float)
    scores = np.array([0.9, 0.2, 0.9, 0.2, 0.9])  # frames 1, 3 are below the 0.5 floor
    blt = TrackNetBallTracker(backend=_StubBallBackend(_raw(pts, scores)))
    track = blt.track_ball(_clip())
    np.testing.assert_allclose(track.positions_2d, [[0, 0], [10, 10], [20, 20], [30, 30], [40, 40]])
    np.testing.assert_array_equal(track.confidence, [0.9, 0.0, 0.9, 0.0, 0.9])


def test_provenance():
    info = TrackNetBallTracker(score_threshold=0.7, smooth_window=5).info()
    assert info.name == "TrackNetV3"
    assert info.backend.value == "local"
    assert info.params["score_threshold"] == 0.7
    assert info.params["smooth_window"] == 5


def test_backends_satisfy_protocol():
    stub = _StubBallBackend(_raw(np.zeros((1, 2)), np.ones(1), frames=(0,)))
    assert isinstance(stub, BallDetectionBackend)
    assert isinstance(TrackNetBackend(), BallDetectionBackend)  # structural: has detect_ball


def test_raw_ball_detections_rejects_ragged():
    with pytest.raises(ValueError, match="ragged"):
        RawBallDetections(
            frames=[0, 1, 2], points_xy=np.zeros((2, 2)), scores=np.ones(3)
        )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is not None, reason="ball extra installed"
)
def test_default_backend_without_extra_is_actionable():
    # No backend injected and the `ball` extra absent → a clear, install-pointing error.
    with pytest.raises(RuntimeError, match=r"pitch3d\[ball\]"):
        TrackNetBallTracker().track_ball(_clip())
