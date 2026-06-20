"""Composition root — ``default_ports`` threads the device/weights knobs into real adapters.

P2.1: the CLI/MCP entry points must be able to pick the inference device (``cpu`` for local
concept validation, ``cuda`` in production) and an RF-DETR weights path, *without* any of the
heavy perception extras installed — adapter construction stays lazy (the heavy import lives in
each backend's ``_load``), so this runs green with no torch/cv2/GPU (the AC-7 discipline).
"""

from __future__ import annotations

from pitch3d.adapters.models import (
    ByteTrackTracker,
    GVHMRPoseEstimator,
    KeypointFieldCalibrator,
    RFDETRDetector,
    TrackNetBallTracker,
)
from pitch3d.app.wiring import default_ports


def _all_real(**kw):
    return default_ports(
        detector="rfdetr", tracker="bytetrack", calibrator="keypoints",
        pose="gvhmr", ball="tracknet", **kw,
    )


def test_device_threads_into_every_real_adapter():
    ports = _all_real(device="cpu")
    assert isinstance(ports.detector, RFDETRDetector)
    assert isinstance(ports.tracker, ByteTrackTracker)
    assert isinstance(ports.calibrator, KeypointFieldCalibrator)
    assert isinstance(ports.pose, GVHMRPoseEstimator)
    assert isinstance(ports.ball, TrackNetBallTracker)
    assert ports.detector.device == "cpu"
    assert ports.tracker.device == "cpu"
    assert ports.calibrator.device == "cpu"
    assert ports.pose.device == "cpu"
    assert ports.ball.device == "cpu"


def test_device_default_is_the_cpu_validation_profile():
    # The composition root defaults to the local CPU profile even though each adapter dataclass
    # defaults to "cuda" (production intent) — the wiring is the seam that picks the deployment.
    assert _all_real().detector.device == "cpu"


def test_cuda_is_forwarded_when_asked():
    assert _all_real(device="cuda").pose.device == "cuda"


def test_detector_weights_path_is_forwarded():
    ports = default_ports(detector="rfdetr", detector_weights="/tmp/rfdetr.pth")
    assert ports.detector.weights == "/tmp/rfdetr.pth"


def test_detector_weights_default_to_none():
    assert default_ports(detector="rfdetr").detector.weights is None
