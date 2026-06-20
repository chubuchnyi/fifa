"""Real ML-model adapters (roadmap M0-6 / M1).

Each class names its intended backend + license and satisfies the port's ABC. **Detection is
wired** (:class:`RFDETRDetector`, behind the optional ``cv`` extra); the remaining classes are
honest stubs whose work methods raise ``NotImplementedError`` until their milestone. Keeping
every class importable with **no torch/cv2 at import time** means the hexagonal wiring,
provenance (``info()``), and tests are complete now. Swap a fake for the matching real adapter
one at a time; each must pass the same port test the fake does (see ``tests`` and ADR-0001).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ...core.ports.io import ClipRef, CropRef
from ...core.ports.perception import (
    BallTracker,
    Detections,
    FieldCalibrator,
    Tracker,
    Tracks,
)
from ...core.ports.pose import PoseEstimator
from ...core.ports.reconstruction import AvatarBuilder, EnvReconstructor
from ...core.scene.assets import RenderAssetRef, SynthViewRef
from ...core.scene.camera import CameraTrack
from ...core.scene.field import FieldCalibration
from ...core.scene.motion import Ball2DTrack, SubjectMotion
from ...core.scene.provenance import Backend, ModelInfo
from ...core.scene.subject import Subject
from .detection import RFDETRBackend, RFDETRDetector


def _todo(what: str) -> NotImplementedError:
    return NotImplementedError(
        f"{what} is not wired yet — install the `models` extra and implement it (roadmap M1). "
        "Use the matching fake in pitch3d.adapters.fakes for tests and the dry-run."
    )


@dataclass
class ByteTrackTracker(Tracker):
    """ByteTrack / BoT-SORT tracking + appearance team clustering (FR-6)."""

    def info(self) -> ModelInfo:
        return ModelInfo(name="ByteTrack+BoT-SORT", backend=Backend.LOCAL, license="MIT")

    def track(self, clip: ClipRef, detections: Detections) -> Tracks:
        raise _todo("ByteTrack tracking")


@dataclass
class KeypointFieldCalibrator(FieldCalibrator):
    """Pitch-keypoint model + ``cv2.findHomography`` + temporal smoothing (FR-7)."""

    def info(self) -> ModelInfo:
        return ModelInfo(name="PitchKeypoints+findHomography", backend=Backend.LOCAL)

    def calibrate(self, clip: ClipRef) -> FieldCalibration:
        raise _todo("field homography calibration")


@dataclass
class GVHMRPoseEstimator(PoseEstimator):
    """GVHMR / WHAM / TRAM SMPL-X HMR + PromptHMR-class re-fit (FR-8, FR-22c)."""

    def info(self) -> ModelInfo:
        return ModelInfo(name="GVHMR", backend=Backend.LOCAL, license="see upstream (non-commercial SMPL-X)")

    def estimate(self, clip: ClipRef, tracks: Tracks, calibration: FieldCalibration) -> dict[int, SubjectMotion]:
        raise _todo("GVHMR pose estimation")

    def refit(self, clip: ClipRef, motion: SubjectMotion, constraints: dict, frames: np.ndarray) -> SubjectMotion:
        raise _todo("constraint-guided pose re-fit")


@dataclass
class TrackNetBallTracker(BallTracker):
    """TrackNet-class 2D ball tracker; 3D lift stays in core (FR-9)."""

    def info(self) -> ModelInfo:
        return ModelInfo(name="TrackNetV3", backend=Backend.LOCAL)

    def track_ball(self, clip: ClipRef) -> Ball2DTrack:
        raise _todo("TrackNet ball tracking")


@dataclass
class SplatEnvReconstructor(EnvReconstructor):
    """3D Gaussian Splatting / NeRF stadium reconstruction (FR-11)."""

    def info(self) -> ModelInfo:
        return ModelInfo(name="3DGS", backend=Backend.LOCAL)

    def reconstruct(
        self, clip: ClipRef, camera: CameraTrack, synth_views: Sequence[SynthViewRef] | None = None
    ) -> RenderAssetRef:
        raise _todo("3DGS environment reconstruction")


@dataclass
class ApiAvatarBuilder(AvatarBuilder):
    """Generative (Rodin-class API) avatar builder (FR-12, strategy #2)."""

    def info(self) -> ModelInfo:
        return ModelInfo(name="Rodin", backend=Backend.API, est_cost_usd=0.0)

    def build(
        self, subject: Subject, ref_crops: Sequence[CropRef], synth_views: Sequence[SynthViewRef] | None = None
    ) -> RenderAssetRef:
        raise _todo("generative avatar building")


__all__ = [
    "ApiAvatarBuilder",
    "ByteTrackTracker",
    "GVHMRPoseEstimator",
    "KeypointFieldCalibrator",
    "RFDETRBackend",
    "RFDETRDetector",
    "SplatEnvReconstructor",
    "TrackNetBallTracker",
]
