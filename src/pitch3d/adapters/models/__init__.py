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

from ...core.ports.io import ClipRef, CropRef
from ...core.ports.reconstruction import AvatarBuilder, EnvReconstructor
from ...core.scene.assets import RenderAssetRef, SynthViewRef
from ...core.scene.camera import CameraTrack
from ...core.scene.provenance import Backend, ModelInfo
from ...core.scene.subject import Subject
from .avatar import AvatarMeshBackend, SmplxTextureBackend, TexturedSmplxAvatarBuilder
from .ball import TrackNetBackend, TrackNetBallTracker
from .calibration import (
    CameraModuleFieldCalibrator,
    KeypointFieldCalibrator,
    PitchKeypointBackend,
)
from .detection import DETECTOR_CLASS_MAPS, RFDETRBackend, RFDETRDetector
from .pose import GVHMRBackend, GVHMRPoseEstimator
from .tracking import ByteTrackBackend, ByteTrackTracker


def _todo(what: str) -> NotImplementedError:
    return NotImplementedError(
        f"{what} is not wired yet — install the `models` extra and implement it (roadmap M1). "
        "Use the matching fake in pitch3d.adapters.fakes for tests and the dry-run."
    )


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
    "DETECTOR_CLASS_MAPS",
    "ApiAvatarBuilder",
    "AvatarMeshBackend",
    "ByteTrackBackend",
    "ByteTrackTracker",
    "CameraModuleFieldCalibrator",
    "GVHMRBackend",
    "GVHMRPoseEstimator",
    "KeypointFieldCalibrator",
    "PitchKeypointBackend",
    "RFDETRBackend",
    "RFDETRDetector",
    "SmplxTextureBackend",
    "SplatEnvReconstructor",
    "TexturedSmplxAvatarBuilder",
    "TrackNetBackend",
    "TrackNetBallTracker",
]
