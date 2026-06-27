"""Photoreal asset builders: environment reconstruction and avatar building.

Both can optionally consume **ViewSynthesizer seam-B** synthesized views as extra
(pseudo-multi-)view input, turning the mono clip into something multi-view-like to
raise quality (ADR-0007). They return :class:`RenderAssetRef` pointers (the heavy data
lives on disk, addressed by the cache).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from ..scene.assets import RenderAssetRef, SynthViewRef
from ..scene.camera import CameraTrack
from ..scene.subject import Subject
from .base import ModelProvider
from .io import ClipRef, CropRef


class EnvReconstructor(ModelProvider):
    """Reconstructs the stadium/pitch environment (FR-11).

    Default: 3D Gaussian Splatting / NeRF driven by camera motion; a generative stadium
    is the fallback when motion is insufficient (R-8). Seam-B synth views may be supplied
    to compensate for the single real viewpoint.
    """

    @abstractmethod
    def reconstruct(
        self,
        clip: ClipRef,
        camera: CameraTrack,
        synth_views: Sequence[SynthViewRef] | None = None,
    ) -> RenderAssetRef:
        """Build an environment render asset (splats/NeRF) and return a ref."""
        raise NotImplementedError


class AvatarBuilder(ModelProvider):
    """Builds a photoreal avatar for one subject (FR-12).

    Three combinable strategies (selected via the returned asset kind): textured SMPL-X
    (#1, MVP base), generative avatar (#2, Rodin-class API), per-subject Gaussian avatar
    (#3). Seam-B inpainted views may be supplied to fill unseen sides (R-1, FR-31).
    """

    @abstractmethod
    def build(
        self,
        subject: Subject,
        ref_crops: Sequence[CropRef],
        synth_views: Sequence[SynthViewRef] | None = None,
        *,
        camera: CameraTrack | None = None,
        clip: ClipRef | None = None,
    ) -> RenderAssetRef:
        """Build an avatar render asset for ``subject`` and return a ref.

        ``camera`` + ``clip`` (when supplied) let a measured builder sample the subject's real
        broadcast pixels (M2-8b): the estimated world→image camera and a reference to the decoded
        source frames. Both optional — without them the builder is geometry-only (R-6: every vertex
        ``measured=0``), never a fabricated appearance.
        """
        raise NotImplementedError
