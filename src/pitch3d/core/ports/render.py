"""RenderPass — produce photoreal frames from the current (resolved) scene state.

A render pass is a pure consumer of the **resolved** scene (SMPL/curves driving avatars,
env splats, ball). It never edits the scene. Two implementations live in
``adapters/render``: a splat/avatar pass (free camera) and a ViewSynthesizer-seam-A pass
(limited orbit, video). Both satisfy this contract (UX-3 render-path choice).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

from ..scene.camera import CameraTrack
from ..scene.scene import Scene
from .base import Port


class RenderQuality(str, Enum):
    """Preview vs final (UX-9 fast low-q preview before the expensive final)."""

    PREVIEW = "preview"
    FINAL = "final"


@dataclass
class RenderResult:
    """Output of a render pass — a pointer to rendered media, plus provenance."""

    uri: str                  # video file or frame directory
    n_frames: int
    quality: RenderQuality
    is_video: bool            # True for ViewSynthesizer seam-A output
    camera: CameraTrack       # the camera path used
    note: str | None = None


class RenderPass(Port):
    """Assembles a photoreal frame/sequence from a scene's resolved state (FR-14)."""

    @abstractmethod
    def render(
        self,
        scene: Scene,
        camera_path: CameraTrack,
        quality: RenderQuality = RenderQuality.PREVIEW,
    ) -> RenderResult:
        """Render ``scene`` along ``camera_path`` at the requested quality.

        Implementations must read only the *resolved* state (single source of truth);
        they must not mutate ``scene``.
        """
        raise NotImplementedError
