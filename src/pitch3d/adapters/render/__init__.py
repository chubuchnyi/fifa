"""Real render passes (FR-14, UX-3).

Three implementations satisfy :class:`RenderPass`. **Wired now:**
:class:`ReprojectionOverlayRenderPass` — a Blender-free pass that reprojects the resolved 3D
scene back onto per-frame PNGs (pure numpy + stdlib), so the mono pipeline is visually
inspectable today. **Stubs (roadmap M2):** a splat/avatar pass for a free camera, and a
ViewSynthesizer-seam-A pass wrapping :meth:`ViewSynthesizer.render_orbit` for limited orbits
(video, not editable). Use :class:`pitch3d.adapters.fakes.FakeRenderPass` for the no-IO dry-run.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.ports.render import RenderPass, RenderQuality, RenderResult
from ...core.ports.view_synthesizer import ViewSynthesizer
from ...core.scene.camera import CameraTrack
from ...core.scene.scene import Scene
from .overlay import ReprojectionOverlayRenderPass


@dataclass
class SplatAvatarRenderPass(RenderPass):
    """Assembles a photoreal frame from env splats + per-subject avatars (free camera)."""

    def render(self, scene: Scene, camera_path: CameraTrack, quality: RenderQuality = RenderQuality.PREVIEW) -> RenderResult:
        raise NotImplementedError(
            "splat/avatar render pass is not wired yet (roadmap M2). "
            "Use pitch3d.adapters.fakes.FakeRenderPass meanwhile."
        )


@dataclass
class ViewSynthOrbitRenderPass(RenderPass):
    """Wraps a :class:`ViewSynthesizer` seam-A orbit as a render pass (video, not editable)."""

    synthesizer: ViewSynthesizer

    def render(self, scene: Scene, camera_path: CameraTrack, quality: RenderQuality = RenderQuality.PREVIEW) -> RenderResult:
        raise NotImplementedError(
            "ViewSynthesizer seam-A render pass is not wired yet (roadmap M2). "
            "It will delegate to ViewSynthesizer.render_orbit and wrap the SynthViewRef as a RenderResult."
        )


__all__ = [
    "ReprojectionOverlayRenderPass",
    "SplatAvatarRenderPass",
    "ViewSynthOrbitRenderPass",
]
