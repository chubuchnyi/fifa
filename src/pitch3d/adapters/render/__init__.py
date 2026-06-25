"""Real render passes (FR-14, UX-3).

Three implementations satisfy :class:`RenderPass`. **Wired now:**
:class:`ReprojectionOverlayRenderPass` — a Blender-free pass that reprojects the resolved 3D
scene back onto per-frame PNGs (pure numpy + stdlib), so the mono pipeline is visually
inspectable today; and :class:`SplatAvatarRenderPass` (M2-3) — splats the MEASURED M2-2 avatar
meshes onto per-frame PNGs with a z-buffer, tinting never-observed vertices (R-6). **Stub (roadmap
M2):** a ViewSynthesizer-seam-A pass wrapping :meth:`ViewSynthesizer.render_orbit` for limited
orbits (video, not editable). Use :class:`pitch3d.adapters.fakes.FakeRenderPass` for the no-IO
dry-run.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.ports.render import RenderPass, RenderQuality, RenderResult
from ...core.ports.view_synthesizer import ViewSynthesizer
from ...core.scene.camera import CameraTrack
from ...core.scene.scene import Scene
from .avatar_splat import SplatAvatarRenderPass
from .overlay import ReprojectionOverlayRenderPass


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
