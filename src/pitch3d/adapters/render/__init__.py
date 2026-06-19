"""Real render passes — honest stubs (FR-14, UX-3, roadmap M2).

Two implementations both satisfy :class:`RenderPass`: a splat/avatar pass for a free camera,
and a ViewSynthesizer-seam-A pass that wraps :meth:`ViewSynthesizer.render_orbit` for limited
orbits (video, not editable). Importable now; ``render`` raises ``NotImplementedError``. Use
:class:`pitch3d.adapters.fakes.FakeRenderPass` for tests and the dry-run.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.ports.render import RenderPass, RenderQuality, RenderResult
from ...core.ports.view_synthesizer import ViewSynthesizer
from ...core.scene.camera import CameraTrack
from ...core.scene.scene import Scene


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


__all__ = ["SplatAvatarRenderPass", "ViewSynthOrbitRenderPass"]
