"""Real render passes (FR-14, UX-3).

Three implementations satisfy :class:`RenderPass`. **Wired now:**
:class:`ReprojectionOverlayRenderPass` — a Blender-free pass that reprojects the resolved 3D
scene back onto per-frame PNGs (pure numpy + stdlib), so the mono pipeline is visually
inspectable today; :class:`SplatAvatarRenderPass` (M2-3) — splats the MEASURED M2-2 avatar
meshes onto per-frame PNGs with a z-buffer, tinting never-observed vertices (R-6); and
:class:`ViewSynthOrbitRenderPass` (M2-5) — wraps a :class:`ViewSynthesizer` seam-A orbit
(:meth:`ViewSynthesizer.render_orbit`) as a render pass producing a limited-orbit **video, not
editable** (ADR-0007). Use :class:`pitch3d.adapters.fakes.FakeRenderPass` for the no-IO dry-run.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.ports.io import ClipRef
from ...core.ports.render import RenderPass, RenderQuality, RenderResult
from ...core.ports.view_synthesizer import ViewSynthesizer
from ...core.scene.assets import SynthViewRef
from ...core.scene.camera import CameraTrack
from ...core.scene.scene import Scene
from .avatar_splat import SplatAvatarRenderPass
from .overlay import ReprojectionOverlayRenderPass


def orbit_render_result(ref: SynthViewRef, quality: RenderQuality) -> RenderResult:
    """Wrap a seam-A :class:`SynthViewRef` as a :class:`RenderResult` (video, not editable).

    Carries the synthesizer's ``frustum_overlap`` and the non-editable flag into the render note
    so a consumer can never mistake an orbit re-shoot for an editable reconstruction (R-15).
    """
    n = int(ref.camera.frames.shape[0]) if ref.camera is not None else 0
    detail = ref.note or "video, not editable"
    return RenderResult(
        uri=ref.uri,
        n_frames=n,
        quality=quality,
        is_video=True,
        camera=ref.camera,
        note=f"seam A (render_orbit) frustum_overlap={ref.frustum_overlap:.2f} — {detail}",
    )


@dataclass
class ViewSynthOrbitRenderPass(RenderPass):
    """Wraps a :class:`ViewSynthesizer` seam-A orbit as a render pass (video, not editable).

    Seam A *re-shoots the source clip* along the prescribed orbit ``camera_path``; the clip-free
    :class:`RenderPass` contract (a consumer of the *resolved* scene) doesn't carry the source
    video, so this pass reconstructs the minimal :class:`ClipRef` the synthesizer needs from the
    scene's ``source_id`` and the orbit camera. That suffices for the dependency-free fake; the
    authoritative real-clip path — with the registered clip's uri/fps **and caching** — is
    :meth:`pitch3d.app.controller.Application.render_orbit`.
    """

    synthesizer: ViewSynthesizer

    def render(
        self,
        scene: Scene,
        camera_path: CameraTrack,
        quality: RenderQuality = RenderQuality.PREVIEW,
    ) -> RenderResult:
        clip = ClipRef(
            source_id=scene.source_id,
            uri="",
            frames=camera_path.frames,
            width=int(camera_path.intrinsics.width),
            height=int(camera_path.intrinsics.height),
            fps=0.0,
        )
        ref = self.synthesizer.render_orbit(clip, camera_path, None)
        return orbit_render_result(ref, quality)


__all__ = [
    "ReprojectionOverlayRenderPass",
    "SplatAvatarRenderPass",
    "ViewSynthOrbitRenderPass",
    "orbit_render_result",
]
