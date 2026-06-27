"""Real render passes (FR-14, UX-3).

These implementations satisfy :class:`RenderPass`. **Wired now:**
:class:`ReprojectionOverlayRenderPass` — a Blender-free pass that reprojects the resolved 3D
scene back onto per-frame PNGs (pure numpy + stdlib), so the mono pipeline is visually
inspectable today; :class:`SplatAvatarRenderPass` (M2-3) — splats the MEASURED M2-2 avatar
meshes onto per-frame PNGs with a z-buffer (the no-dependency debug viz), tinting never-observed
vertices (R-6); :class:`CyclesRenderPass` (M2-7) — the real *photoreal* path, rendering those same
measured meshes through Blender/Cycles out of process (ADR-0003), R-6 tint intact, LBS-posed
(M2-8) on a measured grass pitch under a physical sky (M2-9); and :class:`ViewSynthOrbitRenderPass`
(M2-5) — wraps a :class:`ViewSynthesizer` seam-A orbit (:meth:`ViewSynthesizer.render_orbit`) as a
render pass producing a limited-orbit **video, not editable** (ADR-0007). Two M2-10 adapters round
out the feedback loop: :class:`CyclesSceneObserver` renders photoreal ``SCENE_3D`` snapshots (A-8)
and :class:`CyclesViewSynthesizer` is the non-generative seam-A backend that re-renders the 3D
scene at the orbit cameras (A-9). Use :class:`pitch3d.adapters.fakes.FakeRenderPass` for the
no-IO dry-run.
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
from .cycles import CyclesRenderPass
from .cycles_orbit import CyclesViewSynthesizer
from .observe import CyclesSceneObserver
from .overlay import ReprojectionOverlayRenderPass


def orbit_render_result(ref: SynthViewRef, quality: RenderQuality) -> RenderResult:
    """Wrap a seam-A :class:`SynthViewRef` as a :class:`RenderResult` (video, not editable).

    Carries the synthesizer's ``frustum_overlap``, the re-shot resolution, and the non-editable
    flag into the render note so a consumer can never mistake an orbit re-shoot for an editable
    reconstruction (R-15) and can see the preview/final size it was shot at (UX-9).
    """
    n = int(ref.camera.frames.shape[0]) if ref.camera is not None else 0
    size = (
        f"{int(ref.camera.intrinsics.width)}x{int(ref.camera.intrinsics.height)}"
        if ref.camera is not None
        else "0x0"
    )
    detail = ref.note or "video, not editable"
    return RenderResult(
        uri=ref.uri,
        n_frames=n,
        quality=quality,
        is_video=True,
        camera=ref.camera,
        note=f"seam A (render_orbit) {size} {quality.value} "
        f"frustum_overlap={ref.frustum_overlap:.2f} — {detail}",
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
        # Preview lever for the generative seam (M2-6, UX-9): downscale the orbit so the re-shoot
        # is cheaper, and pass the quality hint so a real synthesizer can also go low-q. FINAL =
        # full-res (scaled(1.0) is the same camera), so the re-shot frame count is always preserved.
        camera = camera_path.scaled(quality.scale)
        clip = ClipRef(
            source_id=scene.source_id,
            uri="",
            frames=camera.frames,
            width=int(camera.intrinsics.width),
            height=int(camera.intrinsics.height),
            fps=0.0,
        )
        ref = self.synthesizer.render_orbit(clip, camera, {"quality": quality.value})
        return orbit_render_result(ref, quality)


__all__ = [
    "CyclesRenderPass",
    "CyclesSceneObserver",
    "CyclesViewSynthesizer",
    "ReprojectionOverlayRenderPass",
    "SplatAvatarRenderPass",
    "ViewSynthOrbitRenderPass",
    "orbit_render_result",
]
