"""Cycles seam-A view synthesizer — a *non-generative* limited orbit (M2-10, A-9, ADR-0007).

Seam A's contract is a photoreal **video, not editable** along a bounded orbit (R-15). The fake and
any future generative backend (ReCamMaster/GEN3C/…) *hallucinate* those frames from pixels and stay
gated (R-8); this backend instead **re-renders the reconstructed 3D scene** at the orbit cameras
through Blender/Cycles. That makes the orbit honest — it shows measured geometry from a new angle,
with R-6 unmeasured regions tinted (the back-sides a single broadcast camera never saw), never an
invented surface — at the cost of being a moderate re-aim, not free-viewpoint.

The :class:`~pitch3d.core.ports.view_synthesizer.ViewSynthesizer` port doesn't carry the resolved
scene (its generative seam works from the clip), so the caller passes it as a 3D *hint*
(``scene_hints["scene"]``). ``frustum_overlap`` falls with the requested deviation so a consumer can
still gate how far it trusts the re-aim. Seam B (generative amplify / inpaint) is not this backend's
job and stays gated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ...core.ports.io import ClipRef, CropRef
from ...core.ports.render import RenderQuality
from ...core.ports.view_synthesizer import ViewSynthesizer
from ...core.scene.assets import SynthViewRef, SynthViewSeam
from ...core.scene.camera import CameraTrack
from ...core.scene.provenance import Backend, ModelInfo
from .cycles import CyclesRenderPass

_ORBIT_MAX_DEVIATION_DEG = 45.0  # the seam-A hard cap (mirrors core.agent.bounded_orbit_camera)


@dataclass
class CyclesViewSynthesizer(ViewSynthesizer):
    """Seam-A backend re-rendering the reconstructed 3D scene at the orbit cameras (no generative).

    Drives a wrapped :class:`CyclesRenderPass`. ``render_orbit`` re-renders the resolved scene
    passed via ``scene_hints["scene"]`` at the (already-scaled) ``target_camera`` and returns a
    ``SynthViewRef`` (``seam=A_RENDER``, ``editable=False``) pointing at the produced per-frame PNG
    directory. The generative seam-B methods are intentionally unsupported here (R-8).
    """

    out_dir: Path = field(default_factory=lambda: Path("out/synth"))
    render: CyclesRenderPass | None = None

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.render is None:
            self.render = CyclesRenderPass(out_dir=self.out_dir / "orbit")

    def info(self) -> ModelInfo:
        return ModelInfo(name="CyclesViewSynthesizer", backend=Backend.LOCAL)

    # --- Seam A: re-render the 3D scene at the bounded orbit (no generative) ----
    def render_orbit(
        self,
        clip: ClipRef,
        target_camera: CameraTrack,
        scene_hints: dict | None = None,
    ) -> SynthViewRef:
        hints = scene_hints or {}
        scene = hints.get("scene")
        if scene is None:
            raise ValueError(
                "CyclesViewSynthesizer.render_orbit re-renders the reconstructed 3D scene, so it "
                "needs scene_hints['scene'] (the resolved Scene); Application.render_orbit "
                "passes it."
            )
        assert self.render is not None  # set in __post_init__
        # The orbit camera is already quality-scaled by the caller; render at FINAL so the pass
        # uses it verbatim (FINAL.scale == 1.0) rather than scaling a second time.
        result = self.render.render(scene, target_camera, RenderQuality.FINAL)
        dev = min(abs(float(hints.get("max_deviation_deg", 20.0))), _ORBIT_MAX_DEVIATION_DEG)
        overlap = max(0.0, min(1.0, 1.0 - dev / 90.0))  # falls as the re-aim strays (R-14)
        return SynthViewRef(
            id=f"orbitA-cycles-{scene.id}",
            seam=SynthViewSeam.A_RENDER,
            uri=result.uri,
            camera=target_camera,
            model=self.info(),
            frustum_overlap=overlap,
            editable=False,
            note="video, not editable — Cycles re-render of the resolved 3D scene (no generative)",
        )

    # --- Seam B: generative amplify / inpaint — not this backend's job (gated) --
    def amplify(self, clip: ClipRef, n_views: int, deviation: float) -> list[SynthViewRef]:
        raise NotImplementedError(
            "CyclesViewSynthesizer only re-renders measured geometry for seam A; generative "
            "seam-B amplification is gated (R-8). Wire a real ViewSynthesizer backend for it."
        )

    def inpaint_occlusions(self, subject_views: Sequence[CropRef]) -> SynthViewRef:
        raise NotImplementedError(
            "CyclesViewSynthesizer only re-renders measured geometry for seam A; generative "
            "seam-B inpainting is gated (R-8). Wire a real ViewSynthesizer backend for it."
        )


__all__ = ["CyclesViewSynthesizer"]
