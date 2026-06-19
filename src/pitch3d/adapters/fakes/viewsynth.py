"""FakeViewSynthesizer — both ADR-0007 seams, no video-diffusion backend.

Seam A (:meth:`render_orbit`) returns a non-editable video ref along the prescribed orbit;
seam B (:meth:`amplify`, :meth:`inpaint_occlusions`) returns extra synthesized views that
feed reconstruction. ``frustum_overlap`` decreases as the synthesized camera strays from the
source (R-14), so callers can gate application exactly as they would with a real backend.
Deterministic, dependency-free, writes tiny placeholder files.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pitch3d.core.ports.io import ClipRef, CropRef
from pitch3d.core.ports.view_synthesizer import ViewSynthesizer
from pitch3d.core.scene.assets import SynthViewRef, SynthViewSeam
from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack
from pitch3d.core.scene.provenance import Backend, ModelInfo


def _prescribed_camera(clip: ClipRef, *, dx: float = 0.0) -> CameraTrack:
    """A prescribed (estimated=False) static camera offset ``dx`` meters from the source."""
    intr = CameraIntrinsics(
        fx=float(clip.width), fy=float(clip.width),
        cx=clip.width / 2.0, cy=clip.height / 2.0,
        width=clip.width, height=clip.height,
    )
    t = clip.n_frames
    return CameraTrack(
        intrinsics=intr,
        frames=clip.frames,
        rotation_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (t, 1)),
        translation=np.tile(np.array([dx, 0.0, 0.0]), (t, 1)),
        estimated=False,
    )


@dataclass
class FakeViewSynthesizer(ViewSynthesizer):
    """Deterministic stand-in for ReCamMaster/GEN3C/TrajectoryCrafter-class backends."""

    out_dir: Path = field(default_factory=lambda: Path("out/synth"))

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def info(self) -> ModelInfo:
        return ModelInfo(name="FakeViewSynthesizer", backend=Backend.FAKE)

    def _write(self, name: str, text: str) -> str:
        path = self.out_dir / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    # --- Seam A ---------------------------------------------------------------
    def render_orbit(
        self,
        clip: ClipRef,
        target_camera: CameraTrack,
        scene_hints: dict | None = None,
    ) -> SynthViewRef:
        uri = self._write(f"{clip.source_id}_orbit.mp4.txt", "fake orbit video")
        return SynthViewRef(
            id=f"orbitA-{clip.source_id}",
            seam=SynthViewSeam.A_RENDER,
            uri=uri,
            camera=target_camera,
            model=self.info(),
            frustum_overlap=0.85,
            editable=False,
            note="video, not editable",
        )

    # --- Seam B ---------------------------------------------------------------
    def amplify(self, clip: ClipRef, n_views: int, deviation: float) -> list[SynthViewRef]:
        overlap = float(max(0.0, 1.0 - deviation))
        out: list[SynthViewRef] = []
        for k in range(n_views):
            dx = deviation * (k + 1) / max(n_views, 1)
            uri = self._write(f"{clip.source_id}_amp{k:02d}.png.txt", f"fake amplified view {k}")
            out.append(
                SynthViewRef(
                    id=f"ampB-{clip.source_id}-{k}",
                    seam=SynthViewSeam.B_AMPLIFY,
                    uri=uri,
                    camera=_prescribed_camera(clip, dx=dx),
                    model=self.info(),
                    frustum_overlap=overlap,
                )
            )
        return out

    def inpaint_occlusions(self, subject_views: Sequence[CropRef]) -> SynthViewRef:
        # A CropRef carries no clip geometry; inpaint just needs *a* prescribed camera ref.
        tid = subject_views[0].subject_track_id if subject_views else None
        clip_ref = ClipRef(
            source_id=f"inpaint-{tid}", uri="", frames=np.arange(1), width=64, height=36, fps=25.0
        )
        uri = self._write(f"inpaint_{tid}.png.txt", "fake inpainted side")
        return SynthViewRef(
            id=f"inpaintB-{tid}",
            seam=SynthViewSeam.B_INPAINT,
            uri=uri,
            camera=_prescribed_camera(clip_ref),
            model=self.info(),
            frustum_overlap=0.6,
            subject_track_id=tid,
            note="plausible, not exact",
        )
