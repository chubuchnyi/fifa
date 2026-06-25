"""Measured-pitch environment reconstructor (M2-1, FR-11) — the honest, dependency-free half.

The environment signal we genuinely *measure* from a broadcast clip is the **pitch plane**: the
field calibration (PnLCalib, M1) fits a homography that aligns the image to the standard FIFA
markings on ``Z = plane_z``. This adapter emits exactly that — the calibration-anchored line
markings as a vertex-coloured PLY in world meters (every vertex ``measured=1``; it is a measured
template, nothing is fabricated). It is the validator anchor M2-0 calls for ("a leg can't pass
through the pitch") and renders through the same splat pass as the avatars.

What it deliberately does **not** do (R-8, kept honest): a photoreal 3DGS/NeRF stadium from camera
motion, or a *generative* stadium when motion is insufficient — those hallucinate unmeasured stands
and stay gated in :class:`~pitch3d.adapters.models.SplatEnvReconstructor` until their milestone.
Pure numpy + the stdlib PLY writer; no torch, no GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...core.ports.io import ClipRef
from ...core.ports.reconstruction import EnvReconstructor
from ...core.scene.assets import RenderAssetKind, RenderAssetRef, SynthViewRef
from ...core.scene.camera import CameraTrack
from ...core.scene.pitch import pitch_line_world_points
from ...core.scene.provenance import Backend, ModelInfo
from ...core.scene.units import FieldDimensions
from .avatar import write_vertex_colored_ply

# Pitch markings are white; this is a Law-of-the-Game fact, not a sampled appearance, so the
# vertices are honestly ``measured`` (their *geometry* is what the calibration anchors).
_LINE_RGB = (235, 235, 235)


@dataclass
class MeasuredPitchEnvReconstructor(EnvReconstructor):
    """Emit the calibration-anchored pitch markings as a measured vertex-coloured PLY (FR-11).

    Attributes:
        out_dir: Where the env PLY is written.
        dimensions: Outer pitch size (the markings template uses fixed Laws metrics inside it).
        plane_z: Ground-plane height in world meters (matches the field model's ``plane_z``).
        spacing: Sample spacing along every marking line (meters).
    """

    out_dir: Path = field(default_factory=lambda: Path("out/assets"))
    dimensions: FieldDimensions = field(default_factory=FieldDimensions)
    plane_z: float = 0.0
    spacing: float = 0.5

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def info(self) -> ModelInfo:
        return ModelInfo(name="MeasuredPitch", backend=Backend.LOCAL)

    def reconstruct(
        self,
        clip: ClipRef,
        camera: CameraTrack,
        synth_views: Sequence[SynthViewRef] | None = None,
    ) -> RenderAssetRef:
        verts = pitch_line_world_points(self.dimensions, plane_z=self.plane_z, spacing=self.spacing)
        n = verts.shape[0]
        rgb = np.tile(np.asarray(_LINE_RGB, dtype=np.uint8), (n, 1))
        measured = np.ones(n, dtype=bool)  # every emitted vertex is a measured pitch marking
        uri = write_vertex_colored_ply(
            self.out_dir / f"{clip.source_id}_pitch.ply", verts, np.zeros((0, 3)), rgb, measured
        )
        return RenderAssetRef(
            id=f"env-{clip.source_id}",
            kind=RenderAssetKind.ENV_PITCH_MESH,
            uri=uri,
            model=self.info(),
            extra={
                "n_vertices": int(n),
                "coverage": 1.0,
                "spacing": float(self.spacing),
                "length": float(self.dimensions.length),
                "width": float(self.dimensions.width),
                "synth_views": 0 if synth_views is None else len(synth_views),
            },
        )
