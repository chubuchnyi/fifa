"""References to *derived* render assets and to synthesized (ViewSynthesizer) views.

These are pointers + provenance, never the heavy data itself. The scene model stays
light and serializable; the actual splats/meshes/videos live on disk, addressed by URI
and reproduced from the content-addressable cache (ADR-0004).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .camera import CameraTrack
from .provenance import ModelInfo


class RenderAssetKind(str, Enum):
    """Kinds of derived render asset (FR-11..FR-13)."""

    ENV_PITCH_MESH = "env_pitch_mesh"          # measured pitch line markings (calibration-anchored)
    ENV_SPLAT = "env_splat"                    # 3D Gaussian Splatting environment
    ENV_NERF = "env_nerf"                      # NeRF environment
    ENV_GENERATIVE = "env_generative"          # generative stadium fallback
    AVATAR_TEXTURED_SMPLX = "avatar_textured"  # strategy #1 (MVP base)
    AVATAR_GENERATIVE = "avatar_generative"    # strategy #2 (Rodin-class)
    AVATAR_GAUSSIAN = "avatar_gaussian"        # strategy #3 (per-subject 3DGS avatar)
    BALL_TEXTURE = "ball_texture"              # textured sphere


@dataclass
class RenderAssetRef:
    """A pointer to a derived render asset, with full provenance (NFR-7, UX-7).

    Attributes:
        id: Stable identifier.
        kind: Which kind of asset.
        uri: Path/URI to the asset payload on disk.
        subject_track_id: For per-subject avatars; ``None`` for environment/ball.
        model: Which model/version/cost produced it.
        extra: Adapter-specific metadata (e.g. splat count, texture resolution).
    """

    id: str
    kind: RenderAssetKind
    uri: str
    model: ModelInfo
    subject_track_id: int | None = None
    extra: dict = field(default_factory=dict)


class SynthViewSeam(str, Enum):
    """Which ViewSynthesizer integration seam produced this view (ADR-0007)."""

    A_RENDER = "A_render"      # seam A: limited-orbit render output (video, NOT editable)
    B_AMPLIFY = "B_amplify"    # seam B: extra views feeding reconstruction (pseudo-multi-view)
    B_INPAINT = "B_inpaint"    # seam B: inpainted unseen sides of a subject


@dataclass
class SynthViewRef:
    """A synthesized view produced by a :class:`ViewSynthesizer` (FR-29..32).

    The same record type serves both seams; ``seam`` disambiguates how it is consumed.
    Seam-A outputs are flagged non-editable in UX (R-15) — there is no path from these
    pixels back into the SMPL/curve source of truth.

    Attributes:
        id: Stable identifier.
        seam: A_render / B_amplify / B_inpaint.
        uri: Path/URI to the synthesized video (seam A) or frame set (seam B).
        camera: The synthesized camera trajectory (prescribed, ``estimated=False``).
        model: Backend model/version/cost (e.g. ReCamMaster, GEN3C).
        frustum_overlap: Estimated overlap with the source frustum, ``[0, 1]`` — low
            overlap means likely hallucination (R-14); used to gate/limit application.
        subject_track_id: For B_INPAINT, which subject this augments.
        editable: Always ``False`` for seam A (render output is video, not geometry).
        note: Free-text (e.g. "video, not editable").
    """

    id: str
    seam: SynthViewSeam
    uri: str
    camera: CameraTrack
    model: ModelInfo
    frustum_overlap: float = 1.0
    subject_track_id: int | None = None
    editable: bool = False
    note: str | None = None
