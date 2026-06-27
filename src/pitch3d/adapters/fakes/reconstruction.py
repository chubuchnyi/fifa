"""Fake env-reconstruction and avatar building — return asset refs, write tiny markers.

Satisfy :class:`EnvReconstructor` / :class:`AvatarBuilder` by emitting valid
:class:`RenderAssetRef`s (with FAKE provenance) pointing at small placeholder files on
disk, so the scene assembles, serializes, and renders with no 3DGS/NeRF/Rodin backend.
The real generative passes are an adapter swap; the cache (ADR-0004) keys them so they are
never recomputed needlessly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pitch3d.core.ports.io import ClipRef, CropRef
from pitch3d.core.ports.reconstruction import AvatarBuilder, EnvReconstructor
from pitch3d.core.scene.assets import RenderAssetKind, RenderAssetRef, SynthViewRef
from pitch3d.core.scene.camera import CameraTrack
from pitch3d.core.scene.provenance import Backend, ModelInfo
from pitch3d.core.scene.subject import Subject


def _marker(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


@dataclass
class FakeEnvReconstructor(EnvReconstructor):
    """Writes a placeholder env asset and returns a splat ref."""

    out_dir: Path = field(default_factory=lambda: Path("out/assets"))

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def info(self) -> ModelInfo:
        return ModelInfo(name="FakeEnvReconstructor", backend=Backend.FAKE)

    def reconstruct(
        self,
        clip: ClipRef,
        camera: CameraTrack,
        synth_views: Sequence[SynthViewRef] | None = None,
    ) -> RenderAssetRef:
        uri = _marker(self.out_dir / f"{clip.source_id}_env.splat.txt", "fake env splat")
        return RenderAssetRef(
            id=f"env-{clip.source_id}",
            kind=RenderAssetKind.ENV_SPLAT,
            uri=uri,
            model=self.info(),
            extra={"synth_views": 0 if synth_views is None else len(synth_views)},
        )


@dataclass
class FakeAvatarBuilder(AvatarBuilder):
    """Writes a placeholder avatar asset and returns a textured-SMPL-X ref (strategy #1)."""

    out_dir: Path = field(default_factory=lambda: Path("out/assets"))

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def info(self) -> ModelInfo:
        return ModelInfo(name="FakeAvatarBuilder", backend=Backend.FAKE)

    def build(
        self,
        subject: Subject,
        ref_crops: Sequence[CropRef],
        synth_views: Sequence[SynthViewRef] | None = None,
        *,
        camera: CameraTrack | None = None,
        clip: ClipRef | None = None,
    ) -> RenderAssetRef:
        uri = _marker(
            self.out_dir / f"avatar_{subject.track_id}.glb.txt", f"fake avatar {subject.track_id}"
        )
        return RenderAssetRef(
            id=f"avatar-{subject.track_id}",
            kind=RenderAssetKind.AVATAR_TEXTURED_SMPLX,
            uri=uri,
            model=self.info(),
            subject_track_id=subject.track_id,
            extra={"ref_crops": len(ref_crops)},
        )
