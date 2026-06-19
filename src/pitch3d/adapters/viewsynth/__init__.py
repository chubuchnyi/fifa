"""Real ViewSynthesizer adapter — honest stub for both seams (ADR-0007, roadmap M2/M3).

Backends differ per seam (seam A favors orbit fidelity, e.g. ReCamMaster; seam B favors
3D-consistency, e.g. GEN3C / TrajectoryCrafter); both are reached through this one port.
Importable now (no diffusion deps at import); each method raises ``NotImplementedError``.
Use :class:`pitch3d.adapters.fakes.FakeViewSynthesizer` for tests and the dry-run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...core.ports.io import ClipRef, CropRef
from ...core.ports.view_synthesizer import ViewSynthesizer
from ...core.scene.assets import SynthViewRef
from ...core.scene.camera import CameraTrack
from ...core.scene.provenance import Backend, ModelInfo


def _todo(seam: str) -> NotImplementedError:
    return NotImplementedError(
        f"generative novel-view ({seam}) is not wired yet — install the `viewsynth` extra "
        "(roadmap M2/M3). Use pitch3d.adapters.fakes.FakeViewSynthesizer meanwhile."
    )


@dataclass
class GenerativeViewSynthesizer(ViewSynthesizer):
    """ReCamMaster / GEN3C / TrajectoryCrafter-class backend (FR-29..32)."""

    name: str = "ReCamMaster"

    def info(self) -> ModelInfo:
        return ModelInfo(name=self.name, backend=Backend.LOCAL)

    def render_orbit(self, clip: ClipRef, target_camera: CameraTrack, scene_hints: dict | None = None) -> SynthViewRef:
        raise _todo("seam A render_orbit")

    def amplify(self, clip: ClipRef, n_views: int, deviation: float) -> list[SynthViewRef]:
        raise _todo("seam B amplify")

    def inpaint_occlusions(self, subject_views: Sequence[CropRef]) -> SynthViewRef:
        raise _todo("seam B inpaint_occlusions")


__all__ = ["GenerativeViewSynthesizer"]
