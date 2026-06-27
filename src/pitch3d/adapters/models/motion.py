"""LearnedMotionPrior — the gated learned temporal denoiser (roadmap M3-8).

Implements :class:`~pitch3d.core.ports.motion_prior.MotionPrior` with a learned motion model
(HTD-Refine / StableMotion-class diffusion denoisers). It is importable with no torch at import
time so the wiring, provenance, and tests stay complete; :meth:`denoise` raises until the model is
wired (R-8). Engaged through the existing ``TEMPORAL_SMOOTHING`` correction with
``method="learned"`` (``--motion-prior learned``); for a GPU-free run use the pure
``moving_average``/``gaussian`` methods or the
:class:`~pitch3d.adapters.fakes.motion_prior.FakeMotionPrior`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.ports.motion_prior import MotionPrior
from ...core.scene.provenance import Backend, ModelInfo


@dataclass
class LearnedMotionPrior(MotionPrior):
    """Learned diffusion/transformer motion denoiser (HTD-Refine / StableMotion), gated (R-8)."""

    weights: str | None = None
    device: str = "cuda"
    _model: object = None

    def info(self) -> ModelInfo:
        return ModelInfo(name="LearnedMotionPrior", backend=Backend.LOCAL)

    def denoise(self, values: np.ndarray, frames: np.ndarray, *, is_rotation: bool) -> np.ndarray:
        raise NotImplementedError(
            "learned motion-prior denoising is not wired yet (roadmap M3-8) and is GPU-bound: "
            "HTD-Refine / StableMotion are research repos (not pip packages), so the `motion` "
            "extra ships no weights/network. Use the pure smoothing methods (moving_average / "
            "gaussian — no model, --motion-prior fake) instead, or inject your own MotionPrior. "
            "Validate any learned completion against the homography anchor "
            "(pitch3d.core.correction.anchor) — off-prior drift is hallucinated, "
            "not measured (R-6)."
        )
