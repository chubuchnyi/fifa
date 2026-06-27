"""FakeMotionPrior — a real, GPU-free temporal denoiser standing in for the learned prior.

This is the "fake" that is genuinely real: it implements :class:`MotionPrior` by reusing the pure,
deterministic zero-phase smoothers the correction engine already ships (gaussian, quaternion-aware
for rotations). It needs no torch/weights, so the whole ``method="learned"`` seam — engine routing,
wiring, the controller preview — runs and is testable end-to-end now; the learned model
(``LearnedMotionPrior``, HTD-Refine/StableMotion) is the gated swap-in (R-8, ADR-0001).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitch3d.core.correction.engine import smooth_rotation, smooth_vector
from pitch3d.core.ports.motion_prior import MotionPrior
from pitch3d.core.scene.provenance import Backend, ModelInfo


@dataclass
class FakeMotionPrior(MotionPrior):
    """Deterministic gaussian zero-phase denoiser (no model, no GPU)."""

    window: int = 7
    sigma: float = 1.5

    def info(self) -> ModelInfo:
        return ModelInfo(name="FakeMotionPrior", backend=Backend.FAKE)

    def denoise(self, values: np.ndarray, frames: np.ndarray, *, is_rotation: bool) -> np.ndarray:
        fn = smooth_rotation if is_rotation else smooth_vector
        return fn(np.asarray(values, dtype=float), window=self.window, method="gaussian",
                  sigma=self.sigma)
