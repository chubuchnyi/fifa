"""Learned motion-prior port: temporal denoising behind the smoothing Correction seam (FR-22).

A motion prior is a temporal denoiser over one per-frame signal block — root translation, root
orientation, or a single body joint's rotations. It is engaged through the **existing**
``TEMPORAL_SMOOTHING`` correction with ``method="learned"`` (ADR-0002), so a learned denoiser stays
a normal, inspectable, disableable correction rather than a hidden post-process. The correction
engine calls :meth:`denoise` through this abstraction, keeping the engine pure; the pure
moving-average / gaussian methods need no port at all — this seam exists only for the learned
denoiser (HTD-Refine / StableMotion-class, roadmap M3-8).
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from .base import ModelProvider


class MotionPrior(ModelProvider):
    """Temporal denoiser for one per-frame signal block (FR-22, M3-8)."""

    @abstractmethod
    def denoise(self, values: np.ndarray, frames: np.ndarray, *, is_rotation: bool) -> np.ndarray:
        """Return a temporally-denoised copy of ``values`` (``(M, D)``, time on axis 0).

        ``frames`` are the integer frame ids of the rows (ascending, possibly non-contiguous, so a
        prior can respect real time gaps). ``is_rotation`` selects rotation space (axis-angle
        ``D == 3`` → denoise via quaternions) versus Euclidean (translation). Must not mutate
        ``values``; the engine splices the result as a TEMPORAL_SMOOTHING correction so it stays
        non-destructive (ADR-0002), and any learned completion should be validated against the
        measured homography anchor (off-prior drift is flagged, not trusted — R-6).
        """
        raise NotImplementedError
