"""Field model and the per-frame homography that anchors the mono scene.

The homography ``H`` maps **image pixels → field-plane world meters**:
``[x, y, 1]^T ~ H @ [u, v, 1]^T`` with ``x, y`` on the pitch plane ``Z = 0``.
This is the world anchor in mono mode (FR-7). Each frame carries its own ``H`` plus a
confidence in ``[0, 1]`` so downstream stages and the editor can flag drift (R-6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .units import FieldDimensions


@dataclass
class FieldCalibration:
    """Per-frame homography track image↔field-plane, with confidence.

    Attributes:
        homographies: Image→world(plane) homographies, shape ``(T, 3, 3)``.
        frames: Frame indices, shape ``(T,)``.
        confidence: Per-frame calibration confidence in ``[0, 1]``, shape ``(T,)``.
        keypoints: Optional detected pitch landmarks per frame (adapter-defined).
    """

    homographies: np.ndarray
    frames: np.ndarray
    confidence: np.ndarray
    keypoints: dict | None = None

    def __post_init__(self) -> None:
        self.homographies = np.asarray(self.homographies, dtype=float).reshape(-1, 3, 3)
        self.frames = np.asarray(self.frames, dtype=int)
        self.confidence = np.asarray(self.confidence, dtype=float).reshape(-1)

    def image_to_world(self, frame_index: int, uv: np.ndarray) -> np.ndarray:
        """Project image points ``uv`` (N,2) to world plane points (N,2) at a frame.

        Used to put player feet onto the pitch (FR-7) and to lift the ball's ground
        contacts. Pure projective transform; no model involved.
        """
        i = int(np.searchsorted(self.frames, frame_index))
        i = min(max(i, 0), self.homographies.shape[0] - 1)
        h = self.homographies[i]
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        ones = np.ones((uv.shape[0], 1))
        hom = np.hstack([uv, ones]) @ h.T
        return hom[:, :2] / hom[:, 2:3]


@dataclass
class FieldModel:
    """The pitch: metric size, ground plane and the calibration track."""

    dimensions: FieldDimensions = field(default_factory=FieldDimensions)
    plane_z: float = 0.0
    calibration: FieldCalibration | None = None
