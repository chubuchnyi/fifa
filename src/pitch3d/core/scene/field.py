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

#: Confidence below which a frame's homography was **carried**, not measured. Both calibrators
#: write exactly ``0.0`` on a frame they could not solve ("carry last good, but flag zero
#: confidence"), so this is a "was it solved at all" test, not a quality bar. Un-projecting a foot
#: through a carried homography is what put roots 3 km apart and a subject at 100 416 m/s on the
#: vertical fan clip (findings/open-items-2026-08-01.md §3.3): the plane is stale by tens of frames
#: of zoom, and near the wrong horizon a pixel is kilometres.
MIN_SOLVED_CONFIDENCE = 0.02


#: |det H| below this is a homography that maps the plane onto a line — unusable, and not a
#: solve. Scaled homographies vary over orders of magnitude, so this is deliberately tiny:
#: it catches genuine rank collapse, not a merely ill-conditioned solve.
_SINGULAR_DET = 1e-12


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
        # A SINGULAR homography is not a low-quality solve, it is not a solve at all: it maps the
        # plane onto a line, so `world_to_image` cannot invert it and `image_to_world` collapses
        # every point to the same place. Found 2026-08-09 on the portrait fan clip, where PnLCalib
        # produced one and `world_to_image` raised `LinAlgError: Singular matrix` inside the ball
        # lift — killing a 236-frame run outright, with no scene written.
        #
        # The repo already has the concept for "this frame was not solved": confidence 0. Say it
        # here, once, so every consumer that honours `solved_mask` skips these for free instead of
        # each one growing its own guard (R-6: mark, never hide).
        if self.homographies.size:
            bad = ~np.isfinite(self.homographies).all(axis=(1, 2))
            with np.errstate(all="ignore"):
                bad |= np.abs(np.linalg.det(self.homographies)) < _SINGULAR_DET
            if bad.any() and self.confidence.shape[0] == bad.shape[0]:
                self.confidence = np.where(bad, 0.0, self.confidence)

    @property
    def degenerate_frames(self) -> np.ndarray:
        """Frames whose homography is singular or non-finite — unusable, and marked unsolved."""
        if not self.homographies.size:
            return np.empty(0, dtype=int)
        with np.errstate(all="ignore"):
            bad = (~np.isfinite(self.homographies).all(axis=(1, 2))
                   | (np.abs(np.linalg.det(self.homographies)) < _SINGULAR_DET))
        return self.frames[bad] if bad.shape[0] == self.frames.shape[0] else np.empty(0, dtype=int)

    def _frame_row(self, frame_index: int) -> int:
        i = int(np.searchsorted(self.frames, frame_index))
        return min(max(i, 0), self.homographies.shape[0] - 1)

    def solved_mask(
        self, frames: np.ndarray, min_confidence: float = MIN_SOLVED_CONFIDENCE
    ) -> np.ndarray:
        """Boolean mask over ``frames``: was that frame's homography actually **measured**?

        The counterpart to :meth:`image_to_world`, and the test a caller must apply before it.
        ``image_to_world`` will happily project through a homography carried from forty frames ago
        — it is pure geometry and has no way to know. Only ``confidence`` records that, so a caller
        that grounds a foot without consulting this is trusting a stale plane (R-6: mark, never
        hide — the mark exists, it just has to be read).
        """
        rows = np.clip(
            np.searchsorted(self.frames, np.asarray(frames, dtype=int).reshape(-1)),
            0, self.confidence.shape[0] - 1,
        )
        return self.confidence[rows] >= float(min_confidence)

    def image_to_world(self, frame_index: int, uv: np.ndarray) -> np.ndarray:
        """Project image points ``uv`` (N,2) to world plane points (N,2) at a frame.

        Used to put player feet onto the pitch (FR-7) and to lift the ball's ground
        contacts. Pure projective transform; no model involved.
        """
        h = self.homographies[self._frame_row(frame_index)]
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        ones = np.ones((uv.shape[0], 1))
        hom = np.hstack([uv, ones]) @ h.T
        return hom[:, :2] / hom[:, 2:3]

    def world_to_image(self, frame_index: int, xy: np.ndarray) -> np.ndarray:
        """Project world plane points ``xy`` (N,2) back to image pixels (N,2) at a frame.

        The inverse of :meth:`image_to_world` (``H⁻¹``). Used to place a known world
        anchor — e.g. a player's foot — into the image so it can be matched against a
        2D detection (the ball), which is how ball ground contacts are found (#206).
        """
        h = self.homographies[self._frame_row(frame_index)]
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        # NaN, never an exception. A singular homography killed a whole 236-frame run from inside
        # the ball lift (2026-08-09, the portrait clip); a caller can skip NaN, it cannot skip a
        # LinAlgError raised four frames deep. Such frames are also marked confidence 0 at
        # construction, so `solved_mask` filters them before it ever gets here.
        with np.errstate(all="ignore"):
            if not np.isfinite(h).all() or abs(float(np.linalg.det(h))) < _SINGULAR_DET:
                return np.full((xy.shape[0], 2), np.nan)
            hinv = np.linalg.inv(h)
        ones = np.ones((xy.shape[0], 1))
        hom = np.hstack([xy, ones]) @ hinv.T
        with np.errstate(all="ignore"):
            return np.where(np.abs(hom[:, 2:3]) < 1e-12, np.nan, hom[:, :2] / hom[:, 2:3])


@dataclass
class FieldModel:
    """The pitch: metric size, ground plane and the calibration track."""

    dimensions: FieldDimensions = field(default_factory=FieldDimensions)
    plane_z: float = 0.0
    calibration: FieldCalibration | None = None
