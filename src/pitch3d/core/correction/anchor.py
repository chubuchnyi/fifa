"""Homography-anchor validation: does the reconstructed root sit where the pitch says (R-6)?

The measured-over-generative check for M3-2 (from the brief). A subject's world root is
*grounded* on the pitch by the field homography — the bbox foot point projected to world metres
(``FieldCalibration.image_to_world``) is a **measured** ground-plane position. Any later
constraint-guided re-fit or generative occlusion-completion that moves the body must still agree
with that measured anchor: a pose that drifts off the player's measured ground track is
hallucinated, not measured, and gets *flagged* (low confidence) rather than silently trusted.

These are pure numpy helpers over arrays plus a small report — no ports, no model — so the refit
adapters and the agent can validate completions cheaply. Only the ground plane (XY) is compared;
the root height Z is the mono vertical ambiguity (R-4) and is not part of the anchor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Default tolerance: how far (m) a root's ground XY may sit from its measured anchor and still
#: count as on-anchor. ~0.5 m absorbs the pelvis-vs-foot lean of a moving player without letting a
#: hallucinated metre-scale drift pass as measured.
DEFAULT_MAX_RESIDUAL_M = 0.5


def _as_anchor_xy(anchor_xy: np.ndarray, t: int) -> np.ndarray:
    """Normalise an anchor spec to ``(t, 2)``: a single ``(2,)`` broadcasts; ``(t, 2)`` passes."""
    a = np.asarray(anchor_xy, dtype=float)
    if a.ndim == 1:
        if a.shape[0] != 2:
            raise ValueError(f"a 1-D anchor must be (2,) world XY, got {a.shape}")
        return np.broadcast_to(a.reshape(1, 2), (t, 2))
    a = a.reshape(-1, 2)
    if a.shape[0] != t:
        raise ValueError(f"per-frame anchor has {a.shape[0]} rows, expected {t}")
    return a


def anchor_residuals(transl: np.ndarray, anchor_xy: np.ndarray) -> np.ndarray:
    """Per-frame ground distance (m) between each root and its measured pitch anchor.

    ``transl`` is the ``(T, 3)`` world root (XY + height Z); ``anchor_xy`` is the measured
    ground-plane position, either ``(T, 2)`` (per frame) or ``(2,)`` (a single locked spot,
    broadcast). Returns ``(T,)`` horizontal distances — Z is ignored (the mono height ambiguity).
    """
    t = np.asarray(transl, dtype=float).reshape(-1, 3)
    a = _as_anchor_xy(anchor_xy, t.shape[0])
    return np.linalg.norm(t[:, :2] - a, axis=1)


def blend_to_anchor(cur_xy: np.ndarray, anchor: np.ndarray, blend: float = 1.0) -> np.ndarray:
    """Pull ground XY toward a measured anchor by ``blend`` ∈ [0, 1] (1 = hard lock).

    Shared by the re-fit adapters (real + fake) so "lock the root to the measured homography
    position" is one pure op, not duplicated. ``cur_xy`` is ``(M, 2)``; ``anchor`` is a single
    ``(2,)`` (broadcast) or per-frame ``(M, 2)`` aligned to the same ascending-frame order the
    correction engine passes the re-fit frames in.
    """
    cur = np.asarray(cur_xy, dtype=float).reshape(-1, 2)
    target = _as_anchor_xy(anchor, cur.shape[0])
    b = float(blend)
    return (1.0 - b) * cur + b * target


@dataclass
class AnchorReport:
    """How well a motion's root tracks its measured homography anchor (R-6 transparency)."""

    frames: np.ndarray
    residuals: np.ndarray  # (T,) per-frame ground distance, metres
    valid: np.ndarray      # (T,) bool: True = on-anchor (residual <= max_residual_m)
    max_residual_m: float  # the tolerance the validity mask was computed against

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def n_off_anchor(self) -> int:
        return self.n_frames - self.n_valid

    @property
    def off_anchor_frames(self) -> np.ndarray:
        """The frames flagged off-anchor — the ones a reviewer/agent should distrust."""
        return self.frames[~self.valid]

    @property
    def mean_residual_m(self) -> float:
        return float(np.mean(self.residuals)) if self.residuals.size else 0.0

    @property
    def worst_residual_m(self) -> float:
        return float(np.max(self.residuals)) if self.residuals.size else 0.0


def validate_against_anchor(
    frames: np.ndarray,
    transl: np.ndarray,
    anchor_xy: np.ndarray,
    *,
    max_residual_m: float = DEFAULT_MAX_RESIDUAL_M,
) -> AnchorReport:
    """Flag frames whose root strays from the measured ground anchor beyond ``max_residual_m``.

    The honest gate on a re-fit/completion: frames within tolerance are measured-consistent;
    the rest are surfaced as off-anchor (R-6) for the attention list or a confidence dip, never
    silently accepted. Pure — no scene, no port; feed it a subject's root and its homography
    ground track.
    """
    f = np.asarray(frames, dtype=int).reshape(-1)
    residuals = anchor_residuals(transl, anchor_xy)
    valid = residuals <= float(max_residual_m)
    return AnchorReport(
        frames=f, residuals=residuals, valid=valid, max_residual_m=float(max_residual_m)
    )
