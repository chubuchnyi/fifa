"""TrackNet ball tracker — fifth real adapter (M1, FR-9).

Self-hosted default for `BallTracker`: a TrackNet-class heatmap network, behind the optional
``ball`` extra. Split like the detector so the *decode* logic is testable with **no torch, no
GPU**:

* :class:`TrackNetBallTracker` — the **pure** half: thresholds the per-frame heatmap peaks,
  linearly interpolates the 2D position across missed (occluded) frames so the core ballistic
  lift gets a dense track, and flags every filled frame at confidence 0 (drift surfaced
  honestly, R-6). Numpy only; unit-tested via an injected backend.
* :class:`TrackNetBackend` — the **heavy** half: *not wired yet* (roadmap M1). Unlike the
  rfdetr/bytetrack reals, the ``ball`` extra ships only torch (no TrackNet weights/decoder), so
  the live network is a stub: :meth:`TrackNetBackend.detect_ball` raises ``NotImplementedError``
  pointing at ``--ball fake`` or an injected backend. The pure threshold/gap-fill half above is
  complete and tested, so the ball path runs end to end on the fake today.

The 2D→3D ballistic lift stays in the core (``orchestration.ball_lift``), not here (FR-9).
Swap it in via ``default_ports(ball="tracknet")`` (wiring) — one fake replaced at a time,
satisfying the very same ``BallTracker`` port test the fake passes (roadmap M1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ...core.ports.io import ClipRef
from ...core.ports.perception import BallTracker
from ...core.scene.motion import Ball2DTrack
from ...core.scene.provenance import Backend, ModelInfo


@dataclass
class RawBallDetections:
    """Backend output for the clip: per-frame ball-candidate peak + heatmap score.

    One row per processed frame; a frame with no confident peak still gets a row, its low
    ``score`` telling the adapter to treat it as a miss (and interpolate the position).

    Attributes:
        frames: ``(T,)`` source frame indices (ascending).
        points_xy: ``(T, 2)`` per-frame peak position, image px.
        scores: ``(T,)`` peak heatmap confidence in ``[0, 1]``.
    """

    frames: np.ndarray
    points_xy: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=int).reshape(-1)
        t = self.frames.shape[0]
        self.points_xy = np.asarray(self.points_xy, dtype=float).reshape(-1, 2)
        self.scores = np.asarray(self.scores, dtype=float).reshape(-1)
        if not (self.points_xy.shape[0] == self.scores.shape[0] == t):
            raise ValueError(
                f"ragged raw ball detections: {t} frames, "
                f"{self.points_xy.shape[0]} points, {self.scores.shape[0]} scores"
            )


@runtime_checkable
class BallDetectionBackend(Protocol):
    """The heavy half: run the heatmap network and return per-frame ball peaks.

    Kept behind this protocol so :class:`TrackNetBallTracker`'s threshold/interpolation logic
    can be tested with a stub returning canned :class:`RawBallDetections` — no GPU required.
    """

    def detect_ball(self, clip: ClipRef) -> RawBallDetections:
        """Return the per-frame ball-candidate track for ``clip``."""
        ...


def _interpolate_track(
    frames: np.ndarray, points: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fill missed frames by linear interpolation; detected frames keep their score, fills get 0.

    Frames scoring ``>= threshold`` are kept as-is. Gaps between them are linearly interpolated
    by frame index; leading/trailing gaps carry the nearest detected point. Every interpolated
    frame is marked confidence 0 so the lift can tell real detections from fills (R-6).
    """
    detected = scores >= threshold
    if not detected.any():
        return points.copy(), np.zeros(frames.shape[0])
    fx = np.interp(frames, frames[detected], points[detected, 0])
    fy = np.interp(frames, frames[detected], points[detected, 1])
    conf = np.where(detected, np.clip(scores, 0.0, 1.0), 0.0)
    return np.column_stack([fx, fy]), conf


def _smooth_xy(points: np.ndarray, window: int) -> np.ndarray:
    """Box-average a ``(T, 2)`` path over a centred frame window (anti-jitter, edge-clamped)."""
    t = points.shape[0]
    if window <= 1 or t <= 2:
        return points
    half = (window if window % 2 else window + 1) // 2
    out = np.empty_like(points)
    for i in range(t):
        out[i] = points[max(0, i - half):min(t, i + half + 1)].mean(axis=0)
    return out


@dataclass
class TrackNetBallTracker(BallTracker):
    """TrackNet 2D ball tracker (FR-9) — pure threshold + gap-fill over an injected backend.

    Attributes:
        backend: The heatmap-network backend. If ``None``, a real :class:`TrackNetBackend` is
            constructed lazily on first use (needs the ``ball`` extra + weights + GPU).
        score_threshold: Heatmap-peak confidence floor; frames below it count as misses and
            have their position interpolated at confidence 0.
        smooth_window: Centred temporal-smoothing window in frames (1 disables smoothing).
        device: Inference device for the default backend.
    """

    backend: BallDetectionBackend | None = None
    score_threshold: float = 0.5
    smooth_window: int = 1
    device: str = "cuda"

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="TrackNetV3",
            backend=Backend.LOCAL,
            params={
                "score_threshold": self.score_threshold,
                "smooth_window": self.smooth_window,
                "device": self.device,
            },
        )

    def track_ball(self, clip: ClipRef) -> Ball2DTrack:
        raw = self._backend().detect_ball(clip)
        points, conf = _interpolate_track(
            raw.frames, raw.points_xy, raw.scores, self.score_threshold
        )
        points = _smooth_xy(points, self.smooth_window)
        return Ball2DTrack(frames=raw.frames, positions_2d=points, confidence=conf)

    def _backend(self) -> BallDetectionBackend:
        return self.backend or TrackNetBackend(device=self.device)


@dataclass
class TrackNetBackend:
    """Real TrackNet inference: lazy torch, no import cost.

    Imports the heavy stack only when :meth:`detect_ball` is first called, so this module stays
    import-safe without the ``ball`` extra installed.
    """

    weights: str | None = None
    device: str = "cuda"
    _model: object = field(default=None, init=False, repr=False)

    def detect_ball(self, clip: ClipRef) -> RawBallDetections:  # pragma: no cover - heavy path
        self._load()
        raise NotImplementedError(
            "TrackNet inference is not wired yet (roadmap M1): unlike the rfdetr/bytetrack "
            "reals, the `ball` extra ships only torch — no TrackNet weights/decoder — so this "
            "backend cannot run even once the extra is installed. The pure half (threshold + "
            "gap-fill) is complete and tested; inject a BallDetectionBackend that yields your "
            "own per-frame heatmap peaks, or keep the fake (`--ball fake`) until it is wired."
        )

    def _load(self) -> object:  # pragma: no cover - exercised only without the extra
        if self._model is None:
            try:
                import torch  # noqa: F401  (stand-in for the TrackNet stack)
            except ImportError as exc:
                raise RuntimeError(
                    "TrackNet is not installed. Install the ball extra: "
                    "`pip install 'pitch3d[ball]'`, or inject a BallDetectionBackend."
                ) from exc
            self._model = object()
        return self._model
