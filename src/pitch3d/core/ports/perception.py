"""Per-frame / 2D perception ports: detection, tracking, field calibration, ball.

These produce the *intermediate* artifacts the orchestration assembles into a scene.
Pose (HMR) is separate (`pose.py`) because it is central and also exposes re-fit.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..scene.field import FieldCalibration
from ..scene.motion import Ball2DTrack
from ..scene.subject import Team
from .base import ModelProvider
from .io import ClipRef


# --- Detection (FR-5) ----------------------------------------------------------
@dataclass
class Detection:
    """One detected object in one frame."""

    bbox_xyxy: np.ndarray  # (4,) image px
    cls: str               # "player" | "goalkeeper" | "referee" | "ball"
    score: float

    def __post_init__(self) -> None:
        self.bbox_xyxy = np.asarray(self.bbox_xyxy, dtype=float).reshape(4)


@dataclass
class FrameDetections:
    frame: int
    items: list[Detection] = field(default_factory=list)


@dataclass
class Detections:
    """Per-frame detections over a clip."""

    frames: list[FrameDetections] = field(default_factory=list)


class Detector(ModelProvider):
    """Detects players/goalkeepers/referees/ball per frame (FR-5).

    Self-hosted default: RF-DETR (`roboflow/sports`); YOLO families for speed.
    """

    @abstractmethod
    def detect(self, clip: ClipRef) -> Detections:
        """Return per-frame detections for the clip."""
        raise NotImplementedError


# --- Tracking + teams (FR-6) ---------------------------------------------------
@dataclass
class Tracklet:
    """A stable identity over time."""

    track_id: int
    frames: np.ndarray      # (T,)
    bboxes_xyxy: np.ndarray  # (T, 4)
    cls: str
    team_id: str | None = None

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=int).reshape(-1)
        self.bboxes_xyxy = np.asarray(self.bboxes_xyxy, dtype=float).reshape(-1, 4)


@dataclass
class Tracks:
    """Tracklets + the team definitions the classifier produced."""

    tracklets: list[Tracklet] = field(default_factory=list)
    teams: list[Team] = field(default_factory=list)


class Tracker(ModelProvider):
    """Associates detections into stable IDs and classifies teams (FR-6).

    Default: ByteTrack / BoT-SORT + a re-id / appearance-clustering team classifier.
    """

    @abstractmethod
    def track(self, clip: ClipRef, detections: Detections) -> Tracks:
        """Return stable tracklets with team labels."""
        raise NotImplementedError


# --- Field calibration (FR-7) --------------------------------------------------
class FieldCalibrator(ModelProvider):
    """Estimates the per-frame pitch homography — the mono world anchor (FR-7).

    Default: a pitch-keypoint model + ``cv2.findHomography`` + temporal smoothing.
    Returns the canonical :class:`FieldCalibration` (image→world-plane, with confidence).
    """

    @abstractmethod
    def calibrate(self, clip: ClipRef) -> FieldCalibration:
        """Return the per-frame homography track with confidence."""
        raise NotImplementedError


# --- Ball 2D tracking (FR-9; 3D lift is core math) -----------------------------
class BallTracker(ModelProvider):
    """Tracks the ball in 2D (FR-9). The 2D→3D ballistic lift lives in the core
    (``orchestration.ball_lift``), not here, so the uncertain height math is testable
    without a model."""

    @abstractmethod
    def track_ball(self, clip: ClipRef) -> Ball2DTrack:
        """Return the 2D ball track with per-frame detection confidence."""
        raise NotImplementedError
