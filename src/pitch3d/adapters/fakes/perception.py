"""Deterministic perception fakes: detector, tracker+teams, field, 2D ball.

No model, no GPU — just numpy laying out a stable, plausible little scene so the whole
reconstruction spine (and the LLM feedback loop) runs in tests and the dry-run. The layout
is intentionally simple and repeatable: ``n_subjects`` people standing across the lower half
of the frame (one goalkeeper, one referee, the rest outfield players split across two teams)
plus a ball sweeping across the pitch. The "perfect tracker" trusts the detector's stable
ordering, which is exactly what a fake should do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitch3d.core.ports.io import ClipRef
from pitch3d.core.ports.perception import (
    Detection,
    Detections,
    Detector,
    FieldCalibrator,
    FrameDetections,
    BallTracker,
    Tracker,
    Tracklet,
    Tracks,
)
from pitch3d.core.scene.field import FieldCalibration
from pitch3d.core.scene.motion import Ball2DTrack
from pitch3d.core.scene.provenance import Backend, ModelInfo
from pitch3d.core.scene.subject import Team

#: Pitch span (m) the full image width maps to in the fake homography.
_FAKE_PITCH_SPAN_M = 30.0


def _role_for(index: int, n: int) -> str:
    """Stable class label for subject ``index`` of ``n`` (matches detector ↔ tracker)."""
    if index == 0:
        return "goalkeeper"
    if index == n - 1 and n >= 3:
        return "referee"
    return "player"


def _subject_bbox(index: int, n: int, frame: int, width: int, height: int) -> np.ndarray:
    """Deterministic (x0,y0,x1,y1) for subject ``index`` at ``frame``."""
    cx = width * (index + 1) / (n + 1) + 0.5 * np.sin(0.1 * frame + index)
    cy = height * 0.6
    hw, hh = width * 0.025, height * 0.12
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh], dtype=float)


def _ball_xy(frame: int, n_frames: int, width: int, height: int) -> np.ndarray:
    """Deterministic 2D ball position sweeping across the frame."""
    u = width * (0.1 + 0.8 * (frame / max(n_frames - 1, 1)))
    v = height * (0.5 - 0.15 * np.sin(np.pi * frame / max(n_frames - 1, 1)))
    return np.array([u, v], dtype=float)


@dataclass
class FakeDetector(Detector):
    """Emits ``n_subjects`` people + one ball per frame, in a stable order."""

    n_subjects: int = 4

    def info(self) -> ModelInfo:
        return ModelInfo(name="FakeDetector", backend=Backend.FAKE)

    def detect(self, clip: ClipRef) -> Detections:
        n = self.n_subjects
        frames: list[FrameDetections] = []
        for f in clip.frames.tolist():
            items = [
                Detection(
                    bbox_xyxy=_subject_bbox(i, n, f, clip.width, clip.height),
                    cls=_role_for(i, n),
                    score=0.95,
                )
                for i in range(n)
            ]
            items.append(
                Detection(
                    bbox_xyxy=np.array([0, 0, 8, 8], dtype=float)
                    + np.concatenate([_ball_xy(f, clip.n_frames, clip.width, clip.height)] * 2),
                    cls="ball",
                    score=0.9,
                )
            )
            frames.append(FrameDetections(frame=int(f), items=items))
        return Detections(frames=frames)


@dataclass
class FakeTracker(Tracker):
    """Perfect tracker: turns the detector's stable per-frame order into tracklets + teams."""

    def info(self) -> ModelInfo:
        return ModelInfo(name="FakeTracker", backend=Backend.FAKE)

    def track(self, clip: ClipRef, detections: Detections) -> Tracks:
        teams = [
            Team(id="A", name="Team A", color_rgb=(0.8, 0.1, 0.1)),
            Team(id="B", name="Team B", color_rgb=(0.1, 0.1, 0.8)),
        ]
        # Subjects = non-ball detections, assumed stable in count + order across frames.
        per_frame = [
            [d for d in fd.items if d.cls != "ball"] for fd in detections.frames
        ]
        n = min((len(p) for p in per_frame), default=0)
        frames = np.array([fd.frame for fd in detections.frames], dtype=int)
        tracklets: list[Tracklet] = []
        for i in range(n):
            cls = per_frame[0][i].cls
            bboxes = np.stack([per_frame[f][i].bbox_xyxy for f in range(len(per_frame))])
            team_id = None if cls == "referee" else ("A" if i % 2 == 0 else "B")
            tracklets.append(
                Tracklet(track_id=i, frames=frames, bboxes_xyxy=bboxes, cls=cls, team_id=team_id)
            )
        return Tracks(tracklets=tracklets, teams=teams)


@dataclass
class FakeFieldCalibrator(FieldCalibrator):
    """A static, invertible image→world(plane) homography with high confidence."""

    span_m: float = _FAKE_PITCH_SPAN_M
    confidence: float = 0.95

    def info(self) -> ModelInfo:
        return ModelInfo(name="FakeFieldCalibrator", backend=Backend.FAKE)

    def calibrate(self, clip: ClipRef) -> FieldCalibration:
        scale = self.span_m / clip.width  # meters per pixel
        cx, cy = clip.width / 2.0, clip.height / 2.0
        h = np.array(
            [[scale, 0.0, -cx * scale], [0.0, -scale, cy * scale], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        t = clip.n_frames
        return FieldCalibration(
            homographies=np.tile(h, (t, 1, 1)),
            frames=clip.frames,
            confidence=np.full(t, self.confidence),
        )


@dataclass
class FakeBallTracker(BallTracker):
    """2D ball track sweeping the frame; the 3D ballistic lift stays in core."""

    confidence: float = 0.9

    def info(self) -> ModelInfo:
        return ModelInfo(name="FakeBallTracker", backend=Backend.FAKE)

    def track_ball(self, clip: ClipRef) -> Ball2DTrack:
        pts = np.stack(
            [_ball_xy(int(f), clip.n_frames, clip.width, clip.height) for f in clip.frames]
        )
        return Ball2DTrack(
            frames=clip.frames,
            positions_2d=pts,
            confidence=np.full(clip.n_frames, self.confidence),
        )
