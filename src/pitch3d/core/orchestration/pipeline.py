"""Reconstruction pipeline — the DAG spine that turns a clip into a proposal.

Dependency-injected over ports only (ADR-0001): the pipeline imports no adapter. Each stage
runs through :func:`run_cached` (cache → queue), so the spine is identical whether the ports
are fakes (tests, dry-run) or real models. The output is the *proposal* layer; corrections
and resolve live in :mod:`pitch3d.core.correction`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ports.cache import Cache
from ..ports.io import ClipRef
from ..ports.jobs import JobQueue
from ..ports.perception import BallTracker, Detector, FieldCalibrator, Tracker
from ..ports.pose import PoseEstimator
from ..scene.field import FieldCalibration
from ..scene.motion import Ball2DTrack, BallTrack, SubjectMotion
from .ball_lift import lift_ball_to_3d
from .stages import Stage, StageRun, clip_hash, run_cached


@dataclass
class ReconstructionResult:
    """Everything the reconstruction stages produce for one clip (the proposal layer)."""

    detections: object
    tracks: object
    calibration: FieldCalibration
    motions: dict[int, SubjectMotion]
    ball_2d: Ball2DTrack
    ball_3d: BallTrack
    runs: list[StageRun] = field(default_factory=list)


@dataclass
class ReconstructionPipeline:
    """Runs DETECT → TRACK → CALIBRATE → POSE → BALL(+3D lift) over injected ports."""

    detector: Detector
    tracker: Tracker
    calibrator: FieldCalibrator
    pose: PoseEstimator
    ball: BallTracker
    cache: Cache
    queue: JobQueue
    model_version: str = "fake-0"

    def run(
        self,
        clip: ClipRef,
        *,
        on_ground=None,
        params: dict[Stage, dict] | None = None,
    ) -> ReconstructionResult:
        """Execute the reconstruction DAG and return the proposal-layer artifacts."""
        params = params or {}
        ih = clip_hash(clip)
        runs: list[StageRun] = []

        def stage(s: Stage, thunk):
            r = run_cached(
                self.queue,
                self.cache,
                s,
                thunk,
                input_hash=ih,
                params=params.get(s, {}),
                model_version=self.model_version,
            )
            runs.append(r)
            return r.result

        det = stage(Stage.DETECT, lambda: self.detector.detect(clip))
        trk = stage(Stage.TRACK, lambda: self.tracker.track(clip, det))
        cal = stage(Stage.CALIBRATE, lambda: self.calibrator.calibrate(clip))
        motions = stage(Stage.POSE, lambda: self.pose.estimate(clip, trk, cal))
        ball2d = stage(Stage.BALL, lambda: self.ball.track_ball(clip))
        ball3d = lift_ball_to_3d(ball2d, cal, on_ground=on_ground, fps=clip.fps)

        return ReconstructionResult(
            detections=det,
            tracks=trk,
            calibration=cal,
            motions=motions,
            ball_2d=ball2d,
            ball_3d=ball3d,
            runs=runs,
        )
