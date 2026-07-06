"""Reconstruction pipeline — the DAG spine that turns a clip into a proposal.

Dependency-injected over ports only (ADR-0001): the pipeline imports no adapter. Each stage
runs through :func:`run_cached` (cache → queue), so the spine is identical whether the ports
are fakes (tests, dry-run) or real models. The output is the *proposal* layer; corrections
and resolve live in :mod:`pitch3d.core.correction`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..ports.cache import Cache
from ..ports.io import ClipRef
from ..ports.jobs import JobQueue
from ..ports.perception import (
    BallTracker,
    Detections,
    Detector,
    FieldCalibrator,
    Tracker,
    Tracks,
)
from ..ports.pose import PoseEstimator
from ..config.gates import IdentityConfig
from ..scene.field import FieldCalibration
from ..scene.motion import Ball2DTrack, BallTrack, SubjectMotion
from .ball_lift import lift_ball_to_3d
from .continuity import StitchConfig, StitchReport, stitch_tracks_with_report
from .identity import AppearanceProvider, IdentityReport, identity_gate
from .stages import Stage, StageRun, clip_hash, run_cached


@dataclass
class ReconstructionResult:
    """Everything the reconstruction stages produce for one clip (the proposal layer)."""

    detections: Detections
    tracks: Tracks
    calibration: FieldCalibration
    motions: dict[int, SubjectMotion]
    ball_2d: Ball2DTrack
    ball_3d: BallTrack
    runs: list[StageRun] = field(default_factory=list)
    stitch: StitchReport | None = None  # set iff continuity stitching ran (else None)
    identity: IdentityReport | None = None  # set iff identity_gate ran


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
    #: Continuity stitching between TRACK and POSE. ``None`` = off (default, legacy behavior);
    #: a config re-links fragmented tracklets so each identity is posed once (see continuity.py).
    stitch_cfg: StitchConfig | None = None
    #: Identity gate (GTA-style intra-track split + cross-track merge) between STITCH and POSE.
    #: When ``identity_cfg is not None`` AND ``appearance_provider is not None``, the gate runs
    #: on the tracks before POSE consumes them, so each identity is posed once with a clean
    #: unimodal appearance distribution.
    identity_cfg: IdentityConfig | None = None
    #: Callable that returns per-frame appearance features for a tracklet
    #: (``(T, D)`` float array or ``None``). Wire a real Re-ID backbone here.
    appearance_provider: AppearanceProvider | None = None

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

        def stage(s: Stage, thunk, *, extra_params: dict | None = None):
            p = {**params.get(s, {}), **extra_params} if extra_params else params.get(s, {})
            r = run_cached(
                self.queue,
                self.cache,
                s,
                thunk,
                input_hash=ih,
                params=p,
                model_version=self.model_version,
            )
            runs.append(r)
            return r.result

        det = stage(Stage.DETECT, lambda: self.detector.detect(clip))
        trk = stage(Stage.TRACK, lambda: self.tracker.track(clip, det))

        # Structural continuity: re-link fragmented tracklets BEFORE pose, so each identity
        # is posed once. Off by default. The config is folded into POSE's cache params so
        # toggling stitch (or its thresholds) correctly invalidates the pose cache.
        stitch_report: StitchReport | None = None
        pose_extra: dict | None = None
        if self.stitch_cfg is not None:
            trk, stitch_report = stitch_tracks_with_report(trk, self.stitch_cfg)
            pose_extra = {"stitch": asdict(self.stitch_cfg)}

        # Identity gate: GTA split + cross-track merge. Runs AFTER stitch so it
        # sees the already-glued fragments and only handles the cases stitch
        # can't reach (intra-track ID swaps + long-gap same-identity pairs).
        identity_report: IdentityReport | None = None
        if self.identity_cfg is not None and self.identity_cfg.enabled:
            trk, identity_report = identity_gate(
                trk, self.identity_cfg, self.appearance_provider,
            )
            pose_extra = pose_extra or {}
            pose_extra["identity"] = asdict(self.identity_cfg)

        cal = stage(Stage.CALIBRATE, lambda: self.calibrator.calibrate(clip))
        motions = stage(
            Stage.POSE, lambda: self.pose.estimate(clip, trk, cal), extra_params=pose_extra
        )
        ball2d = stage(Stage.BALL, lambda: self.ball.track_ball(clip))
        ball3d = lift_ball_to_3d(
            ball2d, cal, on_ground=on_ground, motions=motions, fps=clip.fps
        )

        return ReconstructionResult(
            detections=det,
            tracks=trk,
            calibration=cal,
            motions=motions,
            ball_2d=ball2d,
            ball_3d=ball3d,
            runs=runs,
            stitch=stitch_report,
            identity=identity_report,
        )
