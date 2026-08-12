"""Reconstruction pipeline — the DAG spine that turns a clip into a proposal.

Dependency-injected over ports only (ADR-0001): the pipeline imports no adapter. Each stage
runs through :func:`run_cached` (cache → queue), so the spine is identical whether the ports
are fakes (tests, dry-run) or real models. The output is the *proposal* layer; corrections
and resolve live in :mod:`pitch3d.core.correction`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..config.gates import IdentityConfig
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
from ..scene.field import FieldCalibration
from ..scene.motion import Ball2DTrack, BallTrack, SubjectMotion
from .ball_lift import lift_ball_to_3d
from .continuity import StitchConfig, StitchReport, stitch_tracks_with_report
from .identity import AppearanceProvider, IdentityReport, identity_gate
from .stages import Stage, StageRun, clip_hash, run_cached


class UnsolvedCalibrationError(RuntimeError):
    """Raised when CALIBRATE returned a homography track that solved no frame at all."""


def require_solved_calibration(cal: FieldCalibration, *, min_solved: int = 1) -> int:
    """Refuse a calibration that carries no measurement, and return how many frames were solved.

    Both calibrators already say so honestly: a frame they could not solve is stored as the last
    good homography — or ``eye(3)`` if there has never been one — at **confidence exactly 0**. What
    was missing is anyone reading it. A run whose every frame is unsolved exports ``eye(3)``, i.e.
    "one image pixel is one metre", and every world coordinate downstream is that fiction: #125 got
    34/60 identity frames, 11 subjects instead of 23, and not one line of complaint.

    This is the failure that cannot be caught by eye later, because the ground overlay is drawn
    through the *same* homography that produced the error, so the pitch always looks right — and
    ``apply_rigid_camera.py`` then replaces the calibration outright, which makes a dead scene score
    a healthy 1.0 px. It has to be caught here or not at all.

    ``min_solved`` is the dial (auto default, manual override): 1 refuses only the indefensible
    "no answer anywhere" case. How many *carried* frames are acceptable is a judgment about drift,
    so it is left to the caller rather than guessed here.
    """
    solved = int((np.asarray(cal.confidence, dtype=float) > 0.0).sum())
    if solved < min_solved:
        total = int(np.asarray(cal.confidence).size)
        raise UnsolvedCalibrationError(
            f"calibration solved {solved}/{total} frames (need >= {min_solved}). The homography "
            f"track is the carried fallback, not a measurement — check that the calibrator's "
            f"weights actually loaded before trusting any world coordinate from this run."
        )
    return solved


def describe_calibration_solve(cal: FieldCalibration) -> str:
    """One line saying how much of a calibration was measured rather than carried (#131).

    ``require_solved_calibration`` deliberately leaves the drift judgement to the caller — but no
    caller was given the number to judge. The only calibration figure a run printed was the mean
    confidence over *all* frames, which is a mean over a bimodal distribution: on the vertical fan
    clip it read 0.28 while 43% of frames carried a stale homography, and the two modes were 0.421
    (roots sane) against 0.000 (roots kilometres out). The mean here is over solved frames only,
    for the same reason.
    """
    conf = np.asarray(cal.confidence, dtype=float)
    ok = conf > 0.0
    n, total = int(ok.sum()), int(conf.size)
    line = f"{n}/{total} frame(s) measured, {total - n} carried (confidence 0)"
    if n:
        line += (f" · over measured frames: mean {conf[ok].mean():.3f}, "
                 f"min {conf[ok].min():.3f}, max {conf[ok].max():.3f}")
    return line


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
    #: Minimum CALIBRATE frames that must carry a real solve (confidence > 0) — see
    #: :func:`require_solved_calibration`. Raise it to demand more than "not entirely dead".
    min_solved_frames: int = 1

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

        keys: dict[Stage, str] = {}

        def stage(
            s: Stage, thunk, *, provider=None, upstream=(), extra_params: dict | None = None
        ):
            p = {**params.get(s, {}), **extra_params} if extra_params else params.get(s, {})
            # A stage is keyed on the CLIP, not on what the stage before it produced. So a run
            # that re-tracked but reused the pose cache got poses computed from the OLD tracks:
            # measured 2026-08-12, BoT-SORT re-ran (new track-*.pkl appeared) and the scene still
            # carried ByteTrack's 38 subjects because POSE hit. Naming the upstream entries makes
            # the key a DAG hash instead of a per-stage one.
            if upstream:
                p = {**p, "upstream": list(upstream)}
            # The adapter's own identity belongs in the key. `ModelInfo.params` has been
            # documented as "feeds the cache key" since it was written, but nothing read it —
            # so a run that differed ONLY in an injected backend (`--tracker-backend`,
            # `--pose-backend`, …) hit the previous backend's entry and returned it silently.
            # Measured 2026-08-12: BoT-SORT requested against a ByteTrack cache reproduced
            # ByteTrack's 38 subjects exactly, and nothing in the log said the backend never ran.
            if provider is not None:
                p = {**p, "provider": provider.info().identity()}
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
            keys[s] = r.key
            return r.result

        det = stage(Stage.DETECT, lambda: self.detector.detect(clip), provider=self.detector)
        trk = stage(
            Stage.TRACK, lambda: self.tracker.track(clip, det),
            provider=self.tracker, upstream=(keys[Stage.DETECT],),
        )

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

        cal = stage(
            Stage.CALIBRATE, lambda: self.calibrator.calibrate(clip), provider=self.calibrator
        )
        require_solved_calibration(cal, min_solved=self.min_solved_frames)
        motions = stage(
            Stage.POSE, lambda: self.pose.estimate(clip, trk, cal),
            provider=self.pose, extra_params=pose_extra,
            upstream=(keys[Stage.TRACK], keys[Stage.CALIBRATE]),
        )
        ball2d = stage(Stage.BALL, lambda: self.ball.track_ball(clip), provider=self.ball)
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
