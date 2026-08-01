"""Pose-motion consistency probe.

User's original complaint: "the person stands still by pose, but moves
across the field." The root translates while the body_pose barely
animates — no walk cycle. HMR frequently emits a "T-pose that translates"
when confidence is low.

Metric: correlation between root velocity magnitude and per-joint
angular activity. In real walking:

* Root velocity increases → leg joints rotate more (stride amplitude).
* Root velocity is zero → joints do not swing.

If ``velocity > velocity_threshold`` but ``per-joint angular activity is
below joint_activity_threshold`` for a substantial fraction of frames,
we flag it as pose-motion desync — the standing pose that walks.

Measurement-only. The correction (blend a walk-cycle prior conditioned
on root velocity) is a larger tier of work (PACER+ step 5) and deferred
here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scene.scene import Scene
from .engine import resolve_subject_motion


@dataclass(frozen=True)
class PoseMotionConfig:
    enabled: bool = False
    #: Root speed above which we expect visible joint movement (m/s).
    velocity_threshold_mps: float = 2.0
    #: Per-frame joint activity (Frobenius norm of Δbody_pose across joints).
    joint_activity_threshold: float = 0.10
    #: Fraction of moving-frames without joint activity to flag desync.
    desync_fraction_threshold: float = 0.30


@dataclass
class SubjectPoseMotionReport:
    track_id: int
    n_frames: int = 0
    moving_frames: int = 0
    still_pose_moving_frames: int = 0  # moving root + still joints
    desync_fraction: float = 0.0       # of moving frames
    is_desynced: bool = False


@dataclass
class PoseMotionReport:
    n_subjects: int = 0
    subjects_desynced: int = 0
    max_desync_fraction: float = 0.0
    subjects: list[SubjectPoseMotionReport] = field(default_factory=list)


def pose_motion_probe(
    scene: Scene, cfg: PoseMotionConfig | None = None, *, fps: float = 25.0,
) -> PoseMotionReport:
    """Measure per-subject correlation between root motion and joint activity."""
    cfg = cfg if cfg is not None else PoseMotionConfig()
    report = PoseMotionReport(n_subjects=len(scene.subjects))
    if not cfg.enabled:
        return report
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    for s in scene.subjects:
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        frames = np.asarray(resolved.pose.frames, dtype=float)
        transl = np.asarray(resolved.pose.transl, dtype=float)
        body_pose = np.asarray(resolved.pose.body_pose, dtype=float)
        n = transl.shape[0]
        r = SubjectPoseMotionReport(track_id=int(s.track_id), n_frames=n)
        if n < 2 or body_pose.shape[0] != n or body_pose.ndim < 3:
            report.subjects.append(r)
            continue

        dt = np.diff(frames) / fps
        ok = dt > 0
        if not ok.any():
            report.subjects.append(r)
            continue

        vel = np.diff(transl, axis=0)[ok] / dt[ok, None]
        speed = np.linalg.norm(vel, axis=1)                    # (N-1,)
        # Frobenius norm of per-frame body_pose delta — one scalar per interval
        dpose = np.diff(body_pose, axis=0)                    # (N-1, K, 3)
        dpose_mag = np.linalg.norm(dpose[ok], axis=(1, 2))     # (N-1,)
        # Per-frame activity (rad-scaled)
        moving = speed > cfg.velocity_threshold_mps
        still_pose = dpose_mag < cfg.joint_activity_threshold
        desync = moving & still_pose
        r.moving_frames = int(moving.sum())
        r.still_pose_moving_frames = int(desync.sum())
        r.desync_fraction = float(desync.sum() / max(1, moving.sum()))
        r.is_desynced = (
            r.moving_frames > 0
            and r.desync_fraction > cfg.desync_fraction_threshold
        )
        if r.is_desynced:
            report.subjects_desynced += 1
        report.max_desync_fraction = max(report.max_desync_fraction, r.desync_fraction)
        report.subjects.append(r)

    return report


__all__ = [
    "PoseMotionConfig",
    "PoseMotionReport",
    "SubjectPoseMotionReport",
    "pose_motion_probe",
]
