"""Pose-motion sync — minimal walk-cycle from root velocity.

Simplest possible fix for the "standing pose that walks" symptom detected
by :mod:`.pose_motion_probe`. When a subject's root moves but joints stay
still, we synthesise a small **procedural knee/hip swing** proportional
to the root velocity — enough to break the visible glide, deliberately
short of pretending it's a real walk cycle (that's step 5 / PACER+).

Algorithm per subject:

* Detect "still-pose while moving" frames (same rule as ``pose_motion_probe``).
* For each such frame, add a small oscillation to the knee joints (indices
  4/5 = left/right knee in SMPL-X body_pose) whose amplitude scales with
  root speed and phase advances with cumulative distance. Also add a hip
  counter-swing (indices 1/2) so the CoM stays plausible.

R-6: this is procedural, not measured. Frames whose pose was patched get
their ``subject_frame_conf`` stamped at ``PATCHED_CONF`` (0.20) so
attention lists surface them for the operator.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..config.gates import PoseMotionSyncConfig
from ..scene.layers import ConfidenceMap, Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene
from .engine import make_keyframes, resolve_subject_motion

#: Confidence stamped on synthesized walk-cycle frames — R-6 tag.
PATCHED_CONF = 0.20

#: SMPL-X body_pose joint indices — 21 body joints, 0-indexed.
#:   1 = left hip, 2 = right hip, 4 = left knee, 5 = right knee.
JOINT_HIP_L, JOINT_HIP_R = 1, 2
JOINT_KNEE_L, JOINT_KNEE_R = 4, 5


@dataclass
class SubjectPoseSyncReport:
    track_id: int
    n_frames: int = 0
    patched_frames: int = 0
    max_amplitude_rad: float = 0.0


@dataclass
class PoseMotionSyncReport:
    n_subjects: int = 0
    subjects_patched: int = 0
    corrections_added: int = 0
    total_patched_frames: int = 0
    subjects: list[SubjectPoseSyncReport] = field(default_factory=list)


def _mark_low_conf(
    scene: Scene, track_id: int, frames: np.ndarray,
    row_indices: np.ndarray, conf: float,
) -> None:
    """Stamp confidence map — mutates ``scene.confidence`` in place."""
    if scene.confidence is None:
        scene.confidence = ConfidenceMap()
    frame_conf = dict(scene.confidence.subject_frame_conf)
    existing = frame_conf.get(track_id)
    if existing is None or len(existing) != len(frames):
        existing = np.ones(len(frames), dtype=float)
    else:
        existing = np.asarray(existing, dtype=float).copy()
    for r in row_indices:
        if 0 <= r < existing.shape[0]:
            existing[r] = float(conf)
    frame_conf[track_id] = existing
    scene.confidence = replace(scene.confidence, subject_frame_conf=frame_conf)


def pose_motion_sync_gate(
    scene: Scene, cfg: PoseMotionSyncConfig | None = None, *, fps: float = 25.0,
) -> tuple[Scene, PoseMotionSyncReport]:
    """Patch minimal walk-cycle on desynced frames. Return new scene + report."""
    cfg = cfg if cfg is not None else PoseMotionSyncConfig()
    report = PoseMotionSyncReport(n_subjects=len(scene.subjects))
    if not cfg.enabled or fps <= 0:
        return scene, report

    auto_corrs: list[Correction] = []
    for s in scene.subjects:
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        frames = np.asarray(resolved.pose.frames, dtype=int)
        transl = np.asarray(resolved.pose.transl, dtype=float)
        body_pose = np.asarray(resolved.pose.body_pose, dtype=float)
        n = transl.shape[0]
        r = SubjectPoseSyncReport(track_id=int(s.track_id), n_frames=n)
        if n < 2 or body_pose.shape[0] != n or body_pose.ndim < 3:
            report.subjects.append(r)
            continue

        dt = np.diff(frames.astype(float)) / fps
        vel = np.zeros((n, 3))
        vel[1:] = np.diff(transl, axis=0) / np.where(dt[:, None] > 0, dt[:, None], 1.0)
        speed = np.linalg.norm(vel, axis=1)
        dpose_mag = np.zeros(n)
        dpose_mag[1:] = np.linalg.norm(np.diff(body_pose, axis=0), axis=(1, 2))
        moving = speed > cfg.velocity_threshold_mps
        still_pose = dpose_mag < cfg.joint_activity_threshold
        desync = moving & still_pose
        if not desync.any():
            report.subjects.append(r)
            continue

        # cumulative distance for phase (only over the whole track, not just
        # desync frames — otherwise the phase discontinuously resets)
        cum_dist = np.concatenate([[0.0], np.cumsum(speed[1:] * dt)])
        phase = 2.0 * np.pi * cfg.strides_per_metre * cum_dist

        new_body = body_pose.copy()
        max_amp = 0.0
        for idx in np.where(desync)[0]:
            amplitude = cfg.knee_amplitude_rad * min(
                1.0, speed[idx] / max(cfg.full_speed_mps, 1e-6),
            )
            max_amp = max(max_amp, amplitude)
            new_body[idx, JOINT_KNEE_L, 0] = amplitude * np.sin(phase[idx])
            new_body[idx, JOINT_KNEE_R, 0] = amplitude * np.sin(phase[idx] + np.pi)
            hip_amp = cfg.hip_amplitude_rad * min(
                1.0, speed[idx] / max(cfg.full_speed_mps, 1e-6),
            )
            new_body[idx, JOINT_HIP_L, 0] = -0.5 * hip_amp * np.sin(phase[idx])
            new_body[idx, JOINT_HIP_R, 0] = -0.5 * hip_amp * np.sin(phase[idx] + np.pi)

        # emit per-joint corrections that touch the affected joints
        for j in (JOINT_HIP_L, JOINT_HIP_R, JOINT_KNEE_L, JOINT_KNEE_R):
            if np.allclose(new_body[:, j, :], body_pose[:, j, :]):
                continue
            auto_corrs.append(
                make_keyframes(
                    f"auto-pose-sync-{s.track_id}-j{j}",
                    CorrectionTarget(
                        kind=TargetKind.POSE_BODY_JOINT,
                        subject_track_id=s.track_id,
                        joint_index=j,
                    ),
                    (int(frames[0]), int(frames[-1])),
                    key_frames=frames.astype(float),
                    key_values=new_body[:, j, :],
                    interp="slerp",
                    note=(
                        f"auto pose-motion sync: joint {j}, "
                        f"{int(desync.sum())} desync frames patched"
                    ),
                )
            )
        r.patched_frames = int(desync.sum())
        r.max_amplitude_rad = float(max_amp)
        report.subjects_patched += 1
        report.total_patched_frames += r.patched_frames
        report.subjects.append(r)
        _mark_low_conf(scene, s.track_id, frames, np.where(desync)[0], PATCHED_CONF)

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report


__all__ = [
    "PATCHED_CONF",
    "PoseMotionSyncConfig",
    "PoseMotionSyncReport",
    "pose_motion_sync_gate",
]
