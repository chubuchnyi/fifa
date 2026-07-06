"""Stride cadence probe — running speed must correlate with knee-swing frequency.

A player running at 6 m/s takes ~4 strides per second. At 2 m/s (walking)
they take ~1.5 strides per second. If the reconstructed knee-swing
frequency is way OUT of that range for a given root speed, the pose is
inconsistent with the motion (HMR failure mode).

Metric per subject:

  cadence_hz = knee-swing frequency measured via zero-crossings of
                d/dt(body_pose[knee, 0]).
  expected_cadence = strides_per_metre · root_speed_mps
  ratio = cadence_hz / expected_cadence.

Flag when ratio is outside [0.5, 2.0] over meaningful motion windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..scene.scene import Scene
from .engine import resolve_subject_motion


@dataclass(frozen=True)
class StrideProbeConfig:
    enabled: bool = False
    velocity_threshold_mps: float = 1.5
    #: Only flag subjects whose MEAN speed over moving frames exceeds this —
    #: we don't care about walk-cycle for a subject who barely moves.
    strict_flag_speed_mps: float = 3.0
    #: Below this cadence we consider it "no walk cycle at all"; above,
    #: we look at the speed-vs-cadence ratio.
    min_cadence_hz: float = 1.0
    strides_per_metre: float = 0.7
    ratio_min: float = 0.5
    ratio_max: float = 2.0
    #: Joint index to sample (SMPL-X body_pose left knee = 4).
    knee_joint_idx: int = 4


@dataclass
class SubjectStrideReport:
    track_id: int
    n_frames: int = 0
    moving_frames: int = 0
    cadence_hz: float = 0.0
    expected_cadence_hz: float = 0.0
    ratio: float = 0.0
    is_off: bool = False


@dataclass
class StrideProbeReport:
    n_subjects: int = 0
    subjects_off: int = 0
    subjects: list[SubjectStrideReport] = field(default_factory=list)


def _zero_crossings(x: np.ndarray) -> int:
    """Count sign-change zero crossings."""
    if x.shape[0] < 2:
        return 0
    signs = np.sign(x - x.mean())
    return int(np.sum(np.diff(signs) != 0))


def stride_probe(
    scene: Scene,
    cfg: StrideProbeConfig | None = None,
    *,
    fps: float = 25.0,
) -> StrideProbeReport:
    cfg = cfg if cfg is not None else StrideProbeConfig()
    report = StrideProbeReport(n_subjects=len(scene.subjects))
    if not cfg.enabled or fps <= 0:
        return report

    for s in scene.subjects:
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        frames = np.asarray(resolved.pose.frames, dtype=float)
        transl = np.asarray(resolved.pose.transl, dtype=float)
        body_pose = np.asarray(resolved.pose.body_pose, dtype=float)
        n = transl.shape[0]
        r = SubjectStrideReport(track_id=int(s.track_id), n_frames=n)
        if n < 6 or body_pose.ndim < 3 or body_pose.shape[1] <= cfg.knee_joint_idx:
            report.subjects.append(r)
            continue

        dt = np.diff(frames) / fps
        ok = dt > 0
        vel = np.zeros((n - 1, 3))
        vel[ok] = np.diff(transl, axis=0)[ok] / dt[ok, None]
        speed = np.linalg.norm(vel[:, :2], axis=1)
        moving = speed > cfg.velocity_threshold_mps
        r.moving_frames = int(moving.sum())
        if r.moving_frames < 3:
            report.subjects.append(r)
            continue

        # Knee flex sign series over moving frames only
        knee = body_pose[:, cfg.knee_joint_idx, 0]
        # Restrict to moving frames (drop last row since velocity is intervals)
        moving_ext = np.concatenate([moving, moving[-1:]])
        knee_moving = knee[moving_ext]
        if knee_moving.shape[0] < 4:
            report.subjects.append(r)
            continue
        n_crossings = _zero_crossings(knee_moving)
        window_sec = knee_moving.shape[0] / fps
        r.cadence_hz = float(n_crossings / max(window_sec, 1e-6) / 2.0)
        mean_speed = float(speed[moving].mean())
        r.expected_cadence_hz = cfg.strides_per_metre * mean_speed
        if r.expected_cadence_hz > 1e-6:
            r.ratio = r.cadence_hz / r.expected_cadence_hz
        else:
            r.ratio = 0.0
        # Only flag subjects who are clearly RUNNING (mean speed above the
        # strict threshold) AND whose cadence doesn't match. Walking
        # subjects with low cadence are physiologically normal — don't
        # false-positive them.
        if mean_speed >= cfg.strict_flag_speed_mps:
            r.is_off = (
                r.cadence_hz < cfg.min_cadence_hz
                or r.ratio < cfg.ratio_min
                or r.ratio > cfg.ratio_max
            )
        if r.is_off:
            report.subjects_off += 1
        report.subjects.append(r)

    return report


__all__ = [
    "StrideProbeConfig",
    "StrideProbeReport",
    "SubjectStrideReport",
    "stride_probe",
]
