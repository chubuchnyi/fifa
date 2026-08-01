"""Gravity probe — airborne subject vertical acceleration must be ≈ -9.81 m/s².

For a subject whose foot Z is above the pitch (airborne / mid-jump), gravity
IS the only vertical force. Their vertical velocity must decrease at 9.81 m/s².
If it doesn't — the subject is levitating or accelerating up mid-air without
a jump initiation. Measurement-only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from ..scene.scene import Scene, Subject
from .engine import resolve_subject_motion

FootPositionProvider = Callable[[Subject], "np.ndarray | None"]

G_MPS2 = 9.81


@dataclass(frozen=True)
class GravityConfig:
    enabled: bool = False
    #: Foot Z above this = airborne (must obey gravity).
    airborne_z_threshold_m: float = 0.10
    #: Tolerance around -9.81 m/s²; anything outside flagged.
    tolerance_mps2: float = 5.0
    #: Ignore short air phases (jitter above the floor).
    min_airborne_run_frames: int = 3


@dataclass
class SubjectGravityReport:
    track_id: int
    n_frames: int = 0
    airborne_frames: int = 0
    airborne_runs: int = 0
    mean_vertical_accel_mps2: float = 0.0
    max_deviation_mps2: float = 0.0
    is_violating: bool = False


@dataclass
class GravityReport:
    n_subjects: int = 0
    subjects_violating: int = 0
    total_airborne_frames: int = 0
    max_deviation_mps2: float = 0.0
    subjects: list[SubjectGravityReport] = field(default_factory=list)


def _find_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    n = mask.shape[0]
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if j - i + 1 >= min_len:
            runs.append((i, j))
        i = j + 1
    return runs


def gravity_probe(
    scene: Scene,
    cfg: GravityConfig | None = None,
    foot_position_provider: FootPositionProvider | None = None,
    *,
    fps: float = 25.0,
    floor_z: float = 0.0,
) -> GravityReport:
    cfg = cfg if cfg is not None else GravityConfig()
    report = GravityReport(n_subjects=len(scene.subjects))
    if not cfg.enabled or fps <= 0 or foot_position_provider is None:
        return report

    for s in scene.subjects:
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        resolved_subject = replace(s, proposal=resolved)
        pos = foot_position_provider(resolved_subject)
        if pos is None:
            continue
        pos = np.asarray(pos, dtype=float)
        n = pos.shape[0]
        r = SubjectGravityReport(track_id=int(s.track_id), n_frames=n)
        if n < 4:
            report.subjects.append(r)
            continue

        transl = np.asarray(resolved.pose.transl, dtype=float)
        z = transl[:, 2]
        dt = 1.0 / fps
        vz = np.diff(z) / dt
        az = np.diff(vz) / dt

        airborne = pos[:, 2] > floor_z + cfg.airborne_z_threshold_m
        r.airborne_frames = int(airborne.sum())
        runs = _find_runs(airborne, cfg.min_airborne_run_frames)
        r.airborne_runs = len(runs)

        accels: list[float] = []
        for a, b in runs:
            end = min(b + 1, az.shape[0])
            if a >= end - 1:
                continue
            run_az = az[a:end]
            mean_a = float(run_az.mean())
            accels.append(mean_a)
            deviation = abs(mean_a - (-G_MPS2))
            r.max_deviation_mps2 = max(r.max_deviation_mps2, deviation)
        if accels:
            r.mean_vertical_accel_mps2 = float(np.mean(accels))
        r.is_violating = r.max_deviation_mps2 > cfg.tolerance_mps2
        if r.is_violating:
            report.subjects_violating += 1
        report.max_deviation_mps2 = max(report.max_deviation_mps2, r.max_deviation_mps2)
        report.total_airborne_frames += r.airborne_frames
        report.subjects.append(r)

    return report


__all__ = [
    "GravityConfig",
    "GravityReport",
    "SubjectGravityReport",
    "gravity_probe",
]
