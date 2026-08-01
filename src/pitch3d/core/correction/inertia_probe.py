"""Inertia probe — detects impossible SUBJECT-scale angular acceleration.

A running person's whole-body angular momentum is bounded by the torque
their contact forces can generate (Newton). We proxy that with:

* per-frame torso spin rate ω(t) = rate of change of root global_orient
  yaw;
* per-frame angular acceleration α(t) = dω/dt.

If ``|α|`` exceeds the human ceiling (~15 rad/s² for a whole-body twist —
elite skater during a spin start), the subject "snaps" in yaw — not
kinematically feasible without external torque.

Measurement-only. The orientation gate (T1c) already caps ω magnitude
per interval; α catches the RATE of that cap being applied (a signal M3-9
misses).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scene.scene import Scene
from .engine import resolve_subject_motion


@dataclass(frozen=True)
class InertiaConfig:
    enabled: bool = False
    max_alpha_rad_s2: float = 15.0     # torso angular acceleration ceiling


@dataclass
class SubjectInertiaReport:
    track_id: int
    n_frames: int = 0
    alpha_max_rad_s2: float = 0.0
    alpha_viol: int = 0


@dataclass
class InertiaReport:
    n_subjects: int = 0
    subjects_flagged: int = 0
    max_alpha_rad_s2: float = 0.0
    total_alpha_viol: int = 0
    subjects: list[SubjectInertiaReport] = field(default_factory=list)


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def inertia_probe(
    scene: Scene, cfg: InertiaConfig | None = None, *, fps: float = 25.0,
) -> InertiaReport:
    """Measure per-subject torso angular acceleration; never mutate scene."""
    cfg = cfg if cfg is not None else InertiaConfig()
    report = InertiaReport(n_subjects=len(scene.subjects))
    if not cfg.enabled or fps <= 0:
        return report

    for s in scene.subjects:
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        frames = np.asarray(resolved.pose.frames, dtype=float)
        orient = np.asarray(resolved.pose.global_orient, dtype=float)
        n = orient.shape[0]
        r = SubjectInertiaReport(track_id=int(s.track_id), n_frames=n)
        if n < 3:
            report.subjects.append(r)
            continue

        dt = np.diff(frames) / fps
        ok = dt > 0
        if not ok.any():
            report.subjects.append(r)
            continue

        yaw = orient[:, 2]
        d_yaw = _wrap_to_pi(np.diff(yaw))
        omega = np.zeros(n - 1)
        omega[ok] = d_yaw[ok] / dt[ok]
        d_omega = np.diff(omega)
        dt2 = dt[1:]
        ok2 = (dt2 > 0) & ok[1:]
        alpha = np.zeros(dt2.shape[0])
        alpha[ok2] = d_omega[ok2] / dt2[ok2]
        if alpha.size:
            r.alpha_max_rad_s2 = float(np.abs(alpha).max())
            r.alpha_viol = int((np.abs(alpha) > cfg.max_alpha_rad_s2).sum())
        report.max_alpha_rad_s2 = max(report.max_alpha_rad_s2, r.alpha_max_rad_s2)
        report.total_alpha_viol += r.alpha_viol
        if r.alpha_viol > 0:
            report.subjects_flagged += 1
        report.subjects.append(r)

    return report


__all__ = [
    "InertiaConfig",
    "InertiaReport",
    "SubjectInertiaReport",
    "inertia_probe",
]
