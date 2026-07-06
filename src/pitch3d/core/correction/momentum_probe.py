"""Step 4 (measurement) — CoM/momentum consistency probe.

Motion physics require the **centre-of-mass acceleration** and **linear
momentum** to obey Newton's laws given the (measured) ground forces.
Since we don't measure ground reaction forces, we settle for a simpler
diagnostic: **is the CoM XY smooth?** A CoM that darts around inconsistently
with the root trajectory usually means the pose changed under our feet
(HMR jitter) while the root anchored on the pitch — a "floppy torso"
symptom.

Approximation: treat the SMPL-X pelvis as CoM proxy. Per-frame CoM
acceleration is ``d²/dt² root_transl``. If the pelvis acceleration exceeds
the shared kinematic ceiling (already checked by M3-9) OR the CoM
trajectory has high-frequency chatter (short-window variance ≫ long-window
variance) we flag the subject.

This is measurement-only for now. A follow-up correction can smooth the
CoM via low-pass filter constrained to contact frames (Body Momentum
arXiv 2509.09496 loss).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..scene.scene import Scene
from .engine import resolve_subject_motion


@dataclass(frozen=True)
class MomentumProbeConfig:
    """Knobs for :func:`momentum_probe`."""

    enabled: bool = False
    #: Jerk (m/s³) above which the CoM is flagged as chatty.
    jerk_threshold_mps3: float = 100.0


@dataclass
class SubjectMomentumReport:
    track_id: int
    n_frames: int = 0
    accel_max_mps2: float = 0.0
    jerk_max_mps3: float = 0.0    # max ‖d(accel)/dt‖
    is_chatty: bool = False


@dataclass
class MomentumProbeReport:
    n_subjects: int = 0
    subjects_chatty: int = 0
    max_jerk_mps3: float = 0.0
    max_accel_mps2: float = 0.0
    subjects: list[SubjectMomentumReport] = field(default_factory=list)


def _windowed_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling-window std along axis 0 (returns length-N array; edges hold nearest)."""
    n = x.shape[0]
    if n == 0:
        return np.zeros(0)
    w = min(max(1, window), n)
    if w == 1:
        return np.zeros(n)
    out = np.zeros(n)
    half = w // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = float(x[lo:hi].std())
    return out


def momentum_probe(
    scene: Scene,
    cfg: MomentumProbeConfig | None = None,
    *,
    fps: float = 25.0,
) -> MomentumProbeReport:
    """Measure per-subject CoM chatter + acceleration; never mutate scene."""
    cfg = cfg if cfg is not None else MomentumProbeConfig()
    report = MomentumProbeReport(n_subjects=len(scene.subjects))
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
        n = transl.shape[0]
        r = SubjectMomentumReport(track_id=int(s.track_id), n_frames=n)
        if n < 3:
            report.subjects.append(r)
            continue
        dt = np.diff(frames) / fps
        vel = np.diff(transl, axis=0)[dt > 0] / dt[dt > 0, None]
        accel = np.linalg.norm(np.diff(vel, axis=0), axis=1) / dt[1:][dt[1:] > 0]
        r.accel_max_mps2 = float(accel.max()) if accel.size else 0.0
        report.max_accel_mps2 = max(report.max_accel_mps2, r.accel_max_mps2)

        # jerk = ‖d(accel)/dt‖ per interval — captures HF chatter that stays
        # under the M3-9 accel ceiling but still reads unphysical.
        if accel.shape[0] >= 2:
            dv2 = np.diff(vel, axis=0)   # (N-1, 3)
            dt2 = dt[1:]
            if dv2.shape[0] >= 2:
                d3v = np.diff(dv2, axis=0)
                dt3 = dt2[1:]
                ok = dt3 > 0
                if ok.any():
                    jerk = np.linalg.norm(d3v[ok], axis=1) / (dt3[ok] ** 2)
                    r.jerk_max_mps3 = float(jerk.max())
        r.is_chatty = r.jerk_max_mps3 > cfg.jerk_threshold_mps3
        if r.is_chatty:
            report.subjects_chatty += 1
        report.max_jerk_mps3 = max(report.max_jerk_mps3, r.jerk_max_mps3)
        report.subjects.append(r)

    return report


__all__ = [
    "MomentumProbeConfig",
    "MomentumProbeReport",
    "SubjectMomentumReport",
    "momentum_probe",
]
