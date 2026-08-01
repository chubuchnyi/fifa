"""Per-joint angular-velocity gate (T1b, R-6): clamp impossible pose changes, honestly.

Measured need (§C of ``docs/research/2026-07-06-player-physics.md``): user reports
"players change poses way too fast." HMR under partial occlusion routinely emits
per-joint jitter that componentwise-diff smoothing (MA(5)) cannot fix — the
existing coherence pass smooths ROOT translation only. This module adds a
per-joint quaternion-space gate that clamps each ``body_pose`` joint to a
configurable angular-velocity ceiling (``JointKinematicConfig.max_omega_dps``).

Design mirrors M3-9 (``kinematics.py``):

* One forward sweep per joint in **rotation space**. For each violating interval
  ``t → t+1`` where ``angle(R_t+1, R_t) / dt > max``, the new rotation is
  ``slerp(R_t, R_t+1, max·dt / actual_angle)`` — the truncated endpoint keeps
  the direction of change but slows the rate.
* Result emitted as ONE dense ``KEYFRAME_INTERP`` ``POSE_BODY_JOINT`` correction
  per (subject, joint) that actually needed it — non-destructive, ADR-0002.
* No fabrication (R-6): rotations that were feasible stay verbatim; only the
  intervals whose rate exceeded the limit are pulled back.

The gate reads its config from ``config/physics.yaml`` via
``PhysicsConfig.joint`` — no hard-coded constants here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..config.gates import JointKinematicConfig
from ..scene.layers import Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene
from .engine import make_keyframes, resolve_subject_motion
from .rotations import (
    axis_angle_to_quat,
    quat_mul,
    slerp_axis_angle,
)


@dataclass
class JointViolation:
    """One per-joint over-limit interval preserved for review (R-6)."""

    track_id: int
    joint_index: int
    frame: int
    rate_dps: float          # measured angular rate deg/s BEFORE clamping
    clamped_dps: float       # measured rate AFTER clamping


@dataclass
class JointKinematicReport:
    """Aggregate report — total intervals violated, subjects/joints touched."""

    n_subjects: int = 0
    subjects_corrected: int = 0
    joints_corrected: int = 0
    intervals_over_limit: int = 0
    intervals_clamped: int = 0
    max_rate_before_dps: float = 0.0
    max_rate_after_dps: float = 0.0
    corrections_added: int = 0
    violations: list[JointViolation] = field(default_factory=list)


def _quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion (w, x, y, z) — conjugate."""
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _interval_angle_deg(aa_a: np.ndarray, aa_b: np.ndarray) -> float:
    """Angle in degrees of the rotation from ``aa_a`` to ``aa_b`` (group metric).

    Uses ``delta = q_b · q_a^-1``; angle is ``2·arccos(|w_delta|)`` — robust to
    the sign ambiguity of quaternions (a rotation is the same as its negative).
    """
    qa = axis_angle_to_quat(np.asarray(aa_a, dtype=float))
    qb = axis_angle_to_quat(np.asarray(aa_b, dtype=float))
    delta = quat_mul(qb, _quat_inv(qa))
    w = float(np.clip(abs(delta[0]), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(w)))


def _clamp_joint_track(
    aa_seq: np.ndarray, dt: np.ndarray, max_dps: float, track_id: int, joint_index: int,
) -> tuple[np.ndarray, list[JointViolation], float, float]:
    """One forward sweep per joint. Returns (clamped, violations, max_before, max_after)."""
    n = aa_seq.shape[0]
    out = aa_seq.copy()
    violations: list[JointViolation] = []
    max_before = 0.0
    max_after = 0.0
    for i in range(1, n):
        step = dt[i - 1] if i - 1 < dt.shape[0] else 1.0
        if step <= 0:
            continue
        # measured rate against the *raw* prev pose (so the report is honest)
        rate_before = _interval_angle_deg(aa_seq[i - 1], aa_seq[i]) / step
        max_before = max(max_before, rate_before)
        # actual clamp uses the *already-clamped* prev pose, so violations don't chain
        angle_now = _interval_angle_deg(out[i - 1], out[i])
        rate_now = angle_now / step
        if rate_now > max_dps and angle_now > 0.0:
            alpha = float(max_dps * step / angle_now)
            out[i] = slerp_axis_angle(out[i - 1], out[i], alpha)
            clamped_rate = _interval_angle_deg(out[i - 1], out[i]) / step
            violations.append(JointViolation(
                track_id=int(track_id), joint_index=int(joint_index),
                frame=int(i), rate_dps=float(rate_before),
                clamped_dps=float(clamped_rate),
            ))
            max_after = max(max_after, clamped_rate)
        else:
            max_after = max(max_after, rate_now)
    return out, violations, max_before, max_after


def joint_kinematic_gate(
    scene: Scene, cfg: JointKinematicConfig | None = None, *, fps: float,
) -> tuple[Scene, JointKinematicReport]:
    """Per-joint angular-velocity clamp; return NEW scene + report.

    * ``cfg is None`` or ``cfg.enabled is False``: measure-only path — reports
      how many intervals per subject would violate the limit, no corrections
      emitted.
    * Enabled: emits ONE ``KEYFRAME_INTERP`` ``POSE_BODY_JOINT`` correction per
      (subject, joint) whose forward-swept clamp actually differs from the
      measured rotations. Untouched joints emit nothing.
    """
    cfg = cfg if cfg is not None else JointKinematicConfig()
    report = JointKinematicReport(n_subjects=len(scene.subjects))
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    auto_corrs: list[Correction] = []
    max_dps = float(cfg.max_omega_dps)

    for s in scene.subjects:
        corrs = list(scene.corrections_for(s.track_id))
        resolved = resolve_subject_motion(s.proposal, corrs)
        frames = np.asarray(resolved.pose.frames, dtype=int)
        body = np.asarray(resolved.pose.body_pose, dtype=float)
        n_frames, n_joints = (body.shape[0], body.shape[1]) if body.ndim == 3 else (0, 0)
        if n_frames < 2 or n_joints == 0:
            continue
        dt = np.diff(frames.astype(float)) / fps
        subj_had_correction = False

        for j in range(n_joints):
            aa_seq = body[:, j, :]
            clamped, violations, mx_b, mx_a = _clamp_joint_track(
                aa_seq, dt, max_dps, s.track_id, j,
            )
            over = sum(1 for v in violations if v.rate_dps > max_dps)
            report.intervals_over_limit += over
            report.max_rate_before_dps = max(report.max_rate_before_dps, mx_b)
            report.max_rate_after_dps = max(report.max_rate_after_dps, mx_a)
            if not cfg.enabled:
                # measure-only: violations still counted, but nothing clamped
                report.violations.extend(violations)
                continue
            if not violations:
                continue
            # rebuild full body_pose keyframes for this joint (dense KEYFRAME_INTERP)
            # only emit if the clamp actually differs from the measured rotations
            if np.allclose(clamped, aa_seq, atol=1e-9):
                continue
            report.intervals_clamped += len(violations)
            report.violations.extend(violations)
            report.joints_corrected += 1
            subj_had_correction = True
            auto_corrs.append(
                make_keyframes(
                    f"auto-joint-{s.track_id}-j{j}",
                    CorrectionTarget(
                        kind=TargetKind.POSE_BODY_JOINT,
                        subject_track_id=s.track_id,
                        joint_index=j,
                    ),
                    (int(frames[0]), int(frames[-1])),
                    key_frames=frames.astype(float),
                    key_values=clamped,
                    interp="slerp",
                    note=(
                        f"auto joint-kinematic clamp: |omega| <= {max_dps}°/s @ {fps:.3g}fps "
                        f"(joint {j}, {len(violations)} interval(s) clamped)"
                    ),
                )
            )
        if subj_had_correction:
            report.subjects_corrected += 1

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report
