"""Root-orientation turn-rate gate (T1c, R-6): clamp global_orient jumps.

Measured need (§D of ``docs/research/2026-07-06-player-physics.md``): user reports
"spatial orientation glitches" — HMR's ``global_orient`` can flip 180° between
frames on ambiguous front/back geometry, and coherence's smoothing on
``root_orientation`` is off by default because MA(5) flattens fast real turns.
Fix at the right level: clamp only the intervals whose angular rate exceeds a
configurable ceiling; preserve genuine fast turns below the ceiling.

Design mirrors :mod:`.joint_kinematics`: one forward sweep in rotation space,
each violating ``t → t+1`` slerped to ``max·dt / actual_angle`` alpha. Emits ONE
dense ``KEYFRAME_INTERP`` ``ROOT_ORIENTATION`` correction per subject that
actually needed clamping. Reads its config from
``PhysicsConfig.orientation`` — no hard-coded numbers here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..config.gates import OrientationConfig
from ..scene.layers import Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene
from .engine import make_keyframes, resolve_subject_motion
from .joint_kinematics import _clamp_joint_track, _interval_angle_deg


@dataclass
class OrientationViolation:
    """Root-orientation over-limit interval preserved for review (R-6)."""

    track_id: int
    frame: int
    rate_dps: float
    clamped_dps: float


@dataclass
class OrientationReport:
    """Aggregate root-orientation gate report."""

    n_subjects: int = 0
    subjects_corrected: int = 0
    intervals_over_limit: int = 0
    intervals_clamped: int = 0
    max_rate_before_dps: float = 0.0
    max_rate_after_dps: float = 0.0
    corrections_added: int = 0
    violations: list[OrientationViolation] = field(default_factory=list)


def orientation_gate(
    scene: Scene, cfg: OrientationConfig | None = None, *, fps: float,
) -> tuple[Scene, OrientationReport]:
    """Root-``global_orient`` angular-rate clamp; return NEW scene + report.

    * ``cfg is None`` or ``cfg.enabled is False``: measure-only path (populates
      counts and rates without emitting corrections).
    * Enabled: emits ONE ``KEYFRAME_INTERP`` ``ROOT_ORIENTATION`` correction per
      subject whose forward-swept clamp actually differs from the measured
      rotations.
    """
    cfg = cfg if cfg is not None else OrientationConfig()
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    report = OrientationReport(n_subjects=len(scene.subjects))
    auto_corrs: list[Correction] = []
    max_dps = float(cfg.max_turn_rate_dps)

    for s in scene.subjects:
        corrs = list(scene.corrections_for(s.track_id))
        resolved = resolve_subject_motion(s.proposal, corrs)
        frames = np.asarray(resolved.pose.frames, dtype=int)
        orient = np.asarray(resolved.pose.global_orient, dtype=float)
        if orient.shape[0] < 2:
            continue
        dt = np.diff(frames.astype(float)) / fps

        clamped, joint_viols, mx_b, mx_a = _clamp_joint_track(
            orient, dt, max_dps, s.track_id, joint_index=-1,
        )
        over = sum(1 for v in joint_viols if v.rate_dps > max_dps)
        report.intervals_over_limit += over
        report.max_rate_before_dps = max(report.max_rate_before_dps, mx_b)
        report.max_rate_after_dps = max(report.max_rate_after_dps, mx_a)
        # translate joint-shaped violations to orientation records for the report
        orient_viols = [
            OrientationViolation(
                track_id=v.track_id, frame=v.frame,
                rate_dps=v.rate_dps, clamped_dps=v.clamped_dps,
            ) for v in joint_viols
        ]
        if not cfg.enabled:
            report.violations.extend(orient_viols)
            continue
        if not joint_viols:
            continue
        if np.allclose(clamped, orient, atol=1e-9):
            continue
        report.intervals_clamped += len(joint_viols)
        report.violations.extend(orient_viols)
        report.subjects_corrected += 1
        auto_corrs.append(
            make_keyframes(
                f"auto-orient-{s.track_id}",
                CorrectionTarget(
                    kind=TargetKind.ROOT_ORIENTATION,
                    subject_track_id=s.track_id,
                ),
                (int(frames[0]), int(frames[-1])),
                key_frames=frames.astype(float),
                key_values=clamped,
                interp="slerp",
                note=(
                    f"auto orientation clamp: |omega| <= {max_dps}°/s @ {fps:.3g}fps "
                    f"({len(joint_viols)} interval(s) clamped)"
                ),
            )
        )

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report


__all__ = [
    "OrientationReport",
    "OrientationViolation",
    "orientation_gate",
    "_interval_angle_deg",  # re-export for tests
]
