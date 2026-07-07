"""Facing-align gate: rotate body so it faces its direction of motion.

Complements the pose-motion sync: when a subject moves at ``> velocity_
threshold_mps`` but their global_orient doesn't match the velocity vector
(they're moonwalking / crab-walking), rotate the root orientation so the
body's +X axis aligns with the horizontal velocity direction.

Only touches ``ROOT_ORIENTATION``; body joints untouched. R-6 low-conf
stamp on rewritten frames (this is inferred from motion, not measured
from the video).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..config.gates import FacingAlignConfig
from ..scene.layers import Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene
from .engine import make_keyframes, resolve_subject_motion

FACING_INFERRED_CONF = 0.30


@dataclass
class SubjectFacingReport:
    track_id: int
    n_frames: int = 0
    corrected_frames: int = 0
    max_yaw_delta_rad: float = 0.0


@dataclass
class FacingAlignReport:
    n_subjects: int = 0
    subjects_corrected: int = 0
    corrections_added: int = 0
    max_yaw_delta_rad: float = 0.0
    subjects: list[SubjectFacingReport] = field(default_factory=list)


def _yaw_from_velocity(vel_xy: np.ndarray) -> np.ndarray:
    """atan2(y, x) per frame → yaw (radians)."""
    return np.arctan2(vel_xy[:, 1], vel_xy[:, 0])


def _ewma(x: np.ndarray, window: int) -> np.ndarray:
    """Centered EWMA over ``window`` frames with edge padding."""
    if window <= 1 or x.shape[0] < 2:
        return x.copy()
    alpha = 2.0 / (window + 1)
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, x.shape[0]):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _yaw_from_axis_angle(aa: np.ndarray) -> np.ndarray:
    """Extract the world-Z yaw component from a (T, 3) axis-angle sequence.

    First-order approximation: axis-angle rotation vector's Z component
    IS the yaw for small tilt/roll (which player figures upright mostly are).
    """
    return aa[:, 2].copy()


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def _mark_low_conf(
    scene: Scene, track_id: int, frames: np.ndarray,
    row_indices: np.ndarray, conf: float,
) -> None:
    from ..scene.layers import ConfidenceMap
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


def facing_align_gate(
    scene: Scene, cfg: FacingAlignConfig | None = None, *, fps: float = 25.0,
) -> tuple[Scene, FacingAlignReport]:
    """Align global_orient yaw to motion direction. Return new scene + report."""
    cfg = cfg if cfg is not None else FacingAlignConfig()
    report = FacingAlignReport(n_subjects=len(scene.subjects))
    if not cfg.enabled or fps <= 0:
        return scene, report

    auto_corrs: list[Correction] = []
    for s in scene.subjects:
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        frames = np.asarray(resolved.pose.frames, dtype=int)
        transl = np.asarray(resolved.pose.transl, dtype=float)
        orient = np.asarray(resolved.pose.global_orient, dtype=float)
        n = transl.shape[0]
        r = SubjectFacingReport(track_id=int(s.track_id), n_frames=n)
        if n < 3:
            report.subjects.append(r)
            continue

        dt = np.diff(frames.astype(float)) / fps
        ok = dt > 0
        if not ok.any():
            report.subjects.append(r)
            continue

        vel = np.zeros((n, 3))
        vel[1:] = np.diff(transl, axis=0) / np.where(dt[:, None] > 0, dt[:, None], 1.0)
        speed = np.linalg.norm(vel[:, :2], axis=1)
        moving = speed > cfg.velocity_threshold_mps
        if not moving.any():
            report.subjects.append(r)
            continue
        target_yaw = _yaw_from_velocity(vel[:, :2])
        # Unwrap BEFORE EWMA — averaging +π and -π wrapped values gives 0
        # (180° off), corrupting the smooth target. Wrap back after.
        target_unwrapped = np.unwrap(target_yaw)
        target_yaw_smooth = _wrap_to_pi(_ewma(target_unwrapped, cfg.yaw_ewma_window))
        current_yaw = _yaw_from_axis_angle(orient)
        delta_yaw = _wrap_to_pi(target_yaw_smooth - current_yaw)
        needs_fix = moving & (np.abs(delta_yaw) > cfg.yaw_tolerance_rad)
        if not needs_fix.any():
            report.subjects.append(r)
            continue

        new_orient = orient.copy()
        new_orient[needs_fix, 2] = target_yaw_smooth[needs_fix]
        r.corrected_frames = int(needs_fix.sum())
        r.max_yaw_delta_rad = float(np.abs(delta_yaw[needs_fix]).max())
        report.max_yaw_delta_rad = max(report.max_yaw_delta_rad, r.max_yaw_delta_rad)
        report.subjects_corrected += 1
        report.subjects.append(r)

        auto_corrs.append(
            make_keyframes(
                f"auto-facing-align-{s.track_id}",
                CorrectionTarget(
                    kind=TargetKind.ROOT_ORIENTATION,
                    subject_track_id=s.track_id,
                ),
                (int(frames[0]), int(frames[-1])),
                key_frames=frames.astype(float),
                key_values=new_orient,
                interp="slerp",
                note=(
                    f"auto facing-align: {r.corrected_frames}/{n} frames, "
                    f"max delta {np.degrees(r.max_yaw_delta_rad):.0f}°"
                ),
            )
        )
        _mark_low_conf(
            scene, s.track_id, frames, np.where(needs_fix)[0],
            FACING_INFERRED_CONF,
        )

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report


__all__ = [
    "FACING_INFERRED_CONF",
    "FacingAlignConfig",
    "FacingAlignReport",
    "facing_align_gate",
]
