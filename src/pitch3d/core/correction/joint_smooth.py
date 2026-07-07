"""Joint-pose smoothing — slerp-based low-pass on body_pose joints.

Complements :mod:`.joint_kinematics` (which caps ``|Δpose|/dt``). This
gate smooths OSCILLATIONS in the pose sequence — HMR jitter that stays
below the max-omega threshold but reads as twitching in the render.

Applied per-joint via a moving-average of the axis-angle representation.
For small rotations the componentwise mean is a fine approximation; for
large deltas the pose_kinematics gate already clamped them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..config.gates import JointSmoothConfig
from ..scene.layers import Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene
from .engine import make_keyframes, resolve_subject_motion


@dataclass
class JointSmoothReport:
    n_subjects: int = 0
    subjects_corrected: int = 0
    corrections_added: int = 0
    max_deviation_rad: float = 0.0


def _moving_average_axisangle(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average per component; ``x`` shape ``(T, K, 3)``."""
    n = x.shape[0]
    if n == 0 or window <= 1:
        return x.copy()
    w = min(window, n)
    half = w // 2
    pad_before = np.repeat(x[:1], half, axis=0)
    pad_after = np.repeat(x[-1:], w - half - 1, axis=0)
    padded = np.concatenate([pad_before, x, pad_after], axis=0)
    kernel = np.ones(w) / w
    T, K, D = x.shape
    out = np.zeros_like(x)
    for k in range(K):
        for d in range(D):
            out[:, k, d] = np.convolve(padded[:, k, d], kernel, mode="valid")
    return out


def joint_smooth_gate(
    scene: Scene, cfg: JointSmoothConfig | None = None, *, fps: float = 25.0,
) -> tuple[Scene, JointSmoothReport]:
    cfg = cfg if cfg is not None else JointSmoothConfig()
    report = JointSmoothReport(n_subjects=len(scene.subjects))
    if not cfg.enabled or fps <= 0:
        return scene, report

    auto_corrs: list[Correction] = []
    for s in scene.subjects:
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        frames = np.asarray(resolved.pose.frames, dtype=int)
        body = np.asarray(resolved.pose.body_pose, dtype=float)
        n = body.shape[0]
        if n < 3 or body.ndim < 3:
            continue

        smoothed_all = _moving_average_axisangle(body, cfg.smooth_window)
        subj_emit = False
        for j in range(body.shape[1]):
            per_joint_dev = float(np.abs(smoothed_all[:, j, :] - body[:, j, :]).max())
            report.max_deviation_rad = max(report.max_deviation_rad, per_joint_dev)
            if per_joint_dev < cfg.min_correction_rad:
                continue
            auto_corrs.append(
                make_keyframes(
                    f"auto-joint-smooth-{s.track_id}-j{j}",
                    CorrectionTarget(
                        kind=TargetKind.POSE_BODY_JOINT,
                        subject_track_id=s.track_id,
                        joint_index=j,
                    ),
                    (int(frames[0]), int(frames[-1])),
                    key_frames=frames.astype(float),
                    key_values=smoothed_all[:, j, :],
                    interp="slerp",
                    note=(
                        f"auto joint smooth: joint {j}, window {cfg.smooth_window}, "
                        f"max dev {per_joint_dev:.3f}rad"
                    ),
                )
            )
            subj_emit = True
        if subj_emit:
            report.subjects_corrected += 1

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report


__all__ = [
    "JointSmoothConfig",
    "JointSmoothReport",
    "joint_smooth_gate",
]
