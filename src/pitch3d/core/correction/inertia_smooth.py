"""Inertia smooth — rate-limit yaw angular acceleration.

Correction sibling of :mod:`.inertia_probe`. Where T1c orientation-gate
clamps ω (rate of yaw change) per interval, this pass smooths the
**derivative** of ω (angular acceleration) via a low-pass on the yaw
signal so a physically-impossible snap (α > 15 rad/s²) becomes a
plausible turn.

Uses a centered moving average on yaw with a small window; the pass
respects the wraparound at ±π.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..scene.layers import Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene
from .engine import make_keyframes, resolve_subject_motion


@dataclass(frozen=True)
class InertiaSmoothConfig:
    enabled: bool = False
    smooth_window: int = 3      # centered moving average on yaw
    max_alpha_rad_s2: float = 15.0
    min_correction_rad: float = 1e-3


@dataclass
class InertiaSmoothReport:
    n_subjects: int = 0
    subjects_corrected: int = 0
    corrections_added: int = 0
    max_alpha_before_rad_s2: float = 0.0
    max_alpha_after_rad_s2: float = 0.0


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def _unwrap(yaw: np.ndarray) -> np.ndarray:
    """Unwrap principal-value yaw so diffs cross ±π monotonically."""
    return np.unwrap(np.asarray(yaw, dtype=float))


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    n = x.shape[0]
    if n == 0 or window <= 1:
        return x.copy()
    w = min(window, n)
    half = w // 2
    pad_before = np.repeat(x[:1], half)
    pad_after = np.repeat(x[-1:], w - half - 1)
    padded = np.concatenate([pad_before, x, pad_after])
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode="valid")


def _peak_alpha(yaw_series: np.ndarray, fps: float) -> float:
    n = yaw_series.shape[0]
    if n < 3:
        return 0.0
    dt = 1.0 / fps
    dy = np.diff(yaw_series)
    omega = dy / dt
    alpha = np.diff(omega) / dt
    return float(np.abs(alpha).max()) if alpha.size else 0.0


def inertia_smooth_gate(
    scene: Scene, cfg: InertiaSmoothConfig | None = None, *, fps: float = 25.0,
) -> tuple[Scene, InertiaSmoothReport]:
    cfg = cfg if cfg is not None else InertiaSmoothConfig()
    report = InertiaSmoothReport(n_subjects=len(scene.subjects))
    if not cfg.enabled or fps <= 0:
        return scene, report

    auto_corrs: list[Correction] = []
    for s in scene.subjects:
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        frames = np.asarray(resolved.pose.frames, dtype=int)
        orient = np.asarray(resolved.pose.global_orient, dtype=float)
        n = orient.shape[0]
        if n < 3:
            continue

        yaw = orient[:, 2]
        yaw_unwrapped = _unwrap(yaw)
        alpha_before = _peak_alpha(yaw_unwrapped, fps)
        report.max_alpha_before_rad_s2 = max(
            report.max_alpha_before_rad_s2, alpha_before,
        )
        if alpha_before <= cfg.max_alpha_rad_s2:
            report.max_alpha_after_rad_s2 = max(
                report.max_alpha_after_rad_s2, alpha_before,
            )
            continue

        yaw_smooth = _moving_average(yaw_unwrapped, cfg.smooth_window)
        alpha_after = _peak_alpha(yaw_smooth, fps)
        report.max_alpha_after_rad_s2 = max(
            report.max_alpha_after_rad_s2, alpha_after,
        )
        # rewrap to (-π, π]
        yaw_smooth_wrapped = _wrap_to_pi(yaw_smooth)
        dev = float(np.abs(_wrap_to_pi(yaw_smooth_wrapped - yaw)).max())
        if dev < cfg.min_correction_rad:
            continue

        new_orient = orient.copy()
        new_orient[:, 2] = yaw_smooth_wrapped
        report.subjects_corrected += 1
        auto_corrs.append(
            make_keyframes(
                f"auto-inertia-smooth-{s.track_id}",
                CorrectionTarget(
                    kind=TargetKind.ROOT_ORIENTATION,
                    subject_track_id=s.track_id,
                ),
                (int(frames[0]), int(frames[-1])),
                key_frames=frames.astype(float),
                key_values=new_orient,
                interp="slerp",
                note=(
                    f"auto inertia smooth: α {alpha_before:.0f}→{alpha_after:.0f} rad/s², "
                    f"max yaw dev {np.degrees(dev):.0f}°"
                ),
            )
        )

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report


__all__ = [
    "InertiaSmoothConfig",
    "InertiaSmoothReport",
    "inertia_smooth_gate",
]
