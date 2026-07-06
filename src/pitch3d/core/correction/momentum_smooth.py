"""Step 4b — momentum smooth: low-pass the root translation to kill jerk.

Contact-lock (step 3b) zeroed the mean foot slide during stance phases;
this pass tackles the residual **high-frequency chatter** of the pelvis
CoM that the M3-9 accel gate lets through (M3-9 bounds magnitude, not
rate of change). Simple moving average on ``transl`` with a small window;
the window is short enough to preserve legitimate stride amplitude but
long enough to kill 1-frame HMR jitter.

Applied AFTER contact-lock so the smoothed root still touches down at
the stance-lock anchor at each contact frame (contact frames are locked
by a KEYFRAME_INTERP on the same ``ROOT_TRANSLATION`` target — later
corrections layer over earlier ones and the smooth pass respects that).

Config lives in ``MomentumSmoothConfig`` (parametric YAML): ``enabled``,
``smooth_window``, ``preserve_contact`` (skip frames that are locked so
we don't undo the anti-slide).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..config.gates import MomentumSmoothConfig
from ..scene.layers import Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene
from .contact_probe import FootPositionProvider, _find_runs
from .engine import make_keyframes, resolve_subject_motion


@dataclass
class MomentumSmoothReport:
    n_subjects: int = 0
    subjects_smoothed: int = 0
    corrections_added: int = 0
    max_shift_m: float = 0.0
    max_jerk_before: float = 0.0
    max_jerk_after: float = 0.0


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding by nearest value."""
    n = x.shape[0]
    if n == 0 or window <= 1:
        return x.copy()
    w = min(window, n)
    half = w // 2
    pad_before = np.repeat(x[:1], half, axis=0)
    pad_after = np.repeat(x[-1:], w - half - 1, axis=0)
    padded = np.concatenate([pad_before, x, pad_after], axis=0)
    kernel = np.ones(w) / w
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(padded[:, j], kernel, mode="valid")
    return out


def _peak_jerk(transl: np.ndarray, fps: float) -> float:
    n = transl.shape[0]
    if n < 4:
        return 0.0
    dt = 1.0 / fps
    vel = np.diff(transl, axis=0) / dt
    accel = np.diff(vel, axis=0) / dt
    jerk = np.linalg.norm(np.diff(accel, axis=0), axis=1) / dt
    return float(jerk.max()) if jerk.size else 0.0


def momentum_smooth_gate(
    scene: Scene,
    cfg: MomentumSmoothConfig | None = None,
    foot_position_provider: FootPositionProvider | None = None,
    *,
    fps: float = 25.0,
) -> tuple[Scene, MomentumSmoothReport]:
    """Low-pass root translation, preserving stance frames.

    ``foot_position_provider`` is required when ``preserve_contact`` is True
    (otherwise stance frames can't be identified); when absent or the
    provider returns ``None`` the pass smooths uniformly.
    """
    cfg = cfg if cfg is not None else MomentumSmoothConfig()
    report = MomentumSmoothReport(n_subjects=len(scene.subjects))
    if not cfg.enabled or fps <= 0:
        return scene, report

    auto_corrs: list[Correction] = []
    for s in scene.subjects:
        corrs = list(scene.corrections_for(s.track_id))
        resolved = resolve_subject_motion(s.proposal, corrs)
        frames = np.asarray(resolved.pose.frames, dtype=int)
        transl = np.asarray(resolved.pose.transl, dtype=float)
        n = transl.shape[0]
        if n < 3:
            continue
        report.max_jerk_before = max(report.max_jerk_before, _peak_jerk(transl, fps))

        smoothed = _moving_average(transl, cfg.smooth_window)
        if cfg.preserve_contact and foot_position_provider is not None:
            resolved_subject = replace(s, proposal=resolved)
            pos = foot_position_provider(resolved_subject)
            if pos is not None and np.asarray(pos).shape[0] == n:
                pos = np.asarray(pos, dtype=float)
                contact_mask = pos[:, 2] <= cfg.contact_z_threshold_m
                runs = _find_runs(contact_mask, cfg.min_contact_run_frames)
                # Contact frames retain their pre-smooth position (foot lock)
                for a, b in runs:
                    smoothed[a:b + 1] = transl[a:b + 1]

        dev = float(np.linalg.norm(smoothed - transl, axis=1).max())
        report.max_jerk_after = max(report.max_jerk_after, _peak_jerk(smoothed, fps))
        if dev < cfg.min_correction_m:
            continue
        report.subjects_smoothed += 1
        report.max_shift_m = max(report.max_shift_m, dev)
        auto_corrs.append(
            make_keyframes(
                f"auto-momentum-smooth-{s.track_id}",
                CorrectionTarget(
                    TargetKind.ROOT_TRANSLATION, subject_track_id=s.track_id,
                ),
                (int(frames[0]), int(frames[-1])),
                key_frames=frames.astype(float),
                key_values=smoothed,
                note=(
                    f"auto momentum smooth: window={cfg.smooth_window}f, "
                    f"max shift {dev:.3f}m"
                ),
            )
        )

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report


__all__ = [
    "MomentumSmoothConfig",
    "MomentumSmoothReport",
    "momentum_smooth_gate",
]
