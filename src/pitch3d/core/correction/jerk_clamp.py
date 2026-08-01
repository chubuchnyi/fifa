"""Jerk clamp — iterate low-pass on root translation until peak jerk drops below a ceiling.

momentum_smooth applies a fixed-window moving average once; on very noisy
HMR signals (peak jerk 300k m/s³ on the real pod scene) a single pass
still leaves the residual well above human physiology. This gate:

* Measures the peak jerk on the resolved root translation.
* If above ``max_jerk_mps3``, applies a moving-average pass with the
  configured window, re-measures, and repeats up to ``max_passes``.
* Emits ONE dense KEYFRAME_INTERP per subject whose final transl differs
  from the raw input by more than ``min_correction_m``.

Every pass preserves the endpoints (padding by nearest value). The
correction is layered through the ADR-0002 seam.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..config.gates import JerkClampConfig
from ..scene.layers import Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene
from .engine import make_keyframes, resolve_subject_motion
from .momentum_smooth import _moving_average, _peak_jerk


@dataclass
class JerkClampReport:
    n_subjects: int = 0
    subjects_corrected: int = 0
    corrections_added: int = 0
    total_passes_used: int = 0
    max_jerk_before_mps3: float = 0.0
    max_jerk_after_mps3: float = 0.0
    max_shift_m: float = 0.0


def jerk_clamp_gate(
    scene: Scene, cfg: JerkClampConfig | None = None, *, fps: float = 25.0,
) -> tuple[Scene, JerkClampReport]:
    cfg = cfg if cfg is not None else JerkClampConfig()
    report = JerkClampReport(n_subjects=len(scene.subjects))
    if not cfg.enabled or fps <= 0:
        return scene, report

    auto_corrs: list[Correction] = []
    for s in scene.subjects:
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        frames = np.asarray(resolved.pose.frames, dtype=int)
        transl = np.asarray(resolved.pose.transl, dtype=float)
        n = transl.shape[0]
        if n < 4:
            continue

        peak_before = _peak_jerk(transl, fps)
        report.max_jerk_before_mps3 = max(report.max_jerk_before_mps3, peak_before)
        if peak_before <= cfg.max_jerk_mps3:
            report.max_jerk_after_mps3 = max(report.max_jerk_after_mps3, peak_before)
            continue

        smoothed = transl.copy()
        passes = 0
        for _ in range(cfg.max_passes):
            smoothed = _moving_average(smoothed, cfg.smooth_window)
            passes += 1
            if _peak_jerk(smoothed, fps) <= cfg.max_jerk_mps3:
                break

        report.total_passes_used += passes
        peak_after = _peak_jerk(smoothed, fps)
        report.max_jerk_after_mps3 = max(report.max_jerk_after_mps3, peak_after)
        dev = float(np.linalg.norm(smoothed - transl, axis=1).max())
        if dev < cfg.min_correction_m:
            continue
        report.subjects_corrected += 1
        report.max_shift_m = max(report.max_shift_m, dev)
        auto_corrs.append(
            make_keyframes(
                f"auto-jerk-clamp-{s.track_id}",
                CorrectionTarget(
                    TargetKind.ROOT_TRANSLATION, subject_track_id=s.track_id,
                ),
                (int(frames[0]), int(frames[-1])),
                key_frames=frames.astype(float),
                key_values=smoothed,
                note=(
                    f"auto jerk clamp: {passes} pass(es), "
                    f"jerk {peak_before:.0f}→{peak_after:.0f} m/s³"
                ),
            )
        )

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report


__all__ = [
    "JerkClampConfig",
    "JerkClampReport",
    "jerk_clamp_gate",
]
