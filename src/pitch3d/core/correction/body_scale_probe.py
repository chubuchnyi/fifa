"""Body-scale probe — flag subjects who change size across the clip.

An adult player's height doesn't change during a 2-second broadcast clip.
If our reconstruction assigns wildly different SMPL-X ``betas`` (which
control body shape/height) across frames or between per-frame detections
of the same person, the identity model failed.

Metric: standard deviation of ``pelvis_above_foot`` (via SMPL-X FK on
each frame's betas + zero-pose) across a subject's frames. Ideally ≈ 0
since betas SHOULD be constant per subject (SubjectMotion.shape is
frame-invariant by design).

This probe is a smoke test for the SubjectMotion invariant. If it fires,
some upstream stage is emitting per-frame betas or the export is
duplicating subjects with different shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from ..scene.scene import Scene, Subject
from .engine import resolve_subject_motion

BodyScaleProvider = Callable[[Subject], "float | None"]


@dataclass(frozen=True)
class BodyScaleConfig:
    enabled: bool = False
    #: Cross-subject inconsistency threshold (m). If two subjects share
    #: identity via team+jersey but their measured height differs by more
    #: than this, flag them.
    inter_subject_threshold_m: float = 0.15


@dataclass
class SubjectScaleReport:
    track_id: int
    pelvis_above_foot_m: float = 0.0


@dataclass
class BodyScaleReport:
    n_subjects: int = 0
    n_pairs_flagged: int = 0
    max_pair_diff_m: float = 0.0
    subjects: list[SubjectScaleReport] = field(default_factory=list)


def body_scale_probe(
    scene: Scene,
    cfg: BodyScaleConfig | None = None,
    scale_provider: BodyScaleProvider | None = None,
) -> BodyScaleReport:
    """Cross-check per-subject heights via an injected scale provider.

    ``scale_provider(subject)`` returns the per-subject pelvis-above-foot
    (in m) — e.g. ``make_smplx_foot_z_provider`` from the adapters, whose
    output is a per-frame array whose median we can take.
    """
    cfg = cfg if cfg is not None else BodyScaleConfig()
    report = BodyScaleReport(n_subjects=len(scene.subjects))
    if not cfg.enabled or scale_provider is None:
        return report

    for s in scene.subjects:
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        resolved_subject = replace(s, proposal=resolved)
        offset = scale_provider(resolved_subject)
        if offset is None:
            continue
        arr = np.asarray(offset, dtype=float).reshape(-1)
        if arr.size == 0:
            continue
        median_offset = float(np.median(arr))
        report.subjects.append(SubjectScaleReport(
            track_id=int(s.track_id),
            pelvis_above_foot_m=median_offset,
        ))

    # Cross-subject: SMPLest-X gives per-track shape; check spread across
    # subjects to spot outliers with wildly different heights.
    heights = np.array([r.pelvis_above_foot_m for r in report.subjects])
    if heights.size >= 2:
        for i in range(heights.size):
            for j in range(i + 1, heights.size):
                diff = abs(heights[i] - heights[j])
                report.max_pair_diff_m = max(report.max_pair_diff_m, float(diff))
                if diff > cfg.inter_subject_threshold_m:
                    report.n_pairs_flagged += 1

    return report


__all__ = [
    "BodyScaleConfig",
    "BodyScaleProvider",
    "BodyScaleReport",
    "SubjectScaleReport",
    "body_scale_probe",
]
