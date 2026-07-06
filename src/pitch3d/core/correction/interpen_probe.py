"""Interpenetration probe — capsule overlap statistics across the clip.

Complements the correction-side collision_gate (which pushes pairs apart)
with a passive probe: how much do subjects overlap and for how long. High
overlap is a bad sign — the M3-9 XY track is putting players on top of
each other, which is unphysical.

Metric per subject:

* fraction of frames where at least one other subject sits within
  ``interpen_radius_m`` centres apart;
* deepest overlap (worst interpenetration).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..scene.scene import Scene
from .engine import resolve_subject_motion


@dataclass(frozen=True)
class InterpenConfig:
    enabled: bool = False
    interpen_radius_m: float = 0.5      # 2r for the capsule (r=0.25 rough)


@dataclass
class SubjectInterpenReport:
    track_id: int
    n_frames: int = 0
    overlap_frames: int = 0
    max_overlap_m: float = 0.0
    overlap_fraction: float = 0.0


@dataclass
class InterpenReport:
    n_subjects: int = 0
    subjects_with_overlap: int = 0
    total_overlap_frames: int = 0
    max_overlap_m: float = 0.0
    subjects: list[SubjectInterpenReport] = field(default_factory=list)


def interpen_probe(
    scene: Scene, cfg: InterpenConfig | None = None,
) -> InterpenReport:
    cfg = cfg if cfg is not None else InterpenConfig()
    report = InterpenReport(n_subjects=len(scene.subjects))
    if not cfg.enabled:
        return report

    subj_data = []
    for s in scene.subjects:
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        frames = np.asarray(resolved.pose.frames, dtype=int)
        transl = np.asarray(resolved.pose.transl, dtype=float)
        subj_data.append({
            "track_id": int(s.track_id), "frames": frames, "xy": transl[:, :2],
        })

    per_subject_overlap = {sd["track_id"]: 0 for sd in subj_data}
    per_subject_max = {sd["track_id"]: 0.0 for sd in subj_data}

    all_frames = np.unique(np.concatenate([sd["frames"] for sd in subj_data]) if subj_data else np.zeros(0, int))
    for f in all_frames:
        present = []
        for sd in subj_data:
            idx = np.where(sd["frames"] == f)[0]
            if idx.size:
                present.append((sd["track_id"], sd["xy"][int(idx[0])]))
        if len(present) < 2:
            continue
        for i in range(len(present)):
            tid_i, xy_i = present[i]
            for j in range(i + 1, len(present)):
                tid_j, xy_j = present[j]
                dist = float(np.linalg.norm(xy_i - xy_j))
                if dist < cfg.interpen_radius_m:
                    overlap = cfg.interpen_radius_m - dist
                    per_subject_overlap[tid_i] += 1
                    per_subject_overlap[tid_j] += 1
                    per_subject_max[tid_i] = max(per_subject_max[tid_i], overlap)
                    per_subject_max[tid_j] = max(per_subject_max[tid_j], overlap)
                    report.total_overlap_frames += 1
                    report.max_overlap_m = max(report.max_overlap_m, overlap)

    for sd in subj_data:
        tid = sd["track_id"]
        r = SubjectInterpenReport(
            track_id=tid,
            n_frames=sd["frames"].shape[0],
            overlap_frames=per_subject_overlap[tid],
            max_overlap_m=per_subject_max[tid],
            overlap_fraction=(per_subject_overlap[tid] / max(1, sd["frames"].shape[0])),
        )
        if r.overlap_frames > 0:
            report.subjects_with_overlap += 1
        report.subjects.append(r)

    return report


__all__ = [
    "InterpenConfig",
    "InterpenReport",
    "SubjectInterpenReport",
    "interpen_probe",
]
