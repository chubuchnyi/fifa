"""Beta variance probe — is per-track SMPL-X shape stable?

A SubjectMotion.shape.betas is declared frame-invariant. This probe
verifies the invariant AT ASSEMBLY time — if a track ID has multiple
frames' worth of betas from HMR that were averaged into one, the
variance of those source betas would tell us something about the
consistency of the identity assignment. In our current export the
shape is a scalar tuple, so this probe reports the L2 norm of betas
(a proxy for how "extreme" the shape is) and flags outliers.

Two flag rules:

* An individual subject with ``|betas| > extreme_norm_threshold`` is
  reconstructed with an unusual body shape (small child, giant).
* Two subjects with betas cosine distance < ``similar_cos_threshold``
  are highly similar in body shape — for identity persistence this is
  a hint they may be the same physical player under two track_ids.

Both are measurement-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..scene.scene import Scene


@dataclass(frozen=True)
class BetaVarianceConfig:
    enabled: bool = False
    extreme_norm_threshold: float = 3.0    # |betas| above → extreme
    similar_cos_threshold: float = 0.05    # cos-distance below → similar identity


@dataclass
class SubjectBetaReport:
    track_id: int
    betas_norm: float = 0.0
    is_extreme: bool = False


@dataclass
class SimilarPair:
    track_id_a: int
    track_id_b: int
    cos_distance: float


@dataclass
class BetaVarianceReport:
    n_subjects: int = 0
    subjects_extreme: int = 0
    similar_pairs: list[SimilarPair] = field(default_factory=list)
    subjects: list[SubjectBetaReport] = field(default_factory=list)


def beta_variance_probe(
    scene: Scene, cfg: BetaVarianceConfig | None = None,
) -> BetaVarianceReport:
    cfg = cfg if cfg is not None else BetaVarianceConfig()
    report = BetaVarianceReport(n_subjects=len(scene.subjects))
    if not cfg.enabled:
        return report

    betas_by_id: dict[int, np.ndarray] = {}
    for s in scene.subjects:
        betas = np.asarray(s.proposal.shape.betas, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(betas))
        r = SubjectBetaReport(track_id=int(s.track_id), betas_norm=norm)
        r.is_extreme = norm > cfg.extreme_norm_threshold
        if r.is_extreme:
            report.subjects_extreme += 1
        report.subjects.append(r)
        betas_by_id[int(s.track_id)] = betas

    ids = sorted(betas_by_id.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            b_i, b_j = betas_by_id[ids[i]], betas_by_id[ids[j]]
            na, nb = np.linalg.norm(b_i), np.linalg.norm(b_j)
            if na < 1e-6 or nb < 1e-6:
                continue
            cos = float(np.dot(b_i, b_j) / (na * nb))
            cos = max(-1.0, min(1.0, cos))
            cos_dist = 1.0 - cos
            if cos_dist < cfg.similar_cos_threshold:
                report.similar_pairs.append(SimilarPair(
                    track_id_a=ids[i], track_id_b=ids[j],
                    cos_distance=cos_dist,
                ))
    return report


__all__ = [
    "BetaVarianceConfig",
    "BetaVarianceReport",
    "SimilarPair",
    "SubjectBetaReport",
    "beta_variance_probe",
]
