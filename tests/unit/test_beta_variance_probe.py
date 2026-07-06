"""Beta variance probe — SMPL-X shape consistency & cross-subject similarity."""

from __future__ import annotations

import numpy as np

from pitch3d.core.correction.beta_variance_probe import (
    BetaVarianceConfig,
    beta_variance_probe,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, betas: np.ndarray) -> Subject:
    T = 4
    frames = np.arange(T, dtype=int)
    return Subject(track_id=track_id, proposal=SubjectMotion(
        shape=SmplxShape(betas=betas),
        pose=PoseSequence(
            frames=frames, global_orient=np.zeros((T, 3)),
            body_pose=np.zeros((T, 21, 3)), transl=np.zeros((T, 3)),
        ),
    ))


def _scene(*subjects):
    return Scene(id="s", episode_id="e", source_id="c",
                subjects=list(subjects), corrections=[])


def test_disabled_returns_empty_report():
    s = _subject(1, np.ones(10))
    r = beta_variance_probe(_scene(s), BetaVarianceConfig(enabled=False))
    assert r.subjects == []
    assert r.similar_pairs == []
    assert r.n_subjects == 1


def test_empty_scene():
    r = beta_variance_probe(_scene(), BetaVarianceConfig(enabled=True))
    assert r.n_subjects == 0
    assert r.subjects == []
    assert r.similar_pairs == []


def test_zero_betas_not_extreme():
    s = _subject(1, np.zeros(10))
    r = beta_variance_probe(_scene(s), BetaVarianceConfig(enabled=True))
    assert len(r.subjects) == 1
    assert r.subjects[0].betas_norm == 0.0
    assert not r.subjects[0].is_extreme
    assert r.subjects_extreme == 0


def test_extreme_betas_flagged():
    """|betas| ≈ 3.16 > threshold=3.0 → extreme."""
    s = _subject(1, np.ones(10))     # ‖betas‖ = √10 ≈ 3.16
    r = beta_variance_probe(
        _scene(s), BetaVarianceConfig(enabled=True, extreme_norm_threshold=3.0),
    )
    assert r.subjects[0].is_extreme
    assert r.subjects_extreme == 1


def test_moderate_betas_not_flagged():
    s = _subject(1, 0.2 * np.ones(10))   # ‖betas‖ ≈ 0.63
    r = beta_variance_probe(
        _scene(s), BetaVarianceConfig(enabled=True, extreme_norm_threshold=3.0),
    )
    assert not r.subjects[0].is_extreme
    assert r.subjects_extreme == 0


def test_similar_pair_detected():
    """Two subjects with near-identical betas → cos_dist ≈ 0 → similar pair."""
    b = np.array([1.0, 0.5, -0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    s1 = _subject(1, b)
    s2 = _subject(2, b * 1.001)   # nearly parallel
    r = beta_variance_probe(
        _scene(s1, s2),
        BetaVarianceConfig(enabled=True, similar_cos_threshold=0.05),
    )
    assert len(r.similar_pairs) == 1
    pair = r.similar_pairs[0]
    assert {pair.track_id_a, pair.track_id_b} == {1, 2}
    assert pair.cos_distance < 0.05


def test_dissimilar_pair_not_flagged():
    """Orthogonal betas → cos_dist = 1.0 → not similar."""
    b1 = np.zeros(10); b1[0] = 1.0
    b2 = np.zeros(10); b2[1] = 1.0
    s1, s2 = _subject(1, b1), _subject(2, b2)
    r = beta_variance_probe(
        _scene(s1, s2),
        BetaVarianceConfig(enabled=True, similar_cos_threshold=0.05),
    )
    assert r.similar_pairs == []


def test_zero_norm_pair_skipped():
    """Subject with all-zero betas cannot participate in cos-similarity."""
    s1 = _subject(1, np.zeros(10))
    s2 = _subject(2, np.ones(10))
    r = beta_variance_probe(_scene(s1, s2), BetaVarianceConfig(enabled=True))
    assert r.similar_pairs == []


def test_three_subject_pair_enumeration():
    """Three subjects with same shape → 3 similar pairs enumerated."""
    b = np.arange(10, dtype=float)
    r = beta_variance_probe(
        _scene(_subject(1, b), _subject(2, b), _subject(3, b)),
        BetaVarianceConfig(enabled=True, similar_cos_threshold=0.05),
    )
    assert len(r.similar_pairs) == 3
    pair_ids = {(p.track_id_a, p.track_id_b) for p in r.similar_pairs}
    assert pair_ids == {(1, 2), (1, 3), (2, 3)}
