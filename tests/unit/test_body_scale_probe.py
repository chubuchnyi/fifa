"""Body-scale probe — SMPL-X shape consistency across subjects."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.correction.body_scale_probe import (
    BodyScaleConfig,
    body_scale_probe,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, T: int = 5) -> Subject:
    frames = np.arange(T, dtype=int)
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=np.zeros((T, 3)),
            body_pose=np.zeros((T, 21, 3)), transl=np.zeros((T, 3)),
        ),
    )
    return Subject(track_id=track_id, proposal=motion)


def _scene(*subjects):
    return Scene(id="s", episode_id="e", source_id="c",
                subjects=list(subjects), corrections=[])


def _provider(offsets_by_id):
    def p(subject):
        return offsets_by_id.get(int(subject.track_id))
    return p


def test_disabled_returns_empty():
    r = body_scale_probe(
        _scene(_subject(1)), BodyScaleConfig(enabled=False), _provider({1: 0.92}),
    )
    assert r.subjects == []


def test_no_provider_returns_empty():
    r = body_scale_probe(_scene(_subject(1)), BodyScaleConfig(enabled=True), None)
    assert r.subjects == []


def test_similar_heights_no_flag():
    r = body_scale_probe(
        _scene(_subject(1), _subject(2)),
        BodyScaleConfig(enabled=True, inter_subject_threshold_m=0.15),
        _provider({1: 0.92, 2: 0.95}),
    )
    assert r.n_pairs_flagged == 0
    assert r.max_pair_diff_m == pytest.approx(0.03)


def test_wildly_different_heights_flagged():
    r = body_scale_probe(
        _scene(_subject(1), _subject(2), _subject(3)),
        BodyScaleConfig(enabled=True, inter_subject_threshold_m=0.15),
        _provider({1: 0.92, 2: 1.30, 3: 0.90}),
    )
    # pair (1,2)=0.38, (1,3)=0.02, (2,3)=0.40 → 2 flagged
    assert r.n_pairs_flagged == 2
    assert r.max_pair_diff_m > 0.35


def test_provider_returns_array_uses_median():
    r = body_scale_probe(
        _scene(_subject(1), _subject(2)),
        BodyScaleConfig(enabled=True),
        _provider({1: np.array([0.9, 0.92, 0.94]), 2: np.array([1.1, 1.15])}),
    )
    assert r.subjects[0].pelvis_above_foot_m == pytest.approx(0.92)
    assert r.subjects[1].pelvis_above_foot_m == pytest.approx(1.125)


def test_empty_scene():
    r = body_scale_probe(_scene(), BodyScaleConfig(enabled=True), _provider({}))
    assert r.n_pairs_flagged == 0
