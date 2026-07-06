"""Step 3 (measurement) — foot-contact detection + slide magnitude probe."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.config.gates import ContactProbeConfig
from pitch3d.core.correction.contact_probe import (
    _find_runs,
    contact_probe,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, T: int = 10) -> Subject:
    frames = np.arange(T, dtype=int)
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=np.zeros((T, 3)),
            body_pose=np.zeros((T, 21, 3)), transl=np.zeros((T, 3)),
        ),
    )
    return Subject(track_id=track_id, proposal=motion)


def _scene(*subjects: Subject) -> Scene:
    return Scene(
        id="s", episode_id="e", source_id="c",
        subjects=list(subjects), corrections=[],
    )


def _provider(feet_by_id: dict[int, np.ndarray]):
    def p(subject):
        return feet_by_id.get(int(subject.track_id))
    return p


# ─── _find_runs ────────────────────────────────────────────────────────

def test_find_runs_basic():
    mask = np.array([True, True, False, True, True, True, False, True])
    runs = _find_runs(mask, min_len=2)
    assert runs == [(0, 1), (3, 5)]  # [7:8] single frame → dropped


def test_find_runs_respects_min_len():
    mask = np.array([True, False, True, True, False])
    runs = _find_runs(mask, min_len=2)
    assert runs == [(2, 3)]


def test_find_runs_all_true_returns_one_run():
    mask = np.ones(5, dtype=bool)
    runs = _find_runs(mask, min_len=2)
    assert runs == [(0, 4)]


def test_find_runs_no_true_returns_empty():
    mask = np.zeros(5, dtype=bool)
    assert _find_runs(mask, min_len=2) == []


# ─── contact_probe ───────────────────────────────────────────────────

def test_disabled_returns_empty_report():
    s = _subject(1, T=5)
    feats = np.zeros((5, 3))
    r = contact_probe(_scene(s), ContactProbeConfig(enabled=False),
                     foot_position_provider=_provider({1: feats}))
    assert r.total_contact_frames == 0
    assert r.total_runs == 0


def test_no_provider_returns_empty():
    s = _subject(1, T=5)
    r = contact_probe(_scene(s), ContactProbeConfig(enabled=True))
    assert r.total_contact_frames == 0


def test_perfectly_planted_no_slide():
    """5 frames all Z=0 with foot stationary → 1 run, 0 slides."""
    s = _subject(1, T=5)
    feats = np.zeros((5, 3))
    r = contact_probe(_scene(s), ContactProbeConfig(enabled=True),
                     foot_position_provider=_provider({1: feats}))
    assert r.total_runs == 1
    assert r.total_slides == 0
    assert r.runs[0].slide_m == pytest.approx(0.0)


def test_planted_but_sliding_flagged_as_slide():
    """Foot planted (Z=0) but XY drifts 0.30m → 1 run, 1 slide."""
    s = _subject(1, T=5)
    feats = np.zeros((5, 3))
    feats[:, 0] = np.linspace(0.0, 0.30, 5)
    r = contact_probe(_scene(s), ContactProbeConfig(enabled=True,
                                                    slide_threshold_m=0.05),
                     foot_position_provider=_provider({1: feats}))
    assert r.total_runs == 1
    assert r.total_slides == 1
    assert 0.29 < r.runs[0].slide_m < 0.31


def test_run_below_min_length_ignored():
    """A single-frame contact at Z=0 (min_run=2) is not counted."""
    T = 6
    s = _subject(1, T=T)
    feats = np.zeros((T, 3))
    feats[:, 2] = 1.0  # feet in the air by default
    feats[3, 2] = 0.0  # one contact frame only
    r = contact_probe(_scene(s), ContactProbeConfig(enabled=True,
                                                    min_contact_run_frames=2),
                     foot_position_provider=_provider({1: feats}))
    assert r.total_runs == 0


def test_slide_below_threshold_not_flagged():
    """A 2cm drift within a contact run is NOT flagged (< 5cm threshold)."""
    s = _subject(1, T=5)
    feats = np.zeros((5, 3))
    feats[:, 0] = np.linspace(0.0, 0.02, 5)
    r = contact_probe(_scene(s), ContactProbeConfig(enabled=True,
                                                    slide_threshold_m=0.05),
                     foot_position_provider=_provider({1: feats}))
    assert r.total_runs == 1
    assert r.total_slides == 0


def test_multiple_runs_per_subject():
    """Two contact runs separated by an air interval → both measured."""
    T = 10
    s = _subject(1, T=T)
    feats = np.zeros((T, 3))
    feats[:, 2] = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])   # 3 contact, 3 air, 4 contact
    feats[6:, 0] = np.linspace(0.0, 0.10, 4)                 # slide in the second run
    r = contact_probe(_scene(s), ContactProbeConfig(enabled=True,
                                                    slide_threshold_m=0.05),
                     foot_position_provider=_provider({1: feats}))
    assert r.total_runs == 2
    assert r.total_slides == 1


def test_provider_none_track_skipped():
    s = _subject(1, T=5)
    r = contact_probe(_scene(s), ContactProbeConfig(enabled=True),
                     foot_position_provider=lambda _: None)
    assert r.n_subjects == 1
    assert r.total_runs == 0


def test_bad_shape_raises():
    """Provider returning wrong shape → error, don't silently drop data."""
    s = _subject(1, T=5)
    with pytest.raises(ValueError, match="foot_position_provider"):
        contact_probe(_scene(s), ContactProbeConfig(enabled=True),
                     foot_position_provider=lambda _: np.zeros((5, 2)))


def test_reports_max_slide_across_subjects():
    """Aggregate max_slide_m across multiple subjects."""
    s1, s2 = _subject(1, T=5), _subject(2, T=5)
    feats1 = np.zeros((5, 3))
    feats1[:, 0] = np.linspace(0.0, 0.10, 5)
    feats2 = np.zeros((5, 3))
    feats2[:, 0] = np.linspace(0.0, 0.30, 5)
    r = contact_probe(_scene(s1, s2), ContactProbeConfig(enabled=True),
                     foot_position_provider=_provider({1: feats1, 2: feats2}))
    assert r.subjects_with_slides == 2
    assert r.max_slide_m == pytest.approx(0.30, abs=0.01)
    assert r.mean_slide_m == pytest.approx(0.20, abs=0.01)


def test_empty_scene():
    r = contact_probe(_scene(), ContactProbeConfig(enabled=True),
                     foot_position_provider=lambda _: None)
    assert r.n_subjects == 0
    assert r.total_runs == 0
