"""Interpenetration probe — capsule overlap statistics."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.correction.interpen_probe import (
    InterpenConfig,
    interpen_probe,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, xy: np.ndarray) -> Subject:
    T = xy.shape[0]
    frames = np.arange(T, dtype=int)
    transl = np.zeros((T, 3))
    transl[:, :2] = xy
    return Subject(track_id=track_id, proposal=SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=np.zeros((T, 3)),
            body_pose=np.zeros((T, 21, 3)), transl=transl,
        ),
    ))


def _scene(*subjects):
    return Scene(id="s", episode_id="e", source_id="c",
                subjects=list(subjects), corrections=[])


def test_disabled_returns_empty():
    r = interpen_probe(_scene(_subject(1, np.zeros((5, 2)))),
                      InterpenConfig(enabled=False))
    assert r.subjects == []


def test_far_apart_no_overlap():
    xy1 = np.zeros((5, 2))
    xy2 = np.tile([[10.0, 0.0]], (5, 1))
    r = interpen_probe(_scene(_subject(1, xy1), _subject(2, xy2)),
                      InterpenConfig(enabled=True, interpen_radius_m=0.5))
    assert r.subjects_with_overlap == 0
    assert r.total_overlap_frames == 0


def test_close_pair_flagged():
    xy1 = np.zeros((5, 2))
    xy2 = np.tile([[0.3, 0.0]], (5, 1))   # inside 0.5m
    r = interpen_probe(_scene(_subject(1, xy1), _subject(2, xy2)),
                      InterpenConfig(enabled=True, interpen_radius_m=0.5))
    assert r.subjects_with_overlap == 2
    assert r.max_overlap_m == pytest.approx(0.2, abs=1e-6)


def test_solo_scene_no_overlap():
    r = interpen_probe(_scene(_subject(1, np.zeros((5, 2)))),
                      InterpenConfig(enabled=True))
    assert r.subjects_with_overlap == 0


def test_overlap_fraction_per_subject():
    """Two subjects overlap only on frames 0-2 of 5 → fraction 3/5."""
    xy1 = np.zeros((5, 2))
    xy2 = np.zeros((5, 2))
    xy2[:3, 0] = 0.3       # close on 0-2
    xy2[3:, 0] = 5.0       # far on 3-4
    r = interpen_probe(_scene(_subject(1, xy1), _subject(2, xy2)),
                      InterpenConfig(enabled=True, interpen_radius_m=0.5))
    fr_1 = next(s for s in r.subjects if s.track_id == 1).overlap_fraction
    assert fr_1 == pytest.approx(0.6, abs=1e-6)


def test_empty_scene():
    r = interpen_probe(_scene(), InterpenConfig(enabled=True))
    assert r.n_subjects == 0
