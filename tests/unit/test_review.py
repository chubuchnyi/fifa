"""Attention ranking (UX-4) — severity ordering, thresholds, ball-height, truncation."""

from __future__ import annotations

import numpy as np

from pitch3d.core.scene.layers import ConfidenceMap
from pitch3d.core.scene.motion import BallTrack
from pitch3d.core.scene.review import attention_list


def test_low_confidence_items_sorted_descending(make_scene):
    conf = ConfidenceMap(subject_frame_conf={0: np.array([0.1, 0.9, 0.4])})
    scene = make_scene(confidence=conf)
    items = attention_list(scene)
    assert len(items) == 2  # only frames below the 0.5 threshold
    assert all(it.reason == "low_confidence" for it in items)
    scores = [it.score for it in items]
    assert scores == sorted(scores, reverse=True)
    assert items[0].frame == 0  # 0.1 is the most urgent


def test_reprojection_and_truncation(make_scene):
    conf = ConfidenceMap(reprojection_error_px={0: np.array([2.0, 25.0, 50.0])})
    scene = make_scene(confidence=conf)
    items = attention_list(scene, max_items=1)
    assert len(items) == 1
    assert items[0].reason == "high_reprojection"
    assert items[0].frame == 2  # 50px is worse than 25px


def test_ball_height_flagged(make_scene):
    ball = BallTrack(frames=np.arange(3), positions_3d=np.zeros((3, 3)),
                     height_confidence=np.array([1.0, 0.2, 1.0]))
    items = attention_list(make_scene(ball=ball))
    assert len(items) == 1
    assert items[0].reason == "low_ball_height"
    assert items[0].frame == 1


def test_no_signals_is_empty(make_scene):
    assert attention_list(make_scene()) == []
