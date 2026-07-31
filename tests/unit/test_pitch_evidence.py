"""Pixel evidence for the pitch overlay — a player must never read as a painted line.

Guards the failure that made the penalty arc score a perfect 0.0 px while it visibly
lagged: the "white on grass" mask caught players' socks and shorts, and the arc runs
through a cluster of them. Paint is a few px wide, a player is tens.
"""

from __future__ import annotations

import cv2
import numpy as np
from poseannot import pitch_evidence
from poseannot.pitch_evidence import _bridge, _masks, classify

_CROWD_ROWS = 200
_LINE_V = 302


def _synthetic_frame() -> np.ndarray:
    """Grass under a dark crowd band, with one thin painted line and one fat white blob."""
    hsv = np.zeros((480, 640, 3), np.uint8)
    hsv[:_CROWD_ROWS] = (0, 0, 40)                      # crowd: unsaturated and dark
    hsv[_CROWD_ROWS:] = (60, 180, 120)                  # grass: green, saturated, not bright
    hsv[300:306, 50:600] = (0, 0, 255)                  # painted line: 6 px wide
    hsv[380:420, 100:140] = (0, 0, 255)                 # a player: 40 px across
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _classify(monkeypatch, uv: np.ndarray) -> np.ndarray:
    evidence = _masks(_synthetic_frame())
    monkeypatch.setattr(pitch_evidence, "_evidence_cached", lambda _p, _f: evidence)
    return classify(uv, "unused.mp4", 0)[0]


def test_a_player_is_not_evidence_of_a_painted_line(monkeypatch):
    on_player = np.array([[110.0, 400.0], [120.0, 400.0], [130.0, 400.0]])
    assert list(_classify(monkeypatch, on_player)) == ["off"] * 3


def test_labels_separate_confirmed_from_extrapolated_from_unknown(monkeypatch):
    uv = np.array([
        [300.0, float(_LINE_V)],   # on the paint
        [400.0, 450.0],            # clear grass, no marking anywhere near
        [300.0, 100.0],            # off the grass entirely — no evidence either way
    ])
    assert list(_classify(monkeypatch, uv)) == ["ok", "off", "unknown"]


def test_bridging_needs_confirmation_on_both_sides_of_the_gap():
    occluded = np.array([True, False, False, True])          # a player on the line
    assert list(_bridge(occluded, gap=3)) == [True] * 4
    trailing = np.array([True, False, False, False])         # nothing confirms the tail
    assert list(_bridge(trailing, gap=3)) == [True, False, False, False]
    too_long = np.array([True] + [False] * 5 + [True])
    assert list(_bridge(too_long, gap=3)) == [True] + [False] * 5 + [True]


def test_error_is_nan_where_there_is_no_evidence(monkeypatch):
    evidence = _masks(_synthetic_frame())
    monkeypatch.setattr(pitch_evidence, "_evidence_cached", lambda _p, _f: evidence)
    _, dist = classify(np.array([[300.0, float(_LINE_V)], [300.0, 100.0]]), "unused.mp4", 0)
    assert dist[0] < 1.0
    assert np.isnan(dist[1])
