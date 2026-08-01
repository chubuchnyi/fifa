"""Pixel evidence for the pitch overlay — what counts as paint, and what does not.

Both directions have already burned us. A player's socks reading as a painted line let the
penalty arc score a perfect 0.0 px while it visibly lagged; rejecting those players by
cutting out thick white blobs then deleted the far touchline, which hugs the advertising
boards, and the missing line made the far field measure 37 px out when it was 4.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # provided by [cv]; CI installs opencv-python-headless
from poseannot import pitch_evidence  # noqa: E402  (must follow the cv2 skip guard)
from poseannot.pitch_evidence import _bridge, _masks, classify  # noqa: E402

_CROWD_ROWS = 200
_LINE_V = 302
_BOARD_V = 240
_BESIDE_BOARD_V = 269


def _synthetic_frame() -> np.ndarray:
    """Turf under a dark crowd, with paint, a fat white player, and a white board to dodge."""
    hsv = np.zeros((480, 640, 3), np.uint8)
    hsv[:_CROWD_ROWS] = (0, 0, 40)                      # crowd: unsaturated and dark
    hsv[_CROWD_ROWS:] = (60, 180, 120)                  # turf: green, saturated, not bright
    hsv[220:260, :] = (0, 0, 255)                       # advertising board: 40 px tall
    hsv[268:272, 50:600] = (0, 0, 255)                  # touchline, 9 px clear of the board
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


def test_a_line_running_beside_the_boards_is_still_evidence(monkeypatch):
    """The far touchline hugs the advertising boards, and it anchors the far half of the frame.

    Rejecting players by cutting out thick white blobs plus a margin also cut out this line,
    which is why the far field measured 37 px out when it was 4.
    """
    uv = np.array([[300.0, float(_BOARD_V)], [300.0, float(_BESIDE_BOARD_V)]])
    assert list(_classify(monkeypatch, uv)) == ["off", "ok"]


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
