"""One crop rect for a whole phone clip is a measurement of nothing (`scripts/broadcast_crop.py`).

Measured 2026-08-07 on `14604731_1080_1920_30fps.mp4` in 30-frame windows, the grass band walks:

    f0-29    y 1262..1920   height 658   centre 1591
    f180-209 y  968..1920   height 952   centre 1444
    f330-354 y 1128..1920   height 792   centre 1524

The band centre moves **177 px** and its height changes by **354 px (59 %)** as the fan zooms, so
the single clip-wide rect the script used to emit (`1080x608+0+1200`) keeps stand at the start and
cuts pitch in the middle. Segmenting turns that into four rects at 82.4 / 91.3 / 91.7 / 90.3 %
grass, and — the property that matters for not breaking the broadcast path — a clip on a tripod
still collapses to exactly one segment: the target clip returns `1920x1080+0+0`, unchanged.

The measurement half needs a video, which the repo does not carry; the *grouping* is where the
"did the framing actually change" decision lives, and that is pure.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "broadcast_crop", Path(__file__).resolve().parents[2] / "scripts" / "broadcast_crop.py"
)
pytest.importorskip("cv2", reason="broadcast_crop imports cv2 at module level")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
merge_windows = _MOD.merge_windows
crop_rect = _MOD.crop_rect


def _w(a: int, b: int, y: int):
    """One measured window: frames a..b cropped to 1080x608 at vertical offset y."""
    return (a, b, (1080, 608, 0, y), 0.9)


def test_a_fixed_camera_collapses_to_one_segment():
    """A tripod never moves, so the broadcast path must be untouched by segmentation."""
    windows = [_w(0, 29, 236), _w(30, 59, 236), _w(60, 89, 236)]
    merged = merge_windows(windows, tolerance=48)
    assert len(merged) == 1
    assert merged[0][:3] == (0, 89, (1080, 608, 0, 236))


def test_small_jitter_stays_one_segment():
    """Sub-tolerance drift is measurement noise, not a new framing."""
    merged = merge_windows([_w(0, 29, 1200), _w(30, 59, 1220), _w(60, 89, 1240)], tolerance=48)
    assert len(merged) == 1, "20-px steps must not chop the clip into pieces"


def test_a_real_zoom_splits():
    """The fan clip's own drift: 1287 → 1195 → 1140 → 1209 crosses the tolerance three times."""
    merged = merge_windows(
        [_w(0, 119, 1287), _w(120, 179, 1195), _w(180, 299, 1140), _w(300, 354, 1209)],
        tolerance=48,
    )
    assert [(a, b) for a, b, _r, _k in merged] == [(0, 119), (120, 179), (180, 299), (300, 354)]


def test_segments_tile_the_clip_without_gaps_or_overlaps():
    windows = [_w(i, i + 29, 1200 + i) for i in range(0, 300, 30)]
    merged = merge_windows(windows, tolerance=48)
    assert merged[0][0] == windows[0][0] and merged[-1][1] == windows[-1][1]
    for (_a, prev_last, _r, _k), (next_first, *_rest) in zip(merged, merged[1:], strict=False):
        assert next_first == prev_last + 1


def test_a_size_change_always_splits():
    """Same y, different crop size: the source changed, and gluing those is never right."""
    merged = merge_windows([(0, 29, (1080, 608, 0, 500), 0.9), (30, 59, (720, 405, 0, 500), 0.9)],
                           tolerance=1000)
    assert len(merged) == 2


def test_crop_rect_locks_the_aspect_so_only_y_moves():
    """Why `merge_windows` compares size exactly and drift only in y."""
    wide = crop_rect(1080, 1920, (1262, 1920), 16 / 9)
    zoomed = crop_rect(1080, 1920, (968, 1920), 16 / 9)
    assert wide[:3] == zoomed[:3], "aspect-locked: width, height and x are clip constants"
    assert wide[3] != zoomed[3], "the band moving must move the crop"
