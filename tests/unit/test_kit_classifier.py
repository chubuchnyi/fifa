"""The kit reader must never call the pitch a shirt (W13, 2026-08-08).

`scripts/track_quality.py --kit` classified yellow on `18 <= H <= 48 and S > 90` over the median of
the whole patch. The floodlit pitch on the target clip sits at **H 39-40, S ~150, V ~130**, which
is inside that band — so **64.9 % of every frame classified as "yellow kit"** and any box carrying
a normal amount of grass read `Y`.

It produced four false findings in one session and one that had already reached a committed
findings file (`docs/findings/track-labels-2026-08-07.json`, since corrected). Nobody had ever
pointed the threshold at a negative control; doing so takes two lines, which is what this file is.

Every HSV constant below is **measured off the target clip**, not chosen:

* grass, three patches well away from any player: H 39-40, S 145-170, V 86-141
* the yellow kit: the tracker's own fitted centroid sits at **H ~25**
* the blue kit: sampled median **H ~102**

so 35-48 was never yellow — it was turf.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "track_quality", Path(__file__).resolve().parents[2] / "scripts" / "track_quality.py"
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
classify_kit = _MOD.classify_kit


def _patch(h: int, s: int, v: int, size: int = 40) -> np.ndarray:
    """A solid BGR patch at the given OpenCV HSV, the way a median would see it."""
    hsv = np.full((size, size, 3), (h, s, v), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _mixed(a: tuple[int, int, int], b: tuple[int, int, int], frac_a: float) -> np.ndarray:
    """A patch that is `frac_a` of colour a and the rest b — a box with grass in it."""
    n = 40
    rows_a = int(round(n * frac_a))
    return np.concatenate([_patch(*a, size=n)[:rows_a], _patch(*b, size=n)[rows_a:]], axis=0)


GRASS = (39, 150, 130)      # measured: three pitch patches, H 39-40 S 145-170 V 86-141
YELLOW = (25, 200, 180)     # measured: the tracker's fitted yellow centroid, H ~25
BLUE = (102, 180, 150)      # measured: blue-kit tracks sample H ~102


def test_the_pitch_is_not_a_shirt():
    """The whole bug, in one assertion. `Y` here is what broke four findings."""
    assert classify_kit(_patch(*GRASS)) != 'Y'


@pytest.mark.parametrize("h", [35, 39, 44, 48])
def test_no_hue_in_the_grass_band_reads_as_a_kit(h: int):
    """35-48 is turf across its whole width, not just at the median we happened to measure."""
    assert classify_kit(_patch(h, 150, 130)) not in ('Y', 'B')


@pytest.mark.parametrize("h", [35, 40, 44, 48])
def test_the_yellow_band_itself_excludes_the_grass_hues(h: int):
    """The two halves of the fix are not equally load-bearing, and this is the weaker one.

    Rejecting grass before the median is what actually killed the bug: once it runs, *nothing* at
    H 35-48 with S >= 60 and V >= 40 ever reaches the band, so widening the band back to 18-48
    changes no result and mutating it is invisible. The narrow band only matters for a patch that
    evades the grass rule some other way — here by being dark (V < 40), i.e. a shadowed one.

    Kept because a band that says "yellow" about turf hues is wrong even when something upstream
    happens to hide it, and because the measured yellow centroid is H ~25.
    """
    assert classify_kit(_patch(h, 200, 30)) not in ('Y', 'B')


def test_the_two_real_kits_still_read_correctly():
    """A guard that rejects everything would also pass the test above."""
    assert classify_kit(_patch(*YELLOW)) == 'Y'
    assert classify_kit(_patch(*BLUE)) == 'B'


@pytest.mark.parametrize("grass_frac", [0.3, 0.5, 0.7])
def test_a_shirt_survives_the_grass_around_it(grass_frac: float):
    """The realistic case: a box is part player, part pitch. The player must still win."""
    assert classify_kit(_mixed(GRASS, YELLOW, grass_frac)) == 'Y'
    assert classify_kit(_mixed(GRASS, BLUE, grass_frac)) == 'B'


def test_an_all_pitch_box_reports_no_reading_rather_than_guessing():
    """R-6: a box with no shirt in it should say so, not vote with a handful of stray pixels."""
    assert classify_kit(_patch(*GRASS)) == '-'


def test_an_empty_patch_is_no_reading():
    assert classify_kit(np.zeros((0, 3), dtype=np.uint8)) == '-'
