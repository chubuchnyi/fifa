"""LED strip finishing math — ad-dominance frame choice + per-strip emission calibration.

The clip's LED boards rotate ads inside the render window (dark FIFA panels most of the time,
one white BANK OF AMERICA stretch); the old widest-span frame choice was time-blind and cut the
minority white ad, and a fixed emission x4 clipped panels AND text to PNG white (polarity gone,
2026-07-05). These pin the two pure rules: the dominant-appearance pick (median panel level) and
the emission that saturates only the strip's own bright content.
"""

from __future__ import annotations

import numpy as np

from pitch3d.adapters.render.stadium_backdrop import dominant_strip_index, strip_emission


def _strip(bg: float, text: float, text_frac: float = 0.2) -> np.ndarray:
    s = np.full((8, 100, 3), bg, dtype=np.float32)
    s[:, : int(100 * text_frac)] = text
    return s


def test_white_ad_gets_gentle_emission():
    # BANK OF AMERICA: bright bg .95, navy text .26 — bg may just saturate, text must survive.
    e = strip_emission(_strip(bg=0.95, text=0.26))
    assert 1.0 <= e < 1.2
    assert 0.26 * e < 0.45  # letters stay readable after the multiply


def test_dark_ad_saturates_text_not_panels():
    # FIFA panels: dark bg .30, glowing text .80 — the old x4 pushed the PANELS past white too.
    e = strip_emission(_strip(bg=0.30, text=0.80))
    assert 1.2 < e < 1.5
    assert 0.30 * e < 0.5  # panels keep their measured darkness
    assert 0.80 * e >= 1.0  # text glows (saturates the PNG)


def test_pitch_black_strip_clamps():
    assert strip_emission(np.full((4, 50, 3), 0.05, np.float32)) == 4.0


def test_fascia_calibration_targets_walkway_level():
    # Fascia mode (q=50 -> .40): the dark walkway window emits at the grade-survival level,
    # and a BRIGHT window dims below x1 instead of clamping at it (t13: p90->1.05 was 1.6x hot).
    dark = _strip(bg=0.18, text=0.85, text_frac=0.1)
    e = strip_emission(dark, target=0.40, q=50.0, lo=0.25)
    assert abs(0.18 * e - 0.40) < 0.02
    bright = _strip(bg=0.60, text=0.20, text_frac=0.1)
    assert strip_emission(bright, target=0.40, q=50.0, lo=0.25) < 1.0


def test_dominant_pick_is_the_majority_ad():
    strips = [_strip(0.95, 0.26), _strip(0.30, 0.80), _strip(0.31, 0.78), _strip(0.29, 0.82)]
    assert dominant_strip_index(strips) in {1, 2, 3}


def test_dominant_pick_single_candidate():
    assert dominant_strip_index([_strip(0.95, 0.26)]) == 0
