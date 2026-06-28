"""Measured floodlight-colour estimator (v2 lever 3 auto-detect).

The clip-decoding wrapper needs cv2 + a video, but the estimator math is pure numpy and is checked
here directly with synthetic frames: a bright near-neutral patch (the floodlit white kit / lines)
carrying a known cool tint, surrounded by saturated distractors (green grass, a red shirt) and a
dark night sky. A correct white-patch estimate recovers the tint and *rejects* the coloured/dark
pixels; a clip with nothing neutral and bright falls back to the measured night default (R-6: never
invent a colour we can't measure).
"""

from __future__ import annotations

import numpy as np

from pitch3d.adapters.render.lighting import NIGHT_LIGHT_RGB, estimate_light_color


def _frame(grass, kit, mid, red, dark):
    """A 100×100 RGB frame: grass everywhere, with neutral kit/mid patches + red + dark blocks."""
    img = np.empty((100, 100, 3), dtype=np.float32)
    img[:] = grass
    img[0:30, 0:30] = kit    # bright near-neutral floodlit surface (the illuminant carrier)
    img[0:30, 40:70] = mid   # dimmer neutral surface — must lose to the bright one
    img[60:80, 0:20] = red   # saturated distractor
    img[60:90, 60:90] = dark  # near-black night sky
    return img


def test_estimate_light_color_recovers_cool_floodlight_tint():
    # True floodlight: neutral with a faint cool (blue-up) cast; already peak-normalised to max 1.0.
    light = np.array([0.95, 0.95, 1.0], dtype=np.float32)
    img = _frame(
        grass=(0.10, 0.40, 0.08),
        kit=light,
        mid=(0.40, 0.40, 0.42),
        red=(0.70, 0.05, 0.05),
        dark=(0.02, 0.02, 0.03),
    )
    out = estimate_light_color([img])
    assert np.isclose(out.max(), 1.0)          # peak-normalised
    assert out[2] > out[0] and out[2] > out[1]  # cool: blue is the brightest channel
    assert np.allclose(out, light, atol=0.02)


def test_estimate_light_color_is_peak_normalised_below_one():
    # A dimmer-but-cool light (max channel 0.80) must normalise to a unit peak, same direction.
    light = np.array([0.76, 0.76, 0.80], dtype=np.float32)
    img = _frame(
        grass=(0.10, 0.40, 0.08),
        kit=light,
        mid=(0.30, 0.30, 0.32),
        red=(0.70, 0.05, 0.05),
        dark=(0.02, 0.02, 0.03),
    )
    out = estimate_light_color([img])
    assert np.isclose(out.max(), 1.0)
    assert np.allclose(out, light / light.max(), atol=0.02)


def test_estimate_light_color_falls_back_when_nothing_neutral_and_bright():
    # An all-grass frame: every pixel is saturated, so there is no illuminant carrier to measure.
    img = np.full((40, 40, 3), (0.10, 0.40, 0.08), dtype=np.float32)
    out = estimate_light_color([img])
    assert np.allclose(out, NIGHT_LIGHT_RGB)


def test_estimate_light_color_falls_back_on_empty_input():
    assert np.allclose(estimate_light_color([]), NIGHT_LIGHT_RGB)
