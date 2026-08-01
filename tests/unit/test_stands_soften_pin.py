"""Stands grain-soften pin (t20 lever): blur-blend + saturation quantile map.

The stands band read as saturated confetti — luma local-contrast 2.4x the clip, frac(S>0.5)
.71 vs .43 (t20 measured 2026-07-05). Tone pins land medians, not distribution SHAPE; these
pin the pure helpers: contrast actually drops toward the blur, keep=1 is identity, the S
quantile map lands the reference distribution (a knee can't hit med+p90+mass at once), luma
survives the S remap, and everything is deterministic.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # provided by [cv]; CI installs opencv-python-headless

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from stands_soften_pin import (  # noqa: E402
    apply_sat_lut,
    sat_match_lut,
    sat_of,
    soften_grain,
)


def _confetti(seed: int = 0, h: int = 64, w: int = 256) -> np.ndarray:
    """Saturated block mosaic ~ the rendered crowd band (float32 BGR 0-1)."""
    rng = np.random.default_rng(seed)
    blocks = rng.random((h // 4, w // 4, 3)).astype(np.float32)
    img = np.repeat(np.repeat(blocks, 4, axis=0), 4, axis=1)
    return np.clip(0.1 + 0.8 * img, 0.0, 1.0).astype(np.float32)


def _local_contrast(band: np.ndarray) -> float:
    luma = band.mean(axis=2)
    return float(np.abs(luma - cv2.blur(luma, (5, 5))).mean())


def test_soften_reduces_local_contrast_and_keep1_is_identity():
    band = _confetti()
    lc0 = _local_contrast(band)
    soft = soften_grain(band, ksize=9, keep=0.25)
    assert soft.shape == band.shape and soft.dtype == np.float32
    assert _local_contrast(soft) < 0.5 * lc0
    # keep=1 returns the full detail: blur + 1.0*(band-blur) == band
    assert np.allclose(soften_grain(band, ksize=9, keep=1.0), band, atol=1e-6)
    # monotone in keep
    assert _local_contrast(soften_grain(band, keep=0.5)) > _local_contrast(soft)


def test_sat_lut_lands_reference_distribution():
    src = _confetti(0)
    ref = np.clip(_confetti(1) * np.float32([0.9, 0.85, 0.8]) + 0.1, 0.0, 1.0)  # duller ref
    ss, sm = sat_of(src)
    rs, rm = sat_of(ref)
    assert float(np.median(ss[sm])) > float(np.median(rs[rm]))  # gap exists to close
    lut = sat_match_lut(ss[sm], rs[rm])
    assert lut.shape[0] == 2
    assert (np.diff(lut[0]) >= -1e-6).all() and (np.diff(lut[1]) >= -1e-6).all()  # monotone
    out = apply_sat_lut(src, lut)
    os_, om = sat_of(out)
    for q in (50, 90):
        got = float(np.percentile(os_[om], q))
        want = float(np.percentile(rs[rm], q))
        assert abs(got - want) < 0.05, f"p{q}: {got} vs {want}"
    # saturated-mass proxy lands too
    assert abs(float((os_[om] > 0.5).mean()) - float((rs[rm] > 0.5).mean())) < 0.08


def test_sat_lut_preserves_value_channel():
    src = _confetti(2)
    ref = np.clip(_confetti(3) * 0.7 + 0.15, 0.0, 1.0)
    ss, sm = sat_of(src)
    rs, rm = sat_of(ref)
    out = apply_sat_lut(src, sat_match_lut(ss[sm], rs[rm]))
    v_src = cv2.cvtColor((src * 255).astype(np.uint8), cv2.COLOR_BGR2HSV_FULL)[..., 2]
    v_out = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2HSV_FULL)[..., 2]
    # V (max channel) is untouched by an S-only remap up to uint8 rounding
    assert int(np.abs(v_out.astype(np.int16) - v_src.astype(np.int16)).max()) <= 1


def test_soften_and_lut_deterministic():
    band = _confetti(4)
    ref = _confetti(5) * np.float32(0.8)
    ss, sm = sat_of(band)
    rs, rm = sat_of(ref)
    lut = sat_match_lut(ss[sm], rs[rm])
    a = apply_sat_lut(soften_grain(band), lut)
    b = apply_sat_lut(soften_grain(band), lut)
    assert np.array_equal(a, b)
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0
