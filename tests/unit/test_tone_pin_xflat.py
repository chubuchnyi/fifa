"""Stands hot-edge fix (`grass_pin.py --flatten-val-x`) — the pure gain math.

The floodlit bowl renders the left stands ~1.9x brighter than mid while the clip's stands
never swing past ~1.4x and their profile wanders with the pan, so the pin flattens the band's
V x-profile to its OWN median (framing-independent). These pin the invariants the video pass
relies on: flat profile = no-op, gains invert the measured profile about the median (clamped,
smoothed), and the 2D field touches nothing outside the ROI band while feathering inside it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

_path = Path(__file__).resolve().parents[2] / "scripts" / "grass_pin.py"
_spec = importlib.util.spec_from_file_location("grass_pin", _path)
assert _spec is not None and _spec.loader is not None
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)


def test_flat_profile_is_a_noop():
    g = gp.xflat_gains(np.full(16, 0.27))
    assert np.allclose(g, 1.0, atol=1e-6)


def test_hot_edge_is_inverted_about_the_median_with_clamp():
    # The measured t10 16-bin relative profile: hot left edge 1.62x, healthy middle ~1.0.
    rel = np.array([1.62, 1.51, 1.38, 1.26, 1.08, 1.09, 1.02, 1.04,
                    0.94, 0.95, 0.95, 0.99, 0.99, 1.06, 1.01, 1.03])
    meds = 0.27 * rel
    g = gp.xflat_gains(meds)
    assert g[0] < 0.75  # hot corner strongly dimmed
    assert g.min() >= 0.55 - 1e-6 and g.max() <= 1.3 + 1e-6
    # Bins already at the band median must stay ~1: no relighting of the good stands.
    near_med = np.abs(meds - np.median(meds)) < 0.01
    assert near_med.any() and np.all(np.abs(g[near_med] - 1.0) < 0.1)


def test_smoothing_keeps_monotone_ramp_off_a_step():
    g = gp.xflat_gains(np.array([0.6, 0.6, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]))
    assert g[0] < 1.0 < g[-1] or g[0] < g[-1]  # dark target end gets gain > bright end
    assert np.all(np.diff(g) >= -1e-6)  # 3-tap smooth leaves the step monotone


def test_field_is_identity_outside_roi_and_feathered_inside():
    roi = (0.10, 0.30, 0.0, 1.0)
    gains = gp.xflat_gains(np.array([0.5, 0.25, 0.25, 0.25]))
    f = gp.xflat_field((200, 80), roi, gains)
    assert f.shape == (200, 80)
    assert np.allclose(f[:20], 1.0) and np.allclose(f[60:], 1.0)  # above/below the band
    core = f[30]  # middle of the band: full gain, left column dimmed
    assert core[:10].mean() < 0.8 < core[60:].mean() <= 1.31
    edge_in = f[21]  # feather row: closer to 1 than the core is
    assert abs(edge_in[:10].mean() - 1.0) < abs(core[:10].mean() - 1.0)


def test_sat_max_gate_selects_board_whites_only():
    # One row: glowing white LED (S~0, V hi), dark letter (V lo), yellow kit (S hi, V hi).
    img = np.zeros((4, 3, 3), dtype=np.uint8)
    img[:, 0] = (245, 245, 245)   # board white
    img[:, 1] = (20, 20, 20)      # letter
    img[:, 2] = (0, 220, 235)     # saturated yellow kit (BGR)
    _, gate = gp._gate(img, None, (0.0, 360.0), 0.0, 0.35, sat_max=0.35)
    assert gate[:, 0].all()        # whites pinned
    assert not gate[:, 1].any()    # letters stay dark
    assert not gate[:, 2].any()    # kits outside the desaturated gate


def test_sat_max_default_keeps_legacy_gate():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[:] = (40, 200, 90)  # saturated green
    _, legacy = gp._gate(img, None, (55.0, 140.0), 0.25, 0.10)
    assert legacy.all()
