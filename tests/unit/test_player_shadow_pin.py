"""Player contact-shadow pin (t23 lever): shirt-color detection + ellipse-alpha darken.

The v2v pass smears players into halos that already darken the under-feet zone (-.139 V
below grass on t21_pinned8 f28) but the shadow SHAPE is wrong — the clip has a tight
elliptical patch (-.029), our smear has a diffuse ring. Screen-space pin paints an
elliptical dark alpha at the foot line, gated by a grass mask so the darkening stacks
on unshaded grass rather than the v2v halo.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # provided by [cv]; CI installs opencv-python-headless

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from player_shadow_pin import (  # noqa: E402
    grass_mask,
    paint_shadows,
    player_boxes,
)


def _pitch_bgr(h: int = 200, w: int = 400) -> np.ndarray:
    """Green mid-V pitch bgr uint8."""
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 1] = 130  # G
    img[..., 0] = 60   # B
    img[..., 2] = 40   # R -> green H, S ~.7, V ~.51
    return img


def _add_yellow_player(img: np.ndarray, cx: int, cy: int, ww: int, hh: int) -> None:
    """Paint a yellow rectangle at (cx-w/2, cy-h/2..cy+h/2)."""
    x0, y0 = cx - ww // 2, cy - hh // 2
    img[y0:y0 + hh, x0:x0 + ww] = (30, 220, 220)  # BGR yellow


def test_player_boxes_detects_yellow_and_azure():
    img = _pitch_bgr()
    _add_yellow_player(img, 100, 130, 14, 40)  # yellow
    # azure blob
    img[110:140, 200:214] = (200, 130, 40)     # BGR ~ azure H
    boxes = player_boxes(img)
    assert len(boxes) >= 2, f"expected >=2 players, got {len(boxes)}: {boxes}"
    # each box maps to one of the two centers
    centers = [(x + w // 2, y + h // 2) for (x, y, w, h) in boxes]
    assert any(abs(cx - 100) < 8 for (cx, _) in centers)
    assert any(abs(cx - 207) < 8 for (cx, _) in centers)


def test_grass_mask_is_grass_where_grass_is():
    img = _pitch_bgr()
    _add_yellow_player(img, 100, 130, 14, 40)
    gm = grass_mask(img)
    # grass everywhere except the yellow blob
    assert gm.mean() > 0.85, f"grass frac too low: {gm.mean():.3f}"
    # yellow blob is NOT grass
    assert float(gm[110:150, 93:107].mean()) < 0.05


def test_paint_shadows_darkens_only_below_and_only_on_grass():
    img = _pitch_bgr()
    _add_yellow_player(img, 100, 130, 14, 40)
    boxes = player_boxes(img)
    out = paint_shadows(img, boxes, strength=0.3, feather=3, grass_only=True)
    assert out.shape == img.shape and out.dtype == np.uint8
    # V below the foot (foot_y = 130+20 = 150) is darker
    foot_y = 130 + 20
    v_before = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)[..., 2].astype(np.float32) / 255.0
    v_after = cv2.cvtColor(out, cv2.COLOR_BGR2HSV_FULL)[..., 2].astype(np.float32) / 255.0
    strip_before = float(v_before[foot_y:foot_y + 3, 90:110].mean())
    strip_after = float(v_after[foot_y:foot_y + 3, 90:110].mean())
    assert strip_after < strip_before - 0.02, f"contact strip not darker: {strip_after} vs {strip_before}"
    # yellow shirt V is UNTOUCHED by grass-only gating (blob is not grass)
    shirt_before = float(v_before[115:145, 95:105].mean())
    shirt_after = float(v_after[115:145, 95:105].mean())
    assert abs(shirt_after - shirt_before) < 0.005, f"shirt darkened: {shirt_after} vs {shirt_before}"
    # far-away grass (no ellipse) is untouched
    far_before = float(v_before[20:40, 300:340].mean())
    far_after = float(v_after[20:40, 300:340].mean())
    assert abs(far_after - far_before) < 0.005


def test_paint_shadows_is_identity_when_strength_zero():
    img = _pitch_bgr()
    _add_yellow_player(img, 100, 130, 14, 40)
    boxes = player_boxes(img)
    out = paint_shadows(img, boxes, strength=0.0, feather=3, grass_only=True)
    assert np.array_equal(out, img)


def test_paint_shadows_deterministic():
    img = _pitch_bgr()
    _add_yellow_player(img, 100, 130, 14, 40)
    boxes = player_boxes(img)
    a = paint_shadows(img, boxes, strength=0.25, feather=5, grass_only=True)
    b = paint_shadows(img, boxes, strength=0.25, feather=5, grass_only=True)
    assert np.array_equal(a, b)
