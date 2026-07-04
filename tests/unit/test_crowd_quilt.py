"""Crowd-quilt assembly — the non-repeating stands texture (kaleidoscope fix).

Repeating one small measured tile 40×4 over the bowl reads as a kaleidoscope once the video is
sharpened; the quilt stitches random crops of that tile into ONE bowl-sized texture instead.
These pin what the fix relies on: every canvas pixel is real blended content (no dead zones,
including across the x-wrap), the stitch is deterministic per seed, adjacent windows do NOT
repeat (the whole point), the feather-blend adds no structure of its own on a flat tile, and
the continuous 0–1 unwrap UVs the quilt is sampled by still lift the wrap-seam faces.
"""

from __future__ import annotations

import numpy as np

from pitch3d.adapters.render.stadium_backdrop import (
    apply_stand_structure,
    assemble_crowd_quilt,
)
from pitch3d.core.scene.stadium import bowl_tile_loop_uvs, stadium_bowl_geometry


def _noise_tile(h: int = 24, w: int = 36, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.2 + 0.6 * rng.random((h, w, 3))).astype(np.float32)


def test_quilt_shape_range_and_full_coverage():
    q = assemble_crowd_quilt(_noise_tile(), width=128, height=32, seed=0)
    assert q.shape == (32, 128, 3)
    assert q.dtype == np.float32
    assert np.isfinite(q).all()
    # Tile values live in [0.2, 0.8]; with ±8% gain jitter every blended pixel must stay well
    # inside [0.18, 0.87] — a pixel near 0 would mean an unfilled hole in the canvas.
    assert float(q.min()) >= 0.2 * 0.92 - 1e-3
    assert float(q.max()) <= 0.8 * 1.08 + 1e-3


def test_quilt_is_deterministic_per_seed():
    tile = _noise_tile()
    a = assemble_crowd_quilt(tile, width=96, height=24, seed=3)
    b = assemble_crowd_quilt(tile, width=96, height=24, seed=3)
    c = assemble_crowd_quilt(tile, width=96, height=24, seed=4)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_quilt_is_not_periodic():
    # The legacy mosaic repeated the tile: window k and window k+1 were identical (or mirrored).
    # The quilt must break both symmetries.
    q = assemble_crowd_quilt(_noise_tile(), width=128, height=32, seed=0)
    w = 32
    first, second = q[:, :w], q[:, w : 2 * w]
    assert not np.allclose(first, second, atol=5e-2)
    assert not np.allclose(first, second[:, ::-1], atol=5e-2)


def test_quilt_adds_no_structure_on_a_flat_tile():
    # A constant tile must stay constant up to the per-patch gain jitter: the Hann feathering and
    # weight normalisation may not invent edges or darken seams.
    tile = np.full((20, 30, 3), 0.5, dtype=np.float32)
    q = assemble_crowd_quilt(tile, width=128, height=32, seed=1)
    assert float(q.min()) >= 0.5 * 0.92 - 1e-3
    assert float(q.max()) <= 0.5 * 1.08 + 1e-3


def test_stand_structure_carves_walkway_aisles_and_top_fade():
    # Bowl-v runs 0 at the BOTTOM row but the quilt is stored screen-style (numpy row 0 = top),
    # so the walkway (bowl-v 0.65) must land in the LOWER half of the array and the top fade
    # must darken the FIRST rows. structure=False in the assembler stays the raw stitch.
    flat = np.full((256, 1024, 3), 0.5, dtype=np.float32)
    s = apply_stand_structure(flat)
    h = flat.shape[0]

    def row_of(bowl_v: float) -> int:
        return int(round((1.0 - bowl_v) * h - 0.5))

    walk = s[row_of(0.65)].mean()
    crowd_below = s[row_of(0.10)].mean()  # below fade_start: untouched seated rows
    assert walk < 0.5 * crowd_below  # concourse clearly darker than seated rows

    rail = s[row_of(0.65 + 0.03) - 1].mean()
    assert rail > crowd_below  # bright railing lip just above the walkway

    top, mid_upper = s[0].mean(), s[row_of(0.85)].mean()
    assert top < mid_upper < crowd_below  # fade grows toward the top of the stand

    row = s[row_of(0.10)]  # a plain seated row: only aisle dips act on it
    dips = (row.mean(axis=1) < 0.9 * np.median(row.mean(axis=1))).sum()
    expected = int(1024 * (0.0035 / 0.024))  # aisle columns = width * duty cycle
    assert 0 < dips <= expected + 24  # periodic aisles present, not flooding the row

    raw = assemble_crowd_quilt(_noise_tile(), width=128, height=32, seed=2)
    on = apply_stand_structure(raw)
    assert not np.allclose(on, raw)  # the exporter's overlay actually changes the stitch
    assert float(on.mean()) < float(raw.mean())  # mean drops -> exporter must ship tile_gain


def test_unwrap_uvs_span_zero_one_and_lift_the_wrap_seam():
    # Quilt mode samples the bowl with ONE texture copy: repeat=(1, 1) must give a continuous
    # 0-1 unwrap where only the wrap-seam faces reach u = 1 (the lifted low column).
    n_around, rows = 48, 5
    _, faces, param = stadium_bowl_geometry(n_around=n_around, rows=rows)
    uv = bowl_tile_loop_uvs(faces, param, repeat_around=1.0, repeat_up=1.0)
    u, v = uv[:, 0], uv[:, 1]
    assert float(v.min()) == 0.0 and float(v.max()) == 1.0
    assert float(u.min()) == 0.0
    assert np.isclose(float(u.max()), 1.0)
    interior = u[u < 1.0]
    assert float(interior.max()) <= (n_around - 1) / n_around + 1e-6
