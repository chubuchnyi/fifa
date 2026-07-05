"""Fascia multi-window quilt + scattered red fan recolour (t19 stands levers).

The walkway/fascia band repeated ONE measured window ~19x around the ring — a periodic gold
panel row is the tell (t18 residual); the quilt stitches the pan's candidate windows so no two
repeats match. The crowd quilt's own red fans (3.2% of pixels) minify away at bowl range
(measured beauty: 0.0% red vs clip stands 4.0%), so the scatter re-enters them at cluster
scale. These pin the pure-numpy invariants: measured-only content, vertical band structure
kept, determinism, luma-preserving chroma swap, and the fraction dial.
"""

from __future__ import annotations

import numpy as np

from pitch3d.adapters.render.stadium_backdrop import (
    assemble_fascia_quilt,
    fascia_pool_keep,
    scatter_fan_recolor,
)


def _striped_candidate(seed: int, rows: int = 20, w: int = 60) -> np.ndarray:
    """Fascia-like window: fixed horizontal layers + per-candidate x-texture."""
    rng = np.random.default_rng(seed)
    f = np.empty((rows, w, 3), dtype=np.float32)
    f[: rows // 2] = 0.7  # "crowd/panel" upper layers
    f[rows // 2 :] = 0.2  # "hedge/walkway" lower layers
    f += 0.08 * (rng.random((1, w, 3)) - 0.5)  # x-varying content, constant down columns
    return np.clip(f, 0.0, 1.0).astype(np.float32)


def test_fascia_quilt_shape_and_measured_range():
    pool = [_striped_candidate(s) for s in range(4)]
    q = assemble_fascia_quilt(pool, 3)
    assert q.shape == (20, 180, 3)
    assert q.dtype == np.float32
    assert np.isfinite(q).all()
    lo = min(float(f.min()) for f in pool)
    hi = max(float(f.max()) for f in pool)
    # Hann blending is a convex mix of pool pixels — nothing outside the measured range.
    assert float(q.min()) >= lo - 1e-3
    assert float(q.max()) <= hi + 1e-3


def test_fascia_quilt_keeps_vertical_band_structure():
    # The band is layered (crowd -> panel row -> hedge -> walkway); crops are full-height and
    # x-blended only, so each output row must stay at its layer level.
    q = assemble_fascia_quilt([_striped_candidate(s) for s in range(3)], 4)
    upper, lower = q[: q.shape[0] // 2], q[q.shape[0] // 2 :]
    assert abs(float(upper.mean()) - 0.7) < 0.05
    assert abs(float(lower.mean()) - 0.2) < 0.05
    assert float(upper.min()) > float(lower.max())  # layers never bleed across


def test_fascia_quilt_windows_do_not_repeat():
    pool = [_striped_candidate(s) for s in range(4)]
    q = assemble_fascia_quilt(pool, 3, seed=0)
    w = q.shape[1] // 3
    assert not np.allclose(q[:, :w], q[:, w : 2 * w], atol=1e-2)
    # and the quilt is not the legacy single window tiled
    assert not np.allclose(q[:, :w], pool[0], atol=1e-2)


def test_fascia_quilt_deterministic_and_single_window_passthrough():
    pool = [_striped_candidate(s) for s in range(3)]
    a = assemble_fascia_quilt(pool, 4, seed=2)
    b = assemble_fascia_quilt(pool, 4, seed=2)
    c = assemble_fascia_quilt(pool, 4, seed=3)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)
    one = assemble_fascia_quilt(pool, 1)
    assert np.array_equal(one, pool[0])  # legacy: the dominant window untouched


def test_fascia_quilt_trims_pool_to_common_size():
    pool = [_striped_candidate(0, rows=20, w=60), _striped_candidate(1, rows=19, w=57)]
    q = assemble_fascia_quilt(pool, 2)
    assert q.shape == (19, 114, 3)


def test_pool_keep_drops_the_contaminated_candidate():
    # t19 batch: one candidate caught a red/white flag crossing the band and the quilt spread
    # it into pink blocks around the ring. Consensus pruning must drop it and keep the rest.
    pool = [_striped_candidate(s) for s in range(5)]
    bad = pool[3].copy()
    bad[5:15, 10:50] = np.float32([0.9, 0.3, 0.4])  # big foreign blob
    pool[3] = bad
    keep = fascia_pool_keep(pool)
    assert 3 not in keep
    assert len(keep) >= 3
    q = assemble_fascia_quilt(pool, 4)
    # nothing in the quilt reaches the blob's magenta signature
    magenta = (q[..., 0] > 0.8) & (q[..., 1] < 0.45) & (q[..., 2] > 0.3)
    assert not magenta.any()


def test_pool_keep_passes_small_and_consistent_pools():
    pool = [_striped_candidate(s) for s in range(4)]
    assert fascia_pool_keep(pool) == [0, 1, 2, 3]  # consistent pool: nobody dropped
    assert fascia_pool_keep(pool[:2]) == [0, 1]    # <3: no consensus to test against


def test_scatter_red_fraction_luma_and_determinism():
    rng = np.random.default_rng(0)
    quilt = (0.3 + 0.3 * rng.random((128, 512, 3))).astype(np.float32)
    out = scatter_fan_recolor(quilt, frac=0.05, seed=1, diam_range=(8, 24))
    assert out.shape == quilt.shape and out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    changed = np.abs(out - quilt).max(axis=2) > 1e-3
    assert 0.025 <= float(changed.mean()) <= 0.15  # frac dial actually lands near target
    # Recoloured pixels are RED-dominant (chroma swapped toward the measured shirt colour);
    # core = solidly-swapped px, past the feathered rim where the blend is still partial.
    core = np.abs(out - quilt).max(axis=2) > 0.15
    assert (out[core, 0] > out[core, 1]).mean() > 0.99
    assert (out[core, 0] > out[core, 2]).mean() > 0.99
    # ...but keep the crowd's own light: per-pixel luma is preserved through the swap.
    dl = np.abs(out[core].mean(axis=1) - quilt[core].mean(axis=1))
    assert float(np.median(dl)) < 0.02

    again = scatter_fan_recolor(quilt, frac=0.05, seed=1, diam_range=(8, 24))
    other = scatter_fan_recolor(quilt, frac=0.05, seed=2, diam_range=(8, 24))
    assert np.array_equal(out, again)
    assert not np.allclose(out, other)


def test_scatter_zero_frac_is_a_noop():
    quilt = np.full((32, 64, 3), 0.4, dtype=np.float32)
    out = scatter_fan_recolor(quilt, frac=0.0)
    assert np.array_equal(out, quilt)
