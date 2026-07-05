"""LED ad-board ring — the broadcast perimeter band between grass and crowd.

The clip (floodlit night) reads grass -> bright white LED strip -> dark walkway -> crowd; the
render previously ran grass straight into the crowd wall. These pin the ring geometry contract:
two closed vertical bands (board + gap) with per-vertex colours the renderer feeds into an
emission shader, valid face indices, the board sitting outside the pitch lines, the per-loop
UVs that wrap the measured sponsor strip around the ring, and the robust fit the strip
extractor uses to ride over goalposts/players crossing the boards.
"""

from __future__ import annotations

import numpy as np

from pitch3d.adapters.blender.anim_contract import required_keys_for
from pitch3d.adapters.render.stadium_backdrop import _robust_quadfit
from pitch3d.core.scene.stadium import (
    FieldDimensions,
    adboard_loop_uvs,
    adboard_ring_geometry,
)


def test_boards_are_a_known_contract_artifact():
    # write_manifest rejects unknown artifacts, so the export dies on the pod if this pattern
    # is ever dropped from REQUIRED_KEYS (2026-07-03: exactly that happened on first E2E).
    assert set(required_keys_for("boards.npz")) == {"verts", "faces", "colors"}


def test_ring_counts_and_validity():
    n = 240
    verts, faces, colors = adboard_ring_geometry(n_around=n)
    assert verts.shape == (4 * n, 3)
    assert colors.shape == (4 * n, 3)
    assert faces.shape == (4 * n, 3)
    assert faces.min() >= 0 and faces.max() < len(verts)
    # Every vertex referenced: a closed ring leaves no orphans.
    assert len(np.unique(faces)) == len(verts)


def test_ring_bands_split_at_board_height():
    h, gap = 1.0, 2.2
    verts, _, colors = adboard_ring_geometry(height=h, gap=gap, n_around=64)
    half = len(verts) // 2
    board_z, gap_z = verts[:half, 2], verts[half:, 2]
    assert set(np.round(board_z, 6)) == {0.0, h}
    assert set(np.round(gap_z, 6)) == {h, h + gap}
    # Board band is the bright one, walkway band the dark one (dark-GREY, not near-black:
    # 0.02 albedo crushed to a dead V 0.00 stripe through the generative tail, 2026-07-05).
    assert colors[:half].min() > 0.9
    assert colors[half:].max() < 0.2


def test_ring_sits_outside_the_pitch():
    dims = FieldDimensions()
    offset = 5.0
    verts, _, _ = adboard_ring_geometry(dims, offset=offset, n_around=128)
    x, y = np.abs(verts[:, 0]), np.abs(verts[:, 1])
    # Nothing inside the playing area, nothing past the requested offset.
    assert np.all((x >= dims.length / 2 - 1e-6) | (y >= dims.width / 2 - 1e-6))
    assert x.max() <= dims.length / 2 + offset + 1e-6
    assert y.max() <= dims.width / 2 + offset + 1e-6


def test_loop_uvs_mirror_ring_face_order():
    n = 16
    verts, faces, _ = adboard_ring_geometry(height=1.0, n_around=n)
    uv = adboard_loop_uvs(n, repeat_around=float(n))
    assert uv.shape == (12 * n, 2)
    board_faces = faces[: 2 * n]
    board_uv = uv[: 6 * n].reshape(2 * n, 3, 2)
    # v mirrors the corner height: with height=1 every loop's v equals its vertex z.
    assert np.allclose(board_uv[..., 1], verts[board_faces][..., 2])
    # u walks the ring position backwards (text orientation: ring order runs toward -x on the
    # far touchline, the measured strip toward +x), one repeat per segment at repeat_around=n...
    expect_u = float(n) - (board_faces // 2).astype(np.float32)
    # ...except the closing segment, which wraps down to u=0 instead of rewinding to n.
    expect_u[-2:][expect_u[-2:] == float(n)] = 0.0
    assert np.allclose(board_uv[..., 0], expect_u)
    # Walkway loops pin a constant UV so their near-black vertex tint owns the colour.
    assert np.all(uv[6 * n :] == 0.5)


def test_robust_quadfit_rejects_goalpost_outliers():
    t = np.linspace(0.0, 1.0, 400).astype(np.float32)
    y = 40.0 + 30.0 * t - 18.0 * t**2
    y_noisy = y + np.sin(np.arange(400)).astype(np.float32) * 0.7
    y_noisy[120:160] = 5.0  # a goalpost slicing through the band
    coef, keep = _robust_quadfit(t, y_noisy, np.ones(400, dtype=bool))
    assert float(np.abs(np.polyval(coef, t) - y).max()) < 2.0
    assert int(keep[120:160].sum()) == 0
