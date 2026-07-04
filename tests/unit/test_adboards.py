"""LED ad-board ring — the broadcast perimeter band between grass and crowd.

The clip (floodlit night) reads grass -> bright white LED strip -> dark walkway -> crowd; the
render previously ran grass straight into the crowd wall. These pin the ring geometry contract:
two closed vertical bands (board + gap) with per-vertex colours the renderer feeds into an
emission shader, valid face indices, and the board sitting outside the pitch lines.
"""

from __future__ import annotations

import numpy as np

from pitch3d.core.scene.stadium import FieldDimensions, adboard_ring_geometry


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
    # Board band is the bright one, walkway band the dark one.
    assert colors[:half].min() > 0.9
    assert colors[half:].max() < 0.1


def test_ring_sits_outside_the_pitch():
    dims = FieldDimensions()
    offset = 5.0
    verts, _, _ = adboard_ring_geometry(dims, offset=offset, n_around=128)
    x, y = np.abs(verts[:, 0]), np.abs(verts[:, 1])
    # Nothing inside the playing area, nothing past the requested offset.
    assert np.all((x >= dims.length / 2 - 1e-6) | (y >= dims.width / 2 - 1e-6))
    assert x.max() <= dims.length / 2 + offset + 1e-6
    assert y.max() <= dims.width / 2 + offset + 1e-6
