"""Core stadium-bowl geometry — the procedural scaffold the measured backdrop wraps (M2 stadium).

These pin the contract the texture-bake / hole-fill steps rely on: vertex/face/param counts, the
base→top height span, that the footprint sits outside the touchlines (a stand never overlaps the
pitch), face indices are valid, normals point inward (the side the camera sees), and the bowl keeps
the pitch's four-fold symmetry so hole-fill can copy a half-turn across.
"""

from __future__ import annotations

import numpy as np

from pitch3d.core.scene.stadium import (
    bowl_tile_loop_uvs,
    fill_holes_by_copy,
    stadium_bowl_geometry,
)
from pitch3d.core.scene.units import FieldDimensions


def test_counts_match_the_loop_times_rows_grid():
    n_around, rows = 240, 20
    verts, faces, param = stadium_bowl_geometry(n_around=n_around, rows=rows)
    assert verts.shape == (n_around * (rows + 1), 3)
    assert param.shape == (n_around * (rows + 1), 2)
    assert faces.shape == (n_around * rows * 2, 3)
    assert 0 <= int(faces.min()) and int(faces.max()) < verts.shape[0]
    assert np.isfinite(verts).all()


def test_height_spans_base_to_rake_top():
    rows, rise = 18, 0.8
    verts, _, param = stadium_bowl_geometry(rows=rows, rise=rise, run=0.9)
    np.testing.assert_allclose(verts[:, 2].min(), 0.0)
    np.testing.assert_allclose(verts[:, 2].max(), rows * rise)
    # param is (angle_frac in [0,1), height_frac in [0,1])
    assert param[:, 0].min() >= 0.0 and param[:, 0].max() < 1.0
    np.testing.assert_allclose([param[:, 1].min(), param[:, 1].max()], [0.0, 1.0])


def test_footprint_sits_outside_the_pitch():
    dims = FieldDimensions(length=105.0, width=68.0)
    apron = 7.0
    verts, _, param = stadium_bowl_geometry(dims, apron=apron, corner_radius=16.0)
    base = verts[param[:, 1] == 0.0]
    # No base vertex falls inside the touch/goal-line rectangle — the stand cannot overlap play.
    inside = (np.abs(base[:, 0]) < dims.length / 2.0) & (np.abs(base[:, 1]) < dims.width / 2.0)
    assert int(inside.sum()) == 0
    # Straight-edge extent = pitch half + apron (corners round inward of that).
    np.testing.assert_allclose(np.abs(base[:, 0]).max(), dims.length / 2.0 + apron)
    np.testing.assert_allclose(np.abs(base[:, 1]).max(), dims.width / 2.0 + apron)


def test_tiers_step_outward_away_from_the_pitch():
    # On the +Y straight run, a higher row sits further out (+Y) than the base — the seating rake.
    verts, _, param = stadium_bowl_geometry()
    plusy = (param[:, 0] > 0.22) & (param[:, 0] < 0.28)  # near the +Y edge midpoint
    col = verts[plusy]
    base_y = col[col[:, 2].argmin(), 1]
    top_y = col[col[:, 2].argmax(), 1]
    assert top_y > base_y


def test_faces_wind_inward():
    # The render shows the inside of the bowl, so face normals must point toward the pitch centre.
    verts, faces, _ = stadium_bowl_geometry()
    tri = verts[faces[0]]
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    centroid = tri.mean(axis=0)
    to_centre = np.array([-centroid[0], -centroid[1], 0.0])  # horizontal, toward the axis
    assert float(np.dot(normal[:2], to_centre[:2])) > 0.0


def test_hole_fill_mirrors_far_stand_onto_the_near_stand():
    # Camera sees the +Y stand; the -Y near stand is a hole. Colour each vertex by its sign(y) so we
    # can prove the near stand was filled from its +Y mirror, not from an end corner.
    verts, _, _ = stadium_bowl_geometry(n_around=120, rows=8)
    covered = verts[:, 1] > 5.0  # only the far (+Y) sideline is seen
    colors = np.zeros((verts.shape[0], 3))
    colors[covered] = [0.0, 1.0, 0.0]  # far stand is green
    filled, src = fill_holes_by_copy(verts, colors, covered)
    near = (verts[:, 1] < -5.0)  # the near sideline (a hole)
    # Every near-stand vertex now carries the far stand's colour, sourced from a +Y covered vertex.
    assert np.allclose(filled[near], [0.0, 1.0, 0.0])
    assert (verts[src[near], 1] > 0).all()


def test_hole_fill_is_noop_when_all_or_none_covered():
    verts, _, _ = stadium_bowl_geometry(n_around=60, rows=4)
    colors = np.random.default_rng(0).random((verts.shape[0], 3))
    for covered in (np.ones(verts.shape[0], bool), np.zeros(verts.shape[0], bool)):
        filled, src = fill_holes_by_copy(verts, colors, covered)
        np.testing.assert_array_equal(filled, colors)
        np.testing.assert_array_equal(src, np.arange(verts.shape[0]))


def test_tile_uvs_are_per_loop_and_seam_free():
    n_around, rows = 240, 20
    verts, faces, param = stadium_bowl_geometry(n_around=n_around, rows=rows)
    repeat_around, repeat_up = 40.0, 2.0
    uv = bowl_tile_loop_uvs(faces, param, repeat_around=repeat_around, repeat_up=repeat_up)
    # One UV per face-corner, in face order, ready for a renderer's per-loop foreach_set.
    assert uv.shape == (faces.shape[0] * 3, 2)
    # u spans a full lap of tiles, v the rake; both stay within the requested repeat counts.
    np.testing.assert_allclose([uv[:, 0].min(), uv[:, 0].max()], [0.0, repeat_around], atol=1e-5)
    np.testing.assert_allclose([uv[:, 1].min(), uv[:, 1].max()], [0.0, repeat_up], atol=1e-5)
    # No face runs the texture backwards: within every triangle u jumps at most one tile-step, so
    # the wrap column (angle_frac 1 back to 0) was lifted a full turn, not snapped back to zero.
    per_face_u = uv[:, 0].reshape(-1, 3)
    step = repeat_around / n_around
    assert float((per_face_u.max(axis=1) - per_face_u.min(axis=1)).max()) <= step + 1e-5


def test_bowl_is_four_fold_symmetric():
    verts, _, _ = stadium_bowl_geometry(n_around=240)
    for sign in ([-1.0, 1.0, 1.0], [1.0, -1.0, 1.0]):
        mirrored = verts * np.asarray(sign)
        d = np.linalg.norm(mirrored[:, None, :] - verts[None, :, :], axis=2).min(axis=1)
        assert float(d.max()) < 1.0  # every mirrored vertex has a near original
