"""Core pitch-marking geometry — the measured ground anchor (M2-1).

These are the standard FIFA markings the field calibration aligns to, emitted in world meters on
the ``Z = plane_z`` plane. The tests pin the frame convention (origin = pitch centre, X = length,
Y = width), the on-plane invariant, the four-fold pitch symmetry, and that the named spots land
where the Laws put them — so the env reconstructor can trust this as a *measured* template.
"""

from __future__ import annotations

import numpy as np

from pitch3d.core.scene.pitch import (
    _LINE_LIFT,
    pitch_line_ribbons,
    pitch_line_world_points,
    pitch_line_xy,
)
from pitch3d.core.scene.units import FieldDimensions


def _rounded_set(pts: np.ndarray) -> set:
    return {tuple(np.round(p, 3)) for p in pts}


def _max_mirror_gap(xy: np.ndarray, sign: list[float]) -> float:
    """Largest distance from a mirrored point to its nearest original point (symmetry residual)."""
    mirrored = xy * np.asarray(sign)
    d = np.linalg.norm(mirrored[:, None, :] - xy[None, :, :], axis=2).min(axis=1)
    return float(d.max())


def test_points_lie_on_the_requested_plane():
    pts = pitch_line_world_points(plane_z=0.3, spacing=1.0)
    np.testing.assert_allclose(pts[:, 2], 0.3)


def test_extents_match_pitch_dimensions_centred_at_origin():
    dims = FieldDimensions(length=105.0, width=68.0)
    xy = pitch_line_xy(dims, spacing=0.5)
    # Symmetric extent (min == -max) proves both the scale and the centre-at-origin frame; nothing
    # pokes outside the touchlines/goal lines (the arcs poke *inward*).
    np.testing.assert_allclose([xy[:, 0].min(), xy[:, 0].max()], [-52.5, 52.5])
    np.testing.assert_allclose([xy[:, 1].min(), xy[:, 1].max()], [-34.0, 34.0])


def test_markings_are_four_fold_symmetric():
    # Discretisation makes the sampled point *sets* not bit-identical under mirroring, so check the
    # honest invariant: every mirrored point has a near neighbour in the original (within a sample).
    xy = pitch_line_xy(FieldDimensions(), spacing=1.0)
    assert _max_mirror_gap(xy, [-1.0, 1.0]) < 1.5  # across the halfway line (x → -x)
    assert _max_mirror_gap(xy, [1.0, -1.0]) < 1.5  # across the long axis  (y → -y)


def test_named_spots_present():
    xy = pitch_line_xy(FieldDimensions(length=105.0, width=68.0), spacing=1.0)
    s = _rounded_set(xy)
    assert (0.0, 0.0) in s                       # centre spot
    assert (52.5 - 11.0, 0.0) in s               # right penalty spot (11 m from goal line)
    assert (-(52.5 - 11.0), 0.0) in s            # left penalty spot


def test_spacing_controls_density_and_geometry_is_finite():
    coarse = pitch_line_xy(FieldDimensions(), spacing=2.0)
    fine = pitch_line_xy(FieldDimensions(), spacing=0.5)
    assert fine.shape[0] > coarse.shape[0]
    assert np.isfinite(fine).all()


def test_penalty_arc_pokes_outside_its_box():
    # The "D" must reach toward the centre, past the box's inner line (x = 52.5 - 16.5 = 36).
    xy = pitch_line_xy(FieldDimensions(length=105.0, width=68.0), spacing=0.5)
    box_inner_x = 52.5 - 16.5
    # points on the right half near y≈0 that sit inside the box line are the arc's "D"
    near_axis = xy[(np.abs(xy[:, 1]) < 9.0) & (xy[:, 0] > 0)]
    assert (near_axis[:, 0] < box_inner_x).any()


# --- ribbon geometry: the measured markings given thickness for Cycles (M2-9) ---
def test_ribbons_are_valid_indexed_quads_on_the_lifted_plane():
    # Each marking polyline becomes width-wide quads: 4 verts + 2 triangles per quad. So faces == 2
    # per quad and verts == 4 per quad → faces == verts/2, every index valid, all finite.
    verts, faces = pitch_line_ribbons(FieldDimensions(), plane_z=0.5)
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert verts.shape[0] % 4 == 0 and faces.shape[0] == verts.shape[0] // 2
    assert 0 <= int(faces.min()) and int(faces.max()) < verts.shape[0]
    assert np.isfinite(verts).all()
    # Lifted a hair above the requested plane so the lines don't z-fight the grass.
    np.testing.assert_allclose(verts[:, 2], 0.5 + _LINE_LIFT)


def test_ribbon_quads_wind_face_up():
    # CCW-from-above winding → +Z face normal, so Cycles shades the painted top, not the underside.
    verts, faces = pitch_line_ribbons(FieldDimensions(), plane_z=0.0)
    tri = verts[faces[0]]
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    assert normal[2] > 0.0


def test_ribbons_fit_inside_the_ground_plane():
    # The Cycles ground plane is 140 m; the markings (plus line half-width) must sit well inside it.
    verts, _ = pitch_line_ribbons(FieldDimensions(length=105.0, width=68.0))
    assert np.abs(verts[:, 0]).max() < 70.0
    assert np.abs(verts[:, 1]).max() < 70.0
