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
    GOAL_FRAME_HEIGHT,
    GOAL_INNER_WIDTH,
    GOAL_POST_THICK,
    goal_frame_geometry,
    pitch_line_ribbons,
    pitch_polylines,
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


def test_consecutive_samples_never_jump_a_gap():
    """Neighbours in a polyline are drawn joined, so a big step is a line that isn't painted.

    Building the "D" by boolean-filtering a full circle put its two kept angular blocks side by
    side in the array, and the seam drew as a 14 m chord across the penalty area — a phantom
    marking 0.37 m from the box's front line, which is why the arc looked like it lagged the box.
    """
    for spacing in (0.5, 2.0):
        for poly in pitch_polylines(spacing=spacing):
            if len(poly) < 2:
                continue
            step = np.linalg.norm(np.diff(poly, axis=0), axis=1)
            assert step.max() <= spacing + 1e-6


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


# --- goal frames: posts + crossbar on each goal line (#205) ---
def test_goal_frame_is_two_valid_boxed_goals():
    # Two goals × (2 posts + 1 crossbar) = 6 boxes; each box is 8 verts + 12 triangles.
    verts, faces = goal_frame_geometry(FieldDimensions())
    assert verts.shape == (6 * 8, 3)
    assert faces.shape == (6 * 12, 3)
    assert 0 <= int(faces.min()) and int(faces.max()) < verts.shape[0]
    assert np.isfinite(verts).all()


def test_goals_stand_on_both_goal_lines_at_laws_dimensions():
    dims = FieldDimensions(length=105.0, width=68.0)
    verts, _ = goal_frame_geometry(dims, plane_z=0.0)
    half_post = GOAL_POST_THICK / 2.0
    # A goal frame straddles each goal line X = ±52.5 (±half the post thickness).
    np.testing.assert_allclose(verts[:, 0].max(), 52.5 + half_post)
    np.testing.assert_allclose(verts[:, 0].min(), -(52.5 + half_post))
    # Mouth 7.32 m (posts at ±3.66), frame height 2.44 m (crossbar adds its thickness on top).
    np.testing.assert_allclose(verts[:, 1].max(), GOAL_INNER_WIDTH / 2.0 + half_post)
    np.testing.assert_allclose(verts[:, 2].min(), 0.0)
    np.testing.assert_allclose(verts[:, 2].max(), GOAL_FRAME_HEIGHT + GOAL_POST_THICK)


def test_goals_sit_on_the_requested_plane():
    verts, _ = goal_frame_geometry(FieldDimensions(), plane_z=0.5)
    np.testing.assert_allclose(verts[:, 2].min(), 0.5)
