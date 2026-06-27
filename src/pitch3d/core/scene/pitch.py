"""Standard pitch line markings as world-space points — the measured ground anchor.

This is the geometry the field calibration (PnLCalib, M1) *anchors to*: the homography's whole
job is to align the broadcast image to these standard FIFA markings on the ``Z = plane_z`` plane.
Emitting the template in world meters is therefore the **measured** environment primitive (M2-0
note (a): the pitch plane bounds every heavier appearance/HMR layer — "a leg can't pass through
the pitch"), as opposed to a *hallucinated* stadium (the generative fallback, R-8, stays gated).

Frame convention matches :mod:`~pitch3d.adapters.render.radar`: origin at the pitch centre, ``X``
along the length (``±length/2``), ``Y`` along the width (``±width/2``), ``Z`` up. The box/circle
metrics are the fixed Laws-of-the-Game values (independent of the outer ``length × width``); only
the touchline/goal-line rectangle scales with :class:`FieldDimensions`. Pure numpy, no adapters.
"""

from __future__ import annotations

import numpy as np

from .units import FieldDimensions

# Fixed Laws-of-the-Game marking metrics (metres), independent of the outer pitch size.
CENTRE_CIRCLE_RADIUS = 9.15
PENALTY_BOX_DEPTH = 16.5
PENALTY_BOX_HALF_WIDTH = 20.16   # 40.32 m total (16.5 + 7.32 goal + 16.5)
GOAL_BOX_DEPTH = 5.5
GOAL_BOX_HALF_WIDTH = 9.16       # 18.32 m total (5.5 + 7.32 + 5.5)
PENALTY_SPOT_DIST = 11.0
PENALTY_ARC_RADIUS = 9.15


def _segment(p0: tuple[float, float], p1: tuple[float, float], spacing: float) -> np.ndarray:
    """Sample a straight line ``p0→p1`` every ~``spacing`` m (endpoints included)."""
    a = np.asarray(p0, dtype=float)
    b = np.asarray(p1, dtype=float)
    n = max(int(np.ceil(float(np.linalg.norm(b - a)) / spacing)), 1) + 1
    return np.linspace(a, b, n)


def _arc(
    center: tuple[float, float], radius: float, a0: float, a1: float, spacing: float
) -> np.ndarray:
    """Sample a circular arc ``[a0, a1]`` (radians) about ``center`` every ~``spacing`` m."""
    n = max(int(np.ceil(radius * abs(a1 - a0) / spacing)), 1) + 1
    ang = np.linspace(a0, a1, n)
    return np.column_stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)])


def _box_lines(goal_x: float, inward: float, depth: float, half_w: float, spacing: float) -> list:
    """Three sides of a goal-line box (the goal-line side is the goal line itself).

    ``goal_x`` is the goal line's X; ``inward`` is ``-1``/``+1`` toward the pitch centre.
    """
    inner_x = goal_x + inward * depth
    return [
        _segment((goal_x, -half_w), (inner_x, -half_w), spacing),  # near side
        _segment((inner_x, -half_w), (inner_x, half_w), spacing),  # inner line
        _segment((inner_x, half_w), (goal_x, half_w), spacing),    # far side
    ]


# FIFA touch/goal-line width is 12 cm; markings sit a hair above the grass to dodge z-fighting.
LINE_WIDTH = 0.12
_LINE_LIFT = 0.01


def _pitch_polylines(
    dimensions: FieldDimensions | None = None, *, spacing: float = 0.5
) -> list[np.ndarray]:
    """The standard markings as a list of ``(n, 2)`` polylines (a one-point array = a spot).

    Markings: the touchline/goal-line rectangle, halfway line, centre circle + spot, both penalty
    boxes, both goal areas, both penalty spots and both penalty arcs (the "D", only the portion
    outside its box). ``spacing`` controls sample density along every line. Kept per-polyline (not
    flattened) so consumers that need *connectivity* — e.g. ribbon geometry — don't bridge a gap
    between two unrelated markings.
    """
    dims = dimensions or FieldDimensions()
    hl, hw = dims.length / 2.0, dims.width / 2.0
    parts: list[np.ndarray] = [
        # outer rectangle
        _segment((-hl, -hw), (hl, -hw), spacing),
        _segment((hl, -hw), (hl, hw), spacing),
        _segment((hl, hw), (-hl, hw), spacing),
        _segment((-hl, hw), (-hl, -hw), spacing),
        # halfway line + centre circle + centre spot
        _segment((0.0, -hw), (0.0, hw), spacing),
        _arc((0.0, 0.0), CENTRE_CIRCLE_RADIUS, 0.0, 2.0 * np.pi, spacing),
        np.zeros((1, 2)),
    ]
    for goal_x, inward in ((hl, -1.0), (-hl, 1.0)):
        parts += _box_lines(goal_x, inward, PENALTY_BOX_DEPTH, PENALTY_BOX_HALF_WIDTH, spacing)
        parts += _box_lines(goal_x, inward, GOAL_BOX_DEPTH, GOAL_BOX_HALF_WIDTH, spacing)
        spot_x = goal_x + inward * PENALTY_SPOT_DIST
        parts.append(np.array([[spot_x, 0.0]]))  # penalty spot
        box_inner_x = goal_x + inward * PENALTY_BOX_DEPTH
        circle = _arc((spot_x, 0.0), PENALTY_ARC_RADIUS, 0.0, 2.0 * np.pi, spacing)
        # keep only the "D" — the arc portion poking out past the box's inner line toward centre
        keep = circle[:, 0] <= box_inner_x if inward < 0 else circle[:, 0] >= box_inner_x
        parts.append(circle[keep])
    return parts


def pitch_line_xy(dimensions: FieldDimensions | None = None, *, spacing: float = 0.5) -> np.ndarray:
    """Return ``(N, 2)`` world-XY points sampling the full standard pitch markings.

    Markings: the touchline/goal-line rectangle, halfway line, centre circle + spot, both penalty
    boxes, both goal areas, both penalty spots and both penalty arcs (the "D", only the portion
    outside its box). ``spacing`` controls sample density along every line.
    """
    return np.vstack(_pitch_polylines(dimensions, spacing=spacing))


def pitch_line_world_points(
    dimensions: FieldDimensions | None = None, *, plane_z: float = 0.0, spacing: float = 0.5
) -> np.ndarray:
    """Return ``(N, 3)`` world points for the pitch markings on the ``Z = plane_z`` plane."""
    xy = pitch_line_xy(dimensions, spacing=spacing)
    return np.column_stack([xy, np.full(xy.shape[0], float(plane_z))])


def pitch_line_ribbons(
    dimensions: FieldDimensions | None = None,
    *,
    plane_z: float = 0.0,
    spacing: float = 0.5,
    width: float = LINE_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Flat white-line geometry tracing the measured markings: ``(verts (M, 3), faces (F, 3))``.

    Each measured polyline becomes a strip of ``width``-wide quads (two triangles per segment) on
    the ``Z = plane_z`` plane, lifted ``_LINE_LIFT`` m so it doesn't z-fight the grass; a one-point
    marking (penalty / centre spot) becomes a single small square. Quads are wound CCW seen from
    above (face normal +Z). This is the *measured* pitch geometry the calibration anchors to, just
    given thickness so Cycles can shade it — nothing fabricated (M2-9).
    """
    half = float(width) / 2.0
    z = float(plane_z) + _LINE_LIFT
    verts: list[list[float]] = []
    faces: list[list[int]] = []

    def _quad(corners: list[tuple[float, float]]) -> None:
        base = len(verts)
        verts.extend([cx, cy, z] for cx, cy in corners)
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])

    for poly in _pitch_polylines(dimensions, spacing=spacing):
        pts = np.asarray(poly, dtype=float)
        if pts.shape[0] == 1:
            x, y = float(pts[0, 0]), float(pts[0, 1])
            _quad([(x - half, y - half), (x + half, y - half), (x + half, y + half),
                   (x - half, y + half)])
            continue
        for a, b in zip(pts[:-1], pts[1:], strict=False):
            d = b - a
            length = float(np.hypot(d[0], d[1]))
            if length < 1e-9:
                continue
            nx, ny = -d[1] / length * half, d[0] / length * half
            _quad([(a[0] - nx, a[1] - ny), (b[0] - nx, b[1] - ny),
                   (b[0] + nx, b[1] + ny), (a[0] + nx, a[1] + ny)])

    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)
