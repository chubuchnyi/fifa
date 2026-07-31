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

# The goal FRAME (the structure) — distinct from the goal AREA box above. Laws of the Game: a
# 7.32 m inner mouth, 2.44 m high; posts and crossbar are square in section here (~0.12 m). This is
# measured render geometry the calibrated pitch plane carries (#205), not a hallucinated stadium.
GOAL_INNER_WIDTH = 7.32
GOAL_FRAME_HEIGHT = 2.44
GOAL_POST_THICK = 0.12

# Laws of the Game: a corner flagpost is not less than 1.5 m high. Together with the goal frame it
# is the only fixed thing on a pitch that stands UP off the plane, which is what makes the pair
# worth drawing: a ground homography maps the lawn exactly and says nothing whatever about height,
# so nothing on the plane can tell a right focal from a wrong one.
CORNER_FLAG_HEIGHT = 1.5


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


def pitch_plane_line_segments(
    dimensions: FieldDimensions | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World segments of the 17 straight, ``Z = 0`` pitch lines → ``{class: (A(2,), B(2,))}``.

    Class names are SoccerNet's, which PnLCalib's line head also emits, so a detection keyed by
    name looks up its world line here directly. Only *straight* lawn-plane lines are returned —
    exactly the ones that constrain an image→world homography (circles are curved; goal frames sit
    at ``Z ≠ 0``). The box metrics are the fixed Laws-of-the-Game values; only the outer rectangle
    scales with ``dimensions``.

    The coordinates in the table below are written in SoccerNet's own top-down **template** frame so
    they stay diffable against it, and ``seg`` turns each one into our world on the way out.
    """
    dims = dimensions or FieldDimensions()
    hl, hw = dims.length / 2.0, dims.width / 2.0
    pa_x, ga_x = -hl + PENALTY_BOX_DEPTH, -hl + GOAL_BOX_DEPTH
    pa_hw, ga_hw = PENALTY_BOX_HALF_WIDTH, GOAL_BOX_HALF_WIDTH

    def seg(ax: float, ay: float, bx: float, by: float) -> tuple[np.ndarray, np.ndarray]:
        """Template ``(x, y)`` → world: ``Y`` negated (#118, ``calibration.TEMPLATE_TO_WORLD``).

        "top" and "bottom" are the top and bottom of SoccerNet's template *image*, whose ``Y`` runs
        down it; our world is Z-up right-handed, so the template's top side line is our ``+Y`` one.
        The names and the geometry have to cross that boundary together — split them and the
        line-residual gate throws out its own evidence for disagreeing with the keypoints.
        """
        return np.array([ax, -ay], dtype=float), np.array([bx, -by], dtype=float)

    return {
        "Side line top": seg(-hl, -hw, hl, -hw),
        "Side line bottom": seg(-hl, hw, hl, hw),
        "Side line left": seg(-hl, -hw, -hl, hw),
        "Side line right": seg(hl, -hw, hl, hw),
        "Middle line": seg(0.0, -hw, 0.0, hw),
        "Big rect. left top": seg(-hl, -pa_hw, pa_x, -pa_hw),
        "Big rect. left bottom": seg(-hl, pa_hw, pa_x, pa_hw),
        "Big rect. left main": seg(pa_x, -pa_hw, pa_x, pa_hw),
        "Big rect. right top": seg(-pa_x, -pa_hw, hl, -pa_hw),
        "Big rect. right bottom": seg(-pa_x, pa_hw, hl, pa_hw),
        "Big rect. right main": seg(-pa_x, -pa_hw, -pa_x, pa_hw),
        "Small rect. left top": seg(-hl, -ga_hw, ga_x, -ga_hw),
        "Small rect. left bottom": seg(-hl, ga_hw, ga_x, ga_hw),
        "Small rect. left main": seg(ga_x, -ga_hw, ga_x, ga_hw),
        "Small rect. right top": seg(-ga_x, -ga_hw, hl, -ga_hw),
        "Small rect. right bottom": seg(-ga_x, ga_hw, hl, ga_hw),
        "Small rect. right main": seg(-ga_x, -ga_hw, -ga_x, ga_hw),
    }


def world_line_from_segment(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Homogeneous line ``(a, b, c)`` through two world points, scaled so ``a² + b² == 1``.

    With that scaling ``|a·x + b·y + c|`` is the signed **metre** distance from ``(x, y)`` to the
    line, which is what makes a point-on-line residual comparable to a point-to-point one.
    """
    p, q = np.asarray(a, dtype=float).reshape(2), np.asarray(b, dtype=float).reshape(2)
    coeffs = np.array([p[1] - q[1], q[0] - p[0], p[0] * q[1] - q[0] * p[1]], dtype=float)
    norm = float(np.hypot(coeffs[0], coeffs[1]))
    if norm < 1e-12:
        raise ValueError(f"degenerate segment: {p.tolist()} and {q.tolist()} coincide")
    return coeffs / norm


def pitch_line_coefficients(dimensions: FieldDimensions | None = None) -> dict[str, np.ndarray]:
    """The straight pitch lines as normalised world coefficients → ``{class: (a, b, c)}``."""
    return {
        name: world_line_from_segment(a, b)
        for name, (a, b) in pitch_plane_line_segments(dimensions).items()
    }


# FIFA touch/goal-line width is 12 cm; markings sit a hair above the grass to dodge z-fighting.
LINE_WIDTH = 0.12
_LINE_LIFT = 0.01


def pitch_polylines(
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
        # The "D" is only the arc poking out past the box's inner line, so it must be sampled over
        # that angular range. Filtering a full circle instead splits it into two blocks that array
        # indexing then concatenates, and the seam is drawn as a 14 m chord straight across the
        # penalty area, 0.37 m from — and parallel to — the box's front line.
        box_inner_x = goal_x + inward * PENALTY_BOX_DEPTH
        half = float(np.arccos(abs(box_inner_x - spot_x) / PENALTY_ARC_RADIUS))
        base = 0.0 if inward > 0 else np.pi
        parts.append(_arc((spot_x, 0.0), PENALTY_ARC_RADIUS, base - half, base + half, spacing))
    return parts


def pitch_upright_polylines(
    dimensions: FieldDimensions | None = None, *, plane_z: float = 0.0
) -> list[np.ndarray]:
    """The fixed structures that stand UP off the pitch, as ``(n, 3)`` world polylines.

    Both goal frames traced base → crossbar → base, and the four corner flagposts. Everything
    else this module emits lies on ``Z = plane_z``, where a homography is exact; these are the
    only measured points that leave the plane, so they are the only ones whose drawn position
    depends on the camera's focal — which is exactly what makes them the instrument for checking
    it. Post thickness is ignored: this is a wireframe to align against, not render geometry
    (:func:`goal_frame_geometry` is the solid version).
    """
    dims = dimensions or FieldDimensions()
    hl, hw = dims.length / 2.0, dims.width / 2.0
    half = GOAL_INNER_WIDTH / 2.0
    z0 = float(plane_z)
    parts = [
        np.array([[gx, -half, z0], [gx, -half, z0 + GOAL_FRAME_HEIGHT],
                  [gx, half, z0 + GOAL_FRAME_HEIGHT], [gx, half, z0]])
        for gx in (-hl, hl)
    ]
    parts += [
        np.array([[sx * hl, sy * hw, z0], [sx * hl, sy * hw, z0 + CORNER_FLAG_HEIGHT]])
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]
    return parts


def pitch_line_xy(dimensions: FieldDimensions | None = None, *, spacing: float = 0.5) -> np.ndarray:
    """Return ``(N, 2)`` world-XY points sampling the full standard pitch markings.

    Markings: the touchline/goal-line rectangle, halfway line, centre circle + spot, both penalty
    boxes, both goal areas, both penalty spots and both penalty arcs (the "D", only the portion
    outside its box). ``spacing`` controls sample density along every line.
    """
    return np.vstack(pitch_polylines(dimensions, spacing=spacing))


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

    for poly in pitch_polylines(dimensions, spacing=spacing):
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


def _box(
    lo: tuple[float, float, float], hi: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box ``lo→hi`` as ``(8 verts, 12 triangles)``, every face wound outward."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array(
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
         [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        dtype=float,
    )
    f = np.array(
        [[0, 2, 1], [0, 3, 2],   # bottom (-Z)
         [4, 5, 6], [4, 6, 7],   # top (+Z)
         [0, 1, 5], [0, 5, 4],   # -Y
         [2, 3, 7], [2, 7, 6],   # +Y
         [1, 2, 6], [1, 6, 5],   # +X
         [3, 0, 4], [3, 4, 7]],  # -X
        dtype=int,
    )
    return v, f


def goal_frame_geometry(
    dimensions: FieldDimensions | None = None, *, plane_z: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Both goal frames (two posts + a crossbar each) as ``(verts (N, 3), faces (F, 3))`` in metres.

    A goal stands on each goal line (``X = ±length/2``), centred on ``Y = 0``: two
    ``GOAL_FRAME_HEIGHT``-tall posts at ``Y = ±GOAL_INNER_WIDTH/2`` joined by a crossbar at their
    tops, each square in section (``GOAL_POST_THICK``). Nets and goal depth are deliberately omitted
    (left to the appearance stage). Measured Laws geometry on the calibrated pitch plane (#205).
    """
    dims = dimensions or FieldDimensions()
    hl = dims.length / 2.0
    half = GOAL_INNER_WIDTH / 2.0
    t = GOAL_POST_THICK
    z0 = float(plane_z)
    top = z0 + GOAL_FRAME_HEIGHT
    vlist: list[np.ndarray] = []
    flist: list[np.ndarray] = []
    base = 0
    for goal_x in (hl, -hl):
        members = [
            ((goal_x - t / 2, -half - t / 2, z0), (goal_x + t / 2, -half + t / 2, top)),  # post -Y
            ((goal_x - t / 2, half - t / 2, z0), (goal_x + t / 2, half + t / 2, top)),     # post +Y
            # crossbar joining the post tops
            ((goal_x - t / 2, -half - t / 2, top), (goal_x + t / 2, half + t / 2, top + t)),
        ]
        for lo, hi in members:
            v, f = _box(lo, hi)
            vlist.append(v)
            flist.append(f + base)
            base += v.shape[0]
    return np.vstack(vlist), np.vstack(flist)
