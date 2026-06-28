"""Parametric stadium bowl as world-space geometry — the scaffold the *measured* backdrop wraps.

Unlike :mod:`~pitch3d.core.scene.pitch` (the measured Laws-of-the-Game markings the calibration
anchors to) this is a plausible **seating bowl** around the pitch: a rounded-rectangle footprint
just outside the touchlines, swept upward and outward through ``rows`` raked tiers. It carries no
appearance on its own — the backdrop builder (M2 stadium step) projects the broadcast clip onto
these vertices where the camera saw them and copy-fills the rest, so the bowl is the geometry half
of the hybrid (procedural shell + measured pixels). Pure numpy, no adapters.

Frame convention matches :mod:`~pitch3d.core.scene.pitch`: origin at the pitch centre, ``X`` along
the length, ``Y`` along the width, ``Z`` up. Every vertex carries a ``(angle_frac, height_frac)``
parametrisation: ``angle_frac`` ∈ ``[0, 1)`` walks the bowl loop once (so the near/far stands are a
half-turn apart — what hole-fill exploits), ``height_frac`` ∈ ``[0, 1]`` runs base→top of the rake.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .units import FieldDimensions


def _rounded_rect_loop(
    hx: float, hy: float, r: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """A rounded rectangle (half-extents ``hx, hy``, corner radius ``r``) sampled by arc length.

    Returns ``(pts (n, 2), normals (n, 2))`` walking the loop counter-clockwise from the +X axis;
    ``normals`` is the outward (away-from-centre) horizontal unit normal at each point — radial on
    the corner arcs, axis-aligned on the straight edges.
    """
    r = float(min(r, hx, hy))
    sx, sy = hx - r, hy - r  # corner-circle centres at (±sx, ±sy)

    # Each perimeter piece is (arc_length, sample(d)->(point, outward_normal)) so the whole loop is
    # one uniformly-typed list we can walk by global arc length — CCW from the +X edge, alternating
    # straight edge then corner arc, ×4.
    Sample = Callable[[float], tuple[tuple[float, float], tuple[float, float]]]

    def line(ax: float, ay: float, bx: float, by: float) -> tuple[float, Sample]:
        length = float(np.hypot(bx - ax, by - ay))
        nx, ny = (by - ay), -(bx - ax)  # right-hand outward normal for a CCW loop
        nlen = float(np.hypot(nx, ny)) or 1.0

        def f(d: float) -> tuple[tuple[float, float], tuple[float, float]]:
            t = d / length if length > 0 else 0.0
            return (ax + (bx - ax) * t, ay + (by - ay) * t), (nx / nlen, ny / nlen)

        return length, f

    def arc(cx: float, cy: float, a0: float) -> tuple[float, Sample]:
        def f(d: float) -> tuple[tuple[float, float], tuple[float, float]]:
            a = a0 + d / r  # CCW along the quarter arc; radial normal points outward
            return (cx + r * np.cos(a), cy + r * np.sin(a)), (np.cos(a), np.sin(a))

        return 0.5 * np.pi * r, f

    pieces: list[tuple[float, Sample]] = [
        line(hx, -sy, hx, sy),        # +X edge
        arc(sx, sy, 0.0),             # +X+Y corner
        line(sx, hy, -sx, hy),        # +Y edge
        arc(-sx, sy, 0.5 * np.pi),    # -X+Y corner
        line(-hx, sy, -hx, -sy),      # -X edge
        arc(-sx, -sy, np.pi),         # -X-Y corner
        line(-sx, -hy, sx, -hy),      # -Y edge
        arc(sx, -sy, 1.5 * np.pi),    # +X-Y corner
    ]
    total = sum(length for length, _ in pieces)
    s = np.linspace(0.0, total, n, endpoint=False)
    pts = np.zeros((n, 2))
    nrm = np.zeros((n, 2))
    for i, dist in enumerate(s):
        d = float(dist)
        for length, sample in pieces:
            if d <= length or length == 0.0:
                (px, py), (nx, ny) = sample(d)
                pts[i] = (px, py)
                nrm[i] = (nx, ny)
                break
            d -= length
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return pts, nrm


def stadium_bowl_geometry(
    dimensions: FieldDimensions | None = None,
    *,
    apron: float = 7.0,
    n_around: int = 240,
    rows: int = 20,
    rise: float = 0.80,
    run: float = 0.90,
    corner_radius: float = 16.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A raked seating bowl around the pitch: ``(verts (M, 3), faces (F, 3), param (M, 2))``.

    The footprint is a rounded rectangle ``apron`` metres outside the touch/goal lines; from each
    footprint point the surface steps ``rows`` times, gaining ``rise`` m up and ``run`` m outward
    per step (the seating rake), so the bowl is ``rows*rise`` m tall and ``rows*run`` m deep. Faces
    wind so their normals point **inward** (toward the pitch) — the side the broadcast camera sees
    and the render shows. ``param[:, 0]`` is the loop fraction, ``param[:, 1]`` the base→top one.
    """
    dims = dimensions or FieldDimensions()
    hx = dims.length / 2.0 + apron
    hy = dims.width / 2.0 + apron
    loop, nrm = _rounded_rect_loop(hx, hy, corner_radius, n_around)

    nrows = rows + 1
    verts = np.zeros((n_around * nrows, 3))
    param = np.zeros((n_around * nrows, 2))
    for i in range(n_around):
        for r in range(nrows):
            out = r * run
            verts[i * nrows + r] = (loop[i, 0] + nrm[i, 0] * out,
                                    loop[i, 1] + nrm[i, 1] * out,
                                    r * rise)
            param[i * nrows + r] = (i / n_around, r / rows)

    faces: list[list[int]] = []
    for i in range(n_around):
        j = (i + 1) % n_around
        for r in range(rows):
            a = i * nrows + r
            b = j * nrows + r
            c = j * nrows + r + 1
            d = a + 1
            # wound CW seen from outside ⇒ normal points inward (toward the pitch)
            faces.append([a, c, b])
            faces.append([a, d, c])
    return verts, np.asarray(faces, dtype=int), param


def fill_holes_by_copy(
    verts: np.ndarray, colors: np.ndarray, covered: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fill the colour of bowl vertices the camera never saw by copying from ones it did.

    A broadcast camera sits above one sideline, so the *opposite* stand is seen but the camera's own
    near stand (and some end gaps) are holes. Stadiums are near-symmetric, so for each uncovered
    vertex we copy the covered vertex closest to its **long-axis mirror** ``(x, -y, z)`` — the near
    stand inherits the far stand. Where the mirror is itself uncovered (e.g. an end-strip gap) we
    fall back to the covered vertex nearest the point itself, picking whichever source is closer.

    Returns ``(filled (M, 3), source_idx (M,))``; ``source_idx[i]`` is the vertex each colour came
    from (``i`` itself for already-covered vertices) — useful for diagnostics. No-op if all/none
    covered.
    """
    colors = np.asarray(colors, dtype=float)
    covered = np.asarray(covered, dtype=bool)
    out = colors.copy()
    src = np.arange(verts.shape[0])
    cov_idx = np.flatnonzero(covered)
    unc_idx = np.flatnonzero(~covered)
    if cov_idx.size == 0 or unc_idx.size == 0:
        return out, src
    cov_v = verts[cov_idx]
    mirror = verts[unc_idx] * np.array([1.0, -1.0, 1.0])
    direct = verts[unc_idx]
    # Nearest covered vertex to the mirror query and to the point itself; keep the closer source.
    for start in range(0, unc_idx.size, 1024):
        sl = slice(start, start + 1024)
        dm = np.linalg.norm(mirror[sl, None, :] - cov_v[None, :, :], axis=2)
        dd = np.linalg.norm(direct[sl, None, :] - cov_v[None, :, :], axis=2)
        am, ad = dm.argmin(axis=1), dd.argmin(axis=1)
        pick_mirror = dm[np.arange(am.size), am] <= dd[np.arange(ad.size), ad]
        chosen = np.where(pick_mirror, cov_idx[am], cov_idx[ad])
        out[unc_idx[sl]] = colors[chosen]
        src[unc_idx[sl]] = chosen
    return out, src
