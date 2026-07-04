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


def adboard_ring_geometry(
    dimensions: FieldDimensions | None = None,
    *,
    offset: float = 5.0,
    height: float = 1.0,
    gap: float = 2.2,
    n_around: int = 240,
    corner_radius: float = 14.0,
    board_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    gap_color: tuple[float, float, float] = (0.02, 0.02, 0.03),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The broadcast pitch-perimeter furniture: LED ad-board ring + dark walkway band behind it.

    Measured gap in the target clip (2026-07-03): grass runs straight into the crowd wall in our
    render, while a real night broadcast reads grass → bright white LED boards → dark walkway →
    crowd. Two vertical quad strips on a rounded-rectangle loop ``offset`` metres outside the
    lines: boards ``z ∈ [0, height]`` in ``board_color``, the walkway band
    ``z ∈ [height, height+gap]`` in ``gap_color``. Vertices are NOT shared across the two strips
    so per-vertex colours stay crisp at the boundary. Faces wind to face the pitch, matching
    :func:`stadium_bowl_geometry`. Returns ``(verts (M, 3), faces (F, 3), colors (M, 3))``.
    """
    dims = dimensions or FieldDimensions()
    hx = dims.length / 2.0 + offset
    hy = dims.width / 2.0 + offset
    loop, _ = _rounded_rect_loop(hx, hy, corner_radius, n_around)

    def band(z0: float, z1: float, base: int, color) -> tuple[list, list, list]:
        verts, faces, colors = [], [], []
        for i in range(n_around):
            verts.append((loop[i, 0], loop[i, 1], z0))
            verts.append((loop[i, 0], loop[i, 1], z1))
            colors.extend([color, color])
        for i in range(n_around):
            j = (i + 1) % n_around
            a, d = base + 2 * i, base + 2 * i + 1
            b, c = base + 2 * j, base + 2 * j + 1
            # same winding as the bowl: CW seen from outside ⇒ normal faces the pitch
            faces.append([a, c, b])
            faces.append([a, d, c])
        return verts, faces, colors

    bv, bf, bc = band(0.0, height, 0, board_color)
    wv, wf, wc = band(height, height + gap, len(bv), gap_color)
    return (
        np.asarray(bv + wv, dtype=float),
        np.asarray(bf + wf, dtype=int),
        np.asarray(bc + wc, dtype=float),
    )


def bowl_tile_loop_uvs(
    faces: np.ndarray, param: np.ndarray, *, repeat_around: float, repeat_up: float
) -> np.ndarray:
    """Per-loop UVs that tile a crowd patch over the bowl: ``(3F, 2)`` float, a row per face-corner.

    The mosaic backdrop wears a small measured crowd image repeated over the stands rather than one
    stretched pixel per vertex, so the bowl needs UVs. The ``(angle_frac, height_frac)`` param is
    already a clean unwrap — ``u`` walks the loop, ``v`` runs base to top — we just scale it by how
    many tile copies to lay down: ``u`` in ``[0, repeat_around]``, ``v`` in ``[0, repeat_up]``.

    UVs are emitted **per loop** (in the exact corner order of ``faces``, so a renderer can
    ``foreach_set`` them directly) to fix the wrap seam: the one face-column bridging ``angle_frac``
    1 back to 0 would otherwise run the texture backwards. We detect those faces (corner columns
    span >1 step) and lift the low column by a full turn so ``u`` stays monotonic across them.
    Tile counts (``repeat_*``) derive back to the integer grid from ``param`` alone, so this never
    drifts from :func:`stadium_bowl_geometry`.
    """
    faces = np.asarray(faces, dtype=int)
    param = np.asarray(param, dtype=float)
    n_around = int(np.unique(param[:, 0]).size)  # angle_frac = i / n_around: one value per column
    rows = int(np.unique(param[:, 1]).size) - 1  # height_frac = r / rows: nrows = rows + 1 values
    cols = np.rint(param[:, 0] * n_around).astype(int)  # per-vertex column index 0..n_around-1
    rws = np.rint(param[:, 1] * rows).astype(int)        # per-vertex row index 0..rows

    fcols = cols[faces]  # (F, 3) corner columns
    frws = rws[faces]    # (F, 3) corner rows
    mins = fcols.min(axis=1, keepdims=True)
    wrap = (fcols.max(axis=1, keepdims=True) - mins) > 1  # a face straddling the angle wrap seam
    eff = fcols.astype(float)
    eff[(fcols == mins) & wrap] += n_around  # lift the low column a full turn so u stays monotonic

    u = eff / n_around * repeat_around
    v = frws.astype(float) / rows * repeat_up
    return np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32)


def adboard_loop_uvs(n_around: int, *, repeat_around: float) -> np.ndarray:
    """Per-loop UVs wrapping a measured LED strip around the ad-board ring: ``(12·n, 2)`` float.

    Row order mirrors :func:`adboard_ring_geometry` exactly (board-band faces then walkway faces,
    two triangles per segment, same corner order), so a renderer can ``foreach_set`` directly.
    ``u`` walks the loop in ``[repeat_around, 0]`` — *against* ring vertex order, because the ring
    runs toward −x along the far touchline while a strip cut left→right from the upright clip view
    runs toward +x there (measured 2026-07-04: forward u rendered every board mirror-image). One
    global reversal orients text for any camera inside the ring; ``v`` spans the board height 0→1.
    The wrap segment keeps ``u`` monotonic by running down to 0 instead of folding back to a full
    turn — REPEAT extension samples it like any interior segment.
    Walkway faces get a constant mid-texture UV: their near-black vertex tint owns the colour, the
    sampled texel just cancels out of the product.
    """
    us = (n_around - np.arange(n_around + 1, dtype=np.float32)) / n_around * repeat_around
    quads = []
    for i in range(n_around):
        ui, uj = us[i], us[i + 1]
        # tri [a, c, b] = (bottom-i, top-j, bottom-j); tri [a, d, c] = (bottom-i, top-i, top-j)
        quads.append([(ui, 0.0), (uj, 1.0), (uj, 0.0), (ui, 0.0), (ui, 1.0), (uj, 1.0)])
    board = np.asarray(quads, dtype=np.float32).reshape(-1, 2)
    walkway = np.full_like(board, 0.5)
    return np.concatenate([board, walkway], axis=0)


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
