"""Does the frame actually contain the line the overlay is drawing?

Every projected point is checked against the pixels: is there a painted line under it?
Three answers, and the third is the useful one:

``ok``       a painted line is within tolerance — this part of the calibration is confirmed
``unknown``  the point is off the playing surface (crowd, boards) — no evidence either way
``off``      clear turf, no painted line anywhere near — the overlay is claiming a marking
             that is not there

Marked, not erased (R-6): the extrapolated part is still drawn, just not as if it were
measured. On the target clip every marking that is genuinely in shot lands within 1-4 px of
its paint's centreline; the one thing drawn on empty turf is the sliver of centre circle
clipping into the far corner, 135-288 px from anything painted and some 40 m past the
nearest confirmed marking. So the verdict has to be per marking — one frame-wide number
would either call an honest overlay broken or hide the one part that is invented.

What counts as paint is the whole ballgame, and getting it wrong is silent in both
directions. Thresholding "white on grass" and cutting out thick blobs to reject players
also deletes the far touchline, which runs within a few px of the advertising boards: with
the one marking that pins down the far half of the frame gone, the far field measured 37 px
out when it was 3. And a fixed 25..95 hue band for grass admits a stand full of yellow
shirts, which then reads as turf and swallows the crowd into the playing surface.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

#: Paint is a bright ridge with turf on BOTH sides, and that is what separates it from
#: everything else white in a stadium: an advertising board is flat inside and has board,
#: not turf, on one side of its edge; a player's shorts have shirt or skin beside them.
#: The scales bracket the painted width from the far touchline (~2 px) to the goal area
#: (~14 px) — anything wider than twice the largest scale reads as a blob, not a line.
_RIDGE_SCALES = (2, 4, 7)
_RIDGE_CONTRAST = 16
_RIDGE_MIN_V = 95

#: Turf is one narrow hue per broadcast, so it is measured from the frame instead of being
#: hard-coded: a fixed 25..95 band also admits a stand full of yellow shirts, which then
#: pass the "turf on both sides" test and paint the crowd with phantom markings.
_HUE_HALFWIDTH = 7

#: Default confirmation radius in source pixels. The markings are 12 cm wide (a few px in
#: the near field) and the model draws their centreline, so a few px of slack is the line
#: itself, not error.
DEFAULT_TOLERANCE_PX = 6.0

#: Gaps this many samples long are bridged before judging a run: a defender standing on the
#: touchline breaks the paint, and that is occlusion, not a calibration error.
_GAP_BRIDGE = 6


def _turf(hsv: np.ndarray) -> np.ndarray:
    """Turf pixels, keyed to this frame's own dominant hue."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (s > 80) & (v > 80)
    hist = np.bincount(h[lit].ravel(), minlength=180).astype(np.float32)
    peak = int(np.argmax(cv2.GaussianBlur(hist.reshape(-1, 1), (1, 5), 0)))
    return (np.abs(h.astype(np.int16) - peak) <= _HUE_HALFWIDTH) & (s > 70) & (v > 70)


def _surface(turf: np.ndarray) -> np.ndarray:
    """The playing surface as a filled region — paint and players are on it, the crowd isn't.

    This has to be a region rather than a colour test: a point landing exactly on a painted
    line is not turf-coloured, and must still count as having evidence.
    """
    filled = cv2.morphologyEx(turf.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
    if count < 2:
        return filled
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return cv2.morphologyEx(
        (labels == biggest).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((61, 61), np.uint8)
    )


def _shift(a: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    out = np.full_like(a, fill)
    rows, cols = a.shape
    out[max(0, -dy) : rows + min(0, -dy), max(0, -dx) : cols + min(0, -dx)] = a[
        max(0, dy) : rows + min(0, dy), max(0, dx) : cols + min(0, dx)
    ]
    return out


def _masks(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(distance-to-nearest-painted-line, playing-surface)`` for one BGR frame."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    turf = _turf(hsv)
    surface = _surface(turf)
    val = hsv[..., 2].astype(np.int16)

    ridge = np.full(val.shape, -1000, np.int16)
    for d in _RIDGE_SCALES:
        for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
            side = np.minimum(
                val - _shift(val, d * dy, d * dx, 255), val - _shift(val, -d * dy, -d * dx, 255)
            ).astype(np.int16)
            both = _shift(turf, d * dy, d * dx, False) & _shift(turf, -d * dy, -d * dx, False)
            side[~both] = -1000
            np.maximum(ridge, side, out=ridge)

    lines = ((ridge >= _RIDGE_CONTRAST) & (val >= _RIDGE_MIN_V) & (surface > 0)).astype(np.uint8)

    # Distance is measured to the paint's centreline, not to its nearest pixel. Paint near
    # the goal is 8-10 px wide, so "nearest painted pixel" is satisfied anywhere inside the
    # band: an overlay visibly riding the band's edge scores a perfect 0.0 px, which is how
    # the penalty arc measured flawless while it plainly sat inside its own marking.
    inner = cv2.distanceTransform(lines, cv2.DIST_L2, 5)
    spine = (inner >= cv2.dilate(inner, np.ones((3, 3), np.uint8)) - 1e-3) & (inner > 0)
    dist = cv2.distanceTransform((~spine).astype(np.uint8), cv2.DIST_L2, 5)
    return dist, surface


@lru_cache(maxsize=32)
def _evidence_cached(video_path: str, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    from .video import read_frame

    return _masks(read_frame(video_path, frame_index))


def _bridge(flags: np.ndarray, gap: int = _GAP_BRIDGE) -> np.ndarray:
    """Fill short False gaps that are *enclosed* by confirmed samples on both sides.

    Enclosure is the whole point: a defender standing on the touchline breaks the paint
    between two confirmed stretches, and that is occlusion. A lone confirmed sample with
    nothing beyond it confirms only itself — it must not bless its neighbours.
    """
    if flags.size == 0 or gap < 1:
        return flags
    out = flags.copy()
    idx = np.flatnonzero(flags)
    for a, b in zip(idx[:-1], idx[1:], strict=True):
        if 1 < b - a <= gap + 1:
            out[a + 1 : b] = True
    return out


def classify(
    uv: np.ndarray, video_path: str, frame_index: int, tolerance: float = DEFAULT_TOLERANCE_PX
) -> tuple[np.ndarray, np.ndarray]:
    """Label each projected point ``ok`` / ``unknown`` / ``off``; also return its error.

    ``uv`` is ``(N, 2)`` source pixels. Returns ``(labels, dist)`` where ``dist`` is NaN for
    points with no evidence (off-frame or off the surface) so callers can average honestly.
    """
    dist_map, surface = _evidence_cached(video_path, frame_index)
    hgt, wid = surface.shape
    n = len(uv)
    labels = np.full(n, "unknown", dtype="<U7")
    dist = np.full(n, np.nan)
    if n == 0:
        return labels, dist

    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    inside = (u >= 0) & (u < wid) & (v >= 0) & (v < hgt)
    if not inside.any():
        return labels, dist

    ui, vi = u[inside], v[inside]
    on_pitch = surface[vi, ui] > 0
    d = dist_map[vi, ui]
    sub_dist = np.where(on_pitch, d, np.nan)
    near = _bridge(on_pitch & (d <= tolerance))
    sub_labels = np.where(~on_pitch, "unknown", np.where(near, "ok", "off"))

    labels[inside] = sub_labels
    dist[inside] = sub_dist
    return labels, dist
