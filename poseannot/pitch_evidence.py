"""Does the frame actually contain the line the overlay is drawing?

The solved homography is fit from keypoints that cluster in one region of the pitch, so it
is accurate where the evidence is and extrapolates everywhere else — measured on the target
clip: 2.0 px within 25 m of the near goal line, 37 px beyond it, and a centre circle drawn
onto empty grass. An overlay that renders both with the same confident green line invites
exactly the frustration of trying to hand-align something that was never anchored.

So every projected point is checked against the pixels: is there a painted line under it?
Three answers, and the third is the useful one:

``ok``       a painted line is within tolerance — this part of the calibration is confirmed
``unknown``  the point is off the grass (crowd, boards) — no evidence either way, say so
``off``      clear grass, no painted line anywhere near — the overlay is claiming a marking
             that is not there

Marked, not erased (R-6): the extrapolated part is still drawn, just not as if it were
measured.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

#: A painted line is a few px wide; a player is tens. An opening with this kernel keeps only
#: the thick blobs, which are then cut out — otherwise white socks and shorts count as
#: "painted line" and a marking running through a crowd of players scores a perfect fit.
_PLAYER_KERNEL = np.ones((13, 13), np.uint8)
_PLAYER_MARGIN = np.ones((31, 31), np.uint8)

#: Default confirmation radius in source pixels. The markings are 12 cm wide (a few px in
#: the near field) and the model draws their centreline, so a few px of slack is the line
#: itself, not error.
DEFAULT_TOLERANCE_PX = 6.0

#: Gaps this many samples long are bridged before judging a run: a defender standing on the
#: touchline breaks the paint, and that is occlusion, not a calibration error.
_GAP_BRIDGE = 6


def _masks(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(distance-to-nearest-painted-line, grass)`` for one BGR frame."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = ((h > 25) & (h < 95) & (s > 60)).astype(np.uint8)
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    white = (((v > 140) & (s < 70)).astype(np.uint8)) & grass
    thick = cv2.dilate(cv2.morphologyEx(white, cv2.MORPH_OPEN, _PLAYER_KERNEL), _PLAYER_MARGIN)
    lines = (white & (1 - thick)).astype(np.uint8)
    dist = cv2.distanceTransform((1 - lines).astype(np.uint8), cv2.DIST_L2, 5)
    return dist, grass


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
    points with no evidence (off-frame or off-grass) so callers can average honestly.
    """
    dist_map, grass = _evidence_cached(video_path, frame_index)
    hgt, wid = grass.shape
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
    on_grass = grass[vi, ui] > 0
    d = dist_map[vi, ui]
    sub_dist = np.where(on_grass, d, np.nan)
    near = _bridge(on_grass & (d <= tolerance))
    sub_labels = np.where(~on_grass, "unknown", np.where(near, "ok", "off"))

    labels[inside] = sub_labels
    dist[inside] = sub_dist
    return labels, dist
