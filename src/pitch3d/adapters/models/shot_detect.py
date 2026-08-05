"""Per-frame colour histograms for shot-cut detection — the decode half of `orchestration.shots`.

Kept out of the pure module so the cut logic stays testable with synthetic arrays and this file
owns the only cv2 import. Cheap by design: frames are shrunk hard before histogramming, because a
cut changes the whole colour distribution and needs no detail to see.
"""

from __future__ import annotations

import numpy as np


def clip_histograms(  # pragma: no cover - heavy path (needs cv2 + media)
    uri: str,
    n_frames: int = 0,
    start: int = 0,
    bins: int = 8,
    width: int = 96,
) -> np.ndarray:
    """Decode ``uri`` and return ``(T, bins**3)`` BGR colour histograms, one row per frame.

    Args:
        uri: Video path.
        n_frames: Frames to read from ``start``; ``0`` reads to the end.
        start: First frame index.
        bins: Bins per channel (8 → 512 total), plenty to see a camera change.
        width: Frames are resized to this width first — a cut is a global colour change, so
            detail costs time and buys nothing.
    """
    import cv2

    cap = cv2.VideoCapture(uri)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    rows: list[np.ndarray] = []
    while n_frames <= 0 or len(rows) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        h = frame.shape[0] * width // max(1, frame.shape[1])
        small = cv2.resize(frame, (width, max(1, h)), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([small], [0, 1, 2], None, [bins] * 3, [0, 256] * 3)
        rows.append(hist.reshape(-1))
    cap.release()
    return np.stack(rows) if rows else np.zeros((0, bins**3), dtype=float)
