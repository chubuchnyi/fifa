"""Per-frame colour histograms for shot-cut detection — the decode half of `orchestration.shots`.

Kept out of the pure module so the cut logic stays testable with synthetic arrays and this file
owns the only cv2 import. Cheap by design: frames are shrunk hard before histogramming, because a
cut changes the whole colour distribution and needs no detail to see.
"""

from __future__ import annotations

import numpy as np

#: Bins per channel (8 → 512 total). **Do not raise this to "improve" the detector.** Finer bins
#: make the histograms sparse, so ordinary player motion moves mass between neighbouring bins and
#: starts to look like a camera change: measured on the target clip, the true cut stands at 14x the
#: median distance at 8 bins but only 2.8x at 64, while within-shot motion climbs to 1.6x. The
#: adaptive rule in `core.orchestration.shots` degrades safely (it goes quiet rather than inventing
#: shots), but the separation it has to work with is best here.
VALIDATED_BINS = 8


def clip_histograms(  # pragma: no cover - heavy path (needs cv2 + media)
    uri: str,
    n_frames: int = 0,
    start: int = 0,
    bins: int = VALIDATED_BINS,
    width: int = 96,
) -> np.ndarray:
    """Decode ``uri`` and return ``(T, bins**3)`` BGR colour histograms, one row per frame.

    Args:
        uri: Video path.
        n_frames: Frames to read from ``start``; ``0`` reads to the end.
        start: First frame index.
        bins: Bins per channel; see :data:`VALIDATED_BINS` before changing it.
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
