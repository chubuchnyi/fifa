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


#: Homography inlier ratio above which a frame pair is a **camera move**, not a cut.
#:
#: Measured 2026-08-07 on real footage, ORB(2000) + BF-Hamming(crossCheck) + RANSAC(3 px) over
#: grayscale frames resized to :data:`VERIFY_WIDTH`:
#:
#: ===================================== ==============
#: pair                                  inlier ratio
#: ===================================== ==============
#: fan clip f37→f38 (the whip-pan)       **0.995**
#: fan clip f19→f20, f44→f45, f99→f100   0.998 – 1.000
#: broadcast f29→f30, f58→f59 (pan)      1.000
#: broadcast f235→f236 (**a real cut**)  **0.025**
#: ===================================== ==============
#:
#: A 40x gap with nothing in it, so 0.5 is not a tuned number — it is the middle of an empty
#: interval. The separation survives JPEG and holds down to 240 px wide (0.980 vs 0.082); it
#: collapses at 160 px, which is why the histogram pass may shrink frames to 96 px and this one
#: may not.
MIN_MOVE_INLIER_RATIO = 0.5

#: Frames are resized to this width before feature matching. See :data:`MIN_MOVE_INLIER_RATIO`.
VERIFY_WIDTH = 320

#: Below this many matches there is nothing to fit a homography to, and the answer is "cut" —
#: two frames sharing almost no features is what a cut looks like.
MIN_MOVE_MATCHES = 12


def homography_inlier_ratio(prev_gray, curr_gray, width: int = VERIFY_WIDTH) -> float:
    """Fraction of feature matches between two frames explained by one homography, in ``[0, 1]``.

    A pan, a zoom or a phone whip maps the whole frame through a single homography, so nearly
    every match is an inlier. A cut replaces the content, so the few matches that survive are
    coincidences and no homography explains them. This is the measurement behind
    :data:`MIN_MOVE_INLIER_RATIO`; the frames are handed in already decoded so the discriminator
    can be tested against committed stills rather than a video file.
    """
    import cv2

    def _prep(img):
        img = np.asarray(img)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape[1] != width:
            img = cv2.resize(img, (width, max(1, img.shape[0] * width // img.shape[1])),
                             interpolation=cv2.INTER_AREA)
        return img

    a, b = _prep(prev_gray), _prep(curr_gray)
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(a, None)
    k2, d2 = orb.detectAndCompute(b, None)
    if d1 is None or d2 is None or len(k1) < MIN_MOVE_MATCHES or len(k2) < MIN_MOVE_MATCHES:
        return 0.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    matches = sorted(matches, key=lambda m: m.distance)[:400]
    if len(matches) < MIN_MOVE_MATCHES:
        return 0.0
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    h, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    if h is None or mask is None:
        return 0.0
    return float(mask.sum()) / float(len(matches))


def homography_cut_verifier(  # pragma: no cover - heavy path (needs cv2 + media)
    uri: str, start: int = 0, min_ratio: float = MIN_MOVE_INLIER_RATIO
):
    """Build the ``verify`` callable :func:`~pitch3d.core.orchestration.shots.find_shot_cuts` takes.

    ``verify(i)`` decodes the pair ``(start + i - 1, start + i)`` and answers "is this really a
    cut?". Only candidates reach it — normally none or one per clip — so decoding two frames each
    time is free next to the histogram pass that produced them.

    A decode failure answers **True** (keep the candidate): if we cannot check, the histogram's
    opinion stands rather than being silently overridden.
    """
    import cv2

    def verify(i: int) -> bool:
        cap = cv2.VideoCapture(uri)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start + i - 1))
            ok_a, a = cap.read()
            ok_b, b = cap.read()
        finally:
            cap.release()
        if not (ok_a and ok_b):
            return True
        return homography_inlier_ratio(a, b) < min_ratio

    return verify
