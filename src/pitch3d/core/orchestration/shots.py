"""Shot-cut detection: refuse to reconstruct two cameras as one episode (#132, found on the way).

A broadcast clip is not one continuous view. The target clip cuts at frame 236 — 0–235 is the
wide shot the calibration was solved on, 236–333 a close-up replay from a different camera — and
**nothing in the pipeline notices**. `--frames 334` will track identities across the cut, solve
one camera for both, and blend the two into a single "episode". Every run so far has been safe
only by accident, because they all take 48–60 frames from the start.

The signal is a per-frame colour histogram: within a shot, consecutive frames differ a little as
players and the camera move; across a cut, the whole frame changes at once. :func:`find_shot_cuts`
takes those histograms and returns the frames a new shot begins on.

Pure numpy over a caller-supplied array, so the decode lives in an adapter and this half is
unit-testable with synthetic histograms.
"""

from __future__ import annotations

import numpy as np

#: L1 distance between consecutive normalised histograms above which a cut is declared. Measured
#: on the target clip: the one true cut scores 0.775 and no within-shot pair exceeds ~0.2, so the
#: gap is wide and this sits in the middle of it rather than hugging either side.
DEFAULT_CUT_THRESHOLD = 0.45


def histogram_distances(hists: np.ndarray) -> np.ndarray:
    """``(T, B)`` per-frame histograms → ``(T-1,)`` L1 distance between consecutive frames.

    Each row is L1-normalised first, so the distance is in ``[0, 2]`` and does not depend on how
    many pixels the caller sampled.
    """
    h = np.asarray(hists, dtype=float)
    if h.ndim != 2 or h.shape[0] < 2:
        return np.zeros(max(0, h.shape[0] - 1), dtype=float)
    total = h.sum(axis=1, keepdims=True)
    h = np.divide(h, total, out=np.zeros_like(h), where=total > 0)
    return np.abs(np.diff(h, axis=0)).sum(axis=1)


def find_shot_cuts(
    hists: np.ndarray,
    threshold: float = DEFAULT_CUT_THRESHOLD,
    min_shot_frames: int = 8,
) -> list[int]:
    """Frame indices (into ``hists``) where a new shot begins; ``[]`` for a single-shot clip.

    Args:
        hists: ``(T, B)`` per-frame colour histograms, one row per frame in order.
        threshold: L1 distance above which consecutive frames are a cut, not motion.
        min_shot_frames: Cuts closer together than this are a flash, a replay wipe or a camera
            flare rather than two shots. The *first* of such a burst is kept, so a real cut
            followed by a bright frame still reports one cut, not two.

    Returns:
        Sorted frame indices, each the first frame of a new shot. Index 0 is never returned —
        the clip's own start is not a cut.
    """
    d = histogram_distances(hists)
    if d.size == 0:
        return []
    cuts: list[int] = []
    for i in np.flatnonzero(d > threshold).tolist():
        frame = i + 1  # d[i] compares frame i and i+1, so the new shot starts at i+1
        if cuts and frame - cuts[-1] < min_shot_frames:
            continue
        cuts.append(frame)
    return cuts


def shot_bounds(n_frames: int, cuts: list[int]) -> list[tuple[int, int]]:
    """Cut list → inclusive ``(first, last)`` frame spans, one per shot."""
    edges = [0, *cuts, n_frames]
    return [(a, b - 1) for a, b in zip(edges, edges[1:], strict=False) if b - 1 >= a]


def shot_containing(cuts: list[int], n_frames: int, frame: int) -> tuple[int, int]:
    """The inclusive ``(first, last)`` span of the shot that ``frame`` falls in."""
    for first, last in shot_bounds(n_frames, cuts):
        if first <= frame <= last:
            return first, last
    return 0, max(0, n_frames - 1)
