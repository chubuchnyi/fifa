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

#: A cut must be this many times the clip's own **median** consecutive-frame distance.
#:
#: The first version of this module used an absolute distance (0.45) measured at ``bins=8``, and
#: that was a latent bug rather than a tuning choice: the distance scales with the histogram's bin
#: count, so the same clip that reads 2 shots at ``bins=8`` shredded into **14 shots at bins=32 and
#: 36 at bins=48**. Normalising by the clip's own median removes that coupling and also adapts to
#: clips with more or less camera motion than this one.
#:
#: Measured on the target clip, distance ÷ median, true cut vs the largest within-shot pair:
#:
#: ===== ======== ============
#: bins  cut      within-shot
#: ===== ======== ============
#: 4     20.1     2.44
#: 8     14.1     2.26
#: 16    10.0     2.13
#: 32     4.9     1.89
#: 64     2.8     1.64
#: ===== ======== ============
#:
#: 5 sits in the middle of the gap at the validated :data:`~pitch3d.adapters.models.shot_detect`
#: binning, and note the failure direction if a caller does use a silly bin count: the ratio for a
#: real cut *falls*, so the detector goes quiet rather than inventing shots. A missed cut is
#: recoverable; truncating a healthy clip is not.
DEFAULT_CUT_RATIO = 5.0

#: Second condition, so a near-static clip (median ≈ 0, every ratio enormous) cannot manufacture
#: cuts out of compression noise. Distances are L1 over L1-normalised rows, so this is in the same
#: [0, 2] range for any bin count: a quarter of all colour mass must move at once.
DEFAULT_CUT_FLOOR = 0.25


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


def cut_threshold(
    d: np.ndarray,
    ratio: float = DEFAULT_CUT_RATIO,
    floor: float = DEFAULT_CUT_FLOOR,
) -> float:
    """The distance a pair must beat to count as a cut: ``max(ratio × median, floor)``.

    Both conditions have to hold. The ratio makes the decision scale-free — independent of bin
    count and of how much this particular camera moves — and the floor stops a near-static clip,
    where the median is ~0 and every ratio is enormous, from manufacturing cuts out of noise.
    """
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return float(floor)
    return float(max(ratio * float(np.median(d)), floor))


def find_shot_cuts(
    hists: np.ndarray,
    ratio: float = DEFAULT_CUT_RATIO,
    min_shot_frames: int = 8,
    floor: float = DEFAULT_CUT_FLOOR,
    threshold: float | None = None,
) -> list[int]:
    """Frame indices (into ``hists``) where a new shot begins; ``[]`` for a single-shot clip.

    Args:
        hists: ``(T, B)`` per-frame colour histograms, one row per frame in order.
        ratio: Multiple of the clip's own median distance a cut must exceed
            (:data:`DEFAULT_CUT_RATIO`).
        min_shot_frames: Cuts closer together than this are a flash, a replay wipe or a camera
            flare rather than two shots. The *first* of such a burst is kept, so a real cut
            followed by a bright frame still reports one cut, not two.
        floor: Absolute distance a cut must also exceed (:data:`DEFAULT_CUT_FLOOR`).
        threshold: Escape hatch — an absolute distance that replaces the adaptive rule entirely.
            Auto-detect plus manual override; leave ``None`` unless you have measured this clip.

    Returns:
        Sorted frame indices, each the first frame of a new shot. Index 0 is never returned —
        the clip's own start is not a cut.
    """
    d = histogram_distances(hists)
    if d.size == 0:
        return []
    thr = float(threshold) if threshold is not None else cut_threshold(d, ratio, floor)
    cuts: list[int] = []
    for i in np.flatnonzero(d > thr).tolist():
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
