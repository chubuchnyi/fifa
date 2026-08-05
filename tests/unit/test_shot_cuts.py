"""Shot-cut detection — the guard that stops two cameras being reconstructed as one episode.

The pure half takes caller-supplied histograms, so these run with no video and no cv2. The
numbers the threshold is justified by are measured, not invented: on the target clip the one true
cut scores **0.905** and the largest within-shot distance is **0.145**, a gap of 6x
(`scripts/shot_cuts.py`). The synthetic fixtures below straddle that same gap.
"""

from __future__ import annotations

import numpy as np

from pitch3d.core.orchestration.shots import (
    DEFAULT_CUT_THRESHOLD,
    find_shot_cuts,
    histogram_distances,
    shot_bounds,
    shot_containing,
)


def _shot(n, peak, bins=16, jitter=0.0):
    """n frames whose colour mass sits on one bin, with optional within-shot wobble."""
    h = np.zeros((n, bins), dtype=float)
    h[:, peak] = 100.0
    if jitter:
        # Move a little mass to a neighbouring bin, frame to frame — players and camera moving.
        h[:, (peak + 1) % bins] = jitter * np.arange(n)
    return h


def test_a_single_shot_reports_no_cuts():
    assert find_shot_cuts(_shot(30, 3, jitter=1.0)) == []


def test_a_camera_change_is_found_at_its_first_frame():
    hists = np.vstack([_shot(20, 2), _shot(20, 9)])
    assert find_shot_cuts(hists) == [20], "the cut is the first frame of the NEW shot"


def test_within_shot_motion_does_not_trip_the_threshold():
    # Distances an order of magnitude under the cut must stay under the threshold, or the guard
    # would truncate healthy clips — the failure mode that costs more than a missed cut.
    d = histogram_distances(_shot(40, 5, jitter=2.0))
    assert d.max() < DEFAULT_CUT_THRESHOLD


def test_a_flash_right_after_a_cut_is_still_one_cut():
    # A replay wipe or camera flare puts a bright frame next to the real cut. Reporting two shots
    # of 3 frames each would fragment the clip worse than not detecting anything.
    hists = np.vstack([_shot(20, 2), _shot(2, 15), _shot(20, 9)])
    assert find_shot_cuts(hists, min_shot_frames=8) == [20]


def test_distance_ignores_how_many_pixels_were_sampled():
    a, b = _shot(1, 4), _shot(1, 4) * 37.0  # same distribution, 37x the mass
    assert histogram_distances(np.vstack([a, b])).item() < 1e-12


def test_bounds_and_lookup_cover_every_frame():
    cuts = [20, 50]
    assert shot_bounds(80, cuts) == [(0, 19), (20, 49), (50, 79)]
    assert shot_containing(cuts, 80, 0) == (0, 19)
    assert shot_containing(cuts, 80, 49) == (20, 49)
    assert shot_containing(cuts, 80, 79) == (50, 79)


def test_degenerate_input_does_not_raise():
    assert find_shot_cuts(np.zeros((0, 8))) == []
    assert find_shot_cuts(np.zeros((1, 8))) == []
    assert histogram_distances(np.zeros((3, 8))).tolist() == [0.0, 0.0]  # all-zero rows
