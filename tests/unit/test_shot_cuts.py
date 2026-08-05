"""Shot-cut detection — the guard that stops two cameras being reconstructed as one episode.

The pure half takes caller-supplied histograms, so these run with no video and no cv2. The
numbers the threshold is justified by are measured, not invented: on the target clip the one true
cut scores **0.905** and the largest within-shot distance is **0.145**, a gap of 6x
(`scripts/shot_cuts.py`). The synthetic fixtures below straddle that same gap.
"""

from __future__ import annotations

import numpy as np

from pitch3d.core.orchestration.shots import (
    DEFAULT_CUT_RATIO,
    cut_threshold,
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
    assert d.max() < cut_threshold(d)


def test_a_busy_clip_with_no_cut_yields_no_cut():
    """The regression that broke the first version: the bar has to scale with the clip.

    A fixed absolute threshold cannot do this. Every consecutive pair here moves a *third* of the
    colour mass — far above any sensible constant — yet nothing ever changes camera, and calling
    a cut would truncate a healthy clip.
    """
    rng = np.random.default_rng(0)
    hists = np.abs(rng.normal(100.0, 33.0, size=(60, 16)))
    d = histogram_distances(hists)
    assert d.max() > 0.25, "fixture must exceed the absolute floor, else it proves nothing"
    assert find_shot_cuts(hists) == []


def test_the_bar_scales_with_the_clip_not_with_the_bin_count():
    """Same shots, 4x finer histograms: the *ratio* survives where an absolute distance did not.

    Measured on the target clip, raising bins from 8 to 48 shredded 2 shots into 36 under the old
    fixed 0.45 threshold, because the distance itself scales with bin count.
    """
    coarse = np.vstack([_shot(20, 1, bins=8), _shot(20, 6, bins=8)])
    fine = np.vstack([_shot(20, 4, bins=64), _shot(20, 40, bins=64)])
    assert find_shot_cuts(coarse) == [20]
    assert find_shot_cuts(fine) == [20]
    # ...and the thresholds they used are genuinely different numbers, so this is not a fluke of
    # both landing on the same absolute value.
    assert cut_threshold(histogram_distances(coarse)) == cut_threshold(histogram_distances(fine))


def test_absolute_threshold_remains_available_as_an_override():
    hists = np.vstack([_shot(20, 2), _shot(20, 9)])
    # This fixture moves *all* mass between bins, so its distance is exactly 2.0 — the L1 ceiling
    # for normalised rows. A bar above that can never be met.
    assert find_shot_cuts(hists, threshold=2.5) == []
    assert find_shot_cuts(hists, threshold=0.5) == [20]
    assert DEFAULT_CUT_RATIO > 1.0


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
