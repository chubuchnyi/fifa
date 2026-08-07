"""A whip-pan is not a cut — the second opinion, measured on committed real frames.

The colour histogram in `core.orchestration.shots` cannot tell a cut from a fast camera move, and
on phone footage that is not a corner case. Measured on `14604731_1080_1920_30fps.mp4` (a fan clip
shot from the stand): the phone was whipped right and zoomed in one frame at f38, the histogram
distance jumped to **0.334** against a clip median of **0.049** and a threshold of **0.250**, and
the guard truncated a 60-frame run to 38. Frames 37 and 38 show the same goal, the same players and
the same stands.

The discriminator is that a pan or a zoom is one homography and a cut is not. Measured
2026-08-07, ORB(2000) + BF-Hamming + RANSAC(3 px) at 320 px wide:

    fan clip f37→f38 (the whip)        0.995
    fan clip f19→f20, f44→f45          0.998 – 1.000
    broadcast f29→f30 (an ordinary pan) 1.000
    broadcast f235→f236 (a REAL cut)   0.025

A 40x gap. `tests/data/shots/` holds those three pairs as 320-px grayscale stills (140 kB total)
so the discriminator is checked against real pixels here rather than against a synthetic array —
the sample videos are not in the repo, these frames are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pitch3d.core.orchestration.shots import find_shot_cuts

cv2 = pytest.importorskip("cv2", reason="the verifier is the cv2 half of shot detection")

from pitch3d.adapters.models.shot_detect import (  # noqa: E402  (after importorskip)
    MIN_MOVE_INLIER_RATIO,
    homography_inlier_ratio,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "shots"


def _pair(name: str) -> tuple[np.ndarray, np.ndarray]:
    a = cv2.imread(str(DATA / f"{name}_a.jpg"), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(DATA / f"{name}_b.jpg"), cv2.IMREAD_GRAYSCALE)
    assert a is not None and b is not None, f"missing fixture {name} in {DATA}"
    return a, b


@pytest.mark.parametrize("name", ["whip", "pan"])
def test_camera_moves_are_explained_by_one_homography(name: str):
    """The fan clip's whip-pan and an ordinary broadcast pan both fit a single homography."""
    ratio = homography_inlier_ratio(*_pair(name))
    assert ratio > 0.9, f"{name} should read as a camera move, got inlier ratio {ratio:.3f}"


def test_a_real_cut_is_not_explained_by_a_homography():
    """Broadcast f235→f236: the wide shot becomes a close-up replay from another camera."""
    ratio = homography_inlier_ratio(*_pair("cut"))
    assert ratio < 0.1, f"a real cut should not fit a homography, got {ratio:.3f}"


def test_the_threshold_sits_in_an_empty_gap():
    """Not a tuned number: nothing measured lands anywhere near it."""
    moves = [homography_inlier_ratio(*_pair(n)) for n in ("whip", "pan")]
    cut = homography_inlier_ratio(*_pair("cut"))
    assert cut < MIN_MOVE_INLIER_RATIO < min(moves)
    assert min(moves) / max(cut, 1e-6) > 10, "the separation must stay an order of magnitude"


def _hists_with_one_step() -> np.ndarray:
    """Twenty frames that drift slowly, with one abrupt colour change at frame 10."""
    rows = [np.array([1.0, 0.0, 0.0]) + i * 0.002 for i in range(10)]
    rows += [np.array([0.0, 1.0, 0.0]) + i * 0.002 for i in range(10)]
    return np.stack(rows)


def test_verify_can_veto_a_candidate_and_none_keeps_the_old_behaviour():
    """The hook is a veto, not a second detector: it can only remove candidates."""
    hists = _hists_with_one_step()
    assert find_shot_cuts(hists) == [10], "the synthetic step must be a candidate at all"
    assert find_shot_cuts(hists, verify=lambda _f: True) == [10]
    assert find_shot_cuts(hists, verify=lambda _f: False) == []


def test_verify_sees_the_candidate_frame_not_the_pair_index():
    """`verify(frame)` is called with the frame the new shot would start on."""
    seen: list[int] = []

    def spy(frame: int) -> bool:
        seen.append(frame)
        return False

    find_shot_cuts(_hists_with_one_step(), verify=spy)
    assert seen == [10]
