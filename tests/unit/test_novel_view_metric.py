"""R7 (#99): the novel-view metric (``pitch3d.eval.novel_view``).

The metric's whole claim is that Global MPJPE conflates two errors a viewer experiences very
differently: a common-mode one that re-renders as a slightly different novel camera, and a
per-player one that puts a body in the wrong spot. So the tests are built around error fields
that are *identical* under Global MPJPE and must come apart here — if they don't, the metric
is a rename of the number it was written to replace.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.eval.metrics import mpjpe_global
from pitch3d.eval.novel_view import (
    decompose_global_error,
    fit_rigid,
    local_mpjpe_by_speed,
    per_player_residual_m,
)

T, N, J = 12, 5, 15


def _gt(seed: int = 0) -> np.ndarray:
    """Players milling about the pitch with articulated joints — shape ``(T, N, J, 3)``."""
    rng = np.random.default_rng(seed)
    root = rng.uniform([-40, -25, 0.9], [40, 25, 0.95], size=(1, N, 1, 3))
    drift = np.cumsum(rng.normal(0, 0.06, size=(T, N, 1, 3)) * [1, 1, 0.05], axis=0)
    limbs = rng.normal(0, 0.35, size=(1, N, J, 3))
    swing = np.sin(np.linspace(0, 4 * np.pi, T))[:, None, None, None] * limbs * 0.4
    return root + drift + limbs + swing


def _rigid(yaw_deg: float, shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.asarray(shift, float)


# --- the claim: a common-mode error and a per-player error are not the same defect ---------


def test_a_pure_camera_error_is_large_globally_and_gone_after_the_camera_refit():
    # A wrong camera expresses the true scene in a wrong frame: every player moves by one rigid
    # transform. At a novel viewpoint that IS a novel viewpoint, so the residual must be ~0
    # while Global MPJPE reports metres. This is the whole argument for the metric.
    gt = _gt()
    rot, transl = _rigid(4.0, [1.2, -0.7, 0.05])
    pred = gt @ rot.T + transl

    grid = decompose_global_error(pred, gt)
    assert grid["global_mpjpe_m"] > 1.0
    assert grid["after_static_camera_m"] < 1e-9
    assert grid["after_perframe_camera_m"] < 1e-9
    assert grid["camera_absorbed_frac"] > 0.999


def test_two_error_fields_that_global_mpjpe_calls_equal_are_ranked_here():
    # Same headline number, opposite verdicts. Global MPJPE cannot tell these apart, which is
    # exactly why the briefs' 0.35-0.45 m envelope is the wrong bar for a video judged by eye.
    rng = np.random.default_rng(7)
    gt = _gt()
    rot, transl = _rigid(3.0, [0.9, -0.4, 0.0])
    common = gt @ rot.T + transl

    scatter_dir = rng.normal(size=(1, N, 1, 3))
    scatter_dir /= np.linalg.norm(scatter_dir, axis=-1, keepdims=True)
    scale = mpjpe_global(common, gt)
    scattered = gt + scatter_dir * scale

    assert abs(mpjpe_global(common, gt) - mpjpe_global(scattered, gt)) < 1e-9  # indistinguishable
    assert decompose_global_error(common, gt)["after_perframe_camera_m"] < 1e-9
    assert decompose_global_error(scattered, gt)["after_perframe_camera_m"] > 0.5 * scale


def test_scene_swim_separates_a_wobbling_camera_from_a_merely_wrong_one():
    # A camera error that CHANGES frame to frame slides the whole scene under a shot that should
    # be still — visible, and hidden by any whole-clip fit. It must land in scene_swim_m, not in
    # the free bucket. A constant error of the same size must not.
    gt = _gt(1)
    wobble = np.zeros((T, 1, 1, 3))
    wobble[:, 0, 0, 0] = 0.5 * np.sin(np.linspace(0, 3 * np.pi, T))

    swimming = decompose_global_error(gt + wobble, gt)
    steady = decompose_global_error(gt + np.array([0.5, 0.0, 0.0]), gt)

    assert swimming["scene_swim_m"] > 0.2
    assert swimming["after_perframe_camera_m"] < 1e-9  # no player is individually misplaced
    assert steady["scene_swim_m"] < 1e-9


def test_the_fit_never_absorbs_scale_because_wrong_size_players_are_visible():
    # #61's defect is a ~3x scale. A similarity fit would erase it and report a clean scene; we
    # render against a true-size pitch, so it is not erasable. The rigid headline must still see
    # it, and the diagnostic must name it as scale rather than leaving it as anonymous error.
    gt = _gt(2)
    grid = decompose_global_error(gt * 3.0, gt)

    assert grid["after_perframe_camera_m"] > 1.0
    assert grid["predicted_scale"] == pytest.approx(3.0, rel=1e-6)  # reads as the defect, not 1/3
    assert grid["after_similarity_m"] < 1e-9  # ...which is precisely why we do not use it


def test_one_misplaced_player_shows_up_as_spread_and_not_as_a_smeared_average():
    # The deliverable's failure mode: 4 players right, 1 wrong. The per-player breakout must name
    # the culprit instead of dividing its error by 5 (the mean's way of hiding it).
    gt = _gt(3)
    pred = gt.copy()
    pred[:, 2] += np.array([2.0, 0.0, 0.0])

    residual = per_player_residual_m(pred, gt)
    assert residual.argmax() == 2
    assert residual[2] > 1.0
    assert np.median(np.delete(residual, 2)) < 0.6
    assert decompose_global_error(pred, gt)["player_spread_m"] > 1.0


# --- boundaries: where the metric refuses to answer ----------------------------------------


def test_a_single_subject_is_refused_rather_than_silently_mis_scored():
    # With one body a per-frame rigid fit removes that body's own placement error and would
    # report it as "absorbed by the camera" — a number that flatters us by construction.
    gt = _gt()[:, :1]
    with pytest.raises(ValueError, match="at least 2 subjects"):
        decompose_global_error(gt, gt)


def test_the_rigid_fit_refuses_to_mirror_even_when_a_mirror_fits_better():
    # SVD will return det(R) = -1 given the chance. A mirrored fit is the #50/#64 defect wearing
    # a metric's clothes: it would make a mirrored reconstruction score perfectly.
    rng = np.random.default_rng(11)
    pts = rng.normal(size=(40, 3))
    rot, transl, _ = fit_rigid(pts, pts * np.array([-1.0, 1.0, 1.0]))

    assert np.linalg.det(rot) > 0
    assert np.linalg.norm(_apply(rot, transl, pts) - pts * [-1, 1, 1]) > 1.0


def _apply(rot, transl, pts):
    return pts @ rot.T + transl


# --- the speed stratification ---------------------------------------------------------------


def test_a_smoother_that_wins_on_the_mean_can_lose_where_the_motion_is():
    # The yaw low-pass lesson, as a number. Errors placed ONLY on fast joint-frames are diluted
    # by the mean and concentrated by the top decile — so a smoother trading fast-phase fidelity
    # for overall calm looks free on `local_mpjpe_m` and is caught by `top_decile_penalty`.
    gt = _gt(4)
    rel = gt - gt[:, :, :1, :]
    speed = np.linalg.norm(np.gradient(rel, 1 / 25.0, axis=0), axis=-1)
    fast = speed >= np.quantile(speed, 0.9)

    pred = gt.copy()
    pred[fast] += np.array([0.25, 0.0, 0.0])

    grid = local_mpjpe_by_speed(pred, gt, fps=25.0)
    assert grid["local_mpjpe_m"] < 0.05  # a mean would call this clip fine
    assert grid["local_mpjpe_top_decile_m"] > 0.2
    assert grid["top_decile_penalty"] > 5.0
    assert grid["local_mpjpe_bottom_decile_m"] < 1e-9


def test_the_strata_come_from_ground_truth_so_two_methods_are_compared_on_the_same_frames():
    # If speed were read off the prediction, a smoother would move the goalposts by flattening
    # the very motion that defines the stratum, and could improve its own score by cheating.
    gt = _gt(5)
    honest = local_mpjpe_by_speed(gt + 0.1, gt, fps=25.0)
    flattened = local_mpjpe_by_speed(np.repeat(gt[:1], T, axis=0), gt, fps=25.0)
    assert honest["speed_threshold_m_s"] == pytest.approx(flattened["speed_threshold_m_s"])


def test_speed_is_articulation_speed_not_the_players_run():
    # A player sprinting with a frozen body must register as SLOW: local MPJPE has already
    # subtracted the root, so stratifying by ground speed would sort by an irrelevant axis.
    gt = _gt(6)
    sprint = gt + np.arange(T)[:, None, None, None] * np.array([0.4, 0.0, 0.0])
    assert local_mpjpe_by_speed(gt + 0.1, gt, fps=25.0)[
        "speed_threshold_m_s"
    ] == pytest.approx(local_mpjpe_by_speed(sprint + 0.1, sprint, fps=25.0)["speed_threshold_m_s"])
