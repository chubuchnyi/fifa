"""SoccerNet calibration eval: pitch template, GT parsing, and reprojection metrics.

We cannot ship SoccerNet frames or run PnLCalib here, so these tests pin what is verifiable
*without* the dataset or a GPU: that the re-derived pitch template matches the laws-of-the-game
geometry, that the annotation parser scales normalised GT into pixels and keeps only the straight
pitch-plane lines, and that the reprojection metric is ~0 for the true homography and grows under
perturbation (so a real run's number means what it claims).
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.eval.calib_metrics import (
    evaluate_calibration,
    format_sweep_table,
    frame_metrics,
    frame_pixel_errors,
    frame_world_errors,
    summarize_threshold_sweep,
)
from pitch3d.eval.datasets_soccernet import (
    load_calib_annotation,
    pitch_plane_lines,
    synthetic_calib_frames,
)


def test_pitch_template_geometry() -> None:
    lines = pitch_plane_lines(length=105.0, width=68.0)
    assert len(lines) == 17  # 5 touch/halfway + 6 big-rect + 6 small-rect; no circles/goals
    # Spot-check known laws-of-the-game coordinates (metres, origin at centre mark). "top" is the
    # top of SoccerNet's top-down template, whose Y runs down the image, so in our Z-up
    # right-handed world it is the +Y touchline (#118).
    np.testing.assert_allclose(lines["Side line top"][0], [-52.5, 34.0])
    np.testing.assert_allclose(lines["Side line top"][1], [52.5, 34.0])
    np.testing.assert_allclose(lines["Side line bottom"][0], [-52.5, -34.0])
    np.testing.assert_allclose(lines["Middle line"], [[0.0, 34.0], [0.0, -34.0]])
    # Penalty box "main" line is 16.5 m in from the goal line, half-width 20.16 m.
    np.testing.assert_allclose(lines["Big rect. left main"], [[-36.0, 20.16], [-36.0, -20.16]])
    # Goal-area box "main" line is 5.5 m in, half-width 9.16 m, on the right.
    np.testing.assert_allclose(lines["Small rect. right main"], [[47.0, 9.16], [47.0, -9.16]])


def test_load_annotation_scales_and_filters() -> None:
    ann = {
        "Side line top": [{"x": 0.1, "y": 0.2}, {"x": 0.9, "y": 0.25}],
        "Big rect. left main": [{"x": 0.3, "y": 0.4}, {"x": 0.32, "y": 0.7}],
        "Circle central": [{"x": 0.5, "y": 0.5}, {"x": 0.51, "y": 0.52}],  # curved → dropped
        "Goal left crossbar": [{"x": 0.2, "y": 0.2}, {"x": 0.25, "y": 0.2}],  # Z!=0 → dropped
        "Middle line": [{"x": 0.5, "y": 0.5}],  # < 2 points → dropped
        "Line unknown": [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}],  # not a template line
    }
    gt = load_calib_annotation(ann, width=960, height=540, frame=7)
    names = {ln.name for ln in gt.lines}
    assert names == {"Side line top", "Big rect. left main"}
    assert gt.frame == 7 and gt.width == 960 and gt.height == 540
    top = next(ln for ln in gt.lines if ln.name == "Side line top")
    np.testing.assert_allclose(top.image_uv, [[96.0, 108.0], [864.0, 135.0]])  # normalised x W,H
    np.testing.assert_allclose(top.world_a, [-52.5, 34.0])  # template top → our +Y (#118)


def test_synthetic_oracle_is_near_zero_and_perturbation_grows() -> None:
    frames, true_h = synthetic_calib_frames(n_frames=4, seed=1)
    assert all(f.n_lines >= 4 for f in frames)  # the synthetic view sees plenty of lines

    grid = evaluate_calibration(frames, true_h)
    assert grid["reproj_rms_m"] < 1e-6  # true homography → essentially exact
    assert grid["reproj_rms_px"] < 1e-6
    assert grid["line_acc@5px"] == 1.0

    rng = np.random.default_rng(0)
    bad = np.stack([h * (1.0 + 0.05 * rng.standard_normal((3, 3))) for h in true_h])
    bad = np.stack([h / h[2, 2] for h in bad])
    bad_grid = evaluate_calibration(frames, bad)
    assert bad_grid["reproj_rms_m"] > grid["reproj_rms_m"] + 0.1  # metres of error appear
    assert bad_grid["line_acc@5px"] < 1.0


def test_frame_world_errors_match_known_offset() -> None:
    # Identity-ish homography: image coords already in metres, one line along world X at y=0.
    frames, true_h = synthetic_calib_frames(n_frames=1, seed=2)
    gt = frames[0]
    h = true_h[0]
    errs = frame_world_errors(h, gt)
    assert errs.size == gt.n_points
    assert float(np.max(errs)) < 1e-6


def test_singular_homography_is_an_honest_miss() -> None:
    frames, _ = synthetic_calib_frames(n_frames=1, seed=3)
    gt = frames[0]
    singular = np.zeros((3, 3))
    px = frame_pixel_errors(singular, gt)
    assert all(np.all(~np.isfinite(e)) for e in px.values())  # inf, not a crash
    fm = frame_metrics(singular, gt)
    assert fm["lines_ok@5px"] == 0.0


def test_completeness_from_confidence() -> None:
    frames, true_h = synthetic_calib_frames(n_frames=4, seed=4)
    conf = np.array([1.0, 0.0, 0.8, 0.0])
    grid = evaluate_calibration(frames, true_h, confidence=conf)
    assert grid["completeness"] == pytest.approx(0.5)  # 2 of 4 frames confident


def test_on_completed_excludes_failed_frames() -> None:
    # Two good frames + one "failed" frame whose homography is degenerate-but-invertible — a
    # stand-in for the identity/garbage H the calibrator stores when it cannot lock on. It maps
    # 960x540 pixel coords into a sub-metre blob at the origin while the true pitch lines span
    # tens of metres, so its reprojection error is huge-but-finite: it must pollute the all-frames
    # RMS yet be excluded from the on_completed (confidence > 0) sub-grid.
    frames, true_h = synthetic_calib_frames(n_frames=3, seed=5)
    h = true_h.copy()
    h[2] = np.diag([1e-3, 1e-3, 1.0])
    conf = np.array([1.0, 1.0, 0.0])

    grid = evaluate_calibration(frames, h, confidence=conf)

    assert grid["completeness"] == pytest.approx(2 / 3)
    assert grid["n_completed"] == 2
    assert grid["reproj_rms_m"] > 1.0  # all-frames RMS polluted by the failed frame
    on_completed = grid["on_completed"]
    assert isinstance(on_completed, dict)
    assert on_completed["reproj_rms_m"] < 1e-6  # the two confident frames are essentially exact
    assert on_completed["line_acc@5px"] == 1.0


def test_threshold_sweep_summary_and_table() -> None:
    # Stand in for a box sweep: a higher kp threshold leaves one frame uncalibrated (completeness
    # 0.5), a lower one calibrates both (1.0). The summary must surface completeness against the
    # accuracy where the calibrator locked on, and the table must render one row per threshold.
    frames, true_h = synthetic_calib_frames(n_frames=2, seed=7)
    grid_hi = evaluate_calibration(frames, true_h, confidence=np.array([1.0, 0.0]))
    grid_lo = evaluate_calibration(frames, true_h, confidence=np.array([1.0, 1.0]))

    summary = summarize_threshold_sweep([(0.3434, grid_hi), (0.15, grid_lo)])
    assert [r["kp_threshold"] for r in summary] == [0.3434, 0.15]
    assert summary[0]["completeness"] == pytest.approx(0.5)
    assert summary[1]["completeness"] == pytest.approx(1.0)
    assert summary[0]["n_completed"] == 1.0
    assert summary[0]["on_completed_line_acc@5px"] == 1.0  # accurate where it locked on

    table = format_sweep_table(summary)
    assert "complete" in table  # header present
    assert table.count("\n") == 2  # header + two data rows
