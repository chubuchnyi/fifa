"""#125 — a calibration that solved nothing must stop the run, not become a scene.

Pinned against the real artifact: `out/pod_0801` shipped 34/60 identity homographies and
confidence 0.0 on every frame, produced 11 subjects instead of 23, and reached the clip
switcher looking healthy because `apply_rigid_camera.py` had replaced the calibration.

The gate deliberately refuses only the "no answer anywhere" case; #131 is its other half —
telling the caller how much of a *passing* calibration is carried rather than measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.orchestration.pipeline import (
    UnsolvedCalibrationError,
    describe_calibration_solve,
    require_solved_calibration,
)
from pitch3d.core.scene.field import FieldCalibration


def _calib(confidence: np.ndarray) -> FieldCalibration:
    n = confidence.size
    return FieldCalibration(
        homographies=np.tile(np.eye(3), (n, 1, 1)),
        frames=np.arange(n),
        confidence=confidence,
    )


def test_all_frames_unsolved_is_refused():
    with pytest.raises(UnsolvedCalibrationError, match="0/60"):
        require_solved_calibration(_calib(np.zeros(60)))


def test_one_solved_frame_passes_and_is_counted():
    conf = np.zeros(60)
    conf[17] = 0.42
    assert require_solved_calibration(_calib(conf)) == 1


def test_healthy_calibration_reports_every_solved_frame():
    # The shape of `out/fresh60`: every frame solved, confidence around 0.5.
    assert require_solved_calibration(_calib(np.full(60, 0.53))) == 60


def test_min_solved_is_the_operator_s_dial():
    conf = np.zeros(60)
    conf[:5] = 0.6
    assert require_solved_calibration(_calib(conf), min_solved=5) == 5
    with pytest.raises(UnsolvedCalibrationError, match="5/60"):
        require_solved_calibration(_calib(conf), min_solved=6)


def test_the_carried_fraction_is_reported_where_the_mean_hid_it():
    """#131 — the shape of the vertical fan clip, which passed the gate and was still junk.

    355 frames, 153 of them carried once the fan zoomed past the pitch landmarks. The
    all-frames mean is 0.28, which reads like a weak-but-working calibration and was the
    ONLY calibration figure a run printed. The carried count is what says otherwise.
    """
    conf = np.full(355, 0.496)
    conf[202:] = 0.0
    assert require_solved_calibration(_calib(conf)) == 202  # the gate is satisfied, correctly
    assert round(float(conf.mean()), 2) == 0.28  # ...and the number a run printed says nothing

    line = describe_calibration_solve(_calib(conf))
    assert "202/355 frame(s) measured, 153 carried (confidence 0)" in line
    assert "mean 0.496" in line  # over MEASURED frames — 0.28 is the average of two populations


def test_a_fully_measured_calibration_says_nothing_was_carried():
    line = describe_calibration_solve(_calib(np.full(60, 0.53)))
    assert "60/60 frame(s) measured, 0 carried" in line


def test_an_all_carried_calibration_reports_no_confidence_at_all():
    # It cannot reach here through the gate, but describe() is also called on loaded scenes:
    # a mean over an empty selection is NaN, and NaN in a status line is worse than silence.
    line = describe_calibration_solve(_calib(np.zeros(8)))
    assert line == "0/8 frame(s) measured, 8 carried (confidence 0)"
