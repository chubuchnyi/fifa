"""#125 — a calibration that solved nothing must stop the run, not become a scene.

Pinned against the real artifact: `out/pod_0801` shipped 34/60 identity homographies and
confidence 0.0 on every frame, produced 11 subjects instead of 23, and reached the clip
switcher looking healthy because `apply_rigid_camera.py` had replaced the calibration.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.orchestration.pipeline import (
    UnsolvedCalibrationError,
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
