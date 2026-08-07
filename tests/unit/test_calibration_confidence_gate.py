"""Grounding must refuse a CARRIED homography — measured on the committed real calibration.

The failure this guards against was measured on the vertical fan clip (2026-08-03,
`findings/open-items-2026-08-01.md` §3.3): 153 of 355 frames carried a stale homography, root
spread came out **3079.7 × 3079.7 m** and one subject reached **100 416 m/s**. The mechanism was
not exotic — `GVHMRPoseEstimator._ground_root` called `FieldCalibration.image_to_world` on every
frame of every tracklet, and `image_to_world` is pure geometry: it cannot know that the matrix it
was handed was copied forward from a frame the calibrator failed on.

So the magnitude is measured here on **real data**, not invented.
`calib/Colombia-1-0-Congo-DR1080p.npz` (7 kB, committed, the same file
`tests/e2e/test_golden_real_camera.py` pins) carries the measured
per-frame world→image homography for 60 frames of the target clip. Grounding a foot pixel through a
homography 59 frames stale displaces it by:

    foot (960, 700)  → true (-37.53, 6.72) m · stale (-29.00, 10.66) m · **9.39 m**
    foot (700, 850)  → true (-36.11, -7.27) m · stale (-29.12, -3.54) m · **7.92 m**
    foot (1400, 600) → true (-33.18, 22.07) m · stale (-22.95, 25.55) m · **10.80 m**

That is the *gentle* case: a broadcast tripod panning for two seconds. A phone that zooms turns the
same mechanism into kilometres. Either way a player metres out of position is not a placement, and
R-6 says mark it, not invent it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pitch3d.adapters.models.pose import GVHMRPoseEstimator, RawBodyMotion
from pitch3d.core.ports.io import ClipRef
from pitch3d.core.ports.perception import Tracklet, Tracks
from pitch3d.core.scene.field import FieldCalibration

CALIB = Path(__file__).resolve().parents[2] / "calib" / "Colombia-1-0-Congo-DR1080p.npz"
N_FRAMES = 60
FIRST_CARRIED = 30          # the calibrator "fails" from here on
_J = 21
_FOOT_UV = (960.0, 700.0)   # a foot near the middle of the 1920x1080 frame


def _true_homographies() -> np.ndarray:
    """Measured image→world plane homography per frame, straight out of the committed fit."""
    blob = np.load(CALIB, allow_pickle=True)
    return np.stack([np.linalg.inv(h) for h in np.asarray(blob["world_to_image"])])


def _calibration_with_carry() -> tuple[FieldCalibration, np.ndarray]:
    """A calibration that solved frames 0-29 and CARRIED 30-59 — what `calibration.py:718` writes.

    Returns the calibration and, separately, the frames' *true* homographies, so an assertion can
    ask where the player really was without re-deriving it from the object under test.
    """
    true = _true_homographies()
    carried = true.copy()
    carried[FIRST_CARRIED:] = true[FIRST_CARRIED - 1]        # last good, copied forward
    conf = np.full(N_FRAMES, 0.55)
    conf[FIRST_CARRIED:] = 0.0                               # "carry last good, flag zero" (R-6)
    return FieldCalibration(homographies=carried, frames=np.arange(N_FRAMES), confidence=conf), true


def _tracks() -> Tracks:
    u, v = _FOOT_UV
    boxes = np.tile(np.array([u - 20.0, v - 90.0, u + 20.0, v], dtype=float), (N_FRAMES, 1))
    return Tracks(tracklets=[
        Tracklet(track_id=7, frames=np.arange(N_FRAMES), bboxes_xyxy=boxes, cls="player")
    ])


class _StubHMR:
    def estimate_bodies(self, clip: ClipRef, tracks: Tracks) -> dict[int, RawBodyMotion]:
        return {7: RawBodyMotion(
            track_id=7, frames=np.arange(N_FRAMES),
            global_orient=np.zeros((N_FRAMES, 3)),
            body_pose=np.zeros((N_FRAMES, _J, 3)), betas=np.zeros(10),
        )}


def _clip() -> ClipRef:
    return ClipRef(source_id="s", uri="x", frames=np.arange(N_FRAMES),
                   width=1920, height=1080, fps=29.97)


@pytest.mark.skipif(not CALIB.exists(), reason="committed calibration missing")
def test_carried_frames_would_displace_the_player_by_metres():
    """The regression itself: with the gate off, the carried rows land far from the truth."""
    cal, true = _calibration_with_carry()
    est = GVHMRPoseEstimator(backend=_StubHMR(), min_calib_confidence=0.0)   # pre-fix behaviour
    motion = est.estimate(_clip(), _tracks(), cal)[7]

    assert motion.pose.frames.shape[0] == N_FRAMES, "gate off must ground every frame"
    uv = np.array([_FOOT_UV[0], _FOOT_UV[1], 1.0])
    truth = np.stack([(true[f] @ uv)[:2] / (true[f] @ uv)[2] for f in range(N_FRAMES)])
    err = np.linalg.norm(motion.pose.transl[:, :2] - truth, axis=1)

    assert err[:FIRST_CARRIED].max() < 1e-6, "solved frames must be exact"
    assert err[-1] > 5.0, f"stale plane should displace the foot metres, got {err[-1]:.2f} m"
    assert est.dropped_frames == 0


@pytest.mark.skipif(not CALIB.exists(), reason="committed calibration missing")
def test_gate_drops_carried_frames_instead_of_placing_them():
    """With the gate on, those frames are absent — not placed wrongly, and not silently absent."""
    cal, _ = _calibration_with_carry()
    est = GVHMRPoseEstimator(backend=_StubHMR())        # default floor
    motion = est.estimate(_clip(), _tracks(), cal)[7]

    assert motion.pose.frames.tolist() == list(range(FIRST_CARRIED))
    assert motion.pose.transl.shape[0] == FIRST_CARRIED
    assert est.dropped_frames == N_FRAMES - FIRST_CARRIED
    assert est.dropped_subjects == []


@pytest.mark.skipif(not CALIB.exists(), reason="committed calibration missing")
def test_subject_with_no_solved_frame_is_reported_not_hidden():
    """A subject whose whole life sits on carried frames cannot be placed at all — say so."""
    cal, _ = _calibration_with_carry()
    u, v = _FOOT_UV
    late = np.arange(FIRST_CARRIED, N_FRAMES)
    boxes = np.tile(np.array([u - 20.0, v - 90.0, u + 20.0, v]), (late.shape[0], 1))
    tracks = Tracks(tracklets=[
        Tracklet(track_id=9, frames=late, bboxes_xyxy=boxes, cls="player")
    ])

    class _Late:
        def estimate_bodies(self, clip, tr):
            n = late.shape[0]
            return {9: RawBodyMotion(track_id=9, frames=late, global_orient=np.zeros((n, 3)),
                                     body_pose=np.zeros((n, _J, 3)), betas=np.zeros(10))}

    est = GVHMRPoseEstimator(backend=_Late())
    out = est.estimate(_clip(), tracks, cal)
    assert out == {}, "no ground plane for any of his frames — inventing one is the bug"
    assert est.dropped_subjects == [9]
