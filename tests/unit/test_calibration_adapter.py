"""FieldCalibrator adapters — shared port contract + DLT/scoring/smoothing (FR-7, M1).

Both real calibrators are exercised with an *injected* stub backend, so their pure halves are
verified with **no model, no cv2, no GPU** — the same AC-7 discipline the fakes follow:

* :class:`KeypointFieldCalibrator` (DLT path) — normalized-DLT homography, RANSAC outlier
  rejection, reprojection scoring, last-good carry, temporal smoothing.
* :class:`CameraModuleFieldCalibrator` (camera-module path) — scores + smooths a backend's
  full-solve homographies, plus the ``cam_params`` → image→world conversion round-trip.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from pitch3d.adapters.fakes import FakeFieldCalibrator
from pitch3d.adapters.models.calibration import (
    CameraModuleFieldCalibrator,
    FrameHomography,
    FrameKeypoints,
    HomographyBackend,
    KeypointBackend,
    KeypointFieldCalibrator,
    PitchKeypointBackend,
    _apply_homography,
    _confidence_from_error,
    _temporal_smooth,
    carry_on_motion,
    image_to_world_from_cam_params,
    point_line_residual,
    probe_pixels,
    reprojection_error,
    solve_homography,
    solve_homography_ransac,
)
from pitch3d.core.ports.io import ClipRef
from pitch3d.core.ports.perception import FieldCalibrator
from pitch3d.core.scene.field import FieldCalibration
from pitch3d.core.scene.pitch import world_line_from_segment

# Ground-truth image→world homography (mild perspective) and a non-degenerate landmark set.
_H_GT = np.array(
    [[0.050, 0.002, -30.0], [0.001, -0.050, 18.0], [2e-4, 1e-4, 1.0]], dtype=float
)
_WORLD = np.array(
    [[-30.0, -15.0], [30.0, -15.0], [30.0, 15.0], [-30.0, 15.0],
     [0.0, 0.0], [12.0, -6.0], [-8.0, 9.0]], dtype=float
)
#: Image points that map back to _WORLD under _H_GT (so solve should recover _H_GT).
_IMAGE = _apply_homography(np.linalg.inv(_H_GT), _WORLD)

#: The clean set plus one gross mislocalisation (image shifted 1000 px, world unchanged) — the
#: dominant real-world failure RANSAC/weighting must reject without dragging the homography.
_IMAGE_OUT = np.vstack([_IMAGE, _IMAGE[0] + [1000.0, 1000.0]])
_WORLD_OUT = np.vstack([_WORLD, _WORLD[0]])


def _clip(frames=(0, 1, 2), width=1280, height=720) -> ClipRef:
    return ClipRef(
        source_id="s", uri="x", frames=np.array(frames), width=width, height=height, fps=25.0
    )


class _StubKeypointBackend:
    """Returns canned (image↔world) landmark matches per frame — stands in for the model."""

    def __init__(self, per: dict[int, tuple], lines: tuple | None = None):
        self.per = per
        self.lines = lines

    def detect_keypoints(self, clip: ClipRef) -> list[FrameKeypoints]:
        out = []
        l_uv, l_abc = self.lines if self.lines else (None, None)
        for f in clip.frames.tolist():
            image_uv, world_xy = self.per[int(f)]
            out.append(FrameKeypoints(frame=int(f), image_uv=image_uv, world_xy=world_xy,
                                      line_uv=l_uv, line_abc=l_abc))
        return out


def _keypoints_calibrator(frames=(0, 1, 2), smooth_window=1) -> KeypointFieldCalibrator:
    per = {int(f): (_IMAGE, _WORLD) for f in frames}
    return KeypointFieldCalibrator(backend=_StubKeypointBackend(per), smooth_window=smooth_window)


class _StubHomographyBackend:
    """Returns canned per-frame :class:`FrameHomography` — stands in for the full camera module."""

    def __init__(self, per: dict[int, FrameHomography]):
        self.per = per

    def calibrate_frames(self, clip: ClipRef) -> list[FrameHomography]:
        return [self.per[int(f)] for f in clip.frames.tolist()]


def _camera_calibrator(frames=(0, 1, 2), smooth_window=1) -> CameraModuleFieldCalibrator:
    per = {
        int(f): FrameHomography(frame=int(f), homography=_H_GT, rep_err_px=0.0, n_landmarks=7)
        for f in frames
    }
    return CameraModuleFieldCalibrator(
        backend=_StubHomographyBackend(per), smooth_window=smooth_window
    )


# --- pure homography maths -----------------------------------------------------
def test_solve_homography_round_trips_known_H():
    h = solve_homography(_IMAGE, _WORLD)
    np.testing.assert_allclose(h, _H_GT, atol=1e-6)
    assert reprojection_error(h, _IMAGE, _WORLD) < 1e-6


def test_reprojection_error_grows_with_noise():
    h = solve_homography(_IMAGE, _WORLD)
    rng = np.random.default_rng(0)
    noisy = _IMAGE + rng.normal(scale=3.0, size=_IMAGE.shape)
    assert reprojection_error(h, noisy, _WORLD) > reprojection_error(h, _IMAGE, _WORLD)


def test_solve_homography_needs_four_points():
    with pytest.raises(ValueError, match="≥4"):
        solve_homography(_IMAGE[:3], _WORLD[:3])


def test_weighted_solve_ignores_zero_weighted_outlier():
    # Zeroing the outlier's weight must recover _H_GT exactly; the unweighted fit is wrecked by it.
    weights = np.ones(_IMAGE_OUT.shape[0])
    weights[-1] = 0.0
    h = solve_homography(_IMAGE_OUT, _WORLD_OUT, weights=weights)
    np.testing.assert_allclose(h, _H_GT, atol=1e-6)
    h_unweighted = solve_homography(_IMAGE_OUT, _WORLD_OUT)
    assert reprojection_error(h_unweighted, _IMAGE, _WORLD) > reprojection_error(h, _IMAGE, _WORLD)


def test_weighted_solve_rejects_mismatched_weight_length():
    with pytest.raises(ValueError, match="one weight per correspondence"):
        solve_homography(_IMAGE, _WORLD, weights=np.ones(_IMAGE.shape[0] - 1))


def test_ransac_rejects_outlier_and_recovers_homography():
    h, inliers = solve_homography_ransac(_IMAGE_OUT, _WORLD_OUT, threshold=1.0, seed=0)
    assert inliers.tolist() == [True] * 7 + [False]  # the planted outlier is the only reject
    np.testing.assert_allclose(h, _H_GT, atol=1e-6)


def test_ransac_four_points_is_a_plain_weighted_solve():
    h, inliers = solve_homography_ransac(_IMAGE[:4], _WORLD[:4])
    assert inliers.tolist() == [True] * 4  # nothing to reject with a minimal set
    np.testing.assert_allclose(h, solve_homography(_IMAGE[:4], _WORLD[:4]), atol=1e-9)


def test_ransac_is_deterministic_for_seed():
    h1, m1 = solve_homography_ransac(_IMAGE_OUT, _WORLD_OUT, seed=7)
    h2, m2 = solve_homography_ransac(_IMAGE_OUT, _WORLD_OUT, seed=7)
    np.testing.assert_array_equal(h1, h2)
    np.testing.assert_array_equal(m1, m2)


# --- point-on-line constraints (R3, #95) ---------------------------------------
#: Six world lines and one image point on each: the evidence PnLCalib's line head produces on a
#: frame whose keypoint *intersections* are mostly off-screen or occluded.
_LINE_ABC = np.array([
    world_line_from_segment(*seg)
    for seg in (
        ([-52.5, -34.0], [52.5, -34.0]),   # top touchline
        ([-52.5, 34.0], [52.5, 34.0]),     # bottom touchline
        ([0.0, -34.0], [0.0, 34.0]),       # halfway
        ([-36.0, -20.16], [-36.0, 20.16]), # left penalty box, main
        ([36.0, -20.16], [36.0, 20.16]),   # right penalty box, main
        ([-52.5, -34.0], [-52.5, 34.0]),   # left goal line
    )
])
#: A world point on each of those lines, then pulled back into the image through _H_GT.
_LINE_WORLD = np.array([
    [10.0, -34.0], [-20.0, 34.0], [0.0, 7.0], [-36.0, -5.0], [36.0, 12.0], [-52.5, -2.0],
])
_LINE_UV = _apply_homography(np.linalg.inv(_H_GT), _LINE_WORLD)


def test_line_observations_are_exactly_on_their_lines():
    resid = point_line_residual(_H_GT, _LINE_UV, _LINE_ABC)
    np.testing.assert_allclose(resid, 0.0, atol=1e-9)


def test_two_points_plus_six_lines_recover_the_homography():
    # 2*2 + 6 = 10 DLT rows. Points alone (4 rows) cannot solve this at all.
    h = solve_homography(_IMAGE[:2], _WORLD[:2], line_uv=_LINE_UV, line_abc=_LINE_ABC)
    np.testing.assert_allclose(h, _H_GT, atol=1e-9)


def test_lines_do_not_perturb_an_already_determined_fit():
    h = solve_homography(_IMAGE, _WORLD, line_uv=_LINE_UV, line_abc=_LINE_ABC)
    np.testing.assert_allclose(h, solve_homography(_IMAGE, _WORLD), atol=1e-9)


def test_under_determined_row_count_is_reported():
    with pytest.raises(ValueError, match="7 DLT rows"):  # 2 points + 3 lines
        solve_homography(_IMAGE[:2], _WORLD[:2], line_uv=_LINE_UV[:3], line_abc=_LINE_ABC[:3])


def test_line_uv_and_abc_must_come_together():
    with pytest.raises(ValueError, match="together"):
        solve_homography(_IMAGE, _WORLD, line_uv=_LINE_UV)


def test_line_weights_must_match_observation_count():
    with pytest.raises(ValueError, match="one weight per line"):
        solve_homography(
            _IMAGE, _WORLD, line_uv=_LINE_UV, line_abc=_LINE_ABC, line_weights=np.ones(3)
        )


def test_ransac_drops_a_mislabelled_line():
    # One observation tagged with the wrong pitch line — the failure mode a line *classifier* has,
    # as opposed to the mislocalisation a keypoint head has.
    bad_abc = np.vstack([_LINE_ABC, _LINE_ABC[0]])
    bad_uv = np.vstack([_LINE_UV, _LINE_UV[2]])
    h, _ = solve_homography_ransac(
        _IMAGE, _WORLD, threshold=1.0, line_uv=bad_uv, line_abc=bad_abc
    )
    np.testing.assert_allclose(h, _H_GT, atol=1e-6)


def test_frame_keypoints_validates_and_counts_lines():
    fk = FrameKeypoints(
        frame=0, image_uv=_IMAGE, world_xy=_WORLD, line_uv=_LINE_UV, line_abc=_LINE_ABC
    )
    assert fk.n_lines == 6
    np.testing.assert_allclose(fk.line_confidence, 1.0)  # defaults filled like keypoint confidence
    assert FrameKeypoints(frame=0, image_uv=_IMAGE, world_xy=_WORLD).n_lines == 0
    with pytest.raises(ValueError, match="ragged line observations"):
        FrameKeypoints(
            frame=3, image_uv=_IMAGE, world_xy=_WORLD, line_uv=_LINE_UV, line_abc=_LINE_ABC[:4]
        )


def test_calibrator_solves_a_thin_frame_only_when_lines_are_present():
    """3 keypoints is under-determined on points alone; the line head rescues the frame (R-6)."""
    thin = (_IMAGE[:3], _WORLD[:3])
    without = KeypointFieldCalibrator(
        backend=_StubKeypointBackend({0: thin}), smooth_window=1
    ).calibrate(_clip(frames=(0,)))
    assert without.confidence[0] == 0.0  # carried, honestly flagged as unsolved
    np.testing.assert_allclose(without.homographies[0], np.eye(3))

    with_lines = KeypointFieldCalibrator(
        backend=_StubKeypointBackend({0: thin}, lines=(_LINE_UV, _LINE_ABC)), smooth_window=1
    ).calibrate(_clip(frames=(0,)))
    assert with_lines.confidence[0] > 0.5
    np.testing.assert_allclose(with_lines.homographies[0], _H_GT, atol=1e-6)


def test_calibrator_confidence_is_scored_on_the_lines_that_made_it_solvable():
    """A thin frame must not report false confidence just because 3 points reproject exactly."""
    off = _LINE_UV + np.array([40.0, 40.0])  # line detections that disagree with the keypoints
    good = KeypointFieldCalibrator(
        backend=_StubKeypointBackend({0: (_IMAGE[:3], _WORLD[:3])}, lines=(_LINE_UV, _LINE_ABC))
    ).calibrate(_clip(frames=(0,)))
    bad = KeypointFieldCalibrator(
        backend=_StubKeypointBackend({0: (_IMAGE[:3], _WORLD[:3])}, lines=(off, _LINE_ABC))
    ).calibrate(_clip(frames=(0,)))
    assert bad.confidence[0] < good.confidence[0]


def test_confidence_is_zero_when_the_fit_has_no_redundancy_to_verify_it():
    """A 4-point fit reproduces its own points exactly, so a residual cannot judge it (#105).

    The frame is admitted (2·4 rows == the evidence floor) and its residual is *identically* zero
    however wrong the homography is, so scoring by mean residual made the least-supported frames
    the most confident ones. Normalising by residual degrees of freedom — ``rows − 8`` for a
    homography's 8 DOF — makes that frame report the honest answer: unverifiable, confidence 0.
    """
    skew = np.array([[18.0, -11.0], [-14.0, 16.0], [21.0, 13.0], [-9.0, -19.0]])
    bad_uv = _IMAGE[:4] + skew
    result = KeypointFieldCalibrator(
        backend=_StubKeypointBackend({0: (bad_uv, _WORLD[:4])})
    ).calibrate(_clip(frames=(0,)))

    # The fit really is wrong — the true image points now land metres from their world points …
    off = np.linalg.norm(result.image_to_world(0, _IMAGE[:4]) - _WORLD[:4], axis=1)
    assert off.max() > 1.0
    # … yet its own residual is exactly zero, which is why a count-normalised score read ~1.0.
    assert reprojection_error(result.homographies[0], bad_uv, _WORLD[:4]) == pytest.approx(0.0)
    assert result.confidence[0] == 0.0

    # The correction must not punish frames that *are* over-determined: 7 clean landmarks give
    # 14 rows against 8 DOF, and still score ~1.
    ok = _keypoints_calibrator(frames=(0,)).calibrate(_clip(frames=(0,)))
    assert ok.confidence[0] > 0.99


def test_confidence_decreases_with_error():
    assert _confidence_from_error(0.0, 0.5) == 1.0
    assert _confidence_from_error(0.5, 0.5) == pytest.approx(0.5)
    assert _confidence_from_error(2.0, 0.5) < _confidence_from_error(0.5, 0.5)


def test_temporal_smoothing_box_averages_window():
    a, b, c = (np.eye(3) for _ in range(3))
    a[0, 2], b[0, 2], c[0, 2] = 0.0, 3.0, 0.0
    sm = _temporal_smooth(np.stack([a, b, c]), 3)
    assert sm[1][0, 2] == pytest.approx(1.0)   # mean(0, 3, 0)
    assert sm[0][0, 2] == pytest.approx(1.5)   # endpoint clamps to mean(0, 3)


# --- camera propagation (R2 / #94) ---------------------------------------------
#: A smoothly panning tripod camera: one constant inter-frame pixel homography.
_PAN = np.array([[1.0, 0.0, 9.0], [0.0, 1.0, 1.5], [0.0, 0.0, 1.0]])


def _swimming_track(n=25, sigma=0.3, seed=0):
    """A true panning camera + the independently noisy per-frame solve of it.

    Frame ``k``'s pixels reach frame 0 through ``_PAN**-k``, so the true image→world homography is
    ``_H_GT @ _PAN**-k``. Each frame's *measured* homography is then re-fitted from probe world
    points corrupted independently — which is what the real per-frame calibration does (#104).
    """
    probe = probe_pixels(1920, 1080)
    rng = np.random.default_rng(seed)
    motion = np.stack([_PAN] * (n - 1))
    truth, noisy = [], []
    step = np.linalg.inv(_PAN)
    back = np.eye(3)
    for _ in range(n):
        h = _H_GT @ back
        truth.append(h)
        seen = _apply_homography(h, probe) + rng.normal(0, sigma, (5, 2))
        noisy.append(solve_homography(probe, seen))
        back = back @ step
    return np.stack(truth), np.stack(noisy), motion, probe


def _truth_error(track, truth, probe):
    """Median metres between where a track and the truth put the probe pixels on the pitch."""
    return float(np.median([
        np.median(np.linalg.norm(_apply_homography(t, probe) - _apply_homography(g, probe), axis=1))
        for t, g in zip(track, truth, strict=True)
    ]))


def test_carrying_on_measured_motion_moves_the_track_toward_the_truth():
    """Carrying must improve ACCURACY, not merely smoothness (#94).

    The swim metric R2 was designed against is circular — a frozen camera scores a perfect 0.0 m of
    frame-to-frame disagreement while being metres wrong (#104). So this scores against the known
    true homography instead, and carries the frozen candidate along as the control it must beat.
    """
    truth, noisy, motion, probe = _swimming_track()
    carried = carry_on_motion(noisy, motion, window=8, probe_uv=probe)

    per_frame = _truth_error(noisy, truth, probe)
    assert _truth_error(carried, truth, probe) < 0.5 * per_frame

    # The control: a frozen track is perfectly steady and badly wrong. If the assertion above were
    # really rewarding smoothness, this would pass it too — it must not.
    frozen = np.stack([noisy[0]] * len(noisy))
    assert _truth_error(frozen, truth, probe) > per_frame


def test_carrying_is_off_by_default_and_declines_impossible_input():
    truth, noisy, motion, probe = _swimming_track(n=4)
    assert carry_on_motion(noisy, motion, window=0, probe_uv=probe) is noisy
    with pytest.raises(ValueError, match="need 3 inter-frame motions"):
        carry_on_motion(noisy, motion[:1], window=2, probe_uv=probe)


def test_probe_points_track_the_frame_size():
    assert probe_pixels(1920, 1080).shape == (5, 2)
    assert probe_pixels(1280, 720) * 1.5 == pytest.approx(probe_pixels(1920, 1080))
    # Players' feet: the probe sits in the lower half of the frame, never at the horizon.
    assert (probe_pixels(1920, 1080)[:, 1] > 540).all()


def test_calibrator_carries_only_when_a_motion_source_is_injected():
    class _StubMotion:
        def frame_motion(self, clip):
            return np.stack([_PAN] * (clip.n_frames - 1))

    frames = tuple(range(6))
    plain = _keypoints_calibrator(frames=frames).calibrate(_clip(frames=frames))
    assert plain.homographies.shape == (6, 3, 3)
    assert KeypointFieldCalibrator(motion=None).info().params["carry_window"] == 0

    carried = KeypointFieldCalibrator(
        backend=_StubKeypointBackend({f: (_IMAGE, _WORLD) for f in frames}),
        motion=_StubMotion(),
        carry_window=2,
    ).calibrate(_clip(frames=frames))
    assert carried.homographies.shape == (6, 3, 3)
    # Every frame saw identical clean landmarks, so each frame's own solve is already _H_GT and the
    # carry has nothing to correct — but it must have run, and must not have damaged the track.
    assert _apply_homography(carried.homographies[3], _IMAGE) == pytest.approx(_WORLD, abs=1e-6)


# --- adapter behaviour ---------------------------------------------------------
@pytest.mark.parametrize(
    "make", [FakeFieldCalibrator, _keypoints_calibrator, _camera_calibrator],
    ids=["fake", "keypoints", "camera"],
)
def test_calibrator_port_contract(make):
    cal = make()
    assert isinstance(cal, FieldCalibrator)
    clip = _clip()
    result = cal.calibrate(clip)
    assert isinstance(result, FieldCalibration)
    assert result.homographies.shape == (clip.n_frames, 3, 3)
    assert result.frames.tolist() == clip.frames.tolist()
    assert result.confidence.shape == (clip.n_frames,)
    assert np.all((result.confidence >= 0.0) & (result.confidence <= 1.0))


def test_keypoints_recovers_world_points_and_scores_high():
    result = _keypoints_calibrator(frames=(0,)).calibrate(_clip(frames=(0,)))
    np.testing.assert_allclose(result.image_to_world(0, _IMAGE), _WORLD, atol=1e-6)
    assert result.confidence[0] > 0.99  # exact fit → near-1 confidence


def test_calibrate_rejects_outlier_landmark_and_downweights_confidence():
    per = {0: (_IMAGE_OUT, _WORLD_OUT)}
    cal = KeypointFieldCalibrator(backend=_StubKeypointBackend(per))
    result = cal.calibrate(_clip(frames=(0,)))
    # RANSAC rejects the planted outlier, so the clean points still recover _WORLD exactly …
    np.testing.assert_allclose(result.image_to_world(0, _IMAGE), _WORLD, atol=1e-6)
    # … but confidence is below the all-clean ~1.0, scaled by the 7/8 inlier fraction (R-6).
    assert 0.8 < result.confidence[0] < 0.99


def test_under_detected_frame_carries_last_good_at_zero_confidence():
    per = {0: (_IMAGE, _WORLD), 1: (_IMAGE[:2], _WORLD[:2])}  # frame 1 has only 2 landmarks
    cal = KeypointFieldCalibrator(backend=_StubKeypointBackend(per))
    result = cal.calibrate(_clip(frames=(0, 1)))
    np.testing.assert_allclose(result.homographies[1], result.homographies[0], atol=1e-12)
    assert result.confidence[0] > 0.99 and result.confidence[1] == 0.0


def test_first_frame_under_detected_falls_back_to_identity():
    per = {0: (_IMAGE[:2], _WORLD[:2])}
    cal = KeypointFieldCalibrator(backend=_StubKeypointBackend(per))
    result = cal.calibrate(_clip(frames=(0,)))
    np.testing.assert_allclose(result.homographies[0], np.eye(3))
    assert result.confidence[0] == 0.0


def test_keypoint_provenance():
    info = KeypointFieldCalibrator(smooth_window=5).info()
    assert info.name == "PitchKeypoints+DLT"
    assert info.backend.value == "local"
    assert info.params["smooth_window"] == 5


def test_backends_satisfy_protocol():
    assert isinstance(_StubKeypointBackend({}), KeypointBackend)
    assert isinstance(PitchKeypointBackend(), KeypointBackend)  # structural: has detect_keypoints


def test_frame_keypoints_rejects_ragged():
    with pytest.raises(ValueError, match="ragged"):
        FrameKeypoints(frame=0, image_uv=np.zeros((3, 2)), world_xy=np.zeros((2, 2)))


# --- camera-module calibrator (full PnLCalib solve) ----------------------------
def _cam_params(fx=1400.0, fy=1350.0, px=480.0, py=270.0):
    """A valid pinhole camera (orthonormal tilt, elevated position) → PnLCalib cam_params."""
    theta = 1.2  # tilt about X so the camera looks down at the pitch (ground homography invertible)
    rotation = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(theta), -np.sin(theta)],
         [0.0, np.sin(theta), np.cos(theta)]]
    )
    return {
        "x_focal_length": fx,
        "y_focal_length": fy,
        "principal_point": [px, py],
        "position_meters": [3.0, -10.0, -20.0],
        "rotation_matrix": rotation.tolist(),
    }


def test_image_to_world_from_cam_params_round_trips_ground_plane():
    # Build the forward projection P exactly like PnLCalib's projection_from_cam_params, project
    # centre-origin ground points to image, and assert the converted H recovers world from image.
    cam_params = _cam_params()
    it = np.eye(4)[:-1]
    it[:, -1] = -np.asarray(cam_params["position_meters"])
    k = np.array([[cam_params["x_focal_length"], 0.0, cam_params["principal_point"][0]],
                  [0.0, cam_params["y_focal_length"], cam_params["principal_point"][1]],
                  [0.0, 0.0, 1.0]])
    p = k @ (np.asarray(cam_params["rotation_matrix"]) @ it)
    world = np.array([[0.0, 0.0], [20.0, 10.0], [-15.0, 8.0], [5.0, -12.0], [30.0, 25.0]])
    img = np.array([(p @ [x, y, 0.0, 1.0])[:2] / (p @ [x, y, 0.0, 1.0])[2] for x, y in world])

    h = image_to_world_from_cam_params(cam_params)
    np.testing.assert_allclose(_apply_homography(h, img), world, atol=1e-6)
    assert h[2, 2] == pytest.approx(1.0)  # normalised


def test_image_to_world_from_cam_params_raises_on_degenerate_camera():
    # A camera whose third row is collinear with the plane → singular ground homography.
    bad = _cam_params()
    bad["rotation_matrix"] = np.zeros((3, 3)).tolist()
    with pytest.raises(np.linalg.LinAlgError):
        image_to_world_from_cam_params(bad)


def test_camera_solved_frame_recovers_world_and_scores_high():
    result = _camera_calibrator(frames=(0,)).calibrate(_clip(frames=(0,)))
    np.testing.assert_allclose(result.image_to_world(0, _IMAGE), _WORLD, atol=1e-6)
    assert result.confidence[0] > 0.99  # rep_err 0 px → near-1 confidence


def test_camera_rep_err_scales_confidence():
    # conf_scale_px maps the solve's pixel reprojection error to confidence: 0→1, scale→0.5.
    per = {0: FrameHomography(frame=0, homography=_H_GT, rep_err_px=5.0)}
    cal = CameraModuleFieldCalibrator(backend=_StubHomographyBackend(per), conf_scale_px=5.0)
    result = cal.calibrate(_clip(frames=(0,)))
    assert result.confidence[0] == pytest.approx(0.5)


def test_camera_unsolved_frame_carries_last_good_at_zero_confidence():
    per = {
        0: FrameHomography(frame=0, homography=_H_GT, rep_err_px=0.0),
        1: FrameHomography(frame=1, homography=None),  # solve failed for this view
    }
    cal = CameraModuleFieldCalibrator(backend=_StubHomographyBackend(per))
    result = cal.calibrate(_clip(frames=(0, 1)))
    np.testing.assert_allclose(result.homographies[1], result.homographies[0], atol=1e-12)
    assert result.confidence[0] > 0.99 and result.confidence[1] == 0.0


def test_camera_first_frame_unsolved_falls_back_to_identity():
    per = {0: FrameHomography(frame=0, homography=None)}
    cal = CameraModuleFieldCalibrator(backend=_StubHomographyBackend(per))
    result = cal.calibrate(_clip(frames=(0,)))
    np.testing.assert_allclose(result.homographies[0], np.eye(3))
    assert result.confidence[0] == 0.0


def test_camera_missing_backend_is_actionable():
    with pytest.raises(ValueError, match="HomographyBackend"):
        CameraModuleFieldCalibrator().calibrate(_clip(frames=(0,)))


def test_camera_provenance():
    info = CameraModuleFieldCalibrator(smooth_window=3, conf_scale_px=5.0).info()
    assert info.name == "PnLCalib-Camera"
    assert info.backend.value == "local"
    assert info.params["smooth_window"] == 3
    assert info.params["conf_scale_px"] == 5.0


def test_homography_backend_satisfies_protocol():
    assert isinstance(_StubHomographyBackend({}), HomographyBackend)


def test_frame_homography_reshapes_and_allows_none():
    fh = FrameHomography(frame=0, homography=np.arange(9, dtype=float))
    assert fh.homography.shape == (3, 3)
    assert FrameHomography(frame=1, homography=None).homography is None


@pytest.mark.skipif(
    importlib.util.find_spec("cv2") is not None, reason="cv extra installed"
)
def test_default_backend_without_extra_is_actionable():
    # No backend injected and the `cv` extra absent → a clear, install-pointing error.
    with pytest.raises(RuntimeError, match=r"pitch3d\[cv\]"):
        KeypointFieldCalibrator().calibrate(_clip(frames=(0,)))
