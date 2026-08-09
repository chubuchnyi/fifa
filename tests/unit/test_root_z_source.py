"""A constant stand-in height must not look like a measurement (#142).

`GVHMRPoseEstimator._ground_root` fills root Z with the nominal `pelvis_height_m` whenever the
backend returns no `pelvis_above_foot`. That is legitimate — a pose backend without SMPL-X forward
kinematics cannot know how far the pelvis sits above the foot — but it was **silent**, and a
constant is not a small excursion, it is no excursion.

Measured 2026-08-09: `out/cue/scene_off.json`, the scene the 24 #135 eye labels were judged on,
has **6 of 24 subjects at exactly Z = 0.92 m**, `std(z) = 0`. They drag the scene's median root-Z
range to 0.008 m, which is where "we have no vertical degree of freedom" came from — a claim that
rode four revisions of the plan. At a matched window the GT median is 0.084 m and our current
scene is 0.160 m, so the conclusion reversed once the constants were visible.

Same shape as #140: a substitution indistinguishable from a measurement downstream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pitch3d.adapters.models.pose import GVHMRPoseEstimator, RawBodyMotion
from pitch3d.core.ports.io import ClipRef
from pitch3d.core.ports.perception import Tracklet, Tracks
from pitch3d.core.scene.field import FieldCalibration
from pitch3d.core.scene.motion import PoseSequence, RootZSource
from pitch3d.core.scene.serialization import from_json, to_json


def _pose(**kw) -> PoseSequence:
    return PoseSequence(
        frames=np.arange(4),
        global_orient=np.zeros((4, 3)),
        body_pose=np.zeros((4, 2, 3)),
        transl=np.zeros((4, 3)),
        **kw,
    )


def test_measured_is_the_default_so_an_old_scene_is_not_slandered():
    """Scenes written before #142 carry no field; they must not be reported as invented."""
    assert _pose().root_z_source is RootZSource.MEASURED


def test_a_nominal_height_says_so():
    assert _pose(root_z_source=RootZSource.NOMINAL).root_z_source is RootZSource.NOMINAL


def test_the_mark_survives_scene_json():
    """It has to be on disk. In memory the pose estimator already knew and nobody could read it."""
    back = from_json(to_json(_pose(root_z_source=RootZSource.NOMINAL)))
    assert back.root_z_source is RootZSource.NOMINAL


def test_an_old_scene_without_the_field_still_decodes():
    import json

    payload = json.loads(to_json(_pose()))
    payload["fields"].pop("root_z_source", None)
    assert from_json(json.dumps(payload)).root_z_source is RootZSource.MEASURED


def test_the_enum_values_are_stable_strings():
    """They land in scene.json; renaming one invalidates every scene on disk."""
    assert RootZSource.MEASURED.value == "measured"
    assert RootZSource.NOMINAL.value == "nominal"


# --- the estimator end to end, with a backend that returns no height ---------------------

_CALIB = Path(__file__).resolve().parents[2] / "calib" / "Colombia-1-0-Congo-DR1080p.npz"
_N = 20


def _real_calibration() -> FieldCalibration:
    blob = np.load(_CALIB, allow_pickle=True)
    h = np.stack([np.linalg.inv(m) for m in np.asarray(blob["world_to_image"])])[:_N]
    return FieldCalibration(homographies=h, frames=np.arange(_N), confidence=np.full(_N, 0.55))


def _tracks() -> Tracks:
    box = np.tile(np.array([940.0, 610.0, 980.0, 700.0]), (_N, 1))
    return Tracks(tracklets=[
        Tracklet(track_id=7, frames=np.arange(_N), bboxes_xyxy=box, cls="player")
    ])


class _NoHeight:
    """A backend without SMPL-X FK — exactly the case that triggers the stand-in."""

    def estimate_bodies(self, clip, tracks):
        return {7: RawBodyMotion(
            track_id=7, frames=np.arange(_N), global_orient=np.zeros((_N, 3)),
            body_pose=np.zeros((_N, 21, 3)), betas=np.zeros(10),
        )}


class _WithHeight(_NoHeight):
    def estimate_bodies(self, clip, tracks):
        raw = super().estimate_bodies(clip, tracks)[7]
        raw.pelvis_above_foot = 0.92 + 0.05 * np.sin(np.arange(_N) / 3.0)
        return {7: raw}


def _clip():
    return ClipRef(source_id="s", uri="x", frames=np.arange(_N),
                   width=1920, height=1080, fps=29.97)


@pytest.mark.skipif(not _CALIB.exists(), reason="committed calibration missing")
def test_the_estimator_marks_the_track_and_lists_it():
    """The mutation that deletes the bookkeeping must fail here — a fresh object proves nothing."""
    est = GVHMRPoseEstimator(backend=_NoHeight())
    out = est.estimate(_clip(), _tracks(), _real_calibration())

    assert out[7].pose.root_z_source is RootZSource.NOMINAL
    assert est.nominal_root_z == [7], "the report must name the track, not just count it"
    z = out[7].pose.transl[:, 2]
    assert float(np.std(z)) < 1e-9, "the stand-in is a constant, which is the whole problem"


@pytest.mark.skipif(not _CALIB.exists(), reason="committed calibration missing")
def test_a_backend_that_measures_height_is_not_marked():
    """A guard that flagged everything would be as useless as one that flagged nothing."""
    est = GVHMRPoseEstimator(backend=_WithHeight())
    out = est.estimate(_clip(), _tracks(), _real_calibration())

    assert out[7].pose.root_z_source is RootZSource.MEASURED
    assert est.nominal_root_z == []
    assert float(np.std(out[7].pose.transl[:, 2])) > 1e-6


def test_a_constant_z_is_detectable_the_way_the_register_says():
    """The check published in `docs/findings/landmines.md` must actually find one."""
    z_const = np.full(10, 0.92)
    z_real = 0.92 + np.array([0.0, 0.03, 0.11, 0.05, 0.0, 0.02, 0.09, 0.04, 0.0, 0.01])
    assert float(np.std(z_const)) < 1e-9
    assert float(np.std(z_real)) > 1e-9


def test_a_nominal_track_has_zero_range_which_is_why_it_poisons_aggregates():
    """Not a small number — exactly zero. An average over these measures the stand-in."""
    p = _pose(root_z_source=RootZSource.NOMINAL)
    p.transl[:, 2] = 0.92
    z = p.transl[:, 2]
    assert float(z.max() - z.min()) == pytest.approx(0.0)


# --- the FK fallback: the capability that existed and did not reach this stage (#141) -------

@pytest.mark.skipif(not _CALIB.exists(), reason="committed calibration missing")
def test_the_fk_provider_replaces_the_constant_and_the_track_is_marked_measured():
    """The fix. A backend with no height plus a provider gives a MEASURED, varying Z."""
    def provider(betas, global_orient, body_pose):
        n = np.asarray(global_orient).reshape(-1, 3).shape[0]
        return 0.90 + 0.04 * np.sin(np.arange(n) / 2.0)      # what SMPL-X FK would return

    est = GVHMRPoseEstimator(backend=_NoHeight(), pelvis_height_provider=provider)
    out = est.estimate(_clip(), _tracks(), _real_calibration())

    assert out[7].pose.root_z_source is RootZSource.MEASURED
    assert est.nominal_root_z == [], "nothing was substituted, so nothing should be flagged"
    assert est.fk_root_z == [7], "and the FK path must say it ran"
    assert float(np.std(out[7].pose.transl[:, 2])) > 1e-6


@pytest.mark.skipif(not _CALIB.exists(), reason="committed calibration missing")
def test_a_provider_that_declines_falls_back_to_the_constant_and_says_so():
    """No SMPL-X model on the box is the common case; it must degrade, not crash or lie."""
    est = GVHMRPoseEstimator(backend=_NoHeight(), pelvis_height_provider=lambda *_: None)
    out = est.estimate(_clip(), _tracks(), _real_calibration())

    assert out[7].pose.root_z_source is RootZSource.NOMINAL
    assert est.nominal_root_z == [7] and est.fk_root_z == []


@pytest.mark.skipif(not _CALIB.exists(), reason="committed calibration missing")
def test_a_provider_returning_the_wrong_length_is_ignored_rather_than_trusted():
    """A silent shape mismatch would misalign height against frames — refuse it."""
    est = GVHMRPoseEstimator(backend=_NoHeight(),
                             pelvis_height_provider=lambda *_: np.full(3, 0.9))
    out = est.estimate(_clip(), _tracks(), _real_calibration())

    assert out[7].pose.root_z_source is RootZSource.NOMINAL
    assert est.fk_root_z == []


@pytest.mark.skipif(not _CALIB.exists(), reason="committed calibration missing")
def test_a_backend_that_reports_height_does_not_call_the_provider():
    """FK is the expensive path; it must not run when the backend already answered."""
    calls = []

    def provider(*a):
        calls.append(1)
        return None

    est = GVHMRPoseEstimator(backend=_WithHeight(), pelvis_height_provider=provider)
    est.estimate(_clip(), _tracks(), _real_calibration())
    assert calls == [], "the provider ran even though the backend reported a height"
