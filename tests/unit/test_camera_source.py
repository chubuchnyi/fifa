"""A synthetic camera must be distinguishable from a solve, on disk (#140).

`controller.run_reconstruction` ends with `scene.camera = _measured_camera(...) or
_static_camera(...)`. The refusal on the left is correct and deliberate (#61: a scene once carried
two cameras 12686 px apart for months). The substitution on the right used to be **silent**, and
`PlaneCameraFit` — the only record of which one you got — lived in memory and was never serialized.

Measured 2026-08-08: **nine of nine scenes on disk** carried the fallback at `fx = 772.02 @
1280x720`, against a 1920x1080 clip whose real focal is ~4200. Among them
`out/cue/scene_off.json`, the scene the 24 #135 eye labels were judged on. Every comparison of one
of those scenes to the source pixels was reading a camera 5.4x wrong in focal.

That is a plain R-6 violation — "mark, never hide" is the rule applied to phantom players, and it
was not applied to the camera. These tests pin the mark.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.scene.camera import CameraIntrinsics, CameraSource, CameraTrack
from pitch3d.core.scene.serialization import from_json, to_json


def _track(**kw) -> CameraTrack:
    return CameraTrack(
        intrinsics=CameraIntrinsics(fx=4169.3, fy=4169.3, cx=960, cy=540, width=1920, height=1080),
        frames=np.arange(4),
        rotation_quat=np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
        translation=np.zeros((4, 3)),
        **kw,
    )


def test_a_fallback_camera_says_so_and_a_solve_says_so():
    assert _track().is_measured is True
    assert _track(source=CameraSource.STATIC_FALLBACK).is_measured is False
    assert _track(source=CameraSource.PRESCRIBED).is_measured is False


def test_the_mark_survives_a_round_trip_through_scene_json():
    """The whole point: it has to be on DISK. In memory it already existed and did not help."""
    cam = _track(
        source=CameraSource.STATIC_FALLBACK, fit_reprojection_px=471.1, fit_focal_px=4340.8
    )
    back = from_json(to_json(cam))
    assert back.source is CameraSource.STATIC_FALLBACK
    assert back.is_measured is False
    assert back.fit_reprojection_px == pytest.approx(471.1)
    assert back.fit_focal_px == pytest.approx(4340.8)


def test_the_refused_fit_numbers_are_kept_not_dropped():
    """A sane focal with a huge reprojection is a different diagnosis from a wild focal.

    On `f236_res896` the fit recovered 4340.8 px — within 4 % of the golden 4169.32 — and still
    reprojected at 471 px. Recording only "refused" would have hidden that the *scale* was right
    and the *consistency* was not, which is what pointed at the homography tail.
    """
    cam = _track(
        source=CameraSource.STATIC_FALLBACK, fit_reprojection_px=471.1, fit_focal_px=4340.8
    )
    assert cam.fit_focal_px is not None and cam.fit_reprojection_px is not None


def test_an_old_scene_without_the_field_still_decodes():
    """Scenes written before #140 have no `source` key; they must not fail to load."""
    import json

    payload = json.loads(to_json(_track()))
    for key in ("source", "fit_reprojection_px", "fit_focal_px"):
        payload["fields"].pop(key, None)
    back = from_json(json.dumps(payload))
    assert back.source is CameraSource.PLANE_FIT       # the default
    assert back.fit_reprojection_px is None


def test_the_enum_values_are_stable_strings():
    """They land in scene.json; renaming one silently invalidates every scene on disk."""
    assert CameraSource.PLANE_FIT.value == "plane_fit"
    assert CameraSource.STATIC_FALLBACK.value == "static_fallback"
    assert CameraSource.PRESCRIBED.value == "prescribed"
