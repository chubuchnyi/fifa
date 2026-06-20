"""Blender adapter — pure proxy-plan maths (always) + real subprocess build/render (gated).

The pure half (:mod:`pitch3d.adapters.blender.proxy`) turns a resolved scene into a serializable
:class:`ProxyPlan`; it is unit-tested with **no Blender** and must never import ``bpy``. The
subprocess half (:class:`BlenderProxyBuilder` / :class:`BlenderSceneObserver`) is exercised against
a real binary when one is present (``$PITCH3D_BLENDER`` or ``blender`` on ``PATH``); otherwise those
tests skip. A forced-absent binary must raise an actionable error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from pitch3d.adapters.blender import (
    BlenderProxyBuilder,
    BlenderSceneObserver,
    build_proxy_plan,
)
from pitch3d.adapters.blender.proxy import (
    _BALL_RGB,
    _PLAYER_RGB,
    _REFEREE_RGB,
    _subject_rgb,
    _view_from_camera,
    camera_eye_target,
    load_plan,
    plan_to_json,
    write_plan,
)
from pitch3d.adapters.blender.runner import locate_blender, run_blender
from pitch3d.core.agent.viewpoints import standard_viewpoints
from pitch3d.core.correction.engine import make_offset, resolve_subject_motion
from pitch3d.core.ports.observation import ObservationKind, Viewpoint, ViewpointCamera
from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack
from pitch3d.core.scene.layers import CorrectionTarget, TargetKind
from pitch3d.core.scene.motion import BallTrack
from pitch3d.core.scene.subject import Role, Subject, Team

BLENDER = locate_blender()
needs_blender = pytest.mark.skipif(
    BLENDER is None, reason="Blender binary not found ($PITCH3D_BLENDER or 'blender' on PATH)"
)
_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _viewpoint_camera(*, fx=500.0, width=1000, height=600, frame=7) -> ViewpointCamera:
    """A single-frame world→camera at the origin looking +Z, with known intrinsics."""
    intr = CameraIntrinsics(
        fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0, width=width, height=height
    )
    cam = CameraTrack(
        intrinsics=intr, frames=np.array([frame]),
        rotation_quat=np.array([[1.0, 0.0, 0.0, 0.0]]), translation=np.array([[0.0, 0.0, 0.0]]),
    )
    return ViewpointCamera(viewpoint=Viewpoint.FRONT, camera=cam)


# --- pure plan assembly --------------------------------------------------------------


def test_plan_has_one_object_per_subject_and_a_ball(make_scene, make_motion):
    subjects = [Subject(track_id=t, proposal=make_motion([0, 1, 2])) for t in (3, 7)]
    ball = BallTrack(
        frames=np.arange(3), positions_3d=np.zeros((3, 3)), height_confidence=np.ones(3)
    )
    plan = build_proxy_plan(make_scene(subjects=subjects, ball=ball))
    names = [o.name for o in plan.objects]
    assert names == ["subject_3", "subject_7", "ball"]
    assert [o.kind for o in plan.objects] == ["subject", "subject", "ball"]


def test_subject_object_carries_root_pose_and_shape_channels(make_scene, make_motion):
    subj = Subject(track_id=1, proposal=make_motion([0, 1, 2, 3], n_betas=10))
    plan = build_proxy_plan(make_scene(subjects=[subj]), include_pose=True)
    obj = plan.objects[0]
    assert obj.location.shape == (4, 3)         # root translation F-curve
    assert obj.rotation_aa.shape == (4, 3)      # root orientation F-curve (axis-angle)
    assert obj.betas.shape == (10,)             # β identity channel
    assert obj.body_pose.shape[0] == 4 and obj.body_pose.shape[2] == 3  # (T, J, 3) per-joint pose


def test_include_pose_false_drops_the_body_pose_channel(make_scene, make_motion):
    subj = Subject(track_id=1, proposal=make_motion([0, 1]))
    obj = build_proxy_plan(make_scene(subjects=[subj]), include_pose=False).objects[0]
    assert obj.body_pose is None
    assert obj.rotation_aa is not None and obj.betas is not None  # root + shape still present


def test_ball_object_has_no_subject_channels(make_scene):
    ball = BallTrack(
        frames=np.arange(2), positions_3d=np.zeros((2, 3)), height_confidence=np.ones(2)
    )
    obj = build_proxy_plan(make_scene(ball=ball)).objects[0]
    assert obj.kind == "ball"
    assert obj.rotation_aa is None and obj.betas is None and obj.body_pose is None


def test_subject_corrections_are_baked_into_the_plan(make_scene, make_motion):
    subj = Subject(track_id=3, proposal=make_motion([0, 1, 2, 3], transl_z=1.0))
    target = CorrectionTarget(kind=TargetKind.ROOT_TRANSLATION, subject_track_id=3)
    corr = make_offset("c", target, (0, 1), np.array([0.0, 0.0, 0.5]))
    scene = make_scene(subjects=[subj], corrections=[corr])

    obj = build_proxy_plan(scene).objects[0]
    expected = resolve_subject_motion(subj.proposal, scene.corrections_for(3)).pose.transl
    np.testing.assert_allclose(obj.location, expected)
    np.testing.assert_allclose(obj.location[:2, 2], 1.5)  # frames 0–1 lifted
    np.testing.assert_allclose(obj.location[2:, 2], 1.0)  # frames 2–3 untouched (out of range)


def test_ball_corrections_are_baked_into_the_plan(make_scene):
    ball = BallTrack(
        frames=np.arange(4), positions_3d=np.tile([0.0, 0.0, 2.0], (4, 1)),
        height_confidence=np.ones(4),
    )
    target = CorrectionTarget(kind=TargetKind.BALL_POSITION)
    corr = make_offset("b", target, (0, 3), np.array([1.0, 0.0, 0.0]))
    obj = build_proxy_plan(make_scene(ball=ball, corrections=[corr])).objects[0]
    np.testing.assert_allclose(obj.location[:, 0], 1.0)  # whole trajectory shifted +1 m in X


def test_build_plan_does_not_mutate_the_proposal(make_scene, make_motion):
    subj = Subject(track_id=1, proposal=make_motion([0, 1, 2], transl_z=2.0))
    before = subj.proposal.pose.transl.copy()
    target = CorrectionTarget(kind=TargetKind.ROOT_TRANSLATION, subject_track_id=1)
    corr = make_offset("c", target, (0, 2), np.array([0.0, 0.0, 9.0]))
    build_proxy_plan(make_scene(subjects=[subj], corrections=[corr]))
    np.testing.assert_array_equal(subj.proposal.pose.transl, before)


# --- team colours --------------------------------------------------------------------


def test_subject_rgb_referee_default_and_team(make_motion):
    ref = Subject(track_id=1, proposal=make_motion([0]), role=Role.REFEREE)
    assert _subject_rgb(ref, []) == _REFEREE_RGB

    teamless = Subject(track_id=2, proposal=make_motion([0]))
    assert _subject_rgb(teamless, []) == _PLAYER_RGB  # no team → default blue

    member = Subject(track_id=3, proposal=make_motion([0]), team_id="A")
    teams = [Team(id="A", name="Home", color_rgb=(0.5, 0.25, 0.75))]
    assert _subject_rgb(member, teams) == pytest.approx((0.5, 0.25, 0.75))


def test_ball_uses_the_ball_colour(make_scene):
    ball = BallTrack(
        frames=np.arange(2), positions_3d=np.zeros((2, 3)), height_confidence=np.ones(2)
    )
    assert build_proxy_plan(make_scene(ball=ball)).objects[0].color_rgb == _BALL_RGB


# --- camera recovery & view derivation -----------------------------------------------


def test_camera_eye_target_recovers_centre_and_forward():
    intr = CameraIntrinsics(fx=50.0, fy=50.0, cx=32.0, cy=24.0, width=64, height=48)
    cam = CameraTrack(
        intrinsics=intr, frames=np.array([0]),
        rotation_quat=np.array([[1.0, 0.0, 0.0, 0.0]]), translation=np.array([[2.0, 0.0, 0.0]]),
    )
    eye, target = camera_eye_target(ViewpointCamera(viewpoint=Viewpoint.FRONT, camera=cam))
    np.testing.assert_allclose(eye, [-2.0, 0.0, 0.0])      # centre = -Rᵀt, R = I
    np.testing.assert_allclose(target, [-2.0, 0.0, 1.0])   # +Z forward → one metre ahead


def test_view_from_camera_derives_lens_and_downscales_resolution():
    cam = _viewpoint_camera(fx=500.0, width=1000, height=600, frame=7)
    view = _view_from_camera(cam, max_px=480)
    assert view.lens_mm == pytest.approx(500.0 * 36.0 / 1000.0)  # 18 mm
    assert view.resolution == (480, 288)  # 1000→480 keeps aspect (×0.48)
    assert view.frame == 7
    assert view.viewpoint == Viewpoint.FRONT.value


def test_view_from_camera_keeps_small_resolution_unscaled():
    view = _view_from_camera(_viewpoint_camera(width=320, height=240), max_px=480)
    assert view.resolution == (320, 240)


def test_build_plan_places_requested_views(make_scene, make_motion):
    scene = make_scene(subjects=[Subject(track_id=0, proposal=make_motion([0, 1, 2]))])
    views = standard_viewpoints(scene, which=[Viewpoint.FRONT, Viewpoint.TOP])
    plan = build_proxy_plan(scene, views=views)
    assert [v.viewpoint for v in plan.views] == [Viewpoint.FRONT.value, Viewpoint.TOP.value]


# --- JSON round-trip across the subprocess boundary ----------------------------------


def test_plan_json_round_trips(tmp_path, make_scene, make_motion):
    subj = Subject(track_id=5, proposal=make_motion([0, 1, 2], transl_z=1.0))
    scene = make_scene(subjects=[subj])
    plan = build_proxy_plan(scene, views=standard_viewpoints(scene, which=[Viewpoint.FRONT]))

    reloaded = load_plan(write_plan(plan, tmp_path / "plan.json"))
    assert reloaded.scene_id == plan.scene_id and reloaded.fps == plan.fps
    np.testing.assert_allclose(reloaded.objects[0].location, plan.objects[0].location)
    np.testing.assert_allclose(reloaded.objects[0].rotation_aa, plan.objects[0].rotation_aa)
    np.testing.assert_allclose(reloaded.views[0].eye, plan.views[0].eye)
    assert reloaded.views[0].resolution == plan.views[0].resolution
    assert isinstance(plan_to_json(plan), str)  # serializes without raising


# --- import safety: the pure half must never pull bpy into our interpreter -----------


def test_importing_the_adapter_does_not_load_bpy():
    assert "bpy" not in sys.modules


# --- gating: a missing binary is an actionable error (no Blender needed) --------------


def test_run_blender_without_a_binary_is_actionable(monkeypatch, make_scene, make_motion):
    monkeypatch.delenv("PITCH3D_BLENDER", raising=False)
    monkeypatch.setattr("pitch3d.adapters.blender.runner.shutil.which", lambda _: None)
    subj = Subject(track_id=0, proposal=make_motion([0, 1]))
    plan = build_proxy_plan(make_scene(subjects=[subj]))
    with pytest.raises(RuntimeError, match="PITCH3D_BLENDER"):
        run_blender(plan)


# --- real subprocess: build a .blend and render proxy SCENE_3D (skips without Blender) -


@needs_blender
def test_proxy_builder_writes_a_blend(tmp_path, make_scene, make_motion):
    subj = Subject(track_id=1, proposal=make_motion([0, 1, 2], transl_z=1.0))
    out = tmp_path / "proxy.blend"
    path = BlenderProxyBuilder(blender=BLENDER).build(make_scene(subjects=[subj]), out)
    assert path == out and path.is_file() and path.stat().st_size > 0


@needs_blender
def test_observer_renders_scene3d_pngs(tmp_path, make_scene, make_motion):
    subjects = [Subject(track_id=t, proposal=make_motion([0, 1, 2], transl_z=1.0)) for t in (1, 2)]
    ball = BallTrack(
        frames=np.arange(3), positions_3d=np.tile([0.0, 0.0, 0.2], (3, 1)),
        height_confidence=np.ones(3),
    )
    scene = make_scene(subjects=subjects, ball=ball)
    views = standard_viewpoints(scene, which=[Viewpoint.FRONT, Viewpoint.TOP])

    observer = BlenderSceneObserver(out_dir=tmp_path / "obs", blender=BLENDER)
    images = observer.capture_scene_views(scene, views)

    assert len(images) == 2
    assert [img.viewpoint for img in images] == [Viewpoint.FRONT, Viewpoint.TOP]
    for img in images:
        assert img.kind == ObservationKind.SCENE_3D
        data = Path(img.uri).read_bytes()
        assert data[:8] == _PNG_SIG and len(data) > 8
