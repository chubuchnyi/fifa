"""ReprojectionOverlayRenderPass — pinhole projection maths + per-frame PNG overlay (FR-14, M1).

The projection half (quaternion world→camera rotation, pinhole projection, visibility) is pure
numpy and checked directly; the pass itself is exercised end-to-end (writes a PNG per frame,
draws only visible points, reads only resolved state) with **no Blender, no GPU**.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pitch3d.adapters.fakes import FakeRenderPass
from pitch3d.adapters.render.overlay import (
    _BACKGROUND,
    ReprojectionOverlayRenderPass,
    _draw_marker,
    appearance_alpha,
    confidence_to_color,
    fade_to_background,
    project_world_points,
    quat_to_rotation_matrix,
)
from pitch3d.core.ports.render import RenderPass, RenderResult
from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack
from pitch3d.core.scene.layers import ConfidenceMap
from pitch3d.core.scene.motion import BallTrack
from pitch3d.core.scene.subject import Role, Subject


def _intr(w=64, h=48) -> CameraIntrinsics:
    return CameraIntrinsics(fx=50.0, fy=50.0, cx=w / 2.0, cy=h / 2.0, width=w, height=h)


def _camera(n=4) -> CameraTrack:
    return CameraTrack.identity(_intr(), n)


def _scene_with_subject(make_scene, make_motion, transl_z, role=Role.PLAYER):
    subj = Subject(track_id=1, proposal=make_motion(range(4), transl_z=transl_z), role=role)
    return make_scene(subjects=[subj])


def _frame0(result) -> bytes:
    return (Path(result.uri) / "frame_00000.png").read_bytes()


def _frame(result, i: int) -> bytes:
    return (Path(result.uri) / f"frame_{i:05d}.png").read_bytes()


# --- pure projection maths -----------------------------------------------------
def test_quat_to_rotation_matrix_identity():
    np.testing.assert_allclose(quat_to_rotation_matrix([1, 0, 0, 0]), np.eye(3))


def test_quat_to_rotation_matrix_z_rotation():
    r = quat_to_rotation_matrix([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])  # +90° about Z
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-9)


def test_project_pinhole_in_front_is_visible():
    cam = _camera(1)
    uv, vis = project_world_points(cam, 0, np.array([[0.0, 0.0, 5.0]]))
    np.testing.assert_allclose(uv[0], [32.0, 24.0])  # on-axis point → principal point
    assert bool(vis[0]) is True


def test_project_behind_camera_is_invisible():
    _, vis = project_world_points(_camera(1), 0, np.array([[0.0, 0.0, -5.0]]))
    assert bool(vis[0]) is False


def test_project_offscreen_is_invisible():
    # In front of the camera but way off to the side → outside the image rectangle.
    _, vis = project_world_points(_camera(1), 0, np.array([[100.0, 0.0, 5.0]]))
    assert bool(vis[0]) is False


def test_project_applies_extrinsic_translation():
    cam = CameraTrack(
        intrinsics=_intr(), frames=np.array([0]),
        rotation_quat=np.array([[1.0, 0.0, 0.0, 0.0]]), translation=np.array([[2.0, 0.0, 0.0]]),
    )
    uv, vis = project_world_points(cam, 0, np.array([[0.0, 0.0, 4.0]]))
    np.testing.assert_allclose(uv[0], [57.0, 24.0])  # 50*2/4 + 32 = 57 → translation applied
    assert bool(vis[0]) is True


def test_draw_marker_paints_clipped_square():
    fb = np.zeros((10, 10, 3), dtype=np.uint8)
    _draw_marker(fb, 5, 5, (10, 20, 30), 1)
    np.testing.assert_array_equal(fb[5, 5], [10, 20, 30])
    np.testing.assert_array_equal(fb[0, 0], [0, 0, 0])  # far corner untouched
    _draw_marker(fb, 0, 0, (1, 2, 3), 3)  # clips at the edge without raising
    np.testing.assert_array_equal(fb[0, 0], [1, 2, 3])


# --- render pass behaviour -----------------------------------------------------
@pytest.mark.parametrize(
    "make_rp", [FakeRenderPass, ReprojectionOverlayRenderPass], ids=["fake", "overlay"]
)
def test_render_pass_port_contract(make_rp, tmp_path, make_scene, make_motion):
    rp = make_rp(out_dir=tmp_path / "rp")
    assert isinstance(rp, RenderPass)
    cam = _camera(4)
    result = rp.render(_scene_with_subject(make_scene, make_motion, 5.0), cam)
    assert isinstance(result, RenderResult)
    assert result.n_frames == cam.n_frames
    assert result.camera is cam
    assert Path(result.uri).exists()


def test_render_writes_one_png_per_frame_and_manifest(tmp_path, make_scene, make_motion):
    rp = ReprojectionOverlayRenderPass(out_dir=tmp_path / "r")
    result = rp.render(_scene_with_subject(make_scene, make_motion, 5.0), _camera(4))
    out = Path(result.uri)
    assert len(sorted(out.glob("frame_*.png"))) == 4
    assert (out / "manifest.txt").exists()
    assert _frame0(result)[:8] == b"\x89PNG\r\n\x1a\n"  # valid PNG signature
    assert result.n_frames == 4 and result.is_video is False


def test_render_draws_visible_subject(tmp_path, make_scene, make_motion):
    cam = _camera(4)
    visible = ReprojectionOverlayRenderPass(out_dir=tmp_path / "v").render(
        _scene_with_subject(make_scene, make_motion, 5.0), cam
    )
    empty = ReprojectionOverlayRenderPass(out_dir=tmp_path / "e").render(make_scene(), cam)
    assert _frame0(visible) != _frame0(empty)  # the marker changed the frame


def test_render_skips_points_behind_camera(tmp_path, make_scene, make_motion):
    cam = _camera(4)
    behind = ReprojectionOverlayRenderPass(out_dir=tmp_path / "b").render(
        _scene_with_subject(make_scene, make_motion, -5.0), cam
    )
    empty = ReprojectionOverlayRenderPass(out_dir=tmp_path / "e").render(make_scene(), cam)
    assert _frame0(behind) == _frame0(empty)  # nothing drawn for a point behind the camera


def test_render_includes_the_ball(tmp_path, make_scene):
    frames = np.arange(4)
    ball = BallTrack(
        frames=frames, positions_3d=np.tile([0.0, 0.0, 5.0], (4, 1)),
        height_confidence=np.ones(4),
    )
    cam = _camera(4)
    with_ball = ReprojectionOverlayRenderPass(out_dir=tmp_path / "wb").render(
        make_scene(ball=ball), cam
    )
    empty = ReprojectionOverlayRenderPass(out_dir=tmp_path / "e").render(make_scene(), cam)
    assert _frame0(with_ball) != _frame0(empty)


def test_render_is_non_destructive(tmp_path, make_scene, make_motion):
    scene = _scene_with_subject(make_scene, make_motion, 5.0)
    before = scene.subjects[0].proposal.pose.transl.copy()
    ReprojectionOverlayRenderPass(out_dir=tmp_path / "n").render(scene, _camera(4))
    np.testing.assert_array_equal(scene.subjects[0].proposal.pose.transl, before)


# --- confidence highlighting (UX-3, FR-16) -------------------------------------
def test_confidence_to_color_endpoints_and_clamp():
    base, low = (60, 170, 255), (255, 60, 60)
    assert confidence_to_color(base, 1.0) == base       # full confidence → untouched
    assert confidence_to_color(base, 0.0) == low        # zero confidence → warning colour
    assert confidence_to_color(base, 2.0) == base       # clamps above 1
    assert confidence_to_color(base, -1.0) == low       # clamps below 0
    mid = confidence_to_color(base, 0.5)
    assert all(min(b, low[i]) <= mid[i] <= max(b, low[i]) for i, b in enumerate(base))


def _render(scene, tmp_path, name):
    return _frame0(ReprojectionOverlayRenderPass(out_dir=tmp_path / name).render(scene, _camera(4)))


def test_full_confidence_renders_identically_to_no_confidence(tmp_path, make_scene, make_motion):
    subj = Subject(track_id=1, proposal=make_motion(range(4), transl_z=5.0))
    none = _render(make_scene(subjects=[subj]), tmp_path, "none")
    ones = _render(
        make_scene(subjects=[subj], confidence=ConfidenceMap(subject_frame_conf={1: np.ones(4)})),
        tmp_path, "ones",
    )
    assert none == ones  # the colour-untouched guarantee: existing scenes render unchanged


def test_low_confidence_tints_the_subject_marker(tmp_path, make_scene, make_motion):
    subj = Subject(track_id=1, proposal=make_motion(range(4), transl_z=5.0))
    full = _render(make_scene(subjects=[subj]), tmp_path, "full")
    low = _render(
        make_scene(subjects=[subj], confidence=ConfidenceMap(subject_frame_conf={1: np.zeros(4)})),
        tmp_path, "low",
    )
    assert full != low  # a low-confidence subject is visibly highlighted


def test_low_ball_height_confidence_tints_the_ball(tmp_path, make_scene):
    frames, pos = np.arange(4), np.tile([0.0, 0.0, 5.0], (4, 1))
    high = make_scene(ball=BallTrack(frames=frames, positions_3d=pos, height_confidence=np.ones(4)))
    low = make_scene(ball=BallTrack(frames=frames, positions_3d=pos, height_confidence=np.zeros(4)))
    assert _render(high, tmp_path, "bh") != _render(low, tmp_path, "bl")


# --- entry/exit fade (#98, visual polish) --------------------------------------
def test_fade_to_background_endpoints_and_clamp():
    color = (60, 170, 255)
    assert fade_to_background(color, 1.0) == color        # full opacity → untouched
    assert fade_to_background(color, 0.0) == _BACKGROUND   # zero opacity → background
    assert fade_to_background(color, 2.0) == color         # clamps above 1
    assert fade_to_background(color, -1.0) == _BACKGROUND   # clamps below 0
    mid = fade_to_background(color, 0.5)
    for i, c in enumerate(color):
        assert min(c, _BACKGROUND[i]) <= mid[i] <= max(c, _BACKGROUND[i])


def test_appearance_alpha_present_whole_clip_is_opaque():
    # frames span the rendered clip [0, 5] → neither a genuine entry nor exit → no fade.
    np.testing.assert_array_equal(appearance_alpha(np.arange(6), 0, 5, fade=4), np.ones(6))


def test_appearance_alpha_disabled_is_opaque():
    np.testing.assert_array_equal(appearance_alpha(np.array([2, 3, 4]), 0, 9, fade=0), np.ones(3))


def test_appearance_alpha_ramps_in_at_a_genuine_entry():
    # first seen at frame 3 but present through the clip end (7) → entry ramp only, then opaque.
    a = appearance_alpha(np.array([3, 4, 5, 6, 7]), clip_first=0, clip_last=7, fade=2)
    np.testing.assert_allclose(a, [1 / 3, 2 / 3, 1.0, 1.0, 1.0])
    assert a[0] < a[1] < a[2]  # strictly ramping up


def test_appearance_alpha_ramps_out_at_a_genuine_exit():
    # present from the clip start (2) but last seen at 6, before the clip end (9) → exit ramp only.
    a = appearance_alpha(np.array([2, 3, 4, 5, 6]), clip_first=2, clip_last=9, fade=2)
    np.testing.assert_allclose(a, [1.0, 1.0, 1.0, 2 / 3, 1 / 3])
    assert a[-1] < a[-2] < a[-3]  # strictly ramping down


def test_appearance_alpha_fades_both_sides_of_an_interior_gap():
    # a real (unbridged) gap between 4 and 11 → exit before it, entry after it.
    a = appearance_alpha(np.array([3, 4, 11, 12, 13]), clip_first=0, clip_last=20, fade=1)
    # segment [3,4] exits into the gap; segment [11,12,13] enters out of it (both ends ramp).
    assert a[1] < 1.0 and a[2] < 1.0          # frame 4 (pre-gap) and 11 (post-gap) are dimmed
    assert a[0] < 1.0                          # frame 3 also ramps (short pre-gap segment)
    assert a[3] == 1.0 or a[4] == 1.0          # the post-gap segment settles to opaque


def test_appearance_alpha_does_not_fade_clip_boundaries():
    # touches both clip ends → clipped by the window, not a real entry/exit → stays opaque.
    np.testing.assert_array_equal(appearance_alpha(np.array([0, 1, 2]), 0, 2, fade=4), np.ones(3))


def test_render_fades_a_genuine_entry_but_not_a_settled_frame(tmp_path, make_scene, make_motion):
    # subject present frames [2..7] of an 8-frame clip → genuine entry at 2, present at the end.
    subj = Subject(track_id=1, proposal=make_motion(range(2, 8), transl_z=5.0))
    scene = make_scene(subjects=[subj])
    cam = _camera(8)
    faded = ReprojectionOverlayRenderPass(out_dir=tmp_path / "fade", fade_frames=3).render(
        scene, cam
    )
    plain = ReprojectionOverlayRenderPass(out_dir=tmp_path / "plain", fade_frames=0).render(
        scene, cam
    )
    assert _frame(faded, 2) != _frame(plain, 2)   # entry frame is dimmed toward the background
    assert _frame(faded, 7) == _frame(plain, 7)   # a settled (opaque) frame is untouched


def test_render_full_clip_subject_identical_with_fade_on_or_off(tmp_path, make_scene, make_motion):
    # a subject present across the whole clip never fades → fade on/off render byte-identical.
    subj = Subject(track_id=1, proposal=make_motion(range(4), transl_z=5.0))
    scene = make_scene(subjects=[subj])
    cam = _camera(4)
    on = ReprojectionOverlayRenderPass(out_dir=tmp_path / "on", fade_frames=4).render(scene, cam)
    off = ReprojectionOverlayRenderPass(out_dir=tmp_path / "off", fade_frames=0).render(scene, cam)
    assert _frame0(on) == _frame0(off)
