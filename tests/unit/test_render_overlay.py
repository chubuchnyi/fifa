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
    ReprojectionOverlayRenderPass,
    _draw_marker,
    project_world_points,
    quat_to_rotation_matrix,
)
from pitch3d.core.ports.render import RenderPass, RenderResult
from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack
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
