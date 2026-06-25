"""Splat-avatar render pass — projection, z-buffer, and R-6 unmeasured tinting (M2-3).

The pass is checked the honest way: a tiny written avatar PLY + a synthetic identity camera exercise
the real path (read asset → place at the subject's resolved root → project → z-buffer splat) and we
assert on the *framebuffer pixels* via :func:`render_frame_buffer`, no PNG decode. The crux tests:
a measured vertex paints its colour at the pixel it projects to, an **unmeasured** vertex paints the
distinct R-6 tint (never its fabricated placeholder), and at a shared pixel the nearer splat wins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pitch3d.adapters.models.avatar import _UNMEASURED_RGB, write_vertex_colored_ply
from pitch3d.adapters.render.avatar_splat import (
    _UNMEASURED_TINT,
    SplatAvatarRenderPass,
    render_frame_buffer,
)
from pitch3d.adapters.render.overlay import _BACKGROUND
from pitch3d.core.correction.engine import make_offset
from pitch3d.core.scene.assets import RenderAssetKind, RenderAssetRef
from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack
from pitch3d.core.scene.layers import CorrectionTarget, TargetKind
from pitch3d.core.scene.provenance import Backend, ModelInfo
from pitch3d.core.scene.subject import Subject

_ORIGIN_VERT = np.array([[0.0, 0.0, 0.0]])  # one mesh-local vertex; the root translation places it


def _camera(n: int = 1, w: int = 64, h: int = 48) -> CameraTrack:
    intr = CameraIntrinsics(fx=50.0, fy=50.0, cx=w / 2.0, cy=h / 2.0, width=w, height=h)
    return CameraTrack.identity(intr, n)


def _avatar_ref(out_dir, track_id, rgb, measured, *, verts=_ORIGIN_VERT) -> RenderAssetRef:
    uri = write_vertex_colored_ply(
        Path(out_dir) / f"avatar_{track_id}.ply",
        verts, np.zeros((0, 3)), np.asarray(rgb, dtype=np.uint8), np.asarray(measured),
    )
    return RenderAssetRef(
        id=f"avatar-{track_id}",
        kind=RenderAssetKind.AVATAR_TEXTURED_SMPLX,
        uri=uri,
        model=ModelInfo(name="t", backend=Backend.FAKE),
        subject_track_id=track_id,
    )


def _scene_with(make_scene, *subjects_and_refs):
    """make_scene + attach the given (Subject, RenderAssetRef) pairs onto render_assets."""
    subjects = [s for s, _ in subjects_and_refs]
    scene = make_scene(subjects=subjects)
    scene.render_assets = [r for _, r in subjects_and_refs]
    return scene


# --- end-to-end render() output -----------------------------------------------
def test_render_writes_frames_and_manifest(tmp_path, make_motion, make_scene):
    ref = _avatar_ref(tmp_path, 7, [[200, 100, 50]], [True])
    subj = Subject(track_id=7, proposal=make_motion([0, 1], transl_z=5.0))
    scene = _scene_with(make_scene, (subj, ref))
    res = SplatAvatarRenderPass(out_dir=tmp_path / "render").render(scene, _camera(2))

    assert res.n_frames == 2 and not res.is_video
    target = Path(res.uri)
    assert (target / "frame_00000.png").exists() and (target / "frame_00001.png").exists()
    manifest = (target / "manifest.txt").read_text()
    assert "avatars=1" in manifest and "vertices=1" in manifest and "unmeasured=0" in manifest


# --- framebuffer crux: colour placement, R-6 tint, z-buffer -------------------
def test_measured_vertex_paints_its_color_at_projected_pixel(tmp_path, make_motion, make_scene):
    # Vertex world (0,0,5) projects to the principal point (cx,cy)=(32,24).
    ref = _avatar_ref(tmp_path, 7, [[200, 100, 50]], [True])
    subj = Subject(track_id=7, proposal=make_motion([0], transl_z=5.0))
    scene = _scene_with(make_scene, (subj, ref))
    fb = render_frame_buffer(scene, _camera(1), 0)
    np.testing.assert_array_equal(fb[24, 32], [200, 100, 50])
    np.testing.assert_array_equal(fb[0, 0], list(_BACKGROUND))  # nothing painted elsewhere


def test_unmeasured_vertex_renders_r6_tint_not_placeholder(tmp_path, make_motion, make_scene):
    # The asset stores the grey placeholder colour, but measured=0 → the renderer must show the
    # distinct unmeasured tint, admitting "no appearance data" rather than trusting the placeholder.
    ref = _avatar_ref(tmp_path, 7, [list(_UNMEASURED_RGB)], [False])
    subj = Subject(track_id=7, proposal=make_motion([0], transl_z=5.0))
    scene = _scene_with(make_scene, (subj, ref))
    fb = render_frame_buffer(scene, _camera(1), 0)
    np.testing.assert_array_equal(fb[24, 32], list(_UNMEASURED_TINT))
    assert tuple(int(c) for c in fb[24, 32]) != _UNMEASURED_RGB  # NOT the stored placeholder


def test_zbuffer_nearer_avatar_wins_shared_pixel(tmp_path, make_motion, make_scene):
    # Two subjects project to the same pixel; the nearer (z=3, red) must win over the farther
    # (z=9, blue) — and it does even though the farther one is attached/drawn first.
    near = _avatar_ref(tmp_path, 1, [[255, 0, 0]], [True])
    far = _avatar_ref(tmp_path, 2, [[0, 0, 255]], [True])
    scene = _scene_with(
        make_scene,
        (Subject(track_id=2, proposal=make_motion([0], transl_z=9.0)), far),
        (Subject(track_id=1, proposal=make_motion([0], transl_z=3.0)), near),
    )
    fb = render_frame_buffer(scene, _camera(1), 0)
    np.testing.assert_array_equal(fb[24, 32], [255, 0, 0])


def test_no_avatar_assets_renders_empty_background(tmp_path, make_motion, make_scene):
    # A scene with subjects but no avatar assets paints only the background — honest, no crash.
    scene = make_scene(subjects=[Subject(track_id=7, proposal=make_motion([0], transl_z=5.0))])
    fb = render_frame_buffer(scene, _camera(1), 0)
    assert (fb == np.array(_BACKGROUND, dtype=np.uint8)).all()


# --- M2-4 edit↔render sync: a correction re-projects with no avatar rebuild (AC-5a) -----------
_SIDE_VERT = np.array([[1.0, 0.0, 0.0]])  # 1 m off the root so a root *rotation* is observable


def test_root_orientation_edit_reprojects_without_avatar_rebuild(tmp_path, make_motion, make_scene):
    # AC-5a crux: editing the SMPL root orientation re-projects into the rendered frame with NO
    # avatar rebuild. The off-root vertex starts at world (1,0,5) → pixel (col=42,row=24); a 90°
    # ROOT_ORIENTATION correction about Z swings it to world (0,1,5) → pixel (col=32,row=34). Same
    # PLY on disk — only the resolved pose the splat pass reads changed.
    ref = _avatar_ref(tmp_path, 7, [[200, 100, 50]], [True], verts=_SIDE_VERT)
    subj = Subject(track_id=7, proposal=make_motion([0], transl_z=5.0))
    scene = _scene_with(make_scene, (subj, ref))

    before = render_frame_buffer(scene, _camera(1), 0)
    np.testing.assert_array_equal(before[24, 42], [200, 100, 50])  # placed at identity orient
    ply_mtime = Path(ref.uri).stat().st_mtime_ns

    scene.corrections = [
        make_offset(
            "rot90",
            CorrectionTarget(kind=TargetKind.ROOT_ORIENTATION, subject_track_id=7),
            (0, 0),
            delta=np.array([0.0, 0.0, np.pi / 2]),
        )
    ]
    after = render_frame_buffer(scene, _camera(1), 0)

    np.testing.assert_array_equal(after[34, 32], [200, 100, 50])     # swung to the new pixel
    np.testing.assert_array_equal(after[24, 42], list(_BACKGROUND))  # vacated the old pixel
    assert Path(ref.uri).stat().st_mtime_ns == ply_mtime            # asset never rebuilt


def test_root_translation_edit_reprojects(tmp_path, make_motion, make_scene):
    # The translation DOF of the resolved root rigid transform also syncs: a +2 m world-X offset
    # slides the same vertex from pixel col=42 to col=62 — through the resolved-pose path only.
    ref = _avatar_ref(tmp_path, 7, [[200, 100, 50]], [True], verts=_SIDE_VERT)
    subj = Subject(track_id=7, proposal=make_motion([0], transl_z=5.0))
    scene = _scene_with(make_scene, (subj, ref))

    before = render_frame_buffer(scene, _camera(1), 0)
    np.testing.assert_array_equal(before[24, 42], [200, 100, 50])

    scene.corrections = [
        make_offset(
            "slideX",
            CorrectionTarget(kind=TargetKind.ROOT_TRANSLATION, subject_track_id=7),
            (0, 0),
            delta=np.array([2.0, 0.0, 0.0]),
        )
    ]
    after = render_frame_buffer(scene, _camera(1), 0)
    np.testing.assert_array_equal(after[24, 62], [200, 100, 50])     # slid to the new pixel
    np.testing.assert_array_equal(after[24, 42], list(_BACKGROUND))  # vacated the old pixel
