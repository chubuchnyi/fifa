"""Cycles render pass (M2-7) — the pure camera/placement/lens maths + a Blender-gated smoke render.

The error-prone half (OpenCV→Blender camera conversion, root placement, ``K``→lens mapping, plan
assembly + JSON round-trip) is tested with no Blender via strong invariants: the conversion is
checked by transforming a world point into Blender camera-local coords and matching the OpenCV
optical flip, and the placement is checked against the splat pass's exact ``canonical @ rot.T +
transl`` formula so the two passes cannot drift. One integration test actually drives Cycles when a
Blender binary is present, asserting a non-empty photoreal PNG comes out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pitch3d.adapters.blender import locate_blender
from pitch3d.adapters.blender._cycles_script import _parse_args
from pitch3d.adapters.blender.cycles_plan import (
    CyclesMeshRef,
    blender_lens_params,
    build_cycles_plan,
    cv_to_blender_camera_matrix,
    load_cycles_plan,
    root_object_matrix,
    write_cycles_plan,
)
from pitch3d.adapters.models.avatar import write_vertex_colored_ply
from pitch3d.adapters.models.smplx_lbs import SmplxModel, locate_smplx_model
from pitch3d.adapters.render.cycles import CyclesRenderPass
from pitch3d.core.correction.rotations import axis_angle_to_matrix
from pitch3d.core.ports.render import RenderQuality
from pitch3d.core.scene.assets import RenderAssetKind, RenderAssetRef
from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack
from pitch3d.core.scene.motion import (
    N_SMPLX_BODY_JOINTS,
    PoseSequence,
    SmplxShape,
    SubjectMotion,
)
from pitch3d.core.scene.provenance import Backend, ModelInfo
from pitch3d.core.scene.subject import Subject

_FLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV optical → Blender camera axis flip


def _camera(n: int = 1, w: int = 64, h: int = 48) -> CameraTrack:
    intr = CameraIntrinsics(fx=50.0, fy=50.0, cx=w / 2.0, cy=h / 2.0, width=w, height=h)
    return CameraTrack.identity(intr, n)


# --- camera conversion: the whole OpenCV→Blender invariant ---------------------
def test_cv_to_blender_matches_optical_flip_and_centre():
    # For any world point P: Blender camera-local coords (matrix_worldᵀ applied) must equal the
    # OpenCV camera coords with the y/z optical flip — and the camera centre is the OpenCV -Rᵀt.
    rng = np.random.default_rng(0)
    rot = axis_angle_to_matrix(rng.normal(size=3))
    t = rng.normal(size=3)
    p = rng.normal(size=3)
    matrix = cv_to_blender_camera_matrix(rot, t)
    rot_b, centre = matrix[:3, :3], matrix[:3, 3]

    x_cv = rot @ p + t
    blender_local = rot_b.T @ (p - centre)
    np.testing.assert_allclose(blender_local, _FLIP @ x_cv, atol=1e-9)
    np.testing.assert_allclose(centre, -rot.T @ t, atol=1e-9)


def test_identity_camera_looks_along_world_plus_z():
    # An identity OpenCV camera (R=I, t=0) looks +Z world; the Blender camera looks down local -Z,
    # so its world-space view direction (−column 2 of matrix_world) must be +Z, centre at origin.
    matrix = cv_to_blender_camera_matrix(np.eye(3), np.zeros(3))
    np.testing.assert_allclose(matrix[:3, 3], 0.0, atol=1e-12)
    np.testing.assert_allclose(-matrix[:3, 2], [0.0, 0.0, 1.0], atol=1e-12)


# --- placement: identical to the splat pass ------------------------------------
def test_root_object_matrix_matches_splat_placement():
    rng = np.random.default_rng(1)
    aa, transl = rng.normal(size=3), rng.normal(size=3)
    canonical = rng.normal(size=(5, 3))
    matrix = root_object_matrix(aa, transl)
    homo = np.concatenate([canonical, np.ones((5, 1))], axis=1)
    placed = (homo @ matrix.T)[:, :3]
    expected = canonical @ axis_angle_to_matrix(aa).T + transl  # the splat formula, verbatim
    np.testing.assert_allclose(placed, expected, atol=1e-9)


# --- intrinsics → Blender lens/sensor/shift ------------------------------------
def test_blender_lens_params_square_centred():
    intr = CameraIntrinsics(
        fx=1000.0, fy=1000.0, cx=(1280 - 1) / 2.0, cy=(720 - 1) / 2.0, width=1280, height=720
    )
    p = blender_lens_params(intr)
    assert p["sensor_fit"] == "HORIZONTAL"
    assert p["pixel_aspect_x"] == 1.0 and abs(p["pixel_aspect_y"] - 1.0) < 1e-12
    assert abs(p["lens_mm"] - 1000.0 * 36.0 / 1280.0) < 1e-9
    assert abs(p["shift_x"]) < 1e-12 and abs(p["shift_y"]) < 1e-12


def test_blender_lens_params_anisotropic_pixels_ride_aspect():
    intr = CameraIntrinsics(fx=1200.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)
    p = blender_lens_params(intr)
    assert abs(p["pixel_aspect_y"] / p["pixel_aspect_x"] - 1200.0 / 1000.0) < 1e-9


# --- plan assembly: frame count, present/absent visibility, JSON round-trip ----
def test_build_plan_frames_and_absent_subject_hidden(make_motion, make_scene):
    subj = Subject(track_id=7, proposal=make_motion([0, 2], transl_z=5.0))  # absent at frame 1
    scene = make_scene(subjects=[subj])
    mesh = CyclesMeshRef("avatar_7", "avatar_7.npz", 7)
    plan = build_cycles_plan(scene, _camera(3), meshes=[mesh])
    assert len(plan.frames) == 3
    assert [f.placements[0].visible for f in plan.frames] == [True, False, True]


def test_plan_json_round_trip(tmp_path, make_motion, make_scene):
    subj = Subject(track_id=7, proposal=make_motion([0, 1], transl_z=5.0))
    scene = make_scene(subjects=[subj])
    plan = build_cycles_plan(
        scene, _camera(2), meshes=[CyclesMeshRef("avatar_7", "avatar_7.npz", 7)],
        samples=12, device="CPU",
    )
    back = load_cycles_plan(write_cycles_plan(plan, tmp_path / "plan.json"))
    assert back.scene_id == plan.scene_id and back.samples == 12 and len(back.frames) == 2
    np.testing.assert_allclose(
        back.frames[0].camera_matrix_world, plan.frames[0].camera_matrix_world
    )
    np.testing.assert_allclose(
        back.frames[1].placements[0].matrix_world, plan.frames[1].placements[0].matrix_world
    )


def test_plan_round_trips_posed_mesh_and_vert_index(tmp_path, make_motion, make_scene):
    # A posed mesh (M2-8): root + limbs are baked into per-frame LBS verts, so the placement is an
    # identity matrix plus the pose-row ``vert_index`` (not a root matrix). Both the ``posed`` flag
    # and per-frame ``vert_index`` must survive the JSON subprocess boundary so the script swaps the
    # right geometry per frame.
    subj = Subject(track_id=7, proposal=make_motion([0, 1], transl_z=5.0))
    scene = make_scene(subjects=[subj])
    plan = build_cycles_plan(
        scene, _camera(2), meshes=[CyclesMeshRef("avatar_7", "avatar_7.npz", 7, posed=True)]
    )
    assert plan.meshes[0].posed is True
    pl0 = plan.frames[0].placements[0]
    assert pl0.visible and pl0.vert_index == 0
    np.testing.assert_allclose(pl0.matrix_world, np.eye(4))
    assert [f.placements[0].vert_index for f in plan.frames] == [0, 1]

    back = load_cycles_plan(write_cycles_plan(plan, tmp_path / "posed.json"))
    assert back.meshes[0].posed is True
    assert [f.placements[0].vert_index for f in back.frames] == [0, 1]
    np.testing.assert_allclose(back.frames[0].placements[0].matrix_world, np.eye(4))


def test_plan_carries_pitch_npz_through_json(tmp_path, make_motion, make_scene):
    # The measured pitch markings ride the plan as a single NPZ ref the script loads (M2-9); the ref
    # must survive the JSON subprocess boundary, and stays ``None`` when markings are off.
    subj = Subject(track_id=7, proposal=make_motion([0], transl_z=5.0))
    scene = make_scene(subjects=[subj])
    mesh = [CyclesMeshRef("avatar_7", "avatar_7.npz", 7)]
    plan = build_cycles_plan(scene, _camera(1), meshes=mesh, pitch_npz="pitch.npz")
    assert plan.pitch_npz == "pitch.npz"
    assert load_cycles_plan(write_cycles_plan(plan, tmp_path / "p.json")).pitch_npz == "pitch.npz"
    bare = build_cycles_plan(scene, _camera(1), meshes=mesh)
    assert bare.pitch_npz is None
    assert load_cycles_plan(write_cycles_plan(bare, tmp_path / "bare.json")).pitch_npz is None


# --- the in-Blender script's arg parser is pure (no bpy) -----------------------
def test_cycles_script_parses_flags():
    argv = ["blender", "--python", "x.py", "--",
            "--plan", "p.json", "--mesh-dir", "m", "--render-dir", "r"]
    assert _parse_args(argv) == {"plan": "p.json", "mesh_dir": "m", "render_dir": "r"}


# --- Blender-gated: a real Cycles photoreal frame comes out --------------------
@pytest.mark.skipif(locate_blender() is None, reason="needs a Blender binary (env/PATH)")
def test_cycles_render_produces_a_nonempty_frame(tmp_path, make_motion, make_scene):
    verts = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.6]])
    faces = np.array([[0, 1, 2]])
    rgb = np.array([[210, 90, 40]] * 3, dtype=np.uint8)
    uri = write_vertex_colored_ply(
        tmp_path / "avatar_7.ply", verts, faces, rgb, np.array([True, True, True])
    )
    ref = RenderAssetRef(
        id="a-7", kind=RenderAssetKind.AVATAR_TEXTURED_SMPLX, uri=uri,
        model=ModelInfo(name="t", backend=Backend.FAKE), subject_track_id=7,
    )
    subj = Subject(track_id=7, proposal=make_motion([0], transl_z=8.0))
    scene = make_scene(subjects=[subj])
    scene.render_assets = [ref]

    res = CyclesRenderPass(out_dir=tmp_path / "render", samples=4).render(
        scene, _camera(1, w=160, h=120), RenderQuality.FINAL
    )
    frame = Path(res.uri) / "frame_00000.png"
    assert res.n_frames == 1 and not res.is_video
    assert frame.is_file() and frame.stat().st_size > 0
    manifest = (Path(res.uri) / "manifest.txt").read_text()
    assert "cycles" in res.note and "avatars=1" in manifest
    assert "env=grass+lines+sky" in manifest  # measured grass + line markings + sky path ran (M2-9)


# --- Blender + SMPL-X gated: the limbs actually follow the pose (M2-8a headline) ----
@pytest.mark.skipif(
    locate_blender() is None or locate_smplx_model() is None,
    reason="needs a Blender binary and the SMPL-X model (.npz)",
)
def test_cycles_posed_avatar_limbs_follow_the_pose(tmp_path, make_scene):
    # The headline M2-8a proof: a real SMPL-X avatar rendered through Cycles must *articulate*. The
    # two frames share root, camera and Cycles seed (constant); only the left-elbow angle differs
    # (rest vs a hard bend). A rigid placement would render two identical frames (one canonical mesh
    # at one root) — only per-frame LBS can move pixels, and only where the forearm swung.
    import cv2

    model = SmplxModel.load(locate_smplx_model())
    betas = np.zeros(10)
    verts = model.shaped(betas)
    rgb = np.tile(np.array([[220, 40, 40]], dtype=np.uint8), (verts.shape[0], 1))  # high contrast
    measured = np.ones(verts.shape[0], dtype=bool)
    uri = write_vertex_colored_ply(tmp_path / "avatar_5.ply", verts, model.faces, rgb, measured)

    # The left shoulder is abducted the SAME on both frames (arm held clear of the torso, against
    # the sky); only the left elbow (SMPL-X joint 18 == body_pose row 17) bends on frame 1. So any
    # pixel that changes between the two frames is the forearm articulating — nothing else differs.
    shoulder = [0.0, 0.0, -1.1]
    body = np.zeros((2, N_SMPLX_BODY_JOINTS, 3))
    body[0, 15] = body[1, 15] = shoulder
    body[1, 17] = [1.8, 0.0, 1.8]
    pose = PoseSequence(
        frames=np.array([0, 1]),
        global_orient=np.tile([0.0, 1.2, 0.0], (2, 1)),  # ~70 deg about Y → a 3/4 view of the arm
        body_pose=body,
        transl=np.tile([0.0, 0.2, 2.4], (2, 1)),
    )
    subj = Subject(track_id=5, proposal=SubjectMotion(SmplxShape(betas), pose))
    scene = make_scene(subjects=[subj])
    scene.render_assets = [
        RenderAssetRef(
            id="a-5", kind=RenderAssetKind.AVATAR_TEXTURED_SMPLX, uri=uri,
            model=ModelInfo(name="t", backend=Backend.FAKE), subject_track_id=5,
        )
    ]

    # A zoomed-in camera so the arm fills enough pixels for the swing to be unmistakable.
    intr = CameraIntrinsics(fx=130.0, fy=130.0, cx=80.0, cy=60.0, width=160, height=120)
    res = CyclesRenderPass(out_dir=tmp_path / "render", samples=16).render(
        scene, CameraTrack.identity(intr, 2), RenderQuality.FINAL
    )
    assert res.n_frames == 2
    manifest = (Path(res.uri) / "manifest.txt").read_text()
    assert "avatars=1" in manifest and "posed=1" in manifest  # the posed path ran, not rigid

    f0 = cv2.imread(str(Path(res.uri) / "frame_00000.png"))
    f1 = cv2.imread(str(Path(res.uri) / "frame_00001.png"))
    assert f0 is not None and f1 is not None
    changed = np.abs(f0.astype(int) - f1.astype(int)).max(axis=2) > 20
    n_changed = int(changed.sum())
    # The forearm swing moves a big block of pixels, but only locally — the background is identical
    # (constant Cycles seed), so this is geometry, not render noise. A rigid bug would give 0.
    assert n_changed > 80
    assert n_changed < 0.4 * f0.shape[0] * f0.shape[1]


# --- Blender + SMPL-X gated: an EDIT re-projects with no avatar rebuild (M2-10, AC-4) ----
@pytest.mark.skipif(
    locate_blender() is None or locate_smplx_model() is None,
    reason="needs a Blender binary and the SMPL-X model (.npz)",
)
def test_cycles_pose_edit_reprojects_without_avatar_rebuild(tmp_path, make_scene):
    # The M2-10 AC-4 proof: an *edit* — a Correction layered onto the scene, root *and* a limb —
    # re-projects into the Cycles frame with no avatar rebuild. We render the SAME avatar PLY twice:
    # once clean, once after appending a ROOT_TRANSLATION and a POSE_BODY_JOINT correction. The pass
    # resolves scene.corrections at render time (resolve_subject_motion → LBS), so the second frame
    # shows the moved root and bent elbow though the mesh on disk never changed. A pipeline that
    # baked the proposal into the asset (rebuild-required) would render two identical frames.
    import cv2

    from pitch3d.core.correction.engine import make_offset
    from pitch3d.core.scene.layers import CorrectionTarget, TargetKind

    model = SmplxModel.load(locate_smplx_model())
    betas = np.zeros(10)
    verts = model.shaped(betas)
    rgb = np.tile(np.array([[220, 40, 40]], dtype=np.uint8), (verts.shape[0], 1))
    measured = np.ones(verts.shape[0], dtype=bool)
    uri = write_vertex_colored_ply(tmp_path / "avatar_5.ply", verts, model.faces, rgb, measured)

    # One frame, arm abducted clear of the torso so an elbow edit shows against the sky (3/4 view).
    body = np.zeros((1, N_SMPLX_BODY_JOINTS, 3))
    body[0, 15] = [0.0, 0.0, -1.1]
    pose = PoseSequence(
        frames=np.array([0]),
        global_orient=np.array([[0.0, 1.2, 0.0]]),
        body_pose=body,
        transl=np.array([[0.0, 0.2, 2.4]]),
    )
    subj = Subject(track_id=5, proposal=SubjectMotion(SmplxShape(betas), pose))
    scene = make_scene(subjects=[subj])
    ref = RenderAssetRef(
        id="a-5", kind=RenderAssetKind.AVATAR_TEXTURED_SMPLX, uri=uri,
        model=ModelInfo(name="t", backend=Backend.FAKE), subject_track_id=5,
    )
    scene.render_assets = [ref]

    intr = CameraIntrinsics(fx=130.0, fy=130.0, cx=80.0, cy=60.0, width=160, height=120)
    cam = CameraTrack.identity(intr, 1)

    # Render 1: the proposal as-is (no edits).
    clean = CyclesRenderPass(out_dir=tmp_path / "clean", samples=16).render(
        scene, cam, RenderQuality.FINAL
    )

    # The edit: nudge the root sideways AND bend the left elbow (SMPL-X joint 18 == body_pose row
    # 17). We only append corrections to the SAME scene — the PLY asset is untouched, not rebuilt.
    scene.corrections.append(
        make_offset(
            "e-root",
            CorrectionTarget(kind=TargetKind.ROOT_TRANSLATION, subject_track_id=5),
            (0, 0),
            np.array([0.18, 0.0, 0.0]),
        )
    )
    scene.corrections.append(
        make_offset(
            "e-elbow",
            CorrectionTarget(kind=TargetKind.POSE_BODY_JOINT, subject_track_id=5, joint_index=17),
            (0, 0),
            np.array([1.8, 0.0, 1.8]),
        )
    )
    edited = CyclesRenderPass(out_dir=tmp_path / "edited", samples=16).render(
        scene, cam, RenderQuality.FINAL
    )

    f0 = cv2.imread(str(Path(clean.uri) / "frame_00000.png"))
    f1 = cv2.imread(str(Path(edited.uri) / "frame_00000.png"))
    assert f0 is not None and f1 is not None
    changed = np.abs(f0.astype(int) - f1.astype(int)).max(axis=2) > 20
    n_changed = int(changed.sum())
    # The edit moved the root and swung the forearm → a substantial but localized pixel change; the
    # grass/sky background is identical (default Cycles seed), so this is the edit, not noise.
    assert n_changed > 80
    assert n_changed < 0.6 * f0.shape[0] * f0.shape[1]
