"""Measured textured-SMPL-X avatar builder — projection/sampling/aggregation + asset (M2-2 #1).

The model-independent half is checked directly with **no torch, no GPU, no SMPL-X model**: a
synthetic camera + image + tiny mesh exercise the front-facing test, the in-frustum/z-buffer
visibility, the per-vertex colour averaging, and — the R-6 crux — that a vertex *never seen*
stays ``measured=0`` rather than getting a fabricated colour. The builder itself is exercised
end-to-end over a stub :class:`AvatarMeshBackend`, writing a real vertex-coloured PLY.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pitch3d.adapters.models.avatar import (
    _UNMEASURED_RGB,
    AvatarMeshBackend,
    AvatarMeshObservations,
    FrameObservation,
    SmplxTextureBackend,
    SyntheticAvatarMeshBackend,
    TexturedSmplxAvatarBuilder,
    _even_subset,
    aggregate_observations,
    bake_body_vertex_texture,
    read_vertex_colored_ply,
    sample_vertex_colors,
    vertex_normals,
    write_vertex_colored_ply,
)
from pitch3d.adapters.models.smplx_lbs import N_SMPLX_VERTS, locate_smplx_model
from pitch3d.core.ports.io import ClipRef, CropRef
from pitch3d.core.scene.assets import RenderAssetKind
from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.subject import Subject


def _intr(w: int = 64, h: int = 48) -> CameraIntrinsics:
    return CameraIntrinsics(fx=50.0, fy=50.0, cx=w / 2.0, cy=h / 2.0, width=w, height=h)


def _camera(n: int = 1) -> CameraTrack:
    """Identity camera at the world origin looking along +Z (R=I, t=0)."""
    return CameraTrack.identity(_intr(), n)


# --- vertex_normals -----------------------------------------------------------
def test_vertex_normals_unit_and_axis_aligned():
    # One triangle in the z=0 plane, wound CCW seen from +Z → normal along +Z, unit length.
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    n = vertex_normals(verts, faces)
    assert n.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(n, np.tile([0.0, 0.0, 1.0], (3, 1)), atol=1e-9)


# --- sample_vertex_colors -----------------------------------------------------
def _image_with_pixel(x: int, y: int, color, w: int = 64, h: int = 48) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[y, x] = color
    return img


def test_sample_front_facing_vertex_samples_its_pixel():
    # Vertex straight ahead at (0,0,5) projects to the principal point (cx,cy)=(32,24).
    verts = np.array([[0.0, 0.0, 5.0]])
    normals = np.array([[0.0, 0.0, -1.0]])  # points back toward the camera at the origin
    img = _image_with_pixel(32, 24, (10, 200, 30))
    colors, observed = sample_vertex_colors(verts, normals, _camera(), 0, img)
    assert observed.tolist() == [True]
    np.testing.assert_array_equal(colors[0], [10, 200, 30])


def test_sample_back_facing_vertex_rejected():
    verts = np.array([[0.0, 0.0, 5.0]])
    normals = np.array([[0.0, 0.0, 1.0]])  # faces AWAY from the camera
    img = _image_with_pixel(32, 24, (10, 200, 30))
    colors, observed = sample_vertex_colors(verts, normals, _camera(), 0, img)
    assert observed.tolist() == [False]
    np.testing.assert_array_equal(colors[0], [0, 0, 0])


def test_sample_vertex_behind_camera_rejected():
    verts = np.array([[0.0, 0.0, -5.0]])           # behind the camera
    normals = np.array([[0.0, 0.0, -1.0]])
    img = np.full((48, 64, 3), 7, dtype=np.uint8)
    _colors, observed = sample_vertex_colors(verts, normals, _camera(), 0, img)
    assert observed.tolist() == [False]


def test_sample_zbuffer_keeps_nearest_at_shared_pixel():
    # Two front-facing vertices on the optical axis → same pixel; only the nearer one wins.
    verts = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 9.0]])
    normals = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    img = _image_with_pixel(32, 24, (123, 45, 67))
    colors, observed = sample_vertex_colors(verts, normals, _camera(), 0, img)
    assert observed.tolist() == [True, False]      # nearer (z=3) observed, farther (z=9) occluded
    np.testing.assert_array_equal(colors[0], [123, 45, 67])


# --- aggregate_observations ---------------------------------------------------
def test_aggregate_averages_seen_and_flags_unseen():
    # vertex 0 seen in both frames (avg of 10 and 20 = 15); vertex 1 seen in neither.
    c0 = np.array([[10, 10, 10], [0, 0, 0]], dtype=np.uint8)
    o0 = np.array([True, False])
    c1 = np.array([[20, 20, 20], [0, 0, 0]], dtype=np.uint8)
    o1 = np.array([True, False])
    rgb, count = aggregate_observations([c0, c1], [o0, o1], n_vertices=2)
    assert count.tolist() == [2, 0]
    np.testing.assert_array_equal(rgb[0], [15, 15, 15])
    np.testing.assert_array_equal(rgb[1], list(_UNMEASURED_RGB))  # honest placeholder, not faked


# --- bake_body_vertex_texture (image-level core of the clip baker) -------------
def test_bake_body_vertex_texture_measures_seen_verts_and_zeros_unseen():
    # Same triangle as the builder stub: facing the camera, v0 -> the principal point and v2 -> an
    # in-frame pixel (both measured), v1 projects off the right edge (never measured). Two identical
    # reference frames of one solid colour, so every measured vertex carries that colour in [0,1]
    # while the unseen v1 comes back measured=False with vcolor 0 (R-6), NOT the grey placeholder.
    verts_world = np.array([[0.0, 0.0, 5.0], [5.0, 0.0, 5.0], [0.0, 0.5, 5.0]])
    faces = np.array([[0, 2, 1]])  # wound so the face normal points back toward the camera
    img = np.empty((48, 64, 3), np.uint8)
    img[:] = (200, 100, 50)
    verts_per_frame = np.stack([verts_world, verts_world])  # (2 frames, 3 verts, 3)
    vcolor, measured = bake_body_vertex_texture(
        verts_per_frame, faces, _camera(2), [0, 1], [img, img]
    )
    assert vcolor.shape == (3, 3) and measured.tolist() == [True, False, True]
    np.testing.assert_allclose(vcolor[0], np.array([200, 100, 50]) / 255.0, atol=1e-6)
    np.testing.assert_allclose(vcolor[2], np.array([200, 100, 50]) / 255.0, atol=1e-6)
    np.testing.assert_array_equal(vcolor[1], [0.0, 0.0, 0.0])


# --- write_vertex_colored_ply -------------------------------------------------
def _parse_ply(path: Path) -> tuple[int, int, list[int]]:
    """Return (n_vertices, n_faces, measured flags) from our ASCII PLY."""
    lines = Path(path).read_text().splitlines()
    nv = next(int(ln.split()[-1]) for ln in lines if ln.startswith("element vertex"))
    nf = next(int(ln.split()[-1]) for ln in lines if ln.startswith("element face"))
    body = lines[lines.index("end_header") + 1:]
    measured = [int(body[i].split()[6]) for i in range(nv)]
    return nv, nf, measured


def test_read_vertex_colored_ply_round_trips(tmp_path):
    # The reader is the exact inverse of the writer — geometry, colour and the R-6 measured flag all
    # survive, so a RenderPass consuming the asset sees what the builder wrote.
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    faces = np.array([[0, 1, 2]])
    rgb = np.array([[10, 20, 30], [127, 127, 127], [200, 100, 50]], dtype=np.uint8)
    measured = np.array([True, False, True])
    uri = write_vertex_colored_ply(tmp_path / "rt.ply", verts, faces, rgb, measured)
    v, f, c, m = read_vertex_colored_ply(Path(uri))
    np.testing.assert_allclose(v, verts)
    np.testing.assert_array_equal(f, faces)
    np.testing.assert_array_equal(c, rgb)
    assert m.dtype == bool and m.tolist() == [True, False, True]


def test_write_ply_carries_geometry_color_and_measured_flag(tmp_path):
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    rgb = np.array([[1, 2, 3], [4, 5, 6], [127, 127, 127]], dtype=np.uint8)
    measured = np.array([True, True, False])
    uri = write_vertex_colored_ply(tmp_path / "a.ply", verts, faces, rgb, measured)
    assert "property uchar measured" in Path(uri).read_text()
    nv, nf, flags = _parse_ply(Path(uri))
    assert (nv, nf) == (3, 1)
    assert flags == [1, 1, 0]


# --- TexturedSmplxAvatarBuilder over a stub backend ---------------------------
@dataclass
class _StubBackend:
    """A tiny mesh + one frame: vertex 0 is seen, vertex 1 faces away (never measured)."""

    observations: AvatarMeshObservations

    def observe(  # noqa: ANN001
        self, subject: Subject, ref_crops, *, camera=None, clip=None
    ) -> AvatarMeshObservations:
        return self.observations


def _triangle_obs(seen_color=(200, 100, 50)) -> AvatarMeshObservations:
    # One triangle facing the camera at the origin (face normal points back toward -Z). All three
    # vertices are front-facing, but only two land on the image: v0 → the principal point (the lone
    # coloured pixel) and v2 → an in-frame black pixel, while v1 projects off the right edge and is
    # never measured. So the builder yields genuine *partial* coverage (2/3) with one honestly
    # unmeasured vertex (R-6) — not a fabricated colour.
    verts_world = np.array([[0.0, 0.0, 5.0], [5.0, 0.0, 5.0], [0.0, 0.5, 5.0]])
    canonical = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 2, 1]])  # wound so the face normal points back toward the camera
    img = _image_with_pixel(32, 24, seen_color)
    fo = FrameObservation(frame=0, vertices_world=verts_world, camera=_camera(), image=img)
    return AvatarMeshObservations(canonical_vertices=canonical, faces=faces, frames=[fo])


def test_runtime_checkable_protocol_accepts_stub():
    assert isinstance(_StubBackend(_triangle_obs()), AvatarMeshBackend)


def test_build_writes_textured_asset_with_partial_coverage(tmp_path, make_motion):
    obs = _triangle_obs(seen_color=(200, 100, 50))
    builder = TexturedSmplxAvatarBuilder(backend=_StubBackend(obs), out_dir=tmp_path)
    subject = Subject(track_id=7, proposal=make_motion([0]))
    ref = builder.build(subject, [CropRef(7, "crop.png", 0, [0, 0, 10, 10])])

    assert ref.kind == RenderAssetKind.AVATAR_TEXTURED_SMPLX
    assert ref.subject_track_id == 7
    assert ref.extra["n_vertices"] == 3
    assert ref.extra["frames_used"] == 1
    # Two of three front-facing verts land on the image (v1 projects off-frame), so coverage is a
    # genuine partial 2/3 — the measured fraction of the body surface, in (0, 1].
    assert 0.0 < ref.extra["coverage"] <= 1.0
    assert ref.extra["n_measured"] == int(round(ref.extra["coverage"] * 3))
    nv, nf, flags = _parse_ply(Path(ref.uri))
    assert (nv, nf) == (3, 1)
    assert sum(flags) == ref.extra["n_measured"]


def test_build_with_no_frames_is_geometry_only_and_honest(tmp_path, make_motion):
    # No reference frames → appearance is UNMEASURED everywhere (R-6): coverage 0, all flags 0,
    # but the geometry asset is still written (we know the shape, not the look).
    canonical = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    obs = AvatarMeshObservations(
        canonical_vertices=canonical, faces=np.array([[0, 1, 2]]), frames=[]
    )
    builder = TexturedSmplxAvatarBuilder(backend=_StubBackend(obs), out_dir=tmp_path)
    ref = builder.build(Subject(track_id=2, proposal=make_motion([0])), [])
    assert ref.extra["coverage"] == 0.0
    assert ref.extra["n_measured"] == 0
    _nv, _nf, flags = _parse_ply(Path(ref.uri))
    assert flags == [0, 0, 0]


def test_default_backend_geometry_half_wired_or_gated(make_motion):
    # No backend injected → the real SMPL-X backend. M2-8a wired its geometry: with the model
    # present it returns the canonical shaped mesh, geometry-only (frames=[], every vertex
    # measured=0 until M2-8b). Without the model it raises an actionable RuntimeError — never a
    # silent fabricated mesh.
    builder = TexturedSmplxAvatarBuilder(out_dir=Path("out/assets"))
    backend = builder._backend()
    assert isinstance(backend, SmplxTextureBackend)
    subject = Subject(track_id=1, proposal=make_motion([0]))
    if locate_smplx_model() is None:
        with pytest.raises(RuntimeError):
            backend.observe(subject, [])
        return
    obs = backend.observe(subject, [])
    assert obs.canonical_vertices.shape == (N_SMPLX_VERTS, 3)
    assert obs.faces.shape == (20908, 3)
    assert obs.frames == []


# --- SyntheticAvatarMeshBackend (the injectable, dependency-free measured path) -------------
def test_synthetic_backend_drives_builder_to_measured_ply(tmp_path, make_motion):
    # The injectable no-SMPL-X/no-GPU backend runs the *real* projection → sampling → aggregation
    # path end-to-end: of its 3 synthetic verts, two land on the image and one projects off-frame,
    # so the builder yields a genuine 2/3-coverage textured PLY with one honest measured=0 (R-6).
    builder = TexturedSmplxAvatarBuilder(backend=SyntheticAvatarMeshBackend(), out_dir=tmp_path)
    ref = builder.build(Subject(track_id=3, proposal=make_motion([0])), [])
    assert ref.kind == RenderAssetKind.AVATAR_TEXTURED_SMPLX
    assert ref.extra["n_vertices"] == 3
    assert ref.extra["n_measured"] == 2
    assert 0.0 < ref.extra["coverage"] < 1.0
    nv, nf, flags = _parse_ply(Path(ref.uri))
    assert (nv, nf) == (3, 1)
    assert sum(flags) == 2


# --- _even_subset (reference-frame budget) ------------------------------------
def test_even_subset_caps_and_spreads():
    # Picks endpoints-included even spread when over the cap, all of them when under, none at k=0.
    assert _even_subset(np.arange(10, 21), 3).tolist() == [10, 15, 20]
    assert _even_subset(np.arange(3), 8).tolist() == [0, 1, 2]
    assert _even_subset(np.arange(5), 0).tolist() == []


# --- SmplxTextureBackend measured path (gated: needs the SMPL-X model .npz) ----
def _facing_subject(track_id: int, frames, *, transl_z: float = 4.0) -> Subject:
    """A subject turned 180° about Y and set ``transl_z`` m down the camera's +Z axis.

    With the identity camera at the origin this puts the SMPL-X body in front of the lens facing
    back toward it, so a healthy *fraction* of its surface verts are front-facing + in-frustum — a
    genuine R-6 partial, never the whole body from one view (the back stays unmeasured).
    """
    frames = np.asarray(frames, dtype=int).reshape(-1)
    t = frames.shape[0]
    pose = PoseSequence(
        frames=frames,
        global_orient=np.tile([0.0, np.pi, 0.0], (t, 1)),
        body_pose=np.zeros((t, 21, 3)),
        transl=np.tile([0.0, 0.0, transl_z], (t, 1)),
    )
    motion = SubjectMotion(shape=SmplxShape(betas=np.zeros(10)), pose=pose)
    return Subject(track_id=track_id, proposal=motion)


def _solid_frame_clip(tmp_path, cv2, *, bgr, n: int = 3, w: int = 160, h: int = 120) -> ClipRef:
    """Write ``n`` solid-colour PNG frames (OpenCV BGR); return a ClipRef over that dir."""
    d = tmp_path / "frames"
    d.mkdir()
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = bgr
    for i in range(n):
        cv2.imwrite(str(d / f"{i:03d}.png"), img)
    return ClipRef(source_id="t", uri=str(d), frames=np.arange(n), width=w, height=h, fps=25.0)


@pytest.mark.skipif(locate_smplx_model() is None, reason="needs the SMPL-X model .npz")
def test_smplx_backend_samples_measured_pixels_from_broadcast_frames(tmp_path):
    # The full measured path on a real SMPL-X body: shape -> pose at the measured pose into each of
    # three decoded broadcast frames -> front-facing/z-buffer sampling -> averaged vertex colour.
    # The frames are one solid colour, so every *measured* vertex must carry exactly that colour
    # flipped cv2-BGR -> stored-RGB; the back of the body is never seen (honest partial coverage).
    cv2 = pytest.importorskip("cv2")
    w, h = 160, 120
    intr = CameraIntrinsics(fx=200.0, fy=200.0, cx=w / 2.0, cy=h / 2.0, width=w, height=h)
    camera = CameraTrack.identity(intr, 3)
    clip = _solid_frame_clip(tmp_path, cv2, bgr=(30, 200, 10), n=3, w=w, h=h)  # RGB (10,200,30)
    subject = _facing_subject(5, [0, 1, 2])

    backend = SmplxTextureBackend()
    obs = backend.observe(subject, [], camera=camera, clip=clip)
    assert len(obs.frames) == 3
    assert obs.canonical_vertices.shape == (N_SMPLX_VERTS, 3)
    fo = obs.frames[0]
    assert fo.image.shape == (h, w, 3)
    assert fo.vertices_world.shape == (N_SMPLX_VERTS, 3)

    builder = TexturedSmplxAvatarBuilder(backend=backend, out_dir=tmp_path)
    ref = builder.build(subject, [], camera=camera, clip=clip)
    assert ref.extra["frames_used"] == 3
    assert ref.extra["n_measured"] > 100              # a substantial chunk of the front surface
    assert 0.0 < ref.extra["coverage"] < 1.0          # genuine partial — the back is never measured
    _v, _f, colors, measured = read_vertex_colored_ply(Path(ref.uri))
    # Every measured vertex sampled the one solid colour, proving the cv2-BGR -> RGB flip.
    np.testing.assert_array_equal(np.unique(colors[measured], axis=0), [[10, 200, 30]])


@pytest.mark.skipif(locate_smplx_model() is None, reason="needs the SMPL-X model .npz")
def test_smplx_backend_geometry_only_when_source_not_real():
    # Camera present but the clip URI has no pixels behind it (synthetic/dry-run source) → geometry
    # only: no frames, every vertex measured=0. It must never fabricate appearance.
    intr = CameraIntrinsics(fx=200.0, fy=200.0, cx=80.0, cy=60.0, width=160, height=120)
    camera = CameraTrack.identity(intr, 3)
    clip = ClipRef(
        source_id="t", uri="memory://demo.mp4", frames=np.arange(3), width=160, height=120, fps=25.0
    )
    obs = SmplxTextureBackend().observe(_facing_subject(9, [0, 1, 2]), [], camera=camera, clip=clip)
    assert obs.frames == []
    assert obs.canonical_vertices.shape == (N_SMPLX_VERTS, 3)
