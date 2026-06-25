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
    aggregate_observations,
    sample_vertex_colors,
    vertex_normals,
    write_vertex_colored_ply,
)
from pitch3d.core.ports.io import CropRef
from pitch3d.core.scene.assets import RenderAssetKind
from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack
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


# --- write_vertex_colored_ply -------------------------------------------------
def _parse_ply(path: Path) -> tuple[int, int, list[int]]:
    """Return (n_vertices, n_faces, measured flags) from our ASCII PLY."""
    lines = Path(path).read_text().splitlines()
    nv = next(int(ln.split()[-1]) for ln in lines if ln.startswith("element vertex"))
    nf = next(int(ln.split()[-1]) for ln in lines if ln.startswith("element face"))
    body = lines[lines.index("end_header") + 1:]
    measured = [int(body[i].split()[6]) for i in range(nv)]
    return nv, nf, measured


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

    def observe(self, subject: Subject, ref_crops) -> AvatarMeshObservations:  # noqa: ANN001
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


def test_default_backend_heavy_path_is_gated():
    # No backend injected → the real SMPL-X backend, which is an honest unwired stub. It raises
    # NotImplementedError (model present) or RuntimeError (avatar extra absent) — never silently.
    builder = TexturedSmplxAvatarBuilder(out_dir=Path("out/assets"))
    assert isinstance(builder._backend(), SmplxTextureBackend)
    with pytest.raises((NotImplementedError, RuntimeError)):
        builder._backend().observe(Subject(track_id=1, proposal=None), [])  # type: ignore[arg-type]


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
