"""MeasuredPitchEnvReconstructor — the measured env asset (M2-1).

The reconstructor emits the calibration-anchored pitch markings as a vertex-coloured PLY. These
tests check the honest contract: it is a *measured* template (every vertex ``measured=1``, so
``coverage == 1.0``), the geometry lands on the requested ground plane, and the asset is labelled
with the honest :data:`ENV_PITCH_MESH` kind (not an ``env_splat`` it is not).
"""

from __future__ import annotations

import numpy as np

from pitch3d.adapters.models.avatar import read_vertex_colored_ply
from pitch3d.adapters.models.env import _LINE_RGB, MeasuredPitchEnvReconstructor
from pitch3d.core.ports.io import ClipRef
from pitch3d.core.scene.assets import RenderAssetKind
from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack
from pitch3d.core.scene.units import FieldDimensions


def _clip() -> ClipRef:
    return ClipRef(
        source_id="clipA", uri="mem://clipA.mp4", frames=np.arange(3),
        width=1280, height=720, fps=25.0,
    )


def _camera(n: int = 3) -> CameraTrack:
    intr = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)
    return CameraTrack.identity(intr, n)


def test_reconstruct_writes_measured_pitch_ply(tmp_path):
    env = MeasuredPitchEnvReconstructor(out_dir=tmp_path, spacing=1.0)
    ref = env.reconstruct(_clip(), _camera())

    assert ref.kind is RenderAssetKind.ENV_PITCH_MESH
    assert ref.subject_track_id is None        # environment asset, not per-subject
    assert ref.uri.endswith("clipA_pitch.ply")
    assert ref.extra["coverage"] == 1.0 and ref.extra["n_vertices"] > 0


def test_pitch_ply_is_fully_measured_white_on_plane(tmp_path):
    env = MeasuredPitchEnvReconstructor(
        out_dir=tmp_path, dimensions=FieldDimensions(105.0, 68.0), plane_z=0.0, spacing=1.0
    )
    ref = env.reconstruct(_clip(), _camera())
    verts, faces, rgb, measured = read_vertex_colored_ply(ref.uri)

    assert faces.shape == (0, 3)                       # a point cloud of markings, no triangles
    assert measured.all()                              # nothing fabricated — all measured (R-6)
    np.testing.assert_array_equal(rgb[0], _LINE_RGB)   # white markings
    np.testing.assert_allclose(verts[:, 2], 0.0)       # on the ground plane
    np.testing.assert_allclose(np.abs(verts[:, 0]).max(), 52.5)  # spans the pitch length


def test_plane_z_is_respected(tmp_path):
    env = MeasuredPitchEnvReconstructor(out_dir=tmp_path, plane_z=0.5, spacing=2.0)
    ref = env.reconstruct(_clip(), _camera())
    verts, _f, _c, _m = read_vertex_colored_ply(ref.uri)
    np.testing.assert_allclose(verts[:, 2], 0.5)
