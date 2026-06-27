"""Per-subject Gaussian (3DGS) avatar builder — measured init + gated refiner (M3-1 #3).

Like the textured builder (#1), the *measured* half is checked with no torch/GPU/SMPL-X: one
anchored Gaussian per vertex, colour from the same measured sampling, scale from local spacing, and
— the R-6 crux — a vertex never seen stays ``measured=0`` with a *faint* opacity, never a confident
fabricated splat. The standard-3DGS ``.ply`` round-trips (SH colour / log scale / logit opacity +
our ``measured`` flag), the builder runs end-to-end over the dependency-free synthetic backend, and
the generative refiner (IDOL/LHM/GART) is importable but gated (R-8).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pitch3d.adapters.models.avatar import (
    AvatarMeshObservations,
    SyntheticAvatarMeshBackend,
)
from pitch3d.adapters.models.gaussian_avatar import (
    _MEASURED_OPACITY,
    _UNMEASURED_OPACITY,
    FeedForwardGaussianRefiner,
    GaussianAvatarBuilder,
    GaussianRefiner,
    _f_dc_to_rgb,
    _rgb_to_f_dc,
    _vertex_edge_scale,
    mesh_to_gaussians,
    read_gaussian_splat_ply,
    write_gaussian_splat_ply,
)
from pitch3d.app.wiring import default_ports
from pitch3d.core.ports.io import CropRef
from pitch3d.core.scene.assets import RenderAssetKind
from pitch3d.core.scene.subject import Subject

_TRI = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
_FACES = np.array([[0, 1, 2]])


# --- SH colour <-> f_dc round trip --------------------------------------------
def test_rgb_f_dc_round_trips_exactly():
    rgb = np.array([[0, 127, 255], [10, 20, 30], [200, 100, 50]], dtype=np.uint8)
    np.testing.assert_array_equal(_f_dc_to_rgb(_rgb_to_f_dc(rgb)), rgb)


# --- _vertex_edge_scale -------------------------------------------------------
def test_vertex_edge_scale_is_mean_incident_edge_length():
    # 3-4-5 triangle: edges 3,5,4 → per-vertex mean of its two incident edges, times factor 0.5.
    s = _vertex_edge_scale(_TRI, _FACES, factor=0.5)
    np.testing.assert_allclose(s, [1.75, 2.0, 2.25])


# --- mesh_to_gaussians --------------------------------------------------------
def test_mesh_to_gaussians_anchors_one_measured_splat_per_vertex():
    rgb = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    measured = np.array([True, False, True])
    ga = mesh_to_gaussians(_TRI, _FACES, rgb, measured)
    assert ga.n == 3
    np.testing.assert_allclose(ga.positions, _TRI)        # centre = the mesh vertex
    np.testing.assert_array_equal(ga.colors_rgb, rgb)
    assert ga.measured.tolist() == [True, False, True]
    # Unmeasured vertex stays faint, not invisible and not confident (R-6).
    np.testing.assert_allclose(
        ga.opacities, [_MEASURED_OPACITY, _UNMEASURED_OPACITY, _MEASURED_OPACITY]
    )
    np.testing.assert_array_equal(ga.rotations, np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)))
    assert np.all(ga.scales > 0.0)
    assert ga.coverage == pytest.approx(2 / 3)


# --- splat .ply round trip ----------------------------------------------------
def test_gaussian_splat_ply_round_trips(tmp_path):
    rgb = np.array([[10, 20, 30], [127, 127, 127], [200, 100, 50]], dtype=np.uint8)
    measured = np.array([True, False, True])
    ga = mesh_to_gaussians(_TRI, _FACES, rgb, measured)
    uri = write_gaussian_splat_ply(tmp_path / "rt.ply", ga)
    assert "property uchar measured" in Path(uri).read_text()
    assert "property float f_dc_0" in Path(uri).read_text()   # standard 3DGS props → loadable
    back = read_gaussian_splat_ply(Path(uri))
    np.testing.assert_allclose(back.positions, ga.positions, atol=1e-5)
    np.testing.assert_array_equal(back.colors_rgb, ga.colors_rgb)  # SH round trip is exact
    np.testing.assert_allclose(back.opacities, ga.opacities, atol=1e-5)
    np.testing.assert_allclose(back.scales, ga.scales, atol=1e-5)
    assert back.measured.dtype == bool and back.measured.tolist() == [True, False, True]


# --- GaussianAvatarBuilder over the dependency-free synthetic backend ----------
def test_build_writes_gaussian_asset_with_partial_coverage(tmp_path, make_motion):
    # The injectable no-SMPL-X/no-GPU backend runs the real measured path: 2 of 3 synthetic verts
    # land on the image, so the splat avatar is a genuine 2/3-coverage asset with one measured=0.
    builder = GaussianAvatarBuilder(mesh_backend=SyntheticAvatarMeshBackend(), out_dir=tmp_path)
    ref = builder.build(
        Subject(track_id=4, proposal=make_motion([0])), [CropRef(4, "c.png", 0, [0, 0, 1, 1])]
    )
    assert ref.kind == RenderAssetKind.AVATAR_GAUSSIAN
    assert ref.subject_track_id == 4
    assert ref.extra["n_vertices"] == 3 and ref.extra["n_gaussians"] == 3
    assert ref.extra["n_measured"] == 2
    assert ref.extra["coverage"] == pytest.approx(2 / 3)
    assert ref.extra["refined"] is False               # measured init only, no generative refiner
    back = read_gaussian_splat_ply(Path(ref.uri))
    assert int(back.measured.sum()) == 2


@dataclass
class _GeomOnlyBackend:
    """A backend with canonical geometry but no reference frames (every vertex unmeasured)."""

    def observe(self, subject, ref_crops, *, camera=None, clip=None) -> AvatarMeshObservations:  # noqa: ANN001
        return AvatarMeshObservations(canonical_vertices=_TRI, faces=_FACES, frames=[])


def test_build_geometry_only_is_honest(tmp_path, make_motion):
    # No frames → appearance unmeasured everywhere (R-6): coverage 0, every splat faint, asset still
    # written (we know the shape, not the look).
    builder = GaussianAvatarBuilder(mesh_backend=_GeomOnlyBackend(), out_dir=tmp_path)
    ref = builder.build(Subject(track_id=2, proposal=make_motion([0])), [])
    assert ref.extra["coverage"] == 0.0 and ref.extra["n_measured"] == 0
    back = read_gaussian_splat_ply(Path(ref.uri))
    assert not back.measured.any()
    np.testing.assert_allclose(back.opacities, _UNMEASURED_OPACITY, atol=1e-5)


# --- R-8: generative refiner is importable but gated --------------------------
def test_feedforward_refiner_satisfies_protocol_and_raises(make_motion):
    refiner = FeedForwardGaussianRefiner()
    assert isinstance(refiner, GaussianRefiner)
    ga = mesh_to_gaussians(_TRI, _FACES, np.zeros((3, 3), np.uint8), np.array([True, True, True]))
    with pytest.raises(NotImplementedError, match="avatar"):
        refiner.refine(ga, Subject(track_id=1, proposal=make_motion([0])), [])


def test_builder_with_refiner_engages_the_gated_path(tmp_path, make_motion):
    # A refiner is optional; when injected, the builder routes through it — so the gated generative
    # half surfaces its actionable error instead of silently running.
    builder = GaussianAvatarBuilder(
        mesh_backend=SyntheticAvatarMeshBackend(),
        refiner=FeedForwardGaussianRefiner(),
        out_dir=tmp_path,
    )
    with pytest.raises(NotImplementedError, match="avatar"):
        builder.build(Subject(track_id=3, proposal=make_motion([0])), [])


# --- wiring -------------------------------------------------------------------
def test_wiring_selects_gaussian_builder(tmp_path):
    avt = default_ports(out_dir=tmp_path / "o", avatar="gaussian", device="cuda").avatar
    assert isinstance(avt, GaussianAvatarBuilder)
    assert avt.device == "cuda" and avt.mesh_backend is None and avt.refiner is None


def test_wiring_injects_mesh_backend_for_gaussian(tmp_path):
    avt = default_ports(
        out_dir=tmp_path / "o",
        avatar="gaussian",
        avatar_backend="pitch3d.adapters.models.avatar:SyntheticAvatarMeshBackend",
    ).avatar
    assert isinstance(avt, GaussianAvatarBuilder)
    assert isinstance(avt.mesh_backend, SyntheticAvatarMeshBackend)
