"""Exporter adapters — shared port contract + real SMPL-X npz / glTF assembly (FR-26/28, M1).

The real :class:`GltfExporter` is exercised with **no pygltflib, no GPU**: its dependency-free
halves (canonical JSON via the core serializer, resolved SMPL-X ``.npz``, and the Z-up→Y-up glTF
scene-graph assembly) are checked directly, and the gated ``.gltf``/``.glb`` serialization is only
asserted to fail *actionably* when the ``export`` extra is absent — the same discipline the model
adapters follow.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from pitch3d.adapters.export.gltf import (
    GltfExporter,
    build_gltf_scene,
    zup_to_yup,
)
from pitch3d.adapters.fakes import FakeExporter
from pitch3d.core.correction.engine import make_offset
from pitch3d.core.ports.export import Exporter, ExportFormat, ExportResult
from pitch3d.core.scene.layers import CorrectionTarget, TargetKind
from pitch3d.core.scene.motion import BallTrack
from pitch3d.core.scene.serialization import load_scene
from pitch3d.core.scene.subject import Subject

_UNSUPPORTED = [ExportFormat.USD, ExportFormat.FBX, ExportFormat.ALEMBIC, ExportFormat.THREEJS]


def _subject(make_motion, transl_z=2.0, track_id=1) -> Subject:
    return Subject(track_id=track_id, proposal=make_motion(range(4), transl_z=transl_z))


def _ball() -> BallTrack:
    return BallTrack(
        frames=np.arange(4), positions_3d=np.tile([1.0, 0.0, 3.0], (4, 1)),
        height_confidence=np.ones(4),
    )


# --- pure assembly / axis maths ------------------------------------------------
def test_zup_to_yup_maps_axes():
    np.testing.assert_allclose(zup_to_yup([1.0, 2.0, 3.0])[0], [1.0, 3.0, -2.0])
    np.testing.assert_allclose(zup_to_yup([0.0, 0.0, 1.0])[0], [0.0, 1.0, 0.0])  # +Z → +Y


def test_build_gltf_scene_tracks_subject_and_ball(make_scene, make_motion):
    scene = make_scene(subjects=[_subject(make_motion, transl_z=5.0)], ball=_ball())
    gscene = build_gltf_scene(scene, fps=25.0)
    assert [n.name for n in gscene.nodes] == ["subject_1", "ball"]
    np.testing.assert_allclose(gscene.nodes[0].times, np.arange(4) / 25.0)
    np.testing.assert_allclose(gscene.nodes[0].translations, np.tile([0.0, 5.0, 0.0], (4, 1)))
    np.testing.assert_allclose(gscene.nodes[1].translations, np.tile([1.0, 3.0, 0.0], (4, 1)))


def test_build_gltf_scene_omits_ball_when_absent(make_scene, make_motion):
    gscene = build_gltf_scene(make_scene(subjects=[_subject(make_motion)]))
    assert [n.name for n in gscene.nodes] == ["subject_1"]


# --- adapter behaviour ---------------------------------------------------------
@pytest.mark.parametrize("make_exp", [FakeExporter, GltfExporter], ids=["fake", "gltf"])
def test_exporter_port_contract(make_exp, tmp_path, make_scene, make_motion):
    exp = make_exp()
    assert isinstance(exp, Exporter)
    scene = make_scene(subjects=[_subject(make_motion)])
    path = tmp_path / "scene.json"
    result = exp.export(scene, ExportFormat.JSON, str(path))
    assert isinstance(result, ExportResult) and result.fmt is ExportFormat.JSON
    assert result.paths == [str(path)]
    assert load_scene(str(path)).id == scene.id  # genuinely reloadable
    assert exp.supports(ExportFormat.JSON) is True


def test_gltf_supports_matrix():
    exp = GltfExporter()
    for fmt in (ExportFormat.JSON, ExportFormat.SMPLX_NPZ, ExportFormat.GLTF, ExportFormat.GLB):
        assert exp.supports(fmt) is True
    for fmt in _UNSUPPORTED:
        assert exp.supports(fmt) is False


def test_smplx_npz_roundtrips_resolved_motion(tmp_path, make_scene, make_motion):
    subj = _subject(make_motion, transl_z=2.0)
    scene = make_scene(subjects=[subj])
    result = GltfExporter().export(scene, ExportFormat.SMPLX_NPZ, str(tmp_path / "npz"))
    data = np.load(tmp_path / "npz" / "subject_1.npz")
    np.testing.assert_array_equal(data["frames"], subj.proposal.pose.frames)
    np.testing.assert_allclose(data["betas"], subj.proposal.shape.betas)
    np.testing.assert_allclose(data["global_orient"], subj.proposal.pose.global_orient)
    np.testing.assert_allclose(data["body_pose"], subj.proposal.pose.body_pose)
    np.testing.assert_allclose(data["transl"], subj.proposal.pose.transl)
    assert data["body_model"].item() == "SMPL-X"
    assert result.paths == [str(tmp_path / "npz" / "subject_1.npz")]


def test_smplx_npz_reflects_resolved_corrections(tmp_path, make_scene, make_motion):
    subj = _subject(make_motion, transl_z=2.0)
    delta = np.array([1.0, -0.5, 3.0])
    corr = make_offset(
        "c1", CorrectionTarget(kind=TargetKind.ROOT_TRANSLATION, subject_track_id=1), (0, 3), delta
    )
    scene = make_scene(subjects=[subj], corrections=[corr])
    GltfExporter().export(scene, ExportFormat.SMPLX_NPZ, str(tmp_path / "npz"))
    data = np.load(tmp_path / "npz" / "subject_1.npz")
    np.testing.assert_allclose(data["transl"], subj.proposal.pose.transl + delta)  # proposal ⊕ corr


def test_export_is_non_destructive(tmp_path, make_scene, make_motion):
    subj = _subject(make_motion, transl_z=2.0)
    before = subj.proposal.pose.transl.copy()
    GltfExporter().export(make_scene(subjects=[subj]), ExportFormat.SMPLX_NPZ, str(tmp_path / "n"))
    np.testing.assert_array_equal(subj.proposal.pose.transl, before)


def test_unsupported_format_raises(tmp_path, make_scene):
    with pytest.raises(ValueError, match="unsupported"):
        GltfExporter().export(make_scene(), ExportFormat.USD, str(tmp_path / "x.usd"))


@pytest.mark.skipif(
    importlib.util.find_spec("pygltflib") is not None, reason="export extra installed"
)
def test_gltf_without_extra_is_actionable(tmp_path, make_scene, make_motion):
    # The assembly is real; only the pygltflib serialization is gated → a clear install pointer.
    scene = make_scene(subjects=[_subject(make_motion)])
    with pytest.raises(RuntimeError, match=r"pitch3d\[export\]"):
        GltfExporter().export(scene, ExportFormat.GLB, str(tmp_path / "a.glb"))
