"""Orchestration — cache keying/memoization, fakes satisfy the ports, assemble/resolve_scene."""

from __future__ import annotations

from pathlib import Path

import pytest

from pitch3d.adapters.fakes import InProcessJobQueue, MemoryCache
from pitch3d.core.orchestration import resolve_scene
from pitch3d.core.orchestration.stages import Stage, run_cached
from pitch3d.core.ports.cache import Cache, content_key
from pitch3d.core.ports.export import Exporter
from pitch3d.core.ports.jobs import JobQueue
from pitch3d.core.ports.observation import SceneObserver
from pitch3d.core.ports.perception import BallTracker, Detector, FieldCalibrator, Tracker
from pitch3d.core.ports.pose import PoseEstimator
from pitch3d.core.ports.reconstruction import AvatarBuilder, EnvReconstructor
from pitch3d.core.ports.render import RenderPass
from pitch3d.core.ports.view_synthesizer import ViewSynthesizer
from pitch3d.core.scene.assets import RenderAssetKind


def test_content_key_canonical_and_sensitive():
    k1 = content_key("POSE", "h", {"a": 1, "b": 2}, "v0")
    assert k1 == content_key("POSE", "h", {"b": 2, "a": 1}, "v0")  # key order independent
    assert content_key("POSE", "h", {"a": 1}, "v0") != content_key("POSE", "h", {"a": 2}, "v0")
    assert content_key("POSE", "h", {}, "v0") != content_key("POSE", "h", {}, "v1")
    assert k1.startswith("POSE-")


def test_run_cached_memoizes_thunk():
    calls = {"n": 0}

    def thunk():
        calls["n"] += 1
        return 42

    queue, cache = InProcessJobQueue(), MemoryCache()
    r1 = run_cached(queue, cache, Stage.DETECT, thunk, input_hash="h", params={}, model_version="v0")
    r2 = run_cached(queue, cache, Stage.DETECT, thunk, input_hash="h", params={}, model_version="v0")
    assert r1.result == 42 and r2.result == 42
    assert calls["n"] == 1  # second run is a cache hit, thunk not re-executed


def test_fakes_satisfy_all_ports(tmp_path):
    from pitch3d.app.wiring import default_ports

    p = default_ports(out_dir=tmp_path / "out")
    pairs = [
        (p.detector, Detector), (p.tracker, Tracker), (p.calibrator, FieldCalibrator),
        (p.pose, PoseEstimator), (p.ball, BallTracker), (p.env, EnvReconstructor),
        (p.avatar, AvatarBuilder), (p.viewsynth, ViewSynthesizer), (p.observer, SceneObserver),
        (p.render, RenderPass), (p.exporter, Exporter), (p.cache, Cache), (p.queue, JobQueue),
    ]
    for impl, port in pairs:
        assert isinstance(impl, port), f"{type(impl).__name__} is not a {port.__name__}"


def test_assemble_maps_roles_sorts_and_lifts_ball(reconstructed):
    app, scene_id = reconstructed
    scene = app.get_scene(scene_id)
    track_ids = [s.track_id for s in scene.subjects]
    assert track_ids == sorted(track_ids)  # deterministic order
    by_id = {s.track_id: s.role.value for s in scene.subjects}
    assert by_id[min(by_id)] == "goalkeeper"
    assert by_id[max(by_id)] == "referee"
    assert scene.ball is not None and scene.ball.frames.shape[0] > 0


def test_resolve_scene_bakes_corrections_without_mutating_input(reconstructed):
    app, scene_id = reconstructed
    stored = app.get_scene(scene_id)
    track_id = stored.subjects[0].track_id
    app.apply_offset(scene_id, {"kind": "root_translation", "subject_track_id": track_id},
                     (0, 2), [0, 0, 1.0])
    resolved = resolve_scene(stored, refit_port=app.ports.pose)
    assert len(stored.corrections) == 1   # input scene keeps its stack
    assert len(resolved.corrections) == 0  # resolved copy is baked empty
    dz = (resolved.subject(track_id).proposal.pose.transl[0, 2]
          - stored.subject(track_id).proposal.pose.transl[0, 2])
    assert dz == pytest.approx(1.0)


# --- build_avatars (the AVATAR controller stage) ------------------------------
def test_build_avatars_attaches_one_ref_per_subject(reconstructed):
    app, scene_id = reconstructed
    refs = app.build_avatars(scene_id)
    scene = app.get_scene(scene_id)
    assert len(refs) == len(scene.subjects) > 0
    assert all(r.kind == RenderAssetKind.AVATAR_TEXTURED_SMPLX for r in refs)
    # attached to the STORED scene, one stable id per subject (render/export can consume them)
    assert {a.id for a in scene.render_assets} == {f"avatar-{s.track_id}" for s in scene.subjects}


def test_build_avatars_is_idempotent_same_id_replaces(reconstructed):
    app, scene_id = reconstructed
    app.build_avatars(scene_id)
    first = list(app.get_scene(scene_id).render_assets)
    app.build_avatars(scene_id)                 # rebuild over the same subjects
    second = app.get_scene(scene_id).render_assets
    assert len(second) == len(first)            # same-id refs replace, never accumulate
    assert {a.id for a in second} == {a.id for a in first}


def test_build_avatars_textured_synthetic_backend_writes_measured_ply(tmp_path, clip):
    # End-to-end controller path on the REAL measured-projection adapter, fed by the injectable
    # dependency-free backend (no SMPL-X, no GPU): each subject yields a genuine 2/3-coverage
    # textured PLY carrying the per-vertex measured flag — one vertex honestly unmeasured (R-6).
    from pitch3d.app import build_app
    from pitch3d.app.wiring import default_ports

    backend = "pitch3d.adapters.models.avatar:SyntheticAvatarMeshBackend"
    ports = default_ports(out_dir=tmp_path / "out", avatar="textured", avatar_backend=backend)
    app = build_app(out_dir=tmp_path / "out", ports=ports)
    scene_id = app.run_reconstruction(app.register_clip(clip, name="t").id)

    refs = app.build_avatars(scene_id)
    assert refs, "reconstructed scene has subjects"
    for r in refs:
        assert r.kind == RenderAssetKind.AVATAR_TEXTURED_SMPLX
        assert r.extra["n_vertices"] == 3 and r.extra["n_measured"] == 2
        assert 0.0 < r.extra["coverage"] < 1.0          # partial coverage, not fabricated
        assert "property uchar measured" in Path(r.uri).read_text()
