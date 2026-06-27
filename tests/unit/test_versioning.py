"""Scene versioning — named snapshots + rollback (M3-6).

Corrections are the sole edit path (ADR-0002), so a snapshot checkpoints the whole Scene and a
rollback restores its correction stack. These pin the content-addressed fingerprint (ADR-0004:
equal content → equal digest, stable across deep copies), the store's deep-copy isolation (a
snapshot survives further edits to the live or restored scene), and the Application-level
reconstruct → snapshot → edit → rollback round-trip the LLM agent will drive (ADR-0008/0010).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from pitch3d.core.correction.engine import make_offset
from pitch3d.core.scene.layers import CorrectionTarget, TargetKind
from pitch3d.core.scene.subject import Subject
from pitch3d.core.scene.versioning import SnapshotStore, scene_fingerprint

_ROOT = CorrectionTarget(kind=TargetKind.ROOT_TRANSLATION, subject_track_id=1)


def _scene(make_scene, make_motion, transl_z=2.0):
    proposal = make_motion(range(4), transl_z=transl_z)
    return make_scene(subjects=[Subject(track_id=1, proposal=proposal)])


def _offset(delta=(1.0, 0.0, 0.0)):
    return make_offset("c1", _ROOT, (0, 3), np.asarray(delta, float))


def _z(app, scene_id, tid):
    return float(app.resolved(scene_id).subject(tid).proposal.pose.transl[0, 2])


# --- fingerprint (content-addressed identity, ADR-0004) ------------------------
def test_fingerprint_is_pure_and_stable_across_deep_copy(make_scene, make_motion):
    scene = _scene(make_scene, make_motion)
    fp = scene_fingerprint(scene)
    assert fp == scene_fingerprint(scene)                  # pure: same input, same digest
    assert fp == scene_fingerprint(copy.deepcopy(scene))   # stable across a deep copy


def test_fingerprint_changes_with_content(make_scene, make_motion):
    scene = _scene(make_scene, make_motion)
    fp = scene_fingerprint(scene)
    edited = copy.deepcopy(scene)
    edited.corrections.append(_offset())
    assert scene_fingerprint(edited) != fp  # the correction stack is part of the identity


# --- SnapshotStore (pure, in-memory) ------------------------------------------
def test_store_take_records_fingerprint_and_metadata(make_scene, make_motion):
    scene = _scene(make_scene, make_motion)
    store = SnapshotStore()
    snap = store.take(scene, "v1", note="first")
    assert (snap.name, snap.scene_id, snap.note) == ("v1", scene.id, "first")
    assert snap.fingerprint == scene_fingerprint(scene)
    assert store.names(scene.id) == ["v1"]
    assert store.get(scene.id, "v1") is snap


def test_restore_is_an_independent_deep_copy(make_scene, make_motion):
    scene = _scene(make_scene, make_motion)
    store = SnapshotStore()
    snap = store.take(scene, "v1")
    restored = store.restore(scene.id, "v1")
    assert restored is not snap.scene
    restored.corrections.append(_offset())                 # mutate the restored copy
    assert store.get(scene.id, "v1").scene.corrections == []  # snapshot stays pristine
    assert scene_fingerprint(store.restore(scene.id, "v1")) == snap.fingerprint


def test_take_overwrites_same_name(make_scene, make_motion):
    scene = _scene(make_scene, make_motion)
    store = SnapshotStore()
    store.take(scene, "v")
    scene.corrections.append(_offset())
    store.take(scene, "v")
    assert store.names(scene.id) == ["v"]                          # still one entry
    assert len(store.get(scene.id, "v").scene.corrections) == 1    # latest content wins


def test_missing_snapshot_and_empty_name_raise(make_scene, make_motion):
    store = SnapshotStore()
    with pytest.raises(KeyError):
        store.get("nope", "x")
    with pytest.raises(KeyError):
        store.restore("nope", "x")
    with pytest.raises(ValueError, match="non-empty"):
        store.take(_scene(make_scene, make_motion), "")


# --- Application round-trip (the agent/operator flow) --------------------------
def test_snapshot_then_rollback_reverts_edit(reconstructed):
    app, scene_id = reconstructed
    tid = app.get_scene(scene_id).subjects[0].track_id
    base_z = _z(app, scene_id, tid)
    fp_base = app.scene_fingerprint(scene_id)

    snap = app.snapshot(scene_id, "base")
    assert snap.fingerprint == fp_base and snap.created_at is not None

    app.apply_offset(scene_id, {"kind": "root_translation", "subject_track_id": tid}, (0, 2),
                     [0.0, 0.0, 1.0])
    assert app.scene_fingerprint(scene_id) != fp_base
    assert _z(app, scene_id, tid) == pytest.approx(base_z + 1.0)

    restored = app.rollback(scene_id, "base")
    assert restored.corrections == [] and app.get_scene(scene_id).corrections == []
    assert app.scene_fingerprint(scene_id) == fp_base
    assert _z(app, scene_id, tid) == pytest.approx(base_z)


def test_rollback_repeatable_after_further_edits(reconstructed):
    app, scene_id = reconstructed
    tid = app.get_scene(scene_id).subjects[0].track_id
    fp_base = app.scene_fingerprint(scene_id)
    app.snapshot(scene_id, "base")
    target = {"kind": "root_translation", "subject_track_id": tid}

    app.apply_offset(scene_id, target, (0, 2), [0.0, 0.0, 1.0])
    app.rollback(scene_id, "base")
    app.apply_offset(scene_id, target, (0, 2), [0.0, 0.0, 5.0])
    app.rollback(scene_id, "base")  # the checkpoint survives repeated rollbacks
    assert app.scene_fingerprint(scene_id) == fp_base


def test_list_snapshots_and_missing_scene(reconstructed):
    app, scene_id = reconstructed
    app.snapshot(scene_id, "a")
    app.snapshot(scene_id, "b", note="second")
    names = [s.name for s in app.list_snapshots(scene_id)]
    assert set(names) == {"a", "b"}
    with pytest.raises(KeyError):
        app.snapshot("no-such-scene", "x")
