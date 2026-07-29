"""Both sides of the anim export↔render contract must fail loudly on drift."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from pitch3d.adapters.blender import anim_contract as contract


def _write_subject(dirpath, track_id=3, extra=()):
    keys = dict(
        verts=np.zeros((2, 4, 3), np.float32),
        faces=np.zeros((4, 3), np.int32),
        color=np.zeros(3, np.float32),
        frames=np.arange(2, dtype=np.int64),
        alpha=np.ones(2, np.float32),
        provenance=np.array(["measured", "imputed"]),
    )
    for name in extra:
        keys[name] = np.zeros(2, np.float32)
    fname = f"anim_subject_{track_id}.npz"
    np.savez(os.path.join(dirpath, fname), **keys)
    return fname, sorted(keys)


def test_roundtrip_happy_path(tmp_path):
    fname, keys = _write_subject(tmp_path, extra=("vcolor",))
    contract.write_manifest(str(tmp_path), {fname: keys})
    files = contract.load_manifest(str(tmp_path))
    assert files[fname] == keys


def test_missing_manifest_tells_you_to_re_export(tmp_path):
    _write_subject(tmp_path)
    with pytest.raises(contract.ContractError, match="Re-run anim_export"):
        contract.load_manifest(str(tmp_path))


def test_schema_version_mismatch_is_rejected(tmp_path):
    fname, keys = _write_subject(tmp_path)
    contract.write_manifest(str(tmp_path), {fname: keys})
    p = tmp_path / contract.MANIFEST_NAME
    payload = json.loads(p.read_text())
    payload["schema_version"] = 99
    p.write_text(json.dumps(payload))
    with pytest.raises(contract.ContractError, match="schema"):
        contract.load_manifest(str(tmp_path))


def test_listed_file_must_exist(tmp_path):
    fname, keys = _write_subject(tmp_path)
    contract.write_manifest(str(tmp_path), {fname: keys})
    os.remove(tmp_path / fname)
    with pytest.raises(contract.ContractError, match="missing from"):
        contract.load_manifest(str(tmp_path))


def test_npz_must_carry_the_manifest_keys(tmp_path):
    fname, keys = _write_subject(tmp_path)
    contract.write_manifest(str(tmp_path), {fname: keys + ["vcolor"]})
    with pytest.raises(contract.ContractError, match="vcolor"):
        contract.load_manifest(str(tmp_path))


def test_writer_rejects_missing_required_keys(tmp_path):
    fname, _ = _write_subject(tmp_path)
    with pytest.raises(contract.ContractError, match="alpha"):
        contract.write_manifest(str(tmp_path), {fname: ["verts", "faces", "color", "frames"]})


def test_writer_rejects_unknown_artifacts(tmp_path):
    with pytest.raises(contract.ContractError, match="unknown"):
        contract.write_manifest(str(tmp_path), {"mystery.npz": ["x"]})


def test_manifest_without_subjects_is_an_empty_export(tmp_path):
    np.savez(tmp_path / "ball.npz", frames=np.arange(2), positions_3d=np.zeros((2, 3)),
             height_confidence=np.ones(2), mode=np.array(["on_ground", "ballistic"]))
    keys = ["frames", "positions_3d", "height_confidence", "mode"]
    contract.write_manifest(str(tmp_path), {"ball.npz": keys})
    with pytest.raises(contract.ContractError, match="no anim_subject"):
        contract.load_manifest(str(tmp_path))
