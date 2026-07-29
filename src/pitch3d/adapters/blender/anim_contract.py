"""Versioned data contract between the anim exporter and the Blender animate script.

The deliverable video path is two processes that can only talk through files:
``pitch3d.app.anim_export`` (pipeline venv: torch/smplx) writes a directory of ``.npz``
artifacts, and ``scripts/blender_animate.py`` (Blender's own Python, ``--factory-startup``)
renders them. Until now that contract was implicit — string keys agreed across two files, no
version, no validation — so a drift was only ever caught by eye in a finished render.

This module IS the contract, validated on both sides:

- the exporter registers every artifact it writes and calls :func:`write_manifest`;
- the renderer calls :func:`load_manifest` FIRST and fails loudly (missing manifest, schema
  mismatch, missing file, missing npz key) before building any scene.

Which artifacts exist stays optional by design (no source clip → no ``stadium.npz``); what is
non-negotiable is that the manifest tells the renderer exactly what to expect.

Self-contained on purpose (stdlib + numpy only, like ``scene_builders``): the Blender side
imports this module by file via the same ``sys.path`` shim and must never import ``pitch3d``.
"""

from __future__ import annotations

import fnmatch
import json
import os

import numpy as np

SCHEMA_VERSION = 2  # 2: per-frame provenance / ball mode (R4)
MANIFEST_NAME = "manifest.json"

#: Minimum npz keys per artifact (glob pattern → required keys). Optional keys (vcolor,
#: jersey_number, ...) ride on top and are listed per-file in the manifest itself.
REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "anim_subject_*.npz": ("verts", "faces", "color", "frames", "alpha", "provenance"),
    "ball.npz": ("frames", "positions_3d", "height_confidence", "mode"),
    "pitch.npz": ("pitch_verts", "pitch_faces", "goal_verts", "goal_faces"),
    "stadium.npz": ("verts", "faces", "colors", "uv", "tile"),
    "boards.npz": ("verts", "faces", "colors"),
    "lighting.npz": ("light_rgb",),
    "cameras.npz": ("names", "frames"),
}


class ContractError(RuntimeError):
    """The export directory does not satisfy the anim contract."""


def required_keys_for(filename: str) -> tuple[str, ...]:
    for pattern, keys in REQUIRED_KEYS.items():
        if fnmatch.fnmatch(filename, pattern):
            return keys
    raise ContractError(f"unknown anim artifact {filename!r} (no REQUIRED_KEYS pattern matches)")


def write_manifest(out_dir: str, entries: dict[str, list[str]]) -> str:
    """Validate + record what the exporter wrote. ``entries`` maps filename → npz keys."""
    for fname, keys in entries.items():
        missing = set(required_keys_for(fname)) - set(keys)
        if missing:
            raise ContractError(f"{fname} is missing required keys {sorted(missing)}")
        path = os.path.join(out_dir, fname)
        if not os.path.exists(path):
            raise ContractError(f"manifest entry {fname} was never written to {out_dir}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "files": {name: sorted(keys) for name, keys in sorted(entries.items())},
    }
    dst = os.path.join(out_dir, MANIFEST_NAME)
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return dst


def load_manifest(in_dir: str) -> dict[str, list[str]]:
    """Validate an export dir against its manifest; return the filename → keys map."""
    path = os.path.join(in_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        raise ContractError(
            f"{in_dir} has no {MANIFEST_NAME} — this is not a (current) anim export. "
            "Re-run anim_export (pitch3d.app.anim_export) on the scene first."
        )
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ContractError(
            f"anim export schema {version!r} != expected {SCHEMA_VERSION} — "
            "re-run anim_export from the same checkout as the renderer."
        )
    files: dict[str, list[str]] = payload.get("files", {})
    if not any(fnmatch.fnmatch(f, "anim_subject_*.npz") for f in files):
        raise ContractError(f"{MANIFEST_NAME} lists no anim_subject_*.npz — empty export?")
    for fname, keys in files.items():
        fpath = os.path.join(in_dir, fname)
        if not os.path.exists(fpath):
            raise ContractError(f"{fname} is in {MANIFEST_NAME} but missing from {in_dir}")
        with np.load(fpath) as data:
            present = set(data.files)
        expected = ((set(keys), MANIFEST_NAME), (set(required_keys_for(fname)), "contract"))
        for key_set, source in expected:
            missing = key_set - present
            if missing:
                raise ContractError(f"{fname} lacks {source} keys {sorted(missing)}")
    return files
