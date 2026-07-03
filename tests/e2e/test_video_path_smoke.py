"""Gated E2E smoke of the deliverable video path: dry-run scene → anim export → Blender render.

The two halves of the path live in different processes (exporter under the pipeline venv,
renderer under Blender's own Python) and used to drift silently; this test drives the REAL
boundary end to end. Every stage skips cleanly where its local dependency (torch/smplx,
SMPL-X models, a Blender binary) is missing, so CI without assets still runs nothing false.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pitch3d.adapters.blender import anim_contract, locate_blender
from pitch3d.app import run_dry_run
from pitch3d.env import load_env

REPO = Path(__file__).resolve().parents[2]
RENDER_SCRIPT = REPO / "scripts" / "blender_animate.py"


def _models_dir() -> str | None:
    load_env()
    d = os.environ.get("PITCH3D_SMPLX_MODELS", "SMPL-X/models")
    return d if os.path.exists(os.path.join(d, "smplx", "SMPLX_NEUTRAL.npz")) else None


def _blender() -> str | None:
    load_env()  # PITCH3D_BLENDER lives in the repo-root .env
    found = locate_blender()
    return str(found) if found else None


def _render(in_dir: Path, out_dir: Path, blender: str) -> subprocess.CompletedProcess:
    cmd = [
        blender, "--background", "--factory-startup", "--python", str(RENDER_SCRIPT), "--",
        "--in", str(in_dir), "--out", str(out_dir),
        "--res-x", "320", "--res-y", "180", "--samples", "8",
        "--cameras", "broadcast", "--frame-step", "8",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=REPO)


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory) -> Path:
    pytest.importorskip("torch")
    pytest.importorskip("smplx")
    models = _models_dir()
    if models is None:
        pytest.skip("SMPL-X models not available locally")
    from pitch3d.app.anim_export import main as anim_export_main

    root = tmp_path_factory.mktemp("video_path")
    assert run_dry_run(out_dir=root / "out", n_frames=12, n_subjects=3, export_format="json") == 0
    out = root / "mesh"
    rc = anim_export_main([
        "--scene", str(root / "out" / "export" / "scene.json"),
        "--out", str(out),
        "--smplx-models", models,
        "--source-video", "",  # hermetic: never bake stadium/lighting from an ambient env var
        "--canonical-up",
    ])
    assert rc == 0
    return out


def test_export_satisfies_its_own_contract(export_dir):
    files = anim_contract.load_manifest(str(export_dir))
    assert sum(f.startswith("anim_subject_") for f in files) == 3
    assert "cameras.npz" in files and "pitch.npz" in files


def test_virtual_operator_track_is_complete(export_dir):
    cd = np.load(export_dir / "cameras.npz")
    names = [str(n) for n in cd["names"]]
    assert {"broadcast", "sideline", "goal", "top"} <= set(names)
    n = cd["frames"].shape[0]
    for name in names:
        assert cd[f"{name}_pos"].shape == (3,)
        assert cd[f"{name}_look"].shape == (n, 3)
        assert cd[f"{name}_fov_deg"].shape == (n,)
        assert not np.isnan(cd[f"{name}_look"]).any()


def test_blender_renders_through_the_virtual_operator(export_dir, tmp_path):
    blender = _blender()
    if blender is None:
        pytest.skip("no Blender binary (PITCH3D_BLENDER / PATH)")
    res = _render(export_dir, tmp_path / "frames", blender)
    tail = res.stdout[-2000:] + res.stderr[-1000:]
    assert "BLENDER_ANIM_CAMS virtual-operator" in res.stdout, tail
    assert "BLENDER_ANIM_OK" in res.stdout, tail
    pngs = sorted((tmp_path / "frames" / "broadcast").glob("frame_*.png"))
    assert pngs, tail
    assert pngs[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_renderer_refuses_a_directory_without_manifest(tmp_path):
    blender = _blender()
    if blender is None:
        pytest.skip("no Blender binary (PITCH3D_BLENDER / PATH)")
    res = _render(tmp_path, tmp_path / "frames", blender)
    assert "BLENDER_ANIM_OK" not in res.stdout
    assert "Re-run anim_export" in res.stdout + res.stderr
