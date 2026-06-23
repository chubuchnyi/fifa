"""Local, GPU-free analogue of ``scripts/pod_real_e2e.sh`` — the injected-backend → export → gate.

``pod_real_e2e.sh`` runs, on a GPU, ``python -m pitch3d --pose gvhmr --pose-backend <dotted>
--export gltf --format smplx_npz`` and then gates the export with
``scripts/check_export_dynamic.py``. That exact shape is unrunnable in CI (needs a GPU + the
vendored SMPLest-X), but the **seam** is: inject a dependency-free *dynamic* fake ``HMRBackend`` by
the same dotted-path mechanism (ADR-0006), drive the same golden path on CPU to a real
``smplx_npz`` export, and gate it with the same committed checker. The negative case (a zero-pose
backend) proves the gate actually *rejects* a degenerate export — so the gate guarding the real
pod run is itself under test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from pitch3d.adapters.models.pose import RawBodyMotion
from pitch3d.app.cli import run_dry_run

_REPO = Path(__file__).resolve().parents[2]
_CHECKER = _REPO / "scripts" / "check_export_dynamic.py"


class _DynamicBackend:
    """Dependency-free HMRBackend: person-specific shape + per-frame motion (a 'real-ish' run)."""

    def estimate_bodies(self, clip, tracks):
        out: dict[int, RawBodyMotion] = {}
        for k, tl in enumerate(t for t in tracks.tracklets if t.cls != "ball"):
            t = tl.frames.shape[0]
            phase = np.linspace(0.0, np.pi, t)[:, None, None]
            out[tl.track_id] = RawBodyMotion(
                track_id=tl.track_id, frames=tl.frames,
                global_orient=np.zeros((t, 3)),
                body_pose=0.3 * np.sin(phase + k) * np.ones((t, 21, 3)),  # varies across frames
                betas=0.5 * (np.arange(1, 11) / 10.0) + 0.1 * k,          # nonzero, per-subject
            )
        return out


class _ZeroBackend:
    """Degenerate HMRBackend: zero shape + frozen zero pose — the checker MUST reject its export."""

    def estimate_bodies(self, clip, tracks):
        out: dict[int, RawBodyMotion] = {}
        for tl in (t for t in tracks.tracklets if t.cls != "ball"):
            t = tl.frames.shape[0]
            out[tl.track_id] = RawBodyMotion(
                track_id=tl.track_id, frames=tl.frames,
                global_orient=np.zeros((t, 3)), body_pose=np.zeros((t, 21, 3)), betas=np.zeros(10),
            )
        return out


def make_dynamic() -> _DynamicBackend:
    return _DynamicBackend()


def make_zero() -> _ZeroBackend:
    return _ZeroBackend()


def _run_export(out_dir: Path, factory: str) -> Path:
    rc = run_dry_run(
        out_dir=out_dir, n_frames=8, n_subjects=3, export_format="smplx_npz",
        pose="gvhmr", pose_backend=f"{__name__}:{factory}",
        export="gltf", render="overlay",
    )
    assert rc == 0
    return out_dir / "export" / "scene.smplx_npz"


def _gate(export_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_CHECKER), str(export_dir)], capture_output=True, text=True
    )


def test_injected_dynamic_backend_exports_and_passes_gate(tmp_path):
    export_dir = _run_export(tmp_path / "out", "make_dynamic")
    assert export_dir.is_dir()
    assert len(sorted(export_dir.glob("subject_*.npz"))) == 3
    res = _gate(export_dir)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "ALL DYNAMIC" in res.stdout


def test_zero_pose_backend_export_is_rejected_by_gate(tmp_path):
    export_dir = _run_export(tmp_path / "out", "make_zero")
    res = _gate(export_dir)
    assert res.returncode == 1
    assert "DEGENERATE" in res.stdout
