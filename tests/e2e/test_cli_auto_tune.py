"""End-to-end CLI wiring: --physics --auto-tune persists per-subject profiles.

Uses the dependency-free fake pipeline so the test runs without weights, GPU,
or Blender. Confirms:

* the CLI flag surface (--player-profiles-dir, --player-priors, --auto-tune,
  --ball-id) plumbs through run_dry_run → controller.run_reconstruction →
  kinematic_gate → apply_profile_updates.
* Profiles land on disk after the run, in the layout the store documents.
* The auto-tune line prints in the log for operator inspection.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pitch3d.app.cli import run_dry_run


def _run(tmp_path: Path, *, auto_tune: bool) -> tuple[str, Path]:
    """Run the dry-run in a temp dir with the fake stack + optional auto-tune."""
    profiles_dir = tmp_path / "profiles"
    out_dir = tmp_path / "out"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_dry_run(
            out_dir=out_dir, n_frames=8, n_subjects=4, export_format="json",
            clip_path=None,
            detector="fake", tracker="fake", calibrator="fake", pose="fake",
            ball="fake", env="fake", avatar="fake", render="overlay",
            export="gltf", observer="fake", viewsynth="fake",
            device="cpu",
            physics=True, physics_profile="default",
            player_profiles_dir=str(profiles_dir),
            auto_tune=auto_tune,
            demo_edits=False,
        )
    assert rc == 0
    return buf.getvalue(), profiles_dir


def test_cli_auto_tune_persists_profiles(tmp_path: Path):
    stdout, profiles_dir = _run(tmp_path, auto_tune=True)
    # log surfaces the auto-tune outcome for the operator
    assert "== profiles:" in stdout
    assert "== auto-tune:" in stdout
    assert "'applied':" in stdout
    # profiles reached disk
    assert (profiles_dir / "players").exists()
    saved = list((profiles_dir / "players").rglob("*.json"))
    assert saved, "no profile files written"
    # each file is a valid JSON with the T4a schema shape
    for p in saved:
        with p.open() as fh:
            payload = json.load(fh)
        assert payload["schema_version"] == 1
        assert "kinematics" in payload
        assert "provenance" in payload


def test_cli_without_auto_tune_does_not_write_profiles(tmp_path: Path):
    stdout, profiles_dir = _run(tmp_path, auto_tune=False)
    # profile_provider still loads (log shows the priors line)
    assert "== profiles:" in stdout
    # but no auto-tune sink → no auto-tune line
    assert "== auto-tune:" not in stdout
    # nothing persisted
    if (profiles_dir / "players").exists():
        assert list((profiles_dir / "players").rglob("*.json")) == []
