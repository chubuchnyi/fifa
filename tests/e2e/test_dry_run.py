"""End-to-end — the CLI dry-run drives the whole golden path on fakes and exits 0."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.app import run_dry_run
from pitch3d.core.scene.serialization import load_scene


def test_dry_run_completes_and_exports(tmp_path):
    out = tmp_path / "out"
    rc = run_dry_run(out_dir=out, n_frames=10, n_subjects=4, export_format="json")
    assert rc == 0

    snapshots = list((out / "observations").glob("*.png"))
    assert len(snapshots) >= 4  # multi-viewpoint visual feedback was produced (ADR-0008)

    export = out / "export" / "scene.json"
    assert export.exists()
    scene = load_scene(str(export))
    assert len(scene.subjects) == 4

    # The committed root nudge (frames [0, n//2]) is BAKED into the exported resolved scene.
    z = scene.subject(0).proposal.pose.transl[:, 2]
    assert z[0] - z[-1] == pytest.approx(0.10, abs=1e-6)
    assert not scene.corrections  # export is the resolved scene, stack baked empty


def test_dry_run_is_deterministic(tmp_path):
    run_dry_run(out_dir=tmp_path / "a", n_frames=8, n_subjects=3, export_format="json")
    run_dry_run(out_dir=tmp_path / "b", n_frames=8, n_subjects=3, export_format="json")
    a = load_scene(str(tmp_path / "a" / "export" / "scene.json"))
    b = load_scene(str(tmp_path / "b" / "export" / "scene.json"))
    assert len(a.subjects) == len(b.subjects) == 3
    for sa, sb in zip(a.subjects, b.subjects, strict=True):
        np.testing.assert_allclose(sa.proposal.pose.transl, sb.proposal.pose.transl)
