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


def test_dry_run_drives_real_render_and_export(tmp_path):
    # The same golden path, but with the dependency-free *real* adapters wired in (no GPU):
    # the reprojection-overlay render pass and the glTF exporter's SMPL-X npz path.
    out = tmp_path / "out"
    rc = run_dry_run(
        out_dir=out, n_frames=8, n_subjects=3, export_format="smplx_npz",
        render="overlay", export="gltf",
    )
    assert rc == 0

    frames = sorted(out.glob("render/**/frame_*.png"))
    assert len(frames) == 8  # one real reprojection PNG per frame
    assert frames[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # genuine PNG, not a placeholder

    npzs = sorted((out / "export" / "scene.smplx_npz").glob("subject_*.npz"))
    assert len(npzs) == 3  # one resolved SMPL-X animation per subject
    data = np.load(npzs[0])
    assert data["transl"].shape[1] == 3 and data["body_model"].item() == "SMPL-X"


def test_the_observation_frame_comes_from_a_track_that_actually_has_it(tmp_path, monkeypatch):
    """A subject shorter than the clip must not sink the run (#130).

    Tracks are routinely shorter than the clip — the identity gate splits one, or the player
    leaves frame — and the observation frame was the *clip's* midpoint read out of subject 0's
    own array. On a 355-frame pod clip whose first subject lived 167 frames that raised
    IndexError after the whole reconstruction was paid for and before anything was exported.
    """
    from dataclasses import replace

    from pitch3d.app.controller import Application

    real_get_scene, real_observe = Application.get_scene, Application.observe
    seen: dict[str, object] = {}

    def leaves_early(self, scene_id):
        scene = real_get_scene(self, scene_id)
        if seen.setdefault("truncated", False):
            return scene
        seen["truncated"] = True
        pose = scene.subjects[0].proposal.pose
        t = len(pose.frames)
        short = replace(pose, **{
            k: v[: t // 3] for k, v in vars(pose).items()
            if isinstance(v, np.ndarray) and v.shape[0] == t
        })
        s0 = scene.subjects[0]
        seen["frames"] = short.frames
        return replace(
            scene,
            subjects=[replace(s0, proposal=replace(s0.proposal, pose=short)), *scene.subjects[1:]],
        )

    def spy(self, scene_id, *, frame=None, **kw):
        seen.setdefault("observed", frame)
        return real_observe(self, scene_id, frame=frame, **kw)

    monkeypatch.setattr(Application, "get_scene", leaves_early)
    monkeypatch.setattr(Application, "observe", spy)

    # demo_edits off: the same shape the pod runs, and it keeps the truncation aimed at the one
    # line under test instead of at the synthetic nudge the walkthrough commits.
    assert run_dry_run(
        out_dir=tmp_path / "o", n_frames=12, n_subjects=3, export_format="json", demo_edits=False,
    ) == 0
    assert seen["observed"] in list(seen["frames"])


def test_dry_run_is_deterministic(tmp_path):
    run_dry_run(out_dir=tmp_path / "a", n_frames=8, n_subjects=3, export_format="json")
    run_dry_run(out_dir=tmp_path / "b", n_frames=8, n_subjects=3, export_format="json")
    a = load_scene(str(tmp_path / "a" / "export" / "scene.json"))
    b = load_scene(str(tmp_path / "b" / "export" / "scene.json"))
    assert len(a.subjects) == len(b.subjects) == 3
    for sa, sb in zip(a.subjects, b.subjects, strict=True):
        np.testing.assert_allclose(sa.proposal.pose.transl, sb.proposal.pose.transl)
