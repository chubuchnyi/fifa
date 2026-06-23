"""PoseEvalScene routes real frames to the backend, then scores via the synthetic harness.

The dataset adapter (:mod:`pitch3d.eval.dataset`) lets a *real-frames* scene reuse the synthetic
bake-off harness unchanged: it carries a real clip URI so the backend is pointed at actual RGB,
while the scoring (Global/Local MPJPE against GT world joints) is byte-for-byte the synthetic
path. These tests pin the invariants that make that reuse safe:

* the backend receives the scene's *real* ``clip.uri`` (not the ``memory://synthetic`` stand-in);
* a GT-oracle backend still reconstructs the GT exactly (Condition A MPJPE ~0);
* Condition B (foot-grounded via a perfect GT homography) keeps Local MPJPE ~0 — it is
  root-relative, identical to A — while Global MPJPE stays finite (the grounding cost);
* a plain :class:`SyntheticScene` is unaffected (the harness ``getattr`` fallbacks still fire),
  proving the new plumbing is purely additive.
"""

from __future__ import annotations

import numpy as np

from pitch3d.adapters.models.pose import RawBodyMotion
from pitch3d.eval.dataset import PoseEvalScene, evaluate_dataset
from pitch3d.eval.harness import run_backend
from pitch3d.eval.synthetic import generate_scene


class _SpyBackend:
    """GT-oracle ``HMRBackend`` that also records the clip URI it was handed."""

    def __init__(self, scene):
        self.scene = scene
        self.seen_uri: str | None = None

    def estimate_bodies(self, clip, tracks):
        self.seen_uri = clip.uri
        s = self.scene
        return {
            tl.track_id: RawBodyMotion(
                track_id=tl.track_id,
                frames=s.frames,
                global_orient=s.gt_global_orient[:, n],
                body_pose=s.gt_body_pose[:, n],
                betas=s.gt_betas[n],
            )
            for n, tl in enumerate(tracks.tracklets)
        }


def _eval_scene(**kw) -> PoseEvalScene:
    return PoseEvalScene.from_scene(generate_scene(seed=11), **kw)


def test_real_clip_uri_reaches_backend():
    scene = _eval_scene(clip_uri="file:///frames/seqA", source_id="3dpw_seqA")
    spy = _SpyBackend(scene)
    evaluate_dataset(scene, spy)
    assert spy.seen_uri == "file:///frames/seqA"


def test_oracle_scores_zero_condition_a():
    scene = _eval_scene(clip_uri="file:///frames/seqA")
    grid = evaluate_dataset(scene, _SpyBackend(scene))
    assert grid["A"]["global_mpjpe_m"] < 1e-9
    assert grid["A"]["local_mpjpe_m"] < 1e-9
    assert grid["B"] is None  # no calibration supplied


def test_condition_b_is_root_relative_exact_and_global_finite():
    scene = _eval_scene(clip_uri="file:///frames/seqA")
    grid = evaluate_dataset(scene, _SpyBackend(scene), calibration=scene.field_calibration())
    assert grid["B"] is not None
    # Local MPJPE is root-relative -> identical to Condition A even when the root is re-grounded.
    assert grid["B"]["local_mpjpe_m"] < 1e-9
    # Global MPJPE carries the grounding cost: finite and small, but NOT ~0 (feet != pelvis XY).
    assert np.isfinite(grid["B"]["global_mpjpe_m"])
    assert grid["B"]["global_mpjpe_m"] < 1.0


def test_from_scene_copies_every_gt_field():
    base = generate_scene(seed=11)
    scene = PoseEvalScene.from_scene(base, clip_uri="file:///frames/seqA")
    assert scene.clip_uri == "file:///frames/seqA"
    assert np.shares_memory(scene.joints_world, base.joints_world)
    assert scene.joint_model is base.joint_model


def test_plain_synthetic_scene_still_uses_memory_uri():
    base = generate_scene(seed=12)
    spy = _SpyBackend(base)
    grid = run_backend(base, spy)
    assert spy.seen_uri == "memory://synthetic"
    assert grid["global_mpjpe_m"] < 1e-9
