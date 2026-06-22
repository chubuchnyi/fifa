"""Backend-driven bake-off pass (pitch3d.eval.harness.run_backend) + the FK seam.

The contract the runbook hinges on: a backend that returns the scene's GT articulation
reconstructs the GT exactly under the GT camera (condition A) → MPJPE ~0; a zero-pose backend
gives the finite Local-MPJPE sanity floor. We also round-trip the rotation helpers
(``_rodrigues`` / ``_rotmat_to_aa``) across all angles, since the generator relies on them to
emit a faithful camera-space ``global_orient``.
"""

from __future__ import annotations

import numpy as np

from pitch3d.adapters.models.pose import RawBodyMotion
from pitch3d.eval.bodymodel import PlaceholderJointModel, _rodrigues, _rotmat_to_aa
from pitch3d.eval.harness import run_backend
from pitch3d.eval.synthetic import generate_scene


class PerfectBackend:
    """Oracle ``HMRBackend``: returns the scene's own GT articulation per tracklet."""

    def __init__(self, scene):
        self.scene = scene

    def estimate_bodies(self, clip, tracks):
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


class ZeroPoseBackend:
    """Floor ``HMRBackend``: zero articulation (the sanity baseline for Local MPJPE)."""

    def __init__(self, scene):
        self.scene = scene

    def estimate_bodies(self, clip, tracks):
        s = self.scene
        j = s.joints_world.shape[2]
        return {
            tl.track_id: RawBodyMotion(
                track_id=tl.track_id,
                frames=s.frames,
                global_orient=np.zeros((s.n_frames, 3)),
                body_pose=np.zeros((s.n_frames, j, 3)),
                betas=np.zeros(10),
            )
            for tl in tracks.tracklets
        }


def test_perfect_backend_scores_zero():
    s = generate_scene(seed=7)
    grid = run_backend(s, PerfectBackend(s))
    assert grid["global_mpjpe_m"] < 1e-9
    assert grid["local_mpjpe_m"] < 1e-9


def test_zero_pose_backend_is_finite_and_positive():
    s = generate_scene(seed=8)
    grid = run_backend(s, ZeroPoseBackend(s))
    assert set(grid) == {"global_mpjpe_m", "local_mpjpe_m"}
    assert np.isfinite(grid["global_mpjpe_m"])
    assert grid["local_mpjpe_m"] > 0.0


def test_gt_params_fk_reproduces_world_joints():
    # The generator builds GT *through* the FK seam, so re-running FK on the stored GT params
    # and placing via the GT camera must return joints_world exactly (no inverse, no drift).
    s = generate_scene(seed=9)
    jm = s.joint_model
    for n in range(s.n_subjects):
        fk_cam = jm.joints(s.gt_global_orient[:, n], s.gt_body_pose[:, n], s.gt_betas[n])
        root_cam = s.world_to_camera(s.root_world[:, n])
        placed = s.camera_to_world(root_cam[:, None, :] + fk_cam)
        assert np.allclose(placed, s.joints_world[:, n], atol=1e-9)


def test_placeholder_fk_pelvis_is_invariant():
    jm = PlaceholderJointModel()
    out = jm.joints(np.zeros((5, 3)), np.zeros((5, 16, 3)), np.zeros(10))
    assert np.allclose(out[:, 0], 0.0)  # pelvis stays at the root for any articulation


def test_rodrigues_rotmat_round_trip_all_angles():
    rng = np.random.default_rng(3)
    axes = rng.normal(size=(64, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    # span [0, pi], explicitly including the near-pi and near-zero degeneracies
    angles = np.concatenate([rng.uniform(0.0, np.pi, 62), [np.pi - 1e-7, 1e-9]])
    rot = _rodrigues(axes * angles[:, None])
    rot_round = _rodrigues(_rotmat_to_aa(rot))
    assert np.allclose(rot, rot_round, atol=1e-6)
