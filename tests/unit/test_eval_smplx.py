"""Real SMPL-X FK behind the JointModel seam — anatomy, the 16-joint output contract, and the
synthetic oracle round-trip.

These are the box-free half of bake-off condition A: the same FK seam the product runs, but on
CPU against the synthetic oracle. They are skipped unless both the ``smplx``/``torch`` packages
and the ``SMPLX_NEUTRAL.npz`` asset are present, so the asset-free CI path stays green.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("smplx")
pytest.importorskip("torch")

from pitch3d.adapters.models.pose import RawBodyMotion  # noqa: E402
from pitch3d.eval.bodymodel import JOINT_NAMES, SmplxJointModel  # noqa: E402
from pitch3d.eval.harness import run_backend  # noqa: E402
from pitch3d.eval.synthetic import generate_scene  # noqa: E402

_MODEL = Path(__file__).resolve().parents[2] / "SMPL-X" / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
pytestmark = pytest.mark.skipif(not _MODEL.exists(), reason="SMPL-X NEUTRAL .npz not present")

_J = {name: i for i, name in enumerate(JOINT_NAMES)}


def _jm() -> SmplxJointModel:
    """SMPL-X backend pinned to the repo asset (cwd-independent)."""
    return SmplxJointModel(model_path=str(_MODEL.parents[1]))


class _PerfectBackend:
    """Oracle backend: replays the scene's own GT articulation per tracklet."""

    def __init__(self, scene) -> None:
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


def test_smplx_n_pose_joints_is_21():
    assert _jm().n_pose_joints == 21


def test_smplx_output_is_canonical_16():
    out = _jm().joints(np.zeros((3, 3)), np.zeros((3, 21, 3)), np.zeros(10))
    assert out.shape == (3, len(JOINT_NAMES), 3)  # 21 input joints, 16 canonical out


def test_smplx_rest_pose_anatomy_matches_our_frame():
    # Zero pose / zero global_orient: pelvis at the root, head up (+z), ankles down (-z),
    # anatomical-left limbs at +x and -x for the right — same frame as the placeholder skeleton.
    j = _jm().joints(np.zeros((2, 3)), np.zeros((2, 21, 3)), np.zeros(10))[0]
    assert np.allclose(j[_J["pelvis"]], 0.0, atol=1e-6)
    assert j[_J["head"]][2] > 0.3
    assert j[_J["l_ankle"]][2] < -0.3 and j[_J["r_ankle"]][2] < -0.3
    assert j[_J["l_shoulder"]][0] > 0.05 > j[_J["r_shoulder"]][0]
    assert j[_J["l_hip"]][0] > 0.0 > j[_J["r_hip"]][0]


def test_smplx_oracle_scores_near_zero():
    # A perfect backend reconstructs GT through the GT camera (condition A). Tolerance is ~0.1mm,
    # not exact, because the scene FK runs one (T*N) batch while the harness re-runs per subject —
    # float32 BLAS can differ across batch sizes; physically this is zero.
    s = generate_scene(seed=5, n_subjects=2, n_frames=4, joint_model=_jm())
    grid = run_backend(s, _PerfectBackend(s))
    assert grid["global_mpjpe_m"] < 1e-4
    assert grid["local_mpjpe_m"] < 1e-4


def test_smplx_scene_articulation_is_nontrivial():
    # global_orient is constant across frames, so root-relative motion is pure articulation:
    # the knee/elbow swing must actually move the limbs through the real kinematic tree.
    s = generate_scene(seed=5, n_subjects=1, n_frames=6, joint_model=_jm())
    rel = s.joints_world - s.joints_world[:, :, :1, :]  # strip the walking root translation
    assert rel[:, 0].std(axis=0).max() > 1e-3
