"""Pure-numpy SMPL-X LBS forward (M2-8a).

The skinning maths is gated on the (non-commercial, ~100 MB) SMPL-X model being present — located
the same way the Blender binary is. The headline correctness check is a **cross-check against the
reference ``smplx`` package**: a random shape+pose must match it to float precision, so the
hand-rolled forward kinematics / blendshape / LBS can't silently drift. The cheaper invariants
(rest pose is the shaped template, a single joint articulates only its kinematic subtree) run from
the same model and pin the behaviour the render path relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.adapters.models.smplx_lbs import (
    N_SMPLX_JOINTS,
    N_SMPLX_VERTS,
    SmplxModel,
    locate_smplx_model,
)

_MODEL_PATH = locate_smplx_model()
_needs_model = pytest.mark.skipif(
    _MODEL_PATH is None, reason="needs the SMPL-X model (.npz) via $PITCH3D_SMPLX_MODELS / env"
)


@pytest.fixture(scope="module")
def model() -> SmplxModel:
    return SmplxModel.load(_MODEL_PATH)


@_needs_model
def test_model_has_smplx_topology(model: SmplxModel):
    assert model.n_verts == N_SMPLX_VERTS
    assert model.n_joints == N_SMPLX_JOINTS
    assert model.faces.shape == (20908, 3)


@_needs_model
def test_shaped_zero_betas_is_template(model: SmplxModel):
    np.testing.assert_allclose(model.shaped(np.zeros(10)), model.v_template, atol=1e-9)


@_needs_model
def test_rest_pose_is_shaped_plus_translation(model: SmplxModel):
    # A zero pose is the identity skinning, so posing only translates the shaped (canonical) mesh.
    betas = np.linspace(-0.5, 0.5, 10)
    transl = np.array([3.0, -2.0, 0.5])
    posed = model.pose(betas, np.zeros(3), np.zeros((21, 3)), transl)
    np.testing.assert_allclose(posed, model.shaped(betas) + transl, atol=1e-6)


@_needs_model
def test_body_joint_rotation_stays_within_its_subtree(model: SmplxModel):
    # Rotating the left elbow (body joint 18) must swing the left forearm/hand and leave the right
    # leg (a disjoint kinematic chain) effectively fixed — only the joint's subtree articulates.
    betas = np.zeros(10)
    rest = model.pose(betas, np.zeros(3), np.zeros((21, 3)))
    body_pose = np.zeros((21, 3))
    body_pose[17] = [1.2, 0.0, 0.0]  # body_pose index 17 == SMPL-X joint 18 (left elbow)
    posed = model.pose(betas, np.zeros(3), body_pose)
    disp = np.linalg.norm(posed - rest, axis=1)
    dominant = model.weights.argmax(axis=1)
    left_arm = disp[np.isin(dominant, [18, 20, *range(25, 40)])]  # elbow/wrist/left-hand joints
    right_leg = disp[np.isin(dominant, [2, 5, 8, 11])]  # right hip/knee/ankle/foot
    assert left_arm.mean() > 0.02
    assert right_leg.max() < 0.005
    assert left_arm.mean() > 10 * right_leg.max()


@_needs_model
def test_pose_sequence_matches_per_frame(model: SmplxModel):
    betas = np.full(10, 0.1)
    rng = np.random.default_rng(3)
    go = rng.normal(size=(3, 3)) * 0.2
    bp = rng.normal(size=(3, 21, 3)) * 0.1
    tr = rng.normal(size=(3, 3))
    seq = model.pose_sequence(betas, go, bp, tr)
    assert seq.shape == (3, N_SMPLX_VERTS, 3)
    for i in range(3):
        np.testing.assert_allclose(seq[i], model.pose(betas, go[i], bp[i], tr[i]), atol=1e-12)


@_needs_model
def test_matches_reference_smplx_package(model: SmplxModel):
    smplx = pytest.importorskip("smplx")
    torch = pytest.importorskip("torch")
    import os

    models_dir = os.environ.get("PITCH3D_SMPLX_MODELS") or "models/smplx"
    ref = smplx.create(
        models_dir, model_type="smplx", gender="neutral", use_pca=False,
        flat_hand_mean=True, num_betas=10, use_face_contour=False, ext="npz",
    )
    rng = np.random.default_rng(0)
    betas = rng.normal(size=10) * 0.5
    go = rng.normal(size=3) * 0.3
    bp = rng.normal(size=(21, 3)) * 0.2
    mine = model.pose(betas, go, bp)
    out = ref(
        betas=torch.tensor(betas[None], dtype=torch.float32),
        global_orient=torch.tensor(go[None], dtype=torch.float32),
        body_pose=torch.tensor(bp.reshape(1, -1), dtype=torch.float32),
        return_verts=True,
    )
    np.testing.assert_allclose(mine, out.vertices[0].detach().numpy(), atol=1e-4)
