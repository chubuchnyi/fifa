"""Mirroring a subject across world Y (#118/#120) — pinned against real SMPL-X FK.

A mirrored human is not a pose SMPL-X can represent, so ``scripts/apply_rigid_camera.py`` gets
there by ALSO flipping the body about its own sagittal plane and letting the two improper maps
cancel. That is easy to get subtly wrong and impossible to catch by eye at 40-70 m — the first
two attempts were 1.01 m and 1.89 m out and both produced a body that still stood up and still
faced roughly the right way. So the test is the real forward pass, not the algebra.

The tolerance is not arbitrary: the neutral template is not perfectly symmetric (its rest hips
differ by 0.011 m off the left-right axis), which puts a floor of ~0.04 m under any exact-mirror
claim. Anything materially above that is a bug in the transform, not the model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pitch3d.adapters.models.smplx_lbs import SmplxModel, locate_smplx_model
from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.scene.frames import R_SMPLX_CAMERA_TO_WORLD
from pitch3d.core.scene.layers import (
    Correction,
    CorrectionMode,
    CorrectionTarget,
    FrameRange,
    KeyframePayload,
    OffsetPayload,
    TargetKind,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.apply_rigid_camera import (  # noqa: E402
    M,
    S,
    flip_body_pose,
    local_mirror,
    mirror_corrections,
    mirror_subjects,
)

_MODEL_PATH = locate_smplx_model()
_needs_model = pytest.mark.skipif(
    _MODEL_PATH is None, reason="needs the SMPL-X model (.npz) via $PITCH3D_SMPLX_MODELS / env"
)

#: SMPL-X body joints 1-21 under the left↔right swap, lifted to include the root.
_J_SWAP = np.arange(22)
for _a, _b in ((0, 1), (3, 4), (6, 7), (9, 10), (12, 13), (15, 16), (17, 18), (19, 20)):
    _J_SWAP[_a + 1], _J_SWAP[_b + 1] = _b + 1, _a + 1

#: The template's own asymmetry; see the module docstring.
_FLOOR_M = 0.05


@pytest.fixture(scope="module")
def model() -> SmplxModel:
    return SmplxModel.load(_MODEL_PATH)


def world_joints(model, betas, global_orient, body_pose, transl) -> np.ndarray:
    """The ``(22, 3)`` world joints a scene consumer sees for one frame of stored parameters."""
    verts = model.pose(betas, global_orient, body_pose, None)
    joints = (model.j_regressor @ verts)[:22]
    return (joints - joints[0]) @ R_SMPLX_CAMERA_TO_WORLD.T + np.asarray(transl, dtype=float)


def _case(seed: int):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(scale=0.5, size=10),  # betas
        rng.normal(scale=0.8, size=3),  # global_orient
        rng.normal(scale=0.25, size=(21, 3)),  # body_pose
        np.array([rng.uniform(-30, 30), rng.uniform(-40, 40), 0.0]),  # transl
    )


@_needs_model
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_mirror_reproduces_the_world_reflection(model, seed):
    betas, go, bp, transl = _case(seed)
    target = world_joints(model, betas, go, bp, transl) @ M

    rot = Rotation.from_rotvec(go).as_matrix()
    got = world_joints(
        model,
        betas,
        Rotation.from_matrix(local_mirror() @ rot @ S).as_rotvec(),
        flip_body_pose(bp[None])[0],
        transl @ M,
    )
    # Left and right have swapped places, which is what mirroring a person means.
    assert np.abs(got[_J_SWAP] - target).max() < _FLOOR_M


@_needs_model
def test_conjugation_alone_is_not_a_mirror(model):
    """The pre-#120 transform, pinned as WRONG so it cannot quietly come back."""
    betas, go, bp, transl = _case(0)
    target = world_joints(model, betas, go, bp, transl) @ M
    rot = Rotation.from_rotvec(go).as_matrix()
    got = world_joints(model, betas, Rotation.from_matrix(M @ rot @ M).as_rotvec(), bp, transl @ M)
    assert np.abs(got - target).max() > 10 * _FLOOR_M


def test_mirrored_orientation_is_a_real_rotation():
    """``det = +1``: two improper maps around a rotation compose back to a rotation."""
    for seed in range(5):
        rot = Rotation.from_rotvec(_case(seed)[1]).as_matrix()
        assert np.linalg.det(local_mirror() @ rot @ S) == pytest.approx(1.0)


def test_local_mirror_is_the_world_mirror_seen_from_the_camera_frame():
    m_local = local_mirror()
    assert np.allclose(m_local, np.diag([1.0, 1.0, -1.0]))  # world Y is SMPL-X Z
    assert np.allclose(R_SMPLX_CAMERA_TO_WORLD @ m_local, M @ R_SMPLX_CAMERA_TO_WORLD)


def test_flip_body_pose_is_an_involution():
    bp = np.random.default_rng(7).normal(size=(4, 21, 3))
    assert np.allclose(flip_body_pose(flip_body_pose(bp)), bp)


def _motion(seed: int, t: int = 6) -> SubjectMotion:
    rng = np.random.default_rng(seed)
    return SubjectMotion(
        shape=SmplxShape(betas=rng.normal(scale=0.5, size=10)),
        pose=PoseSequence(
            frames=np.arange(t),
            global_orient=rng.normal(scale=0.8, size=(t, 3)),
            body_pose=rng.normal(scale=0.25, size=(t, 21, 3)),
            transl=np.column_stack(
                [rng.uniform(-30, 30, t), rng.uniform(-40, 40, t), np.zeros(t)]
            ),
        ),
    )


def _keyframes(kind, values, *, joint_index=None, t: int = 6) -> Correction:
    return Correction(
        id=f"{kind.value}{'' if joint_index is None else joint_index}",
        target=CorrectionTarget(kind=kind, subject_track_id=1, joint_index=joint_index),
        frame_range=FrameRange(0, t - 1),
        mode=CorrectionMode.KEYFRAME_INTERP,
        payload=KeyframePayload(key_frames=np.arange(t), key_values=values),
    )


def test_corrections_mirror_with_the_subject():
    """The *resolved* motion has to mirror, not just the proposal — the #120 regression.

    A replayed scene carries its motion in corrections, so mirroring only the proposal left the
    default clip with right-handed subjects, a right-handed camera and left-handed motion: every
    player drawn across the halfway line from where they stood.
    """
    rng = np.random.default_rng(11)
    motion, joint = _motion(0), 3
    corrections = [
        _keyframes(TargetKind.ROOT_TRANSLATION, np.column_stack(
            [rng.uniform(-30, 30, 6), rng.uniform(-40, 40, 6), np.zeros(6)])),
        _keyframes(TargetKind.ROOT_ORIENTATION, rng.normal(scale=0.8, size=(6, 3))),
        _keyframes(TargetKind.POSE_BODY_JOINT, rng.normal(scale=0.25, size=(6, 3)),
                   joint_index=joint),
    ]
    before = resolve_subject_motion(motion, corrections)

    scene = SimpleNamespace(subjects=[SimpleNamespace(proposal=motion)], corrections=corrections)
    mirror_subjects(scene)
    mirror_corrections(scene)
    after = resolve_subject_motion(motion, corrections)

    assert corrections[2].target.joint_index != joint, "the corrected joint must change side"
    assert np.allclose(after.pose.transl, before.pose.transl @ M)
    assert np.allclose(after.pose.body_pose, flip_body_pose(before.pose.body_pose))
    assert np.allclose(
        Rotation.from_rotvec(after.pose.global_orient).as_matrix(),
        local_mirror() @ Rotation.from_rotvec(before.pose.global_orient).as_matrix() @ S,
    )


def test_a_correction_the_flip_has_no_rule_for_is_refused():
    """R-6: passing an unmirrorable correction through unchanged is what caused the bug."""
    corr = Correction(
        id="beta", target=CorrectionTarget(kind=TargetKind.SHAPE_BETA, subject_track_id=1),
        frame_range=FrameRange(0, 5), mode=CorrectionMode.CONSTANT_OFFSET,
        payload=OffsetPayload(delta=np.zeros(10)),
    )
    with pytest.raises(NotImplementedError, match="no rule"):
        mirror_corrections(SimpleNamespace(corrections=[corr]))
