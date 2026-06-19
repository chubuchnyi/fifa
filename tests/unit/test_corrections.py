"""Correction engine — the four propagation modes, layer resolve, non-destructiveness."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.adapters.fakes import FakePoseEstimator
from pitch3d.core.correction.engine import (
    apply_offset_vector,
    make_keyframes,
    make_offset,
    make_refit,
    make_smoothing,
    preview_subject_motion,
    resolve_ball,
    resolve_subject_motion,
    smooth_vector,
)
from pitch3d.core.correction.rotations import axis_angle_to_matrix, compose_axis_angle
from pitch3d.core.scene.layers import CorrectionTarget, TargetKind
from pitch3d.core.scene.motion import BallTrack


def _tgt(kind, **kw):
    return CorrectionTarget(kind=kind, **kw)


def test_offset_vector_adds():
    out = apply_offset_vector(np.zeros((3, 3)), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out, np.tile([1.0, 2.0, 3.0], (3, 1)))


def test_constant_offset_applies_to_range_only(make_motion):
    m = make_motion([0, 1, 2, 3])
    before = m.pose.transl.copy()
    corr = make_offset("c", _tgt(TargetKind.ROOT_TRANSLATION, subject_track_id=0), (0, 1),
                       np.array([0, 0, 1.0]))
    res = resolve_subject_motion(m, [corr])
    np.testing.assert_allclose(res.pose.transl[:, 2], [1, 1, 0, 0])
    np.testing.assert_allclose(m.pose.transl, before)  # proposal never mutated (ADR-0002)


def test_rotation_offset_composes_in_rotation_space(make_motion):
    m = make_motion([0, 1])
    j, base, off = 2, np.array([0.0, 0.0, 0.3]), np.array([0.0, 0.0, 0.4])
    m.pose.body_pose[:, j, :] = base
    corr = make_offset("c", _tgt(TargetKind.POSE_BODY_JOINT, subject_track_id=0, joint_index=j),
                       (0, 1), off)
    res = resolve_subject_motion(m, [corr])
    np.testing.assert_allclose(
        axis_angle_to_matrix(res.pose.body_pose[0, j]),
        axis_angle_to_matrix(compose_axis_angle(off, base)),
        atol=1e-7,
    )


def test_pose_joint_requires_joint_index(make_motion):
    m = make_motion([0, 1])
    corr = make_offset("c", _tgt(TargetKind.POSE_BODY_JOINT, subject_track_id=0), (0, 1),
                       np.array([0.1, 0, 0]))
    with pytest.raises(ValueError):
        resolve_subject_motion(m, [corr])


def test_keyframe_interp_fills_range_linearly(make_motion):
    m = make_motion([0, 1, 2, 3, 4])
    corr = make_keyframes("c", _tgt(TargetKind.ROOT_TRANSLATION, subject_track_id=0), (0, 4),
                          np.array([0, 4]), np.array([[0, 0, 0.0], [0, 0, 4.0]]))
    res = resolve_subject_motion(m, [corr])
    np.testing.assert_allclose(res.pose.transl[:, 2], [0, 1, 2, 3, 4], atol=1e-9)


def test_temporal_smoothing_reduces_variance():
    noisy = np.array([[0.0], [2.0], [0.0], [2.0], [0.0]])
    out = smooth_vector(noisy, window=3, method="moving_average")
    assert out.var() < noisy.var()


def test_shape_beta_offset_only(make_motion):
    m = make_motion([0, 1])
    tgt = _tgt(TargetKind.SHAPE_BETA, subject_track_id=0)
    with pytest.raises(ValueError):  # β is frame-invariant: smoothing/keyframe not allowed
        resolve_subject_motion(m, [make_smoothing("c", tgt, (0, 1))])
    res = resolve_subject_motion(m, [make_offset("c", tgt, (0, 1), np.array([0.5] + [0] * 9))])
    assert res.shape.betas[0] == pytest.approx(0.5)


def test_disabled_correction_skipped(make_motion):
    m = make_motion([0, 1])
    corr = make_offset("c", _tgt(TargetKind.ROOT_TRANSLATION, subject_track_id=0), (0, 1),
                       np.array([0, 0, 1.0]))
    corr.enabled = False
    np.testing.assert_allclose(resolve_subject_motion(m, [corr]).pose.transl[:, 2], [0, 0])


def test_refit_splices_and_is_nondestructive(make_motion, clip):
    m = make_motion([0, 1, 2, 3])
    m.pose.body_pose[:] = 0.4
    corr = make_refit("c", _tgt(TargetKind.POSE_BODY_JOINT, subject_track_id=0, joint_index=0),
                      (1, 2), {"root_z_nudge": 0.5})
    res = resolve_subject_motion(m, [corr], refit_port=FakePoseEstimator(), clip=clip)
    np.testing.assert_allclose(res.pose.body_pose[1], 0.2)   # fake halves selected frames
    np.testing.assert_allclose(res.pose.body_pose[0], 0.4)   # outside range untouched
    assert res.pose.transl[1, 2] - m.pose.transl[1, 2] == pytest.approx(0.5)
    np.testing.assert_allclose(m.pose.body_pose[1], 0.4)     # proposal preserved


def test_refit_requires_port_and_clip(make_motion):
    m = make_motion([0, 1])
    corr = make_refit("c", _tgt(TargetKind.ROOT_TRANSLATION, subject_track_id=0), (0, 1), {})
    with pytest.raises(ValueError):
        resolve_subject_motion(m, [corr])


def test_resolve_ball_offsets_range():
    ball = BallTrack(frames=np.arange(4), positions_3d=np.zeros((4, 3)),
                     height_confidence=np.ones(4))
    corr = make_offset("c", _tgt(TargetKind.BALL_POSITION, subject_track_id=None), (1, 2),
                       np.array([0, 0, 2.0]))
    np.testing.assert_allclose(resolve_ball(ball, [corr]).positions_3d[:, 2], [0, 2, 2, 0])


def test_preview_does_not_mutate(make_motion):
    m = make_motion([0, 1, 2])
    cand = make_offset("c", _tgt(TargetKind.ROOT_TRANSLATION, subject_track_id=0), (0, 2),
                       np.array([0, 0, 1.0]))
    out = preview_subject_motion(m, [], cand)
    np.testing.assert_allclose(out.pose.transl[:, 2], 1.0)
    np.testing.assert_allclose(m.pose.transl[:, 2], 0.0)
