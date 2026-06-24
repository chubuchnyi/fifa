"""Temporal coherence: structural gap-fill (slerp/lerp) + auto-smoothing corrections (R-6)."""

from __future__ import annotations

import numpy as np

from pitch3d.core.correction.coherence import (
    CoherenceConfig,
    add_temporal_coherence,
    coherence_corrections,
    fill_motion_gaps,
    fill_pose_gaps,
)
from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.scene.layers import CorrectionMode, TargetKind
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.subject import Subject


def _pose(frames, *, gz=None, j0z=None, tz=None) -> PoseSequence:
    """A pose over ``frames`` whose root/orient/joint-0 vary so interpolation is testable."""
    frames = np.asarray(frames, dtype=int).reshape(-1)
    t = frames.shape[0]
    go = np.zeros((t, 3))
    bp = np.zeros((t, 3, 3))
    tr = np.zeros((t, 3))
    if gz is not None:
        go[:, 2] = gz
    if j0z is not None:
        bp[:, 0, 2] = j0z
    if tz is not None:
        tr[:, 2] = tz
    return PoseSequence(frames=frames, global_orient=go, body_pose=bp, transl=tr)


def _motion(pose: PoseSequence) -> SubjectMotion:
    return SubjectMotion(shape=SmplxShape(betas=np.zeros(10)), pose=pose)


# --- fill_pose_gaps -----------------------------------------------------------------


def test_fill_bridges_interior_gap_linearly_and_slerp():
    # frames [0, 4]: one interior gap of 3 missing frames (1,2,3), all <= max_gap.
    pose = _pose([0, 4], tz=[0.0, 4.0], gz=[0.0, 1.0], j0z=[0.0, 1.0])
    out, filled = fill_pose_gaps(pose, max_gap=12)
    np.testing.assert_array_equal(out.frames, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(filled, [1, 2, 3])
    # translation: linear lerp 0..4
    np.testing.assert_allclose(out.transl[:, 2], [0, 1, 2, 3, 4], atol=1e-9)
    # rotation about a single axis: slerp is linear in angle, so midframe == half-angle
    np.testing.assert_allclose(out.global_orient[:, 2], [0, 0.25, 0.5, 0.75, 1.0], atol=1e-7)
    np.testing.assert_allclose(out.body_pose[:, 0, 2], [0, 0.25, 0.5, 0.75, 1.0], atol=1e-7)


def test_fill_preserves_measured_rows_verbatim():
    pose = _pose([0, 4], tz=[0.0, 4.0], gz=[0.0, 1.0])
    out, _ = fill_pose_gaps(pose, max_gap=12)
    # the measured endpoints keep their exact original values
    np.testing.assert_array_equal(out.transl[[0, 4], 2], [0.0, 4.0])
    np.testing.assert_array_equal(out.global_orient[[0, 4], 2], [0.0, 1.0])


def test_fill_leaves_long_gap_intact():
    pose = _pose([0, 20], tz=[0.0, 20.0])  # 19 missing > max_gap
    out, filled = fill_pose_gaps(pose, max_gap=12)
    np.testing.assert_array_equal(out.frames, [0, 20])
    assert filled.size == 0


def test_fill_no_gap_passthrough():
    pose = _pose([0, 1, 2, 3], tz=[0.0, 1.0, 2.0, 3.0])
    out, filled = fill_pose_gaps(pose, max_gap=12)
    np.testing.assert_array_equal(out.frames, [0, 1, 2, 3])
    assert filled.size == 0


def test_fill_only_short_gaps_when_mixed():
    # gap 1->5 is 3 missing (bridged); gap 5->20 is 14 missing (left intact)
    pose = _pose([1, 5, 20], tz=[1.0, 5.0, 20.0])
    out, filled = fill_pose_gaps(pose, max_gap=12)
    np.testing.assert_array_equal(out.frames, [1, 2, 3, 4, 5, 20])
    np.testing.assert_array_equal(filled, [2, 3, 4])


def test_fill_does_not_mutate_input():
    pose = _pose([0, 4], tz=[0.0, 4.0])
    before_frames = pose.frames.copy()
    before_tr = pose.transl.copy()
    fill_pose_gaps(pose, max_gap=12)
    np.testing.assert_array_equal(pose.frames, before_frames)
    np.testing.assert_array_equal(pose.transl, before_tr)


def test_fill_single_and_empty_passthrough():
    single = _pose([7], tz=[2.0])
    out, filled = fill_pose_gaps(single, max_gap=12)
    np.testing.assert_array_equal(out.frames, [7])
    assert filled.size == 0


def test_fill_preserves_optional_hand_jaw_shapes():
    frames = np.array([0, 4])
    pose = PoseSequence(
        frames=frames,
        global_orient=np.zeros((2, 3)),
        body_pose=np.zeros((2, 3, 3)),
        transl=np.zeros((2, 3)),
        left_hand_pose=np.zeros((2, 15, 3)),
        right_hand_pose=np.zeros((2, 15, 3)),
        jaw_pose=np.zeros((2, 3)),
    )
    out, _ = fill_pose_gaps(pose, max_gap=12)
    assert out.left_hand_pose.shape == (5, 15, 3)
    assert out.right_hand_pose.shape == (5, 15, 3)
    assert out.jaw_pose.shape == (5, 3)


# --- fill_motion_gaps ---------------------------------------------------------------


def test_fill_motion_lifts_and_keeps_shape():
    motion = _motion(_pose([0, 4], tz=[0.0, 4.0]))
    motion.shape.betas[:] = 0.7
    out, filled = fill_motion_gaps(motion, max_gap=12)
    np.testing.assert_array_equal(out.pose.frames, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(filled, [1, 2, 3])
    np.testing.assert_allclose(out.shape.betas, 0.7)  # β is identity, carried through


# --- coherence_corrections ----------------------------------------------------------


def test_corrections_default_translation_only():
    corrs = coherence_corrections(3, (0, 10), CoherenceConfig())
    assert len(corrs) == 1
    c = corrs[0]
    assert c.mode == CorrectionMode.TEMPORAL_SMOOTHING
    assert c.target.kind == TargetKind.ROOT_TRANSLATION
    assert c.target.subject_track_id == 3
    assert (c.frame_range.start, c.frame_range.end) == (0, 10)


def test_corrections_both_when_orientation_enabled():
    cfg = CoherenceConfig(smooth_root_orientation=True)
    kinds = {c.target.kind for c in coherence_corrections(1, (0, 5), cfg)}
    assert kinds == {TargetKind.ROOT_TRANSLATION, TargetKind.ROOT_ORIENTATION}


def test_corrections_none_when_both_disabled():
    cfg = CoherenceConfig(smooth_root_translation=False, smooth_root_orientation=False)
    assert coherence_corrections(1, (0, 5), cfg) == []


# --- add_temporal_coherence ---------------------------------------------------------


def _scene_with(make_scene, subjects):
    return make_scene(subjects=subjects)


def test_add_densifies_and_flags_filled_frames(make_scene):
    sub = Subject(track_id=2, proposal=_motion(_pose([0, 4], tz=[0.0, 4.0])))
    scene = make_scene(subjects=[sub])
    out, report = add_temporal_coherence(scene)

    posed = out.subject(2).proposal.pose
    np.testing.assert_array_equal(posed.frames, [0, 1, 2, 3, 4])

    conf = out.confidence.subject_frame_conf[2]
    cfg = CoherenceConfig()
    # measured frames keep real_confidence; bridged frames get the low filled_confidence
    np.testing.assert_allclose(conf[[0, 4]], cfg.real_confidence)
    np.testing.assert_allclose(conf[[1, 2, 3]], cfg.filled_confidence)

    assert report.filled_frames == 3
    assert report.subjects_filled == 1
    assert report.n_subjects == 1
    assert report.corrections_added == 1


def test_add_appends_smoothing_correction(make_scene):
    sub = Subject(track_id=2, proposal=_motion(_pose([0, 4], tz=[0.0, 4.0])))
    out, _ = add_temporal_coherence(make_scene(subjects=[sub]))
    autos = [c for c in out.corrections if c.mode == CorrectionMode.TEMPORAL_SMOOTHING]
    assert len(autos) == 1
    assert autos[0].target.subject_track_id == 2


def test_add_is_nondestructive(make_scene):
    sub = Subject(track_id=2, proposal=_motion(_pose([0, 4], tz=[0.0, 4.0])))
    scene = make_scene(subjects=[sub])
    n_corr_before = len(scene.corrections)
    frames_before = scene.subject(2).proposal.pose.frames.copy()
    add_temporal_coherence(scene)
    assert len(scene.corrections) == n_corr_before          # original stack untouched
    np.testing.assert_array_equal(scene.subject(2).proposal.pose.frames, frames_before)


def test_add_keeps_existing_corrections(make_scene):
    from pitch3d.core.correction.engine import make_offset
    from pitch3d.core.scene.layers import CorrectionTarget

    sub = Subject(track_id=2, proposal=_motion(_pose([0, 4], tz=[0.0, 4.0])))
    existing = make_offset(
        "manual",
        CorrectionTarget(TargetKind.ROOT_TRANSLATION, subject_track_id=2),
        (0, 4),
        np.array([0, 0, 1.0]),
    )
    out, _ = add_temporal_coherence(make_scene(subjects=[sub], corrections=[existing]))
    ids = [c.id for c in out.corrections]
    assert "manual" in ids
    assert any(i.startswith("auto-coh") for i in ids)


def test_add_smoothing_resolves_over_dense_proposal(make_scene):
    # a jittery root path on every frame: after gap-fill the proposal is dense, and the
    # auto TEMPORAL_SMOOTHING correction must actually reduce its variance when resolved.
    jit = [0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0]
    sub = Subject(track_id=5, proposal=_motion(_pose(range(7), tz=jit)))
    out, _ = add_temporal_coherence(make_scene(subjects=[sub]), CoherenceConfig(smooth_window=3))
    posed = out.subject(5).proposal
    resolved = resolve_subject_motion(posed, out.corrections_for(5))
    assert resolved.pose.transl[:, 2].var() < posed.pose.transl[:, 2].var()


def test_add_empty_scene_reports_zero(make_scene):
    out, report = add_temporal_coherence(make_scene(subjects=[]))
    assert report.n_subjects == 0
    assert report.filled_frames == 0
    assert report.corrections_added == 0
    assert out.subjects == []
