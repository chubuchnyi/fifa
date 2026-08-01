"""Capsule collision post-process (T3): overlap → soft push, R-6 bounded."""

from __future__ import annotations

import numpy as np

from pitch3d.core.config.gates import CollisionConfig
from pitch3d.core.correction.collision import (
    _jacobi_pass,
    _resolve_frame,
    collision_gate,
)
from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, xy_track: np.ndarray) -> Subject:
    T = xy_track.shape[0]
    frames = np.arange(T, dtype=int)
    transl = np.zeros((T, 3))
    transl[:, :2] = xy_track
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=np.zeros((T, 3)),
            body_pose=np.zeros((T, 21, 3)), transl=transl,
        ),
    )
    return Subject(track_id=track_id, proposal=motion)


def _scene(*subjects: Subject) -> Scene:
    return Scene(
        id="s", episode_id="e", source_id="c",
        subjects=list(subjects), corrections=[],
    )


# ─── jacobi pass ──────────────────────────────────────────────────────────────

def test_jacobi_pass_pushes_two_overlapping_apart():
    """Two subjects at distance 0.5 m and r=0.35 (2r=0.70): strength=1.0 resolves fully."""
    xy = np.array([[0.0, 0.0], [0.5, 0.0]])
    delta, pairs, overlap = _jacobi_pass(xy, radius=0.35, strength=1.0)
    assert pairs == 1
    assert abs(overlap - 0.20) < 1e-6  # 2*0.35 - 0.5 = 0.20
    # each moved 0.5 * overlap = 0.10 along ±x
    assert delta[0, 0] < 0.0
    assert delta[1, 0] > 0.0
    assert abs(delta[0, 0] + 0.10) < 1e-9
    assert abs(delta[1, 0] - 0.10) < 1e-9


def test_jacobi_pass_none_when_apart():
    xy = np.array([[0.0, 0.0], [2.0, 0.0]])
    delta, pairs, overlap = _jacobi_pass(xy, radius=0.35, strength=1.0)
    assert pairs == 0
    assert overlap == 0.0
    assert np.allclose(delta, 0.0)


def test_jacobi_pass_coincident_split_along_x():
    """Perfectly overlapping subjects split along +X so passes converge."""
    xy = np.array([[0.0, 0.0], [0.0, 0.0]])
    delta, pairs, overlap = _jacobi_pass(xy, radius=0.35, strength=1.0)
    assert pairs == 1
    assert delta[0, 0] < 0 and delta[1, 0] > 0


def test_resolve_frame_converges_in_a_few_passes():
    """After n_passes, the max overlap between any pair is well below 2r."""
    xy = np.array([[0.0, 0.0], [0.5, 0.0], [0.3, 0.3]])
    cfg = CollisionConfig(enabled=True, capsule_radius_m=0.35, strength=0.5, n_passes=8)
    new_xy, pairs_before, max_over_before, max_push = _resolve_frame(xy, cfg)
    assert pairs_before > 0
    # Check every pair after resolve
    for i in range(3):
        for j in range(i + 1, 3):
            d = float(np.linalg.norm(new_xy[i] - new_xy[j]))
            assert d > 0.60, f"pair {i},{j} still {d:.3f} m apart (< 0.60)"


def test_resolve_frame_max_push_cap():
    """The per-frame safety cap prevents launching subjects across the pitch."""
    # 6 overlapping subjects in a tight blob would blow past 0.30 without the cap
    xy = np.zeros((6, 2))
    xy[:, 0] = np.arange(6) * 0.05
    cfg = CollisionConfig(
        enabled=True, capsule_radius_m=0.35, strength=1.0, n_passes=20,
        max_push_per_frame_m=0.20,
    )
    new_xy, _, _, max_push = _resolve_frame(xy, cfg)
    assert max_push <= 0.20 + 1e-9


# ─── collision_gate ──────────────────────────────────────────────────────────

def test_disabled_gate_measures_but_emits_no_corrections():
    """Disabled → count overlaps but leave the scene unchanged (R-6 audit-only)."""
    s1 = _subject(1, np.zeros((5, 2)))
    s2 = _subject(2, np.tile([0.4, 0.0], (5, 1)))
    new_scene, report = collision_gate(_scene(s1, s2), CollisionConfig(enabled=False))
    assert report.corrections_added == 0
    assert new_scene.corrections == []
    assert report.frames_with_overlap == 5
    assert report.pairs_resolved > 0
    assert report.max_overlap_before_m > 0.0


def test_enabled_gate_pushes_overlapping_subjects_apart():
    fps_ignored = None
    s1 = _subject(1, np.zeros((5, 2)))
    s2 = _subject(2, np.tile([0.4, 0.0], (5, 1)))
    cfg = CollisionConfig(
        enabled=True, capsule_radius_m=0.35, strength=1.0, n_passes=4,
        max_push_per_frame_m=1.0, min_correction_m=1e-4,
    )
    new_scene, report = collision_gate(_scene(s1, s2), cfg)
    assert report.corrections_added == 2
    assert report.subjects_moved == 2
    r1 = resolve_subject_motion(s1.proposal, new_scene.corrections_for(1))
    r2 = resolve_subject_motion(s2.proposal, new_scene.corrections_for(2))
    for k in range(5):
        d = float(np.linalg.norm(
            np.asarray(r1.pose.transl)[k, :2] - np.asarray(r2.pose.transl)[k, :2]
        ))
        assert d >= 0.60, f"frame {k}: distance {d:.3f} still under 2r"


def test_gate_leaves_far_apart_subjects_untouched():
    s1 = _subject(1, np.zeros((5, 2)))
    s2 = _subject(2, np.tile([5.0, 0.0], (5, 1)))
    new_scene, report = collision_gate(_scene(s1, s2), CollisionConfig(enabled=True))
    assert report.corrections_added == 0
    assert report.frames_with_overlap == 0
    assert report.subjects_moved == 0


def test_gate_z_axis_preserved():
    """Only XY is pushed; Z stays as measured (foot floor is a separate gate)."""
    frames = np.arange(5, dtype=int)
    t1 = np.zeros((5, 3)); t1[:, 2] = 0.92
    t2 = np.zeros((5, 3)); t2[:, 0] = 0.4; t2[:, 2] = 0.92
    s1 = Subject(track_id=1, proposal=SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(frames=frames, global_orient=np.zeros((5, 3)),
                          body_pose=np.zeros((5, 21, 3)), transl=t1),
    ))
    s2 = Subject(track_id=2, proposal=SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(frames=frames, global_orient=np.zeros((5, 3)),
                          body_pose=np.zeros((5, 21, 3)), transl=t2),
    ))
    new_scene, _ = collision_gate(
        _scene(s1, s2),
        CollisionConfig(enabled=True, strength=1.0, min_correction_m=1e-4),
    )
    r1 = resolve_subject_motion(s1.proposal, new_scene.corrections_for(1))
    r2 = resolve_subject_motion(s2.proposal, new_scene.corrections_for(2))
    assert np.allclose(np.asarray(r1.pose.transl)[:, 2], 0.92)
    assert np.allclose(np.asarray(r2.pose.transl)[:, 2], 0.92)


def test_disjoint_frame_ranges_no_false_overlap():
    """Subjects present at different frames must not be counted as overlapping."""
    frames1 = np.arange(0, 5, dtype=int)
    frames2 = np.arange(10, 15, dtype=int)
    def build(track_id, frames):
        transl = np.zeros((frames.shape[0], 3))
        return Subject(track_id=track_id, proposal=SubjectMotion(
            shape=SmplxShape(betas=np.zeros(10)),
            pose=PoseSequence(frames=frames, global_orient=np.zeros((frames.shape[0], 3)),
                              body_pose=np.zeros((frames.shape[0], 21, 3)), transl=transl),
        ))
    _, report = collision_gate(
        _scene(build(1, frames1), build(2, frames2)),
        CollisionConfig(enabled=True),
    )
    assert report.frames_with_overlap == 0
    assert report.corrections_added == 0


def test_gate_is_idempotent():
    """Applied twice, the second call finds nothing left to push."""
    s1 = _subject(1, np.zeros((5, 2)))
    s2 = _subject(2, np.tile([0.4, 0.0], (5, 1)))
    cfg = CollisionConfig(enabled=True, strength=1.0, n_passes=6,
                          max_push_per_frame_m=1.0, min_correction_m=1e-4)
    once_scene, _ = collision_gate(_scene(s1, s2), cfg)
    twice_scene, twice_report = collision_gate(once_scene, cfg)
    # After the first resolve there is no residual overlap → no new corrections
    assert twice_report.corrections_added == 0


def test_empty_scene_returns_empty_report():
    _, report = collision_gate(_scene(), CollisionConfig(enabled=True))
    assert report.n_subjects == 0
    assert report.corrections_added == 0


def test_none_config_defaults_to_disabled():
    s1 = _subject(1, np.zeros((5, 2)))
    s2 = _subject(2, np.tile([0.4, 0.0], (5, 1)))
    new_scene, report = collision_gate(_scene(s1, s2))
    assert report.corrections_added == 0
    assert new_scene.corrections == []
