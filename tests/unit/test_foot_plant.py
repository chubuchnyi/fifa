"""Foot plant gate (T6.a): median-lock Z bias fix without killing stride variance."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.config.gates import FootPlantConfig
from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.correction.foot_plant import foot_plant_gate
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, z: np.ndarray) -> Subject:
    T = len(z)
    frames = np.arange(T, dtype=int)
    transl = np.zeros((T, 3))
    transl[:, 2] = z
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


def test_invalid_mode_raises():
    s = _subject(1, np.full(5, 0.92))
    with pytest.raises(ValueError, match="mode"):
        foot_plant_gate(_scene(s), FootPlantConfig(enabled=True, mode="wat"))


def test_off_measures_only():
    """mode=off + enabled=True still measures the bias but emits no corrections."""
    s = _subject(1, np.full(10, 1.10))  # 18 cm above 0.92
    _, report = foot_plant_gate(_scene(s), FootPlantConfig(enabled=True, mode="off"))
    assert report.corrections_added == 0
    assert abs(report.max_abs_bias_m - 0.18) < 1e-9
    assert report.subjects[0].z_bias_m < 0  # need to shift DOWN


def test_disabled_measures_only():
    s = _subject(1, np.full(10, 1.10))
    _, report = foot_plant_gate(_scene(s), FootPlantConfig(enabled=False, mode="median_lock"))
    assert report.corrections_added == 0
    assert report.subjects[0].n_frames == 10


def test_median_lock_recenters_bias():
    """Median-lock: subject at Z=1.10 flat → shift down 0.18 → median = target 0.92."""
    s = _subject(1, np.full(10, 1.10))
    new_scene, report = foot_plant_gate(
        _scene(s), FootPlantConfig(enabled=True, mode="median_lock", target_pelvis_m=0.92),
    )
    assert report.corrections_added == 1
    resolved = resolve_subject_motion(s.proposal, new_scene.corrections_for(1))
    got = np.asarray(resolved.pose.transl)[:, 2]
    assert abs(float(np.median(got)) - 0.92) < 1e-9


def test_median_lock_preserves_stride_variance():
    """Stride amplitude (peak-to-peak Z variation) survives the shift."""
    z = 0.92 + 0.15 * np.sin(np.linspace(0, np.pi * 4, 30))  # ±15cm stride around 0.92
    # add a +0.20 bias so the median drifts to 1.12
    z_biased = z + 0.20
    s = _subject(1, z_biased)
    new_scene, _ = foot_plant_gate(
        _scene(s), FootPlantConfig(enabled=True, mode="median_lock"),
    )
    resolved = resolve_subject_motion(s.proposal, new_scene.corrections_for(1))
    got = np.asarray(resolved.pose.transl)[:, 2]
    # median moved back to target
    assert abs(float(np.median(got)) - 0.92) < 1e-6
    # amplitude preserved (peak-to-peak ~ 0.30m)
    assert abs((got.max() - got.min()) - (z_biased.max() - z_biased.min())) < 1e-6


def test_median_lock_skips_when_bias_below_threshold():
    """Subject already near target → no correction (skip cost)."""
    z = 0.92 + 0.02 * np.random.default_rng(0).random(20)  # ~0.93 median
    s = _subject(1, z)
    _, report = foot_plant_gate(
        _scene(s), FootPlantConfig(enabled=True, mode="median_lock", bias_threshold_m=0.05),
    )
    assert report.corrections_added == 0
    assert report.subjects_corrected == 0


def test_hard_lock_kills_variance():
    """hard_lock: every frame gets clamped to target — no stride survives."""
    z = 0.92 + 0.15 * np.sin(np.linspace(0, np.pi * 4, 30)) + 0.20
    s = _subject(1, z)
    new_scene, _ = foot_plant_gate(
        _scene(s), FootPlantConfig(enabled=True, mode="hard_lock", target_pelvis_m=0.92),
    )
    resolved = resolve_subject_motion(s.proposal, new_scene.corrections_for(1))
    got = np.asarray(resolved.pose.transl)[:, 2]
    assert np.allclose(got, 0.92)


def test_negative_bias_lifts_up():
    """Subject sinking below target → gate lifts UP (median → 0.92)."""
    s = _subject(1, np.full(10, 0.60))  # 32 cm below
    new_scene, report = foot_plant_gate(
        _scene(s), FootPlantConfig(enabled=True, mode="median_lock"),
    )
    assert report.corrections_added == 1
    resolved = resolve_subject_motion(s.proposal, new_scene.corrections_for(1))
    got = np.asarray(resolved.pose.transl)[:, 2]
    assert abs(float(np.median(got)) - 0.92) < 1e-9


def test_multi_subject_independent():
    hovering = _subject(1, np.full(10, 1.15))
    fine = _subject(2, np.full(10, 0.92))
    new_scene, report = foot_plant_gate(
        _scene(hovering, fine), FootPlantConfig(enabled=True, mode="median_lock"),
    )
    assert report.corrections_added == 1  # only hovering fixed
    # subject 2 unchanged
    r2 = resolve_subject_motion(fine.proposal, new_scene.corrections_for(2))
    assert np.allclose(np.asarray(r2.pose.transl)[:, 2], 0.92)


def test_xy_and_pose_untouched():
    """Foot plant only shifts Z; XY, joints, orient stay."""
    T = 10
    frames = np.arange(T, dtype=int)
    transl = np.zeros((T, 3))
    transl[:, 0] = np.arange(T) * 0.5  # walking +X
    transl[:, 2] = 1.10                # hovering
    body_pose = np.ones((T, 21, 3)) * 0.3  # non-zero pose
    global_orient = np.ones((T, 3)) * 0.1
    s = Subject(track_id=1, proposal=SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(frames=frames, global_orient=global_orient,
                          body_pose=body_pose, transl=transl),
    ))
    new_scene, _ = foot_plant_gate(_scene(s), FootPlantConfig(enabled=True, mode="median_lock"))
    resolved = resolve_subject_motion(s.proposal, new_scene.corrections_for(1))
    got_xy = np.asarray(resolved.pose.transl)[:, :2]
    got_body = np.asarray(resolved.pose.body_pose)
    got_orient = np.asarray(resolved.pose.global_orient)
    assert np.allclose(got_xy[:, 0], transl[:, 0])
    assert np.allclose(got_body, body_pose)
    assert np.allclose(got_orient, global_orient)


def test_idempotent():
    s = _subject(1, np.full(10, 1.15))
    cfg = FootPlantConfig(enabled=True, mode="median_lock")
    once, _ = foot_plant_gate(_scene(s), cfg)
    twice, twice_report = foot_plant_gate(once, cfg)
    assert twice_report.corrections_added == 0


def test_empty_scene():
    _, report = foot_plant_gate(_scene(), FootPlantConfig(enabled=True))
    assert report.n_subjects == 0
    assert report.corrections_added == 0


def test_none_config_defaults_to_disabled():
    s = _subject(1, np.full(5, 1.15))
    new_scene, report = foot_plant_gate(_scene(s))
    assert report.corrections_added == 0
    assert new_scene.corrections == []
