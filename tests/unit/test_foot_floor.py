"""Foot-floor gate (T1a): clamp below-floor Z, flag plateaus (R-6), never invent motion."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.config.gates import FootFloorConfig
from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.correction.foot_floor import foot_floor_gate
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, z: np.ndarray, n: int = None) -> Subject:
    """Build a small subject with root Z = the given array (n=len(z) frames)."""
    n = len(z) if n is None else n
    frames = np.arange(n, dtype=int)
    transl = np.zeros((n, 3))
    transl[:, 2] = z
    global_orient = np.zeros((n, 3))
    body_pose = np.zeros((n, 21, 3))
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=global_orient, body_pose=body_pose,
            transl=transl,
        ),
    )
    return Subject(track_id=track_id, proposal=motion)


def _scene(*subjects: Subject) -> Scene:
    return Scene(
        id="test-scene", episode_id="test-ep", source_id="test-clip",
        subjects=list(subjects), corrections=[],
    )


def test_disabled_config_measures_but_emits_no_corrections():
    """Disabled gate is measurement-only — the report is truthful, the scene is untouched."""
    s = _subject(1, np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))  # under the floor (0.92)
    new_scene, report = foot_floor_gate(_scene(s), FootFloorConfig(enabled=False))
    # No corrections added
    assert report.corrections_added == 0
    assert new_scene.corrections == []
    # But the report still tells the truth
    assert report.n_subjects == 1
    assert report.subjects_below_floor == 1
    assert report.subjects[0].below_floor_frames == 6
    assert report.subjects[0].corrected is False


def test_enabled_config_clamps_below_floor_z():
    """Enabled gate emits ONE dense KEYFRAME_INTERP that lifts Z to the floor."""
    # 6 frames: mixed above/below the floor 0.92
    z = np.array([0.92, 0.10, 0.92, -0.20, 0.92, 0.92])
    s = _subject(1, z)
    new_scene, report = foot_floor_gate(_scene(s), FootFloorConfig(enabled=True))
    assert report.corrections_added == 1
    assert report.subjects_corrected == 1
    # Resolve subject → its Z is clamped, other axes untouched
    resolved = resolve_subject_motion(
        s.proposal, new_scene.corrections_for(s.track_id))
    got_z = np.asarray(resolved.pose.transl)[:, 2]
    assert np.all(got_z >= 0.92 - 1e-9)
    # frames that were already above stay exactly the same
    assert got_z[0] == pytest.approx(0.92)
    assert got_z[2] == pytest.approx(0.92)
    # frames that sank are lifted to the floor exactly
    assert got_z[1] == pytest.approx(0.92)
    assert got_z[3] == pytest.approx(0.92)


def test_plateau_flag_when_constant_z():
    """Constant Z (fake-HMR helicopter symptom) → subject flagged plateau, no clamp emitted."""
    z = np.full(20, 0.92)  # perfectly flat
    s = _subject(1, z)
    _, report = foot_floor_gate(_scene(s), FootFloorConfig(enabled=True))
    assert report.subjects_plateau == 1
    assert report.subjects[0].plateau is True
    # Not below floor → nothing to correct
    assert report.corrections_added == 0


def test_plateau_flag_off_when_z_actually_varies():
    """Real motion (crouch/stride/jump variation) must NOT be flagged as plateau."""
    z = 0.92 + 0.1 * np.sin(np.linspace(0, np.pi * 4, 30))
    s = _subject(1, z)
    _, report = foot_floor_gate(_scene(s), FootFloorConfig(enabled=True))
    assert report.subjects_plateau == 0
    assert report.subjects[0].plateau is False


def test_hover_count_matches_config_warn_hover():
    """Hover = Z above floor + pelvis + warn_hover_m — configurable, not hard-coded."""
    z = np.array([0.92, 0.92, 1.30, 1.30, 1.30, 0.92])  # 3 frames >1.22 (0.92+0.30)
    s = _subject(1, z)
    _, report = foot_floor_gate(
        _scene(s), FootFloorConfig(enabled=True, warn_hover_m=0.30)
    )
    assert report.subjects[0].hover_frames == 3


def test_gate_is_idempotent():
    """Applying the gate twice yields the same corrected Z."""
    z = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    s = _subject(1, z)
    once_scene, _ = foot_floor_gate(_scene(s), FootFloorConfig(enabled=True))
    twice_scene, twice_report = foot_floor_gate(once_scene, FootFloorConfig(enabled=True))
    # After the first pass, resolved Z is already at the floor; the second pass finds
    # nothing below the floor → no NEW correction to emit
    assert twice_report.corrections_added == 0


def test_disabled_gate_still_reports_plateau_and_below_floor():
    """Both symptoms are reported in disabled mode — this is how motion_stats/probe uses it."""
    s = _subject(1, np.full(10, 0.5))  # below-floor plateau
    _, report = foot_floor_gate(_scene(s), FootFloorConfig(enabled=False))
    assert report.subjects_below_floor == 1
    assert report.subjects_plateau == 1
    assert report.corrections_added == 0


def test_empty_scene_returns_empty_report():
    _, report = foot_floor_gate(_scene(), FootFloorConfig(enabled=True))
    assert report.n_subjects == 0
    assert report.corrections_added == 0


def test_multiple_subjects_isolated():
    """A per-subject correction never touches OTHER subjects."""
    good = _subject(1, np.full(6, 0.92))
    bad = _subject(2, np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1]))
    new_scene, report = foot_floor_gate(
        _scene(good, bad), FootFloorConfig(enabled=True)
    )
    assert report.corrections_added == 1  # only subject 2 gets one
    assert report.subjects_corrected == 1
    # Subject 1's Z is unchanged
    r1 = resolve_subject_motion(good.proposal, new_scene.corrections_for(1))
    assert np.allclose(np.asarray(r1.pose.transl)[:, 2], 0.92)
    r2 = resolve_subject_motion(bad.proposal, new_scene.corrections_for(2))
    assert np.all(np.asarray(r2.pose.transl)[:, 2] >= 0.92 - 1e-9)


def test_none_config_uses_defaults():
    """Passing cfg=None must work (defaults come from FootFloorConfig())."""
    s = _subject(1, np.full(5, 0.5))  # below the default floor
    _, report = foot_floor_gate(_scene(s))
    # default has enabled=False → no correction, but the diagnosis still surfaces
    assert report.corrections_added == 0
    assert report.subjects_below_floor == 1
