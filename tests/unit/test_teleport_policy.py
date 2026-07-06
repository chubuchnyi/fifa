"""T2.b — teleport_policy=hold vs interpolate: preserve vs smooth, always mark R-6."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pitch3d.core.config import load_physics_config
from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.correction.kinematics import (
    TELEPORT_INTERPOLATED_CONF,
    KinematicConfig,
    kinematic_gate,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject_with_teleport(track_id: int) -> Subject:
    """A subject whose XY takes one giant ID-swap step in the middle."""
    T = 10
    frames = np.arange(T, dtype=int)
    transl = np.zeros((T, 3))
    transl[:5, 0] = np.linspace(0, 2, 5)     # cruise 0..2 m
    transl[5:, 0] = np.linspace(50, 52, 5)   # jump to 50, cruise 50..52
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


def test_invalid_policy_rejected_at_construction():
    with pytest.raises(ValueError, match="teleport_policy"):
        KinematicConfig(teleport_policy="wat")


def test_hold_policy_preserves_teleport_jump():
    """Default hold: the resolved motion keeps the giant jump verbatim."""
    fps = 30.0
    s = _subject_with_teleport(1)
    new_scene, report = kinematic_gate(
        _scene(s), KinematicConfig(teleport_policy="hold"), fps=fps,
    )
    assert report.teleports  # jump detected
    assert report.teleport_interpolated_frames == 0
    resolved = resolve_subject_motion(s.proposal, new_scene.corrections_for(1))
    got = np.asarray(resolved.pose.transl)
    # jump between frame 4 and 5 preserved (~48 m step)
    step = float(np.linalg.norm(got[5] - got[4]))
    assert step > 40.0, step


def test_interpolate_policy_smooths_the_gap():
    """Interpolate: resolved motion is a straight anchored path — no big jump."""
    fps = 30.0
    s = _subject_with_teleport(1)
    new_scene, report = kinematic_gate(
        _scene(s), KinematicConfig(teleport_policy="interpolate"), fps=fps,
    )
    assert report.teleports  # still recorded — audit trail preserved
    assert report.teleport_interpolated_frames > 0
    resolved = resolve_subject_motion(s.proposal, new_scene.corrections_for(1))
    got = np.asarray(resolved.pose.transl)
    # every consecutive step in the resolved motion is much smaller than the raw 48m jump
    steps = np.linalg.norm(np.diff(got, axis=0), axis=1)
    assert steps.max() < 20.0, steps


def test_interpolate_policy_stamps_low_confidence_on_interpolated_rows():
    """Interpolated frames land in scene.confidence at TELEPORT_INTERPOLATED_CONF."""
    fps = 30.0
    s = _subject_with_teleport(1)
    new_scene, report = kinematic_gate(
        _scene(s), KinematicConfig(teleport_policy="interpolate"), fps=fps,
    )
    assert new_scene.confidence is not None
    conf = new_scene.confidence.subject_frame_conf.get(1)
    assert conf is not None
    # at least one row hit the interpolated tag
    assert (np.asarray(conf) == TELEPORT_INTERPOLATED_CONF).any()
    # and the untouched anchor rows stay at 1.0 (measured)
    assert (np.asarray(conf) == 1.0).any()


def test_teleport_event_still_recorded_in_interpolate_mode():
    """R-6: the audit trail (TeleportEvent) survives even when we smooth the jump."""
    fps = 30.0
    s = _subject_with_teleport(1)
    _, report = kinematic_gate(
        _scene(s), KinematicConfig(teleport_policy="interpolate"), fps=fps,
    )
    ev = report.teleports[0]
    assert ev.track_id == 1
    assert ev.jump_m > 40.0


def test_teleport_policy_survives_config_yaml_roundtrip():
    """Shipped 'humanize_teleports' profile maps to teleport_policy='interpolate'."""
    cfg = load_physics_config(profile="humanize_teleports", env={})
    assert cfg.kinematic.teleport_policy == "interpolate"
    # lineage records where the value came from
    assert cfg.lineage["kinematic.teleport_policy"] == "profile:humanize_teleports"


def test_default_profile_uses_hold():
    cfg = load_physics_config(profile="default", env={})
    assert cfg.kinematic.teleport_policy == "hold"
    assert cfg.lineage["kinematic.teleport_policy"] == "base"


def test_no_teleport_no_interpolation_regardless_of_policy():
    """A clean subject with no teleport gets zero interpolated rows in either mode."""
    fps = 30.0
    T = 8
    frames = np.arange(T, dtype=int)
    transl = np.zeros((T, 3))
    transl[:, 0] = np.linspace(0, 2, T)  # slow cruise
    s = Subject(
        track_id=1,
        proposal=SubjectMotion(
            shape=SmplxShape(betas=np.zeros(10)),
            pose=PoseSequence(
                frames=frames, global_orient=np.zeros((T, 3)),
                body_pose=np.zeros((T, 21, 3)), transl=transl,
            ),
        ),
    )
    for policy in ("hold", "interpolate"):
        _, report = kinematic_gate(
            _scene(s), KinematicConfig(teleport_policy=policy), fps=fps,
        )
        assert report.teleports == []
        assert report.teleport_interpolated_frames == 0
