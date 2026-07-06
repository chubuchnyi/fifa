"""Gravity project — airborne Z rewritten to a ballistic parabola."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.correction.gravity_project import (
    G_MPS2,
    GravityProjectConfig,
    gravity_project_gate,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, transl: np.ndarray) -> Subject:
    T = transl.shape[0]
    frames = np.arange(T, dtype=int)
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=np.zeros((T, 3)),
            body_pose=np.zeros((T, 21, 3)), transl=transl,
        ),
    )
    return Subject(track_id=track_id, proposal=motion)


def _scene(*subjects):
    return Scene(id="s", episode_id="e", source_id="c",
                subjects=list(subjects), corrections=[])


def _feats(feats_by_id):
    def p(subject):
        return feats_by_id.get(int(subject.track_id))
    return p


def test_disabled_passthrough():
    T = 10
    scene, report = gravity_project_gate(
        _scene(_subject(1, np.zeros((T, 3)))),
        GravityProjectConfig(enabled=False),
        _feats({1: np.zeros((T, 3))}), fps=30,
    )
    assert report.corrections_added == 0
    assert scene.corrections == []


def test_grounded_no_correction():
    T = 20
    _, report = gravity_project_gate(
        _scene(_subject(1, np.zeros((T, 3)))),
        GravityProjectConfig(enabled=True),
        _feats({1: np.zeros((T, 3))}), fps=30,
    )
    assert report.corrections_added == 0


def test_levitating_subject_projected_to_parabola():
    """Airborne subject with linear Z gets rewritten to a ballistic arc."""
    T = 20
    transl = np.zeros((T, 3))
    transl[:, 2] = np.linspace(1.0, 3.0, T)     # linear rise
    feats = np.zeros((T, 3))
    feats[:, 2] = 0.5                            # airborne throughout
    scene, report = gravity_project_gate(
        _scene(_subject(1, transl)),
        GravityProjectConfig(enabled=True, min_airborne_run_frames=3),
        _feats({1: feats}), fps=30,
    )
    assert report.corrections_added == 1
    resolved = resolve_subject_motion(
        _subject(1, transl).proposal, scene.corrections_for(1),
    )
    got_z = np.asarray(resolved.pose.transl)[:, 2]
    # Endpoints preserved
    assert got_z[0] == pytest.approx(1.0, abs=1e-6)
    assert got_z[-1] == pytest.approx(3.0, abs=1e-6)
    # Interior no longer linear — check the max deviation from linear
    linear = np.linspace(1.0, 3.0, T)
    dev = np.abs(got_z - linear).max()
    assert dev > 0.1


def test_endpoints_preserved():
    T = 15
    transl = np.zeros((T, 3))
    transl[:, 2] = 2.0
    feats = np.zeros((T, 3))
    feats[:, 2] = 0.5
    scene, _ = gravity_project_gate(
        _scene(_subject(1, transl)),
        GravityProjectConfig(enabled=True, min_airborne_run_frames=3),
        _feats({1: feats}), fps=30,
    )
    resolved = resolve_subject_motion(
        _subject(1, transl).proposal, scene.corrections_for(1),
    )
    got = np.asarray(resolved.pose.transl)[:, 2]
    assert got[0] == pytest.approx(2.0)
    assert got[-1] == pytest.approx(2.0)


def test_no_provider_no_correction():
    T = 10
    _, report = gravity_project_gate(
        _scene(_subject(1, np.zeros((T, 3)))),
        GravityProjectConfig(enabled=True), None, fps=30,
    )
    assert report.corrections_added == 0


def test_empty_scene():
    _, report = gravity_project_gate(
        _scene(), GravityProjectConfig(enabled=True),
        _feats({}), fps=30,
    )
    assert report.corrections_added == 0
