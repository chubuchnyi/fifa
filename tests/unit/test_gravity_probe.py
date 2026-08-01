"""Gravity probe — vertical acceleration = -9.81 m/s² during airborne."""

from __future__ import annotations

import numpy as np

from pitch3d.core.correction.gravity_probe import (
    GravityConfig,
    _find_runs,
    gravity_probe,
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


def test_find_runs():
    mask = np.array([True, True, True, False, True, True])
    assert _find_runs(mask, 2) == [(0, 2), (4, 5)]


def test_disabled_returns_empty():
    T = 10
    r = gravity_probe(_scene(_subject(1, np.zeros((T, 3)))),
                     GravityConfig(enabled=False), _feats({1: np.zeros((T, 3))}), fps=30)
    assert r.n_subjects == 1
    assert r.subjects_violating == 0
    assert r.subjects == []


def test_no_provider_returns_empty():
    T = 10
    r = gravity_probe(_scene(_subject(1, np.zeros((T, 3)))),
                     GravityConfig(enabled=True), None, fps=30)
    assert r.n_subjects == 1
    assert r.subjects == []


def test_grounded_subject_no_airborne():
    """A subject with foot always on floor has 0 airborne frames."""
    T = 20
    r = gravity_probe(_scene(_subject(1, np.zeros((T, 3)))),
                     GravityConfig(enabled=True), _feats({1: np.zeros((T, 3))}), fps=30)
    assert r.subjects[0].airborne_frames == 0
    assert r.subjects_violating == 0


def test_gravity_obeyed_by_ballistic_arc():
    """Vertical motion under gravity: az≈-9.81 → not flagged."""
    T = 20
    dt = 1.0 / 30.0
    z = 0.5 * (-9.81) * (dt * np.arange(T)) ** 2 + 5 * dt * np.arange(T) + 0.5
    transl = np.zeros((T, 3))
    transl[:, 2] = z
    feats = np.zeros((T, 3))
    feats[:, 2] = z - 0.9   # foot below pelvis; roughly airborne when pelvis is high
    r = gravity_probe(_scene(_subject(1, transl)),
                     GravityConfig(enabled=True, airborne_z_threshold_m=0.05,
                                  min_airborne_run_frames=3, tolerance_mps2=3.0),
                     _feats({1: feats}), fps=30)
    assert r.subjects[0].mean_vertical_accel_mps2 < -5.0
    assert r.subjects_violating == 0


def test_levitating_subject_flagged():
    """Airborne subject with UPWARD accel (levitating) → violation."""
    T = 30
    transl = np.zeros((T, 3))
    transl[:, 2] = np.linspace(1.0, 3.0, T)   # rising linearly = 0 accel
    feats = np.zeros((T, 3))
    feats[:, 2] = 0.5   # airborne
    r = gravity_probe(_scene(_subject(1, transl)),
                     GravityConfig(enabled=True, airborne_z_threshold_m=0.05,
                                  min_airborne_run_frames=3, tolerance_mps2=3.0),
                     _feats({1: feats}), fps=30)
    assert r.subjects_violating == 1
    assert r.subjects[0].max_deviation_mps2 > 5.0


def test_short_airborne_run_ignored():
    """A one-frame air blip is not counted."""
    T = 20
    feats = np.zeros((T, 3))
    feats[10, 2] = 0.5   # only frame 10 airborne
    r = gravity_probe(_scene(_subject(1, np.zeros((T, 3)))),
                     GravityConfig(enabled=True, min_airborne_run_frames=3),
                     _feats({1: feats}), fps=30)
    assert r.subjects[0].airborne_runs == 0
