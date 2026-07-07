"""Orient verticality gate — forces body-up to world-up on tilted HMR frames."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pitch3d.core.config.gates import OrientVerticalityConfig
from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.correction.orient_verticality import (
    _body_up_world_z,
    _upright_rotvec,
    _world_yaw_from_rotvec,
    orient_verticality_gate,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Scene, Subject


def _subject(track_id: int, global_orient: np.ndarray) -> Subject:
    T = global_orient.shape[0]
    frames = np.arange(T, dtype=int)
    return Subject(track_id=track_id, proposal=SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=global_orient,
            body_pose=np.zeros((T, 21, 3)), transl=np.zeros((T, 3)),
        ),
    ))


def _scene(*subjects):
    return Scene(id="s", episode_id="e", source_id="c",
                subjects=list(subjects), corrections=[])


def test_upright_rotvec_yields_world_up_z_1():
    """The rotvecs from _upright_rotvec produce bodies with world-up-z ≈ 1."""
    yaws = np.linspace(-np.pi, np.pi, 12)
    rv = _upright_rotvec(yaws)
    up = _body_up_world_z(rv)
    assert np.allclose(up, 1.0, atol=1e-5), up


def test_upright_rotvec_preserves_yaw():
    yaws = np.linspace(-np.pi, np.pi, 8)
    rv = _upright_rotvec(yaws)
    recovered = _world_yaw_from_rotvec(rv)
    # atan2 wraps at ±π; compare via wrap-to-π
    wrap = np.mod(recovered - yaws + np.pi, 2 * np.pi) - np.pi
    assert np.all(np.abs(wrap) < 1e-5), wrap


def test_disabled_passthrough():
    orient = np.tile(np.array([np.pi, 0.0, 0.0]), (5, 1))   # upside-down all frames
    s = _subject(1, orient)
    scene, r = orient_verticality_gate(
        _scene(s), OrientVerticalityConfig(enabled=False),
    )
    assert r.corrections_added == 0


def test_upright_body_not_corrected():
    """Body already upright → gate leaves it alone."""
    orient = _upright_rotvec(np.zeros(5))
    s = _subject(1, orient)
    scene, r = orient_verticality_gate(
        _scene(s), OrientVerticalityConfig(enabled=True),
    )
    assert r.corrections_added == 0
    assert r.subjects_corrected == 0


def test_zero_rotation_is_inverted_in_our_world():
    """SMPL-X canonical (rotvec=0) has +Y_body-up = +Y_smplx → world_z = -1.
    This is INVERTED in our world frame. The gate must flag & correct it."""
    orient = np.zeros((5, 3))   # SMPL-X canonical → inverted in world
    s = _subject(1, orient)
    scene, r = orient_verticality_gate(
        _scene(s), OrientVerticalityConfig(enabled=True),
    )
    assert r.corrections_added == 1
    resolved = resolve_subject_motion(s.proposal, scene.corrections_for(1))
    up = _body_up_world_z(np.asarray(resolved.pose.global_orient))
    assert np.all(up > 0.99), up


def test_sideways_body_corrected():
    """Rotate π/2 around Z lays body sideways → world_z ≈ 0. Must be corrected."""
    orient = np.tile(np.array([0.0, 0.0, np.pi / 2]), (6, 1))
    s = _subject(1, orient)
    scene, r = orient_verticality_gate(
        _scene(s), OrientVerticalityConfig(enabled=True),
    )
    assert r.corrections_added == 1
    resolved = resolve_subject_motion(s.proposal, scene.corrections_for(1))
    up = _body_up_world_z(np.asarray(resolved.pose.global_orient))
    assert np.all(up > 0.99)


def test_mixed_frames_partial_correction():
    """Some frames upright, some inverted → only bad frames rewritten."""
    upright = _upright_rotvec(np.zeros(3))     # already upright
    inverted = np.zeros((3, 3))                # SMPL-X canonical = inverted
    orient = np.concatenate([upright, inverted], axis=0)
    s = _subject(1, orient)
    scene, r = orient_verticality_gate(
        _scene(s), OrientVerticalityConfig(enabled=True),
    )
    assert r.corrections_added == 1
    resolved = resolve_subject_motion(s.proposal, scene.corrections_for(1))
    up = _body_up_world_z(np.asarray(resolved.pose.global_orient))
    # all 6 frames should now be upright (either untouched or newly-upright)
    assert np.all(up > 0.99)


def test_slight_lean_within_tolerance_not_corrected():
    """15° forward lean off vertical stays untouched (max_tilt_rad default ≈35°).

    Build orient = R_lean_smplx @ R_upright where R_lean rotates by 15° about
    the smplx X axis — that gives a tilt of 15° from world vertical, well
    inside the default max_tilt_rad=0.61 rad (~35°).
    """
    lean_rad = np.deg2rad(15)
    upright = _upright_rotvec(np.zeros(5))
    R_upright = Rotation.from_rotvec(upright).as_matrix()
    R_lean = Rotation.from_rotvec(
        np.tile(np.array([lean_rad, 0.0, 0.0]), (5, 1))
    ).as_matrix()
    # Compose in smplx-native order: lean applied FIRST in body, then upright
    R = R_lean @ R_upright
    orient = Rotation.from_matrix(R).as_rotvec()
    up = _body_up_world_z(orient)
    # sanity: this construction really is close to upright
    assert np.all(up > np.cos(np.deg2rad(20))), np.degrees(np.arccos(up))
    s = _subject(1, orient)
    scene, r = orient_verticality_gate(
        _scene(s), OrientVerticalityConfig(enabled=True, max_tilt_rad=0.61),
    )
    assert r.corrections_added == 0


def test_empty_scene():
    scene, r = orient_verticality_gate(
        _scene(), OrientVerticalityConfig(enabled=True),
    )
    assert r.n_subjects == 0
    assert r.corrections_added == 0


def test_single_frame_subject_upright():
    """One-frame subject with pre-upright orient → no correction."""
    orient = _upright_rotvec(np.array([0.0]))   # (1, 3)
    s = _subject(1, orient)
    scene, r = orient_verticality_gate(
        _scene(s), OrientVerticalityConfig(enabled=True),
    )
    assert r.corrections_added == 0
