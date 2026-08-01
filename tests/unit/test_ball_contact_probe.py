"""Ball-contact probe — measure orphan direction changes."""

from __future__ import annotations

import numpy as np

from pitch3d.core.correction.ball_contact_probe import (
    BallContactConfig,
    ball_contact_probe,
)
from pitch3d.core.scene.motion import (
    BallTrack,
    PoseSequence,
    SmplxShape,
    SubjectMotion,
)
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


def _scene_with_ball(subjects, ball_positions: np.ndarray) -> Scene:
    frames = np.arange(ball_positions.shape[0], dtype=int)
    ball = BallTrack(
        frames=frames, positions_3d=ball_positions,
        height_confidence=np.ones(ball_positions.shape[0]),
    )
    return Scene(
        id="s", episode_id="e", source_id="c",
        subjects=list(subjects), corrections=[], ball=ball,
    )


def _feats(feats_by_id):
    def p(subject):
        return feats_by_id.get(int(subject.track_id))
    return p


def test_disabled_returns_empty():
    T = 5
    r = ball_contact_probe(
        _scene_with_ball([_subject(1, np.zeros((T, 3)))], np.zeros((T, 3))),
        BallContactConfig(enabled=False), _feats({1: np.zeros((T, 3))}), fps=30,
    )
    assert r.n_direction_changes == 0
    assert r.n_orphan_hits == 0


def test_straight_ball_no_direction_change():
    T = 20
    ball = np.zeros((T, 3))
    ball[:, 0] = np.linspace(0, 5, T)
    r = ball_contact_probe(
        _scene_with_ball([_subject(1, np.zeros((T, 3)))], ball),
        BallContactConfig(enabled=True), _feats({1: np.zeros((T, 3))}), fps=30,
    )
    assert r.n_direction_changes == 0


def test_hit_with_nearby_player_not_orphan():
    """Ball changes direction; a player foot is close → not orphan."""
    T = 20
    ball = np.zeros((T, 3))
    for i in range(T):
        if i < 10:
            ball[i] = [0.5 * i, 0.0, 0.11]        # +X for 10 frames
        else:
            ball[i] = [5.0 - 0.5 * (i - 10), 0.5 * (i - 10), 0.11]  # -X, +Y
    subj = _subject(1, np.tile([5.0, 0.0, 1.1], (T, 1)))
    feats = np.zeros((T, 3))
    feats[:, 0] = 5.0
    r = ball_contact_probe(
        _scene_with_ball([subj], ball),
        BallContactConfig(enabled=True, direction_change_deg=45.0,
                         contact_radius_m=2.0,
                         min_speed_before_mps=1.0), _feats({1: feats}), fps=30,
    )
    assert r.n_direction_changes >= 1
    assert r.n_orphan_hits == 0


def test_orphan_hit_flagged_when_no_player_close():
    T = 20
    ball = np.zeros((T, 3))
    for i in range(T):
        if i < 10:
            ball[i] = [0.5 * i, 0.0, 0.11]
        else:
            ball[i] = [5.0 - 0.5 * (i - 10), 0.5 * (i - 10), 0.11]
    subj = _subject(1, np.tile([50.0, 50.0, 1.1], (T, 1)))
    feats = np.tile([50.0, 50.0, 0.0], (T, 1))
    r = ball_contact_probe(
        _scene_with_ball([subj], ball),
        BallContactConfig(enabled=True, direction_change_deg=45.0,
                         contact_radius_m=1.0,
                         min_speed_before_mps=1.0), _feats({1: feats}), fps=30,
    )
    assert r.n_direction_changes >= 1
    assert r.n_orphan_hits >= 1
    assert r.max_orphan_distance_m > 10.0


def test_no_ball_returns_empty():
    r = ball_contact_probe(
        Scene(id="s", episode_id="e", source_id="c",
              subjects=[_subject(1, np.zeros((5, 3)))], corrections=[]),
        BallContactConfig(enabled=True), _feats({1: np.zeros((5, 3))}), fps=30,
    )
    assert r.n_direction_changes == 0


def test_no_provider_returns_empty():
    T = 20
    ball = np.zeros((T, 3))
    ball[:, 0] = np.linspace(0, 5, T)
    r = ball_contact_probe(
        _scene_with_ball([_subject(1, np.zeros((T, 3)))], ball),
        BallContactConfig(enabled=True), None, fps=30,
    )
    assert r.n_direction_changes == 0
