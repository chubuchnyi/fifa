"""Ball profile wiring (T4.c): proposals from resolved motion + storage roundtrip."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pitch3d.adapters.profiles import LocalJsonPlayerStore
from pitch3d.core.scene.player_profile import (
    ProfileField,
    ProfileSource,
    ProfileUpdateProposal,
    apply_profile_updates,
    default_ball_profile,
    emit_ball_proposals,
    load_priors,
    set_operator_field,
)


def test_emit_ball_proposals_from_ballistic_motion():
    """A trajectory that peaks at ~15 m/s should show up as p95 speed near that."""
    fps = 30.0
    T = 30
    frames = np.arange(T)
    # simple linear motion at 15 m/s along X
    positions = np.zeros((T, 3))
    positions[:, 0] = (15.0 / fps) * np.arange(T)
    proposals = emit_ball_proposals(
        ball_track_id=-1, frames=frames, positions_3d=positions, fps=fps,
        default_peak_speed=34.0, default_peak_accel=190.0,
    )
    keys = {p.field_key for p in proposals}
    assert keys == {"peak_speed_mps", "peak_accel_mps2"}
    speed = next(p for p in proposals if p.field_key == "peak_speed_mps")
    assert abs(speed.observation - 15.0) < 0.5
    assert speed.default_value == 34.0
    assert speed.domain == "ball"


def test_emit_ball_proposals_short_track_returns_empty():
    """Two-frame ball is not enough for accel — helper returns empty."""
    p = emit_ball_proposals(
        ball_track_id=-1, frames=np.array([0, 1]),
        positions_3d=np.zeros((2, 3)), fps=30.0,
    )
    assert p == []


def test_apply_profile_updates_persists_ball_speed(tmp_path: Path):
    """Ball-domain proposals write through to the ball store; player store untouched."""
    store = LocalJsonPlayerStore(tmp_path)
    priors = load_priors()
    updates = [
        ProfileUpdateProposal(
            track_id=-1, domain="ball", field_key="peak_speed_mps",
            observation=34.2, confidence=1.0, default_value=34.0,
        )
    ] * priors.policy.min_promote_n
    counts = apply_profile_updates(
        store, priors,
        subject_lookup={},
        updates=updates,
        ball_id_lookup={-1: "match_ball_1"},
    )
    assert counts["applied"] == priors.policy.min_promote_n
    saved = store.load_ball("match_ball_1")
    assert saved is not None
    field = saved.kinematics["peak_speed_mps"]
    assert field.source is ProfileSource.MEASURED
    # value settled near 34.2 and above the 0.90*34 = 30.6 floor
    assert 30.6 <= field.value <= 34.5


def test_apply_ball_updates_creates_default_when_missing(tmp_path: Path):
    """First observation for an unseen ball seeds from priors."""
    store = LocalJsonPlayerStore(tmp_path)
    priors = load_priors()
    assert store.load_ball("match_ball_1") is None
    counts = apply_profile_updates(
        store, priors,
        subject_lookup={},
        updates=[ProfileUpdateProposal(
            track_id=-1, domain="ball", field_key="peak_speed_mps",
            observation=30.0, confidence=1.0, default_value=34.0,
        )],
        ball_id_lookup={-1: "match_ball_1"},
    )
    assert counts["applied"] == 1
    saved = store.load_ball("match_ball_1")
    assert saved is not None
    assert saved.kinematics["peak_speed_mps"].n == 1


def test_apply_ball_updates_operator_lock_holds(tmp_path: Path):
    """A ball field an operator set is never overwritten by auto-tune."""
    store = LocalJsonPlayerStore(tmp_path)
    priors = load_priors()
    profile = default_ball_profile("match_ball_1", priors=priors)
    from dataclasses import replace as _replace
    kin = dict(profile.kinematics)
    kin["peak_speed_mps"] = set_operator_field(kin["peak_speed_mps"], value=30.0)
    store.save_ball(_replace(profile, kinematics=kin))
    counts = apply_profile_updates(
        store, priors, subject_lookup={},
        updates=[ProfileUpdateProposal(
            track_id=-1, domain="ball", field_key="peak_speed_mps",
            observation=35.0, confidence=1.0, default_value=34.0,
        )] * 5,
        ball_id_lookup={-1: "match_ball_1"},
    )
    assert counts["operator_locked"] == 5
    saved = store.load_ball("match_ball_1")
    assert saved.kinematics["peak_speed_mps"].source is ProfileSource.OPERATOR
    assert saved.kinematics["peak_speed_mps"].value == 30.0


def test_apply_ball_updates_unknown_ball_id_is_skipped(tmp_path: Path):
    store = LocalJsonPlayerStore(tmp_path)
    priors = load_priors()
    counts = apply_profile_updates(
        store, priors, subject_lookup={},
        updates=[ProfileUpdateProposal(
            track_id=-1, domain="ball", field_key="peak_speed_mps",
            observation=30.0, confidence=1.0,
        )],
        ball_id_lookup={},  # no id for track -1
    )
    assert counts["skipped"] == 1


def test_apply_ball_updates_unknown_field_key_skipped(tmp_path: Path):
    store = LocalJsonPlayerStore(tmp_path)
    priors = load_priors()
    counts = apply_profile_updates(
        store, priors, subject_lookup={},
        updates=[ProfileUpdateProposal(
            track_id=-1, domain="ball", field_key="mystery_ball_knob",
            observation=1.0, confidence=1.0,
        )],
        ball_id_lookup={-1: "match_ball_1"},
    )
    assert counts["skipped"] == 1


def test_mixed_player_and_ball_updates_route_correctly(tmp_path: Path):
    """Player + ball proposals in one batch each land in their own store."""
    store = LocalJsonPlayerStore(tmp_path)
    priors = load_priors()
    from pitch3d.core.scene.player_profile import Position
    updates = [
        ProfileUpdateProposal(
            track_id=1, domain="player", field_key="peak_speed_mps",
            observation=9.0, confidence=1.0, default_value=9.5,
        ),
        ProfileUpdateProposal(
            track_id=-1, domain="ball", field_key="peak_speed_mps",
            observation=30.0, confidence=1.0, default_value=34.0,
        ),
    ]
    counts = apply_profile_updates(
        store, priors,
        subject_lookup={1: ("A", 10, Position.MID)},
        updates=updates,
        ball_id_lookup={-1: "match_ball_1"},
    )
    assert counts["applied"] == 2
    assert store.load_player("A", 10) is not None
    assert store.load_ball("match_ball_1") is not None
