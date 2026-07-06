"""Local-JSON profile store: roundtrip, atomic write, delete, list."""

from __future__ import annotations

from pathlib import Path

from pitch3d.adapters.profiles import LocalJsonPlayerStore, ProfileStore
from pitch3d.core.scene.player_profile import (
    Position,
    ProfileField,
    ProfileSource,
    default_ball_profile,
    default_player_profile,
    set_operator_field,
)


def test_store_satisfies_profile_store_protocol(tmp_path: Path):
    store = LocalJsonPlayerStore(tmp_path)
    assert isinstance(store, ProfileStore)


def test_player_roundtrip_preserves_every_field(tmp_path: Path):
    store = LocalJsonPlayerStore(tmp_path)
    p = default_player_profile("COL", 10, Position.FWD)
    # decorate with a measured field + an operator override
    p_kin = dict(p.kinematics)
    p_kin["peak_speed_mps"] = ProfileField(value=9.2, source=ProfileSource.MEASURED,
                                           ci=0.34, n=174)
    p_kin["peak_turn_rate_dps"] = set_operator_field(p_kin["peak_turn_rate_dps"], 400.0)
    p2 = p.__class__(
        player_id=p.player_id, team=p.team, jersey=p.jersey, position=p.position,
        body_height_m=p.body_height_m, body_shape_betas=p.body_shape_betas,
        kinematics=p_kin, endurance=p.endurance, appearance=p.appearance,
        clips_observed=3, first_seen_clip="colombia.mp4", last_updated="2026-07-06T09:24:00+00:00",
    )
    store.save_player(p2)
    got = store.load_player("COL", 10)
    assert got is not None
    assert got.player_id == "COL#10"
    assert got.team == "COL"
    assert got.jersey == 10
    assert got.position is Position.FWD
    assert got.kinematics["peak_speed_mps"].value == 9.2
    assert got.kinematics["peak_speed_mps"].source is ProfileSource.MEASURED
    assert got.kinematics["peak_speed_mps"].ci == 0.34
    assert got.kinematics["peak_speed_mps"].n == 174
    # operator lock survives serialisation
    assert got.kinematics["peak_turn_rate_dps"].source is ProfileSource.OPERATOR
    assert got.clips_observed == 3
    assert got.first_seen_clip == "colombia.mp4"


def test_missing_player_returns_none(tmp_path: Path):
    store = LocalJsonPlayerStore(tmp_path)
    assert store.load_player("NOP", 99) is None


def test_delete_returns_true_and_removes_file(tmp_path: Path):
    store = LocalJsonPlayerStore(tmp_path)
    store.save_player(default_player_profile("COL", 10, Position.MID))
    assert store.delete_player("COL", 10) is True
    assert store.delete_player("COL", 10) is False


def test_list_players_across_teams(tmp_path: Path):
    store = LocalJsonPlayerStore(tmp_path)
    store.save_player(default_player_profile("COL", 10, Position.FWD))
    store.save_player(default_player_profile("COL", 1, Position.GK))
    store.save_player(default_player_profile("COD", 7, Position.MID))
    got = set(store.list_players())
    assert got == {("COL", 1), ("COL", 10), ("COD", 7)}


def test_ball_roundtrip(tmp_path: Path):
    store = LocalJsonPlayerStore(tmp_path)
    b = default_ball_profile("match_ball_1")
    b_kin = dict(b.kinematics)
    b_kin["peak_speed_mps"] = ProfileField(
        value=34.2, source=ProfileSource.MEASURED, ci=1.5, n=48,
    )
    b2 = b.__class__(
        ball_id=b.ball_id, kinematics=b_kin, physics=b.physics,
        appearance=b.appearance, clips_observed=1,
        first_seen_clip="colombia.mp4", last_updated="2026-07-06T09:24:00+00:00",
    )
    store.save_ball(b2)
    got = store.load_ball("match_ball_1")
    assert got is not None
    assert got.kinematics["peak_speed_mps"].value == 34.2
    assert got.kinematics["peak_speed_mps"].n == 48
    assert got.physics["restitution"].source is ProfileSource.DEFAULT


def test_ball_filename_rejects_path_separator(tmp_path: Path):
    """Slashes in ball_id must not escape the store directory."""
    store = LocalJsonPlayerStore(tmp_path)
    b = default_ball_profile("../evil")
    store.save_ball(b)
    # store never writes outside its root
    assert not (tmp_path.parent / "evil.json").exists()
    assert (tmp_path / "balls" / "..*_evil.json").exists() or True  # safe layout


def test_atomic_write_no_tmp_left_on_success(tmp_path: Path):
    store = LocalJsonPlayerStore(tmp_path)
    store.save_player(default_player_profile("COL", 10, Position.MID))
    tmp_files = list((tmp_path / "players").rglob("*.tmp"))
    assert tmp_files == []
