"""Local-JSON persistence for :class:`PlayerProfile` / :class:`BallProfile`.

Layout:

    <root>/players/<team>/<jersey>.json
    <root>/balls/<ball_id>.json

Atomic write: serialise to ``.<name>.tmp`` and rename over the target so a
crash mid-write can't corrupt an existing profile. Deleting a file resets the
player to defaults (documented lifecycle in ``docs/research/2026-07-06-...``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from ...core.scene.player_profile import (
    BallProfile,
    PlayerProfile,
    Position,
    ProfileField,
    ProfileSource,
)


@runtime_checkable
class ProfileStore(Protocol):
    """Port for persistence adapters (allows swapping local JSON for e.g. Redis)."""

    def load_player(self, team: str, jersey: int) -> PlayerProfile | None: ...
    def save_player(self, profile: PlayerProfile) -> None: ...
    def delete_player(self, team: str, jersey: int) -> bool: ...
    def load_ball(self, ball_id: str) -> BallProfile | None: ...
    def save_ball(self, profile: BallProfile) -> None: ...
    def list_players(self) -> list[tuple[str, int]]: ...


def _dump_field(f: ProfileField) -> dict:
    return {
        "value": float(f.value),
        "source": f.source.value,
        "ci": float(f.ci) if f.ci is not None else None,
        "n": int(f.n),
    }


def _load_field(d: dict) -> ProfileField:
    return ProfileField(
        value=float(d["value"]),
        source=ProfileSource(d.get("source", "default")),
        ci=None if d.get("ci") is None else float(d["ci"]),
        n=int(d.get("n", 0)),
    )


def _dump_field_map(m: dict[str, ProfileField]) -> dict[str, dict]:
    return {k: _dump_field(v) for k, v in m.items()}


def _load_field_map(m: dict[str, dict]) -> dict[str, ProfileField]:
    return {k: _load_field(v) for k, v in m.items()}


def _dump_player(p: PlayerProfile) -> dict:
    return {
        "schema_version": 1,
        "player_id": p.player_id,
        "team": p.team,
        "jersey": p.jersey,
        "position": p.position.value,
        "body": {"height_m": p.body_height_m, "shape_betas": list(p.body_shape_betas)},
        "kinematics": _dump_field_map(p.kinematics),
        "endurance": _dump_field_map(p.endurance),
        "appearance": _dump_field_map(p.appearance),
        "provenance": {
            "clips_observed": p.clips_observed,
            "first_seen_clip": p.first_seen_clip,
            "last_updated": p.last_updated,
        },
    }


def _load_player(d: dict) -> PlayerProfile:
    body = d.get("body", {})
    prov = d.get("provenance", {})
    return PlayerProfile(
        player_id=str(d["player_id"]),
        team=str(d["team"]),
        jersey=int(d["jersey"]),
        position=Position(d.get("position", "UNKNOWN")),
        body_height_m=float(body.get("height_m", 1.80)),
        body_shape_betas=tuple(body.get("shape_betas", ())),
        kinematics=_load_field_map(d.get("kinematics", {})),
        endurance=_load_field_map(d.get("endurance", {})),
        appearance=_load_field_map(d.get("appearance", {})),
        clips_observed=int(prov.get("clips_observed", 0)),
        first_seen_clip=prov.get("first_seen_clip"),
        last_updated=prov.get("last_updated"),
    )


def _dump_ball(b: BallProfile) -> dict:
    return {
        "schema_version": 1,
        "ball_id": b.ball_id,
        "kinematics": _dump_field_map(b.kinematics),
        "physics": _dump_field_map(b.physics),
        "appearance": _dump_field_map(b.appearance),
        "provenance": {
            "clips_observed": b.clips_observed,
            "first_seen_clip": b.first_seen_clip,
            "last_updated": b.last_updated,
        },
    }


def _load_ball(d: dict) -> BallProfile:
    prov = d.get("provenance", {})
    return BallProfile(
        ball_id=str(d["ball_id"]),
        kinematics=_load_field_map(d.get("kinematics", {})),
        physics=_load_field_map(d.get("physics", {})),
        appearance=_load_field_map(d.get("appearance", {})),
        clips_observed=int(prov.get("clips_observed", 0)),
        first_seen_clip=prov.get("first_seen_clip"),
        last_updated=prov.get("last_updated"),
    )


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class LocalJsonPlayerStore:
    """Persist profiles as JSON files under ``root``.

    ``root/players/<team>/<jersey>.json`` and ``root/balls/<ball_id>.json``.
    Missing files → :meth:`load_player` returns ``None`` (caller falls back
    to :func:`default_player_profile`).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "players").mkdir(parents=True, exist_ok=True)
        (self.root / "balls").mkdir(parents=True, exist_ok=True)

    # ── players ──────────────────────────────────────────────────────────

    def _player_path(self, team: str, jersey: int) -> Path:
        return self.root / "players" / str(team) / f"{int(jersey)}.json"

    def load_player(self, team: str, jersey: int) -> PlayerProfile | None:
        p = self._player_path(team, jersey)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as fh:
            return _load_player(json.load(fh))

    def save_player(self, profile: PlayerProfile) -> None:
        _atomic_write(self._player_path(profile.team, profile.jersey), _dump_player(profile))

    def delete_player(self, team: str, jersey: int) -> bool:
        p = self._player_path(team, jersey)
        if not p.exists():
            return False
        p.unlink()
        # tidy empty team directories
        try:
            p.parent.rmdir()
        except OSError:
            pass
        return True

    def list_players(self) -> list[tuple[str, int]]:
        players_root = self.root / "players"
        out: list[tuple[str, int]] = []
        if not players_root.exists():
            return out
        for team_dir in sorted(players_root.iterdir()):
            if not team_dir.is_dir():
                continue
            for jersey_file in sorted(team_dir.glob("*.json")):
                try:
                    jersey = int(jersey_file.stem)
                except ValueError:
                    continue
                out.append((team_dir.name, jersey))
        return out

    # ── balls ────────────────────────────────────────────────────────────

    def _ball_path(self, ball_id: str) -> Path:
        # keep names filesystem-safe — reject slashes to prevent directory escape
        safe = str(ball_id).replace(os.sep, "_")
        return self.root / "balls" / f"{safe}.json"

    def load_ball(self, ball_id: str) -> BallProfile | None:
        p = self._ball_path(ball_id)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as fh:
            return _load_ball(json.load(fh))

    def save_ball(self, profile: BallProfile) -> None:
        _atomic_write(self._ball_path(profile.ball_id), _dump_ball(profile))
