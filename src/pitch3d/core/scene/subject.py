"""Subjects: players, goalkeepers, referees — carriers of a parametric body.

A subject holds its **proposal** motion (raw model output). Operator deltas live in
the scene's correction stack (see ``layers.py``); the *resolved* motion is computed by
the correction engine, never hand-stored on the subject (ADR-0002, FR-21).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .motion import SubjectMotion


class Role(str, Enum):
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"


@dataclass
class Team:
    """A team / officiating group, assigned by the tracker's classifier (FR-6)."""

    id: str
    name: str | None = None
    color_rgb: tuple[float, float, float] | None = None


@dataclass
class Subject:
    """One tracked person in the scene.

    Attributes:
        track_id: Stable tracker id (FR-6).
        role: player / goalkeeper / referee.
        team_id: Reference to a :class:`Team` (None for referees).
        jersey_number: Optional shirt number (FR-10, OCR/re-id).
        proposal: Raw SMPL-X motion from HMR — the non-destructive base for edits.
    """

    track_id: int
    proposal: SubjectMotion
    role: Role = Role.PLAYER
    team_id: str | None = None
    jersey_number: int | None = None
