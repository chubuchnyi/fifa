"""Per-player and per-ball physical characteristics with online auto-tuning (T4, R-6).

Motivation (§4 of ``docs/research/2026-07-06-player-physics.md``): a single
``KinematicConfig`` with population-scale limits over-permits a goalkeeper's
motion and under-permits a winger's. Each player is a stateful instance that
carries its own ceilings and cruise numbers, seeded from position priors
(``config/player_priors.yaml``) and refined by measurement.

Every field is tagged with (value, source, ci, n) so an operator can always
trace WHERE a number came from — same "parametric stays parametric" rule the
physics config already follows.

Update rule (see :func:`update_field`) — seven layers of filtering:

1. Only update from **resolved** (post-gate) motion. The M3-9 gate is the
   feasibility floor; tuning from raw jitter would train the ceiling on noise.
2. Use a robust estimator (``p95`` of a window, not per-frame ``max``).
3. Confidence-weight the update; low ``subject_frame_conf`` contributes less.
4. Wait for ``min_promote_n`` samples before promoting ``default → measured``.
5. Quarantine incoming observations outside ``mean ± quarantine_ci_mult · ci``
   — probable ID swap; the gate catches it as a teleport, the tuner refuses
   to learn from it.
6. Enforce a floor on ceilings: ``max(value, ceiling_floor_mult · default)``.
   A player who never sprinted this clip should still be allowed to next clip.
7. ``operator``-sourced fields are IMMUTABLE — auto-tune never overwrites a
   human decision.

All logic is a pure function of its inputs — no I/O, no logging side effects.
The adapter (``adapters/profiles/local_json.py``) handles persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PRIORS_PATH = Path(__file__).resolve().parents[4] / "config" / "player_priors.yaml"


class ProfileSource(str, Enum):
    """Where the value in a :class:`ProfileField` came from."""

    DEFAULT = "default"       # population prior; never observed
    MEASURED = "measured"     # promoted after ``min_promote_n`` clean samples
    OPERATOR = "operator"     # human-set; auto-tune must not touch


class Position(str, Enum):
    """Coarse position tag driving population priors."""

    GK = "GK"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ProfileField:
    """A single characteristic with lineage (value, source, ci, n)."""

    value: float
    source: ProfileSource = ProfileSource.DEFAULT
    ci: float | None = None          # 1-σ estimate; ``None`` until we have ≥ 2 obs
    n: int = 0                       # number of accepted observations so far

    def is_operator_locked(self) -> bool:
        return self.source is ProfileSource.OPERATOR


@dataclass(frozen=True)
class AutoTunePolicy:
    """Knobs for :func:`update_field`; comes from ``player_priors.yaml`` / overrides."""

    min_promote_n: int = 30
    ewma_time_constant: float = 500.0
    quarantine_ci_mult: float = 3.0
    ceiling_floor_mult: float = 0.90
    min_confidence: float = 0.5


@dataclass(frozen=True)
class PopulationPriors:
    """Loaded ``player_priors.yaml`` — positions map + shared body + policy + ball."""

    positions: dict[Position, dict[str, float]]
    body_height_m: float
    n_betas: int
    policy: AutoTunePolicy
    ball: dict[str, float]

    def for_position(self, pos: Position) -> dict[str, float]:
        return self.positions.get(pos, self.positions[Position.UNKNOWN])


@dataclass(frozen=True)
class PlayerProfile:
    """Stateful per-player physical + appearance model."""

    player_id: str
    team: str
    jersey: int
    position: Position = Position.UNKNOWN
    body_height_m: float = 1.80
    body_shape_betas: tuple[float, ...] = ()
    kinematics: dict[str, ProfileField] = field(default_factory=dict)
    endurance: dict[str, ProfileField] = field(default_factory=dict)
    appearance: dict[str, ProfileField] = field(default_factory=dict)
    clips_observed: int = 0
    first_seen_clip: str | None = None
    last_updated: str | None = None  # ISO-8601 UTC


@dataclass(frozen=True)
class BallProfile:
    """Stateful per-ball physical + appearance model (§4.3)."""

    ball_id: str
    kinematics: dict[str, ProfileField] = field(default_factory=dict)
    physics: dict[str, ProfileField] = field(default_factory=dict)
    appearance: dict[str, ProfileField] = field(default_factory=dict)
    clips_observed: int = 0
    first_seen_clip: str | None = None
    last_updated: str | None = None


def load_priors(path: str | Path | None = None) -> PopulationPriors:
    """Load ``player_priors.yaml`` into a :class:`PopulationPriors`."""
    p = Path(path) if path else DEFAULT_PRIORS_PATH
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw.get("version") != 1:
        raise ValueError(f"{p}: unsupported version {raw.get('version')!r}")
    positions = {
        Position(pos_key): {k: float(v) for k, v in fields.items()}
        for pos_key, fields in raw["positions"].items()
    }
    body = raw.get("body", {})
    at = raw.get("auto_tune", {})
    ball = {k: float(v) for k, v in raw.get("ball", {}).items()}
    return PopulationPriors(
        positions=positions,
        body_height_m=float(body.get("height_m", 1.80)),
        n_betas=int(body.get("n_betas", 10)),
        policy=AutoTunePolicy(
            min_promote_n=int(at.get("min_promote_n", 30)),
            ewma_time_constant=float(at.get("ewma_time_constant", 500.0)),
            quarantine_ci_mult=float(at.get("quarantine_ci_mult", 3.0)),
            ceiling_floor_mult=float(at.get("ceiling_floor_mult", 0.90)),
            min_confidence=float(at.get("min_confidence", 0.5)),
        ),
        ball=ball,
    )


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _default_field(value: float) -> ProfileField:
    return ProfileField(value=float(value), source=ProfileSource.DEFAULT, ci=None, n=0)


def default_player_profile(
    team: str, jersey: int, position: Position = Position.UNKNOWN,
    priors: PopulationPriors | None = None,
) -> PlayerProfile:
    """Build a fresh profile seeded from the population priors for ``position``."""
    priors = priors or load_priors()
    pos_defaults = priors.for_position(position)
    kin_keys = ("peak_speed_mps", "cruise_speed_mps", "peak_accel_mps2",
                "peak_turn_rate_dps", "peak_joint_omega_dps")
    end_keys = ("sprint_budget_s", "recover_tau_s")
    kinematics = {k: _default_field(pos_defaults[k]) for k in kin_keys if k in pos_defaults}
    endurance = {k: _default_field(pos_defaults[k]) for k in end_keys if k in pos_defaults}
    return PlayerProfile(
        player_id=f"{team}#{jersey}",
        team=str(team),
        jersey=int(jersey),
        position=position,
        body_height_m=float(priors.body_height_m),
        kinematics=kinematics,
        endurance=endurance,
        last_updated=_now_iso(),
    )


def default_ball_profile(
    ball_id: str, priors: PopulationPriors | None = None,
) -> BallProfile:
    priors = priors or load_priors()
    kin = {
        "peak_speed_mps": _default_field(priors.ball.get("peak_speed_mps", 34.0)),
        "typical_pass_mps": _default_field(priors.ball.get("typical_pass_mps", 18.0)),
        "peak_accel_mps2": _default_field(priors.ball.get("peak_accel_mps2", 190.0)),
    }
    phys = {
        "restitution": _default_field(priors.ball.get("restitution", 0.85)),
        "drag_coeff": _default_field(priors.ball.get("drag_coeff", 0.28)),
    }
    appearance = {
        "radius_m": _default_field(priors.ball.get("radius_m", 0.11)),
    }
    return BallProfile(
        ball_id=str(ball_id),
        kinematics=kin,
        physics=phys,
        appearance=appearance,
        last_updated=_now_iso(),
    )


# ─── auto-tune ────────────────────────────────────────────────────────────────


class UpdateOutcome(str, Enum):
    """Why an update was accepted or rejected — for logging/audit."""

    APPLIED = "applied"                       # updated in-place (or promoted)
    QUARANTINED = "quarantined"               # outside CI×mult, ignored
    LOW_CONFIDENCE = "low_confidence"         # subject_frame_conf below threshold
    OPERATOR_LOCKED = "operator_locked"       # source=operator → immutable


def _accumulate_ci(prev_ci: float | None, prev_value: float, obs: float,
                   n: int, alpha: float) -> float:
    """Update the running 1-σ estimate with the new observation.

    Uses a simple EWMA of squared deviation → sqrt at the end. When there is
    only one prior sample, seed the variance from the absolute delta.
    """
    if prev_ci is None or n <= 1:
        return abs(obs - prev_value)
    var_prev = prev_ci ** 2
    var_new = (1 - alpha) * var_prev + alpha * (obs - prev_value) ** 2
    return float(var_new ** 0.5)


def update_field(
    field_: ProfileField,
    observation: float,
    *,
    confidence: float = 1.0,
    policy: AutoTunePolicy,
    default_value: float | None = None,
) -> tuple[ProfileField, UpdateOutcome]:
    """Apply one observation through the seven-layer filter and return a new field.

    * ``field_`` — current state.
    * ``observation`` — a per-frame or per-window robust value (e.g. p95 speed
      over the last 5 s).  Caller decides the robust estimator; this function
      just accepts the number and does the filtering.
    * ``confidence`` — ``subject_frame_conf`` for the source frame(s).
    * ``policy`` — knobs from :class:`AutoTunePolicy`.
    * ``default_value`` — the population prior for that field (used for the
      ceiling floor, layer 6). ``None`` skips the floor guard.
    """
    if field_.is_operator_locked():
        return field_, UpdateOutcome.OPERATOR_LOCKED
    if confidence < policy.min_confidence:
        return field_, UpdateOutcome.LOW_CONFIDENCE

    if (field_.ci is not None and field_.n > 1
            and abs(observation - field_.value)
            > policy.quarantine_ci_mult * field_.ci):
        return field_, UpdateOutcome.QUARANTINED

    if field_.n < policy.min_promote_n:
        # arithmetic mean until we've seen enough samples to promote
        new_value = (field_.value * field_.n + observation * confidence) / max(
            field_.n + confidence, 1e-9
        )
        new_ci = _accumulate_ci(field_.ci, field_.value, observation, field_.n, alpha=0.5)
        new_n = field_.n + 1
        new_source = ProfileSource.DEFAULT
        if new_n >= policy.min_promote_n:
            new_source = ProfileSource.MEASURED
    else:
        alpha = 1.0 / (1.0 + policy.ewma_time_constant)
        alpha_conf = alpha * confidence
        new_value = (1 - alpha_conf) * field_.value + alpha_conf * observation
        new_ci = _accumulate_ci(field_.ci, field_.value, observation, field_.n, alpha)
        new_n = field_.n + 1
        new_source = ProfileSource.MEASURED

    if default_value is not None:
        floor = policy.ceiling_floor_mult * float(default_value)
        new_value = max(new_value, floor)

    return replace(field_, value=float(new_value), ci=float(new_ci),
                   n=int(new_n), source=new_source), UpdateOutcome.APPLIED


def set_operator_field(field_: ProfileField, value: float) -> ProfileField:
    """Human override: set the value + mark it OPERATOR (immutable from now on)."""
    return replace(field_, value=float(value), source=ProfileSource.OPERATOR)
