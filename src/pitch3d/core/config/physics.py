"""Physics config loader — thresholds live in ``config/physics.yaml``, not in code.

The gates (``KinematicConfig`` / ``CoherenceConfig``) stay pure dataclasses; this
module is the ONE place that decides which numbers they get. That keeps parametric
data parametric — auditable, versionable, comparable across profiles — while the
implementation remains a plain function of its config.

Precedence (lowest → highest):

1. **base defaults** (``base:`` block of the YAML)
2. **profile overrides** (``profiles.<name>.overrides``)
3. **env var overrides** (``PITCH3D_KIN_MAX_SPEED`` etc. — legacy ops surface)
4. **explicit Python overrides** (``load_physics_config(overrides=…)``, tests/harness)

Every constructed field carries a *lineage* string ("base" / "profile:X" /
"env:VAR" / "override:KEY") so a comparison harness — or an operator inspecting
a run — can trace WHERE each number came from. No hidden constants.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..correction.coherence import CoherenceConfig
from ..correction.kinematics import KinematicConfig

#: Repo-relative default location. ``load_physics_config(path=…)`` overrides.
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "physics.yaml"

#: Env vars → (section, field name in YAML). Legacy ops surface kept working.
ENV_MAP: dict[str, tuple[str, str, type]] = {
    "PITCH3D_KIN_MAX_SPEED":     ("kinematic", "max_speed",       float),
    "PITCH3D_KIN_MAX_ACCEL":     ("kinematic", "max_accel",       float),
    "PITCH3D_KIN_TELEPORT":      ("kinematic", "teleport_factor", float),
    "PITCH3D_COH_SMOOTH_WINDOW": ("coherence", "smooth_window",   int),
    "PITCH3D_COH_COAST_SPEED":   ("coherence", "coast_max_speed", float),
    "PITCH3D_FOOT_FLOOR":        ("foot_floor", "floor_m",        float),
    "PITCH3D_FOOT_FLOOR_ON":     ("foot_floor", "enabled",        _bool := lambda s: str(s).lower() in ("1", "true", "yes", "on")),
    "PITCH3D_JOINT_ON":          ("joint",      "enabled",        _bool),
    "PITCH3D_ORIENT_ON":         ("orientation", "enabled",       _bool),
}


@dataclass(frozen=True)
class FootFloorConfig:
    """Auto-default foot-floor clamp (Tier 1a). Not yet applied by the pipeline."""

    enabled: bool = False
    floor_m: float = 0.0
    warn_hover_m: float = 0.30


@dataclass(frozen=True)
class JointKinematicConfig:
    """Per-joint angular-velocity gate (Tier 1b). Schema reserved, not built."""

    enabled: bool = False
    max_omega_dps: float = 600.0


@dataclass(frozen=True)
class OrientationConfig:
    """Root-orientation turn-rate gate (Tier 1c). Schema reserved, not built."""

    enabled: bool = False
    max_turn_rate_dps: float = 720.0


@dataclass(frozen=True)
class BallConfig:
    """Ball measurement thresholds — used by probes, not a gate yet."""

    max_speed: float = 36.0
    max_accel: float = 200.0


@dataclass(frozen=True)
class ProbeConfig:
    """Thresholds only ``scripts/motion_stats.py`` (and future probes) consume."""

    turn_min_speed: float = 2.0
    joint_min_omega_dps: float = 600.0
    orient_min_dps: float = 720.0
    foot_hover_m: float = 0.30


@dataclass(frozen=True)
class PhysicsConfig:
    """Fully-resolved physics config + per-field lineage.

    ``lineage`` maps ``"section.field"`` → provenance string (e.g. ``"base"``,
    ``"profile:conservative"``, ``"env:PITCH3D_KIN_MAX_SPEED"``,
    ``"override:kinematic.max_speed"``). Every scalar in every sub-config has an
    entry — an operator asking "why 9.0 not 10.5" always gets an answer.
    """

    kinematic: KinematicConfig
    coherence: CoherenceConfig
    foot_floor: FootFloorConfig
    joint: JointKinematicConfig
    orientation: OrientationConfig
    ball: BallConfig
    probe: ProbeConfig
    profile_name: str
    profile_description: str
    lineage: dict[str, str] = field(default_factory=dict)
    source_path: str = ""

    def summary(self) -> str:
        """Human-readable one-line summary for logs (profile + a few headline fields)."""
        k = self.kinematic
        c = self.coherence
        return (
            f"profile={self.profile_name!r} "
            f"kinematic(max_speed={k.max_speed}m/s, max_accel={k.max_accel}m/s²) "
            f"coherence(smooth_window={c.smooth_window}, coast={c.coast_max_speed}m/s) "
            f"foot_floor={'on' if self.foot_floor.enabled else 'off'} "
            f"joint={'on' if self.joint.enabled else 'off'} "
            f"orientation={'on' if self.orientation.enabled else 'off'}"
        )


def _deep_merge(dst: dict[str, Any], src: dict[str, Any], lineage: dict[str, str],
                origin: str, path: str = "") -> None:
    """In-place merge ``src`` onto ``dst``, recording lineage for every leaf touched."""
    for k, v in src.items():
        key = f"{path}.{k}" if path else k
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v, lineage, origin, key)
        else:
            dst[k] = v
            lineage[key] = origin


def _apply_env(cfg: dict[str, Any], lineage: dict[str, str],
               env: dict[str, str]) -> None:
    """Apply ENV_MAP overrides onto ``cfg`` in place; record ``env:VAR`` lineage."""
    for var, (section, field_name, cast) in ENV_MAP.items():
        if var not in env:
            continue
        section_dict = cfg.setdefault(section, {})
        try:
            section_dict[field_name] = cast(env[var])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"env {var}={env[var]!r} won't cast to {cast.__name__}: {exc}") from exc
        lineage[f"{section}.{field_name}"] = f"env:{var}"


def _apply_overrides(cfg: dict[str, Any], lineage: dict[str, str],
                     overrides: dict[str, Any]) -> None:
    """Apply Python-level overrides (harness / tests); record ``override:KEY`` lineage."""
    _deep_merge(cfg, overrides, lineage, origin="override")
    # rewrite origin to include the key path so lineage stays traceable
    for key in _flatten_keys(overrides):
        lineage[key] = f"override:{key}"


def _flatten_keys(d: dict[str, Any], prefix: str = "") -> list[str]:
    out: list[str] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_flatten_keys(v, key))
        else:
            out.append(key)
    return out


def _build_dataclass(cls, section: dict[str, Any], name: str):
    """Instantiate a frozen dataclass from a subset dict; unknown keys raise."""
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(section) - fields
    if unknown:
        raise ValueError(
            f"config section {name!r} has unknown key(s) {sorted(unknown)}; "
            f"known: {sorted(fields)}"
        )
    return cls(**section)


def load_physics_config(
    path: str | Path | None = None,
    profile: str = "default",
    env: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> PhysicsConfig:
    """Load ``config/physics.yaml`` into a resolved :class:`PhysicsConfig`.

    * ``path`` — defaults to the repo's ``config/physics.yaml``.
    * ``profile`` — must exist under ``profiles:``; ``"default"`` is required to exist.
    * ``env`` — defaults to ``os.environ``; pass ``{}`` in tests to isolate.
    * ``overrides`` — Python-level nested dict for the harness / tests.

    Every scalar's provenance is recorded in :attr:`PhysicsConfig.lineage`.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path}: root must be a mapping, got {type(raw).__name__}")
    if raw.get("version") != 1:
        raise ValueError(f"{cfg_path}: unsupported schema version {raw.get('version')!r}")

    base = copy.deepcopy(raw.get("base", {}))
    lineage: dict[str, str] = {k: "base" for k in _flatten_keys(base)}

    profiles = raw.get("profiles", {})
    if profile not in profiles:
        raise KeyError(
            f"profile {profile!r} not in {cfg_path.name}; available: {sorted(profiles)}"
        )
    prof = profiles[profile]
    description = str(prof.get("description", ""))
    prof_overrides = prof.get("overrides") or {}
    _deep_merge(base, prof_overrides, lineage, origin=f"profile:{profile}")

    _apply_env(base, lineage, env if env is not None else dict(os.environ))
    if overrides:
        _apply_overrides(base, lineage, overrides)

    kin = _build_dataclass(KinematicConfig, base.get("kinematic", {}), "kinematic")
    coh = _build_dataclass(CoherenceConfig, base.get("coherence", {}), "coherence")
    foot = _build_dataclass(FootFloorConfig, base.get("foot_floor", {}), "foot_floor")
    joint = _build_dataclass(JointKinematicConfig, base.get("joint", {}), "joint")
    orient = _build_dataclass(OrientationConfig, base.get("orientation", {}), "orientation")
    ball = _build_dataclass(BallConfig, base.get("ball", {}), "ball")
    probe = _build_dataclass(ProbeConfig, base.get("probe", {}), "probe")

    return PhysicsConfig(
        kinematic=kin,
        coherence=coh,
        foot_floor=foot,
        joint=joint,
        orientation=orient,
        ball=ball,
        probe=probe,
        profile_name=profile,
        profile_description=description,
        lineage=lineage,
        source_path=str(cfg_path),
    )
