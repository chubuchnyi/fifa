"""Pure per-gate config dataclasses — no imports beyond stdlib.

Kept separate from ``physics.py`` so the correction modules (``foot_floor``,
``joint_kinematics``, ``orientation``) can consume their configs without pulling
the YAML loader (which itself imports ``KinematicConfig`` / ``CoherenceConfig``
from ``core.correction``). Breaking that cycle lives here — this module has no
package-internal imports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FootFloorConfig:
    """Auto-default foot-floor clamp (Tier 1a) — read from ``config/physics.yaml``."""

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
class CollisionConfig:
    """Capsule-collision post-process (Tier 3).

    Not a physics sim: at each frame the subjects are treated as vertical
    capsules of radius ``capsule_radius_m`` on the pitch plane; overlapping
    pairs get a soft push apart, one Jacobi iteration per pass. ``strength=1.0``
    fully resolves each overlap in one pass; ``0.5`` halves it (softer, less
    twitchy). ``n_passes`` iterates the Jacobi step so a stack of three near
    each other converges.
    """

    enabled: bool = False
    capsule_radius_m: float = 0.35     # ~ shoulder half-width for a standing player
    strength: float = 0.5              # fraction of overlap resolved per pass
    n_passes: int = 4                  # bounded iterations per frame
    max_push_per_frame_m: float = 0.30  # safety cap — never move a subject more than this
    min_correction_m: float = 1e-4      # skip emitting a correction below this net push


@dataclass(frozen=True)
class ProbeConfig:
    """Thresholds only ``scripts/motion_stats.py`` (and future probes) consume."""

    turn_min_speed: float = 2.0
    joint_min_omega_dps: float = 600.0
    orient_min_dps: float = 720.0
    foot_hover_m: float = 0.30
