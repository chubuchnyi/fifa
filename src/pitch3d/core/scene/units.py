"""World frame, units and physical constants for the canonical scene model.

Convention (ADR-0005, ADR-0002):
    * Right-handed, **Z-up**, units in **meters**.
    * The football pitch is the plane ``Z = 0``; height is ``+Z``.
    * Gravity acts along ``-Z`` with magnitude :data:`GRAVITY`.

Rationale: this matches Blender (the editing platform, ADR-0001) and the natural
"ground plane + height" intuition for sports. Exporters convert to Y-up where the
target format requires it (glTF/USD); that conversion lives in ``adapters/export``,
never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

GRAVITY: float = 9.80665
"""Standard gravitational acceleration, m/s**2 (used by the ball 3D lift)."""


class UpAxis(str, Enum):
    """Which world axis points up. We standardise on Z."""

    Z = "Z"
    Y = "Y"


class Handedness(str, Enum):
    RIGHT = "right"
    LEFT = "left"


@dataclass
class WorldFrame:
    """Defines the metric world coordinate system of a :class:`~pitch3d.core.scene.scene.Scene`.

    Attributes:
        up_axis: Axis pointing away from gravity. Defaults to Z.
        units: Length unit name. Always ``"m"`` in the canonical model.
        handedness: Coordinate system handedness. Defaults to right-handed.
        origin: Human-readable description of where the origin sits
            (default: the pitch centre spot, on the ground plane).
    """

    up_axis: UpAxis = UpAxis.Z
    units: str = "m"
    handedness: Handedness = Handedness.RIGHT
    origin: str = "field_center_on_ground"
    meters_per_unit: float = 1.0

    def gravity_vector(self) -> tuple[float, float, float]:
        """Return gravity as a 3-vector in this frame."""
        if self.up_axis is UpAxis.Z:
            return (0.0, 0.0, -GRAVITY)
        return (0.0, -GRAVITY, 0.0)


@dataclass
class FieldDimensions:
    """Playing-field size in meters (FIFA default 105 x 68)."""

    length: float = 105.0
    width: float = 68.0


@dataclass
class TimeBase:
    """Frame-rate / timing metadata shared by a source and its scenes."""

    fps: float = 25.0
    start_timecode: str | None = None

    def frame_to_seconds(self, frame_index: int) -> float:
        return frame_index / self.fps


@dataclass
class Settings:
    """Project-level knobs that are not model parameters."""

    extra: dict = field(default_factory=dict)
