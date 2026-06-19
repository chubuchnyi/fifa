"""Exporter — write a scene to interchange formats and the web viewer.

Covers FR-26 (glTF/USD/FBX/Alembic + SMPL-X .npz), FR-27 (three.js / R3F viewer) and
FR-28 (intermediate JSON; synthesized videos). Axis/scale conversion (Z-up→Y-up where a
target requires it) happens in the export adapter, never in the core.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from ..scene.scene import Scene
from .base import Port


class ExportFormat(str, Enum):
    GLTF = "gltf"
    GLB = "glb"
    USD = "usd"
    FBX = "fbx"
    ALEMBIC = "abc"
    JSON = "json"            # canonical scene JSON (intermediate data, FR-28)
    SMPLX_NPZ = "smplx_npz"  # SMPL-X animation alone (FR-26)
    THREEJS = "threejs"      # web viewer bundle (FR-27)


@dataclass
class ExportResult:
    """What an export produced."""

    fmt: ExportFormat
    paths: list[str] = field(default_factory=list)
    note: str | None = None


class Exporter(Port):
    """Writes a scene to a target format (FR-26..28)."""

    @abstractmethod
    def supports(self, fmt: ExportFormat) -> bool:
        """Whether this exporter handles ``fmt``."""
        raise NotImplementedError

    @abstractmethod
    def export(self, scene: Scene, fmt: ExportFormat, out_path: str) -> ExportResult:
        """Export ``scene`` to ``fmt`` at ``out_path`` (file or directory)."""
        raise NotImplementedError
