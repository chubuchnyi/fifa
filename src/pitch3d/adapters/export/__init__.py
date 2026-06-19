"""Real exporters — honest stubs (FR-26..28, roadmap M1/M3).

glTF/USD/FBX/Alembic + SMPL-X ``.npz`` + the three.js web bundle, with the Z-up→Y-up
axis/scale conversion that those targets need (done here, never in core). Importable now;
``export`` raises ``NotImplementedError``. The canonical-JSON path (FR-28) is already real in
:class:`pitch3d.adapters.fakes.FakeExporter`, which the dry-run uses.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.ports.export import ExportFormat, Exporter, ExportResult
from ...core.scene.scene import Scene

#: Formats the real glTF exporter intends to handle (JSON is handled by the core serializer).
_GLTF_FORMATS = frozenset(
    {ExportFormat.GLTF, ExportFormat.GLB, ExportFormat.USD, ExportFormat.FBX,
     ExportFormat.ALEMBIC, ExportFormat.SMPLX_NPZ, ExportFormat.THREEJS}
)


@dataclass
class GltfExporter(Exporter):
    """Writes interchange geometry/animation + the web viewer (axis/scale converted)."""

    def supports(self, fmt: ExportFormat) -> bool:
        return fmt in _GLTF_FORMATS

    def export(self, scene: Scene, fmt: ExportFormat, out_path: str) -> ExportResult:
        raise NotImplementedError(
            f"real {fmt.value} export is not wired yet (roadmap M1/M3). "
            "Use pitch3d.adapters.fakes.FakeExporter (canonical JSON is real there)."
        )


__all__ = ["GltfExporter"]
