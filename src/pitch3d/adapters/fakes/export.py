"""FakeExporter — writes the canonical scene JSON for real, placeholders for the rest.

The JSON path (FR-28) is *not* faked: it uses the core serializer, so the dry-run produces
a genuinely reloadable scene. The mesh/animation formats (glTF/USD/FBX/SMPL-X npz/three.js)
need axis/scale conversion and real geometry, so here they write a small labelled
placeholder — honest about being a stand-in for the real ``adapters/export`` work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pitch3d.core.ports.export import ExportFormat, Exporter, ExportResult
from pitch3d.core.scene.scene import Scene
from pitch3d.core.scene.serialization import save_scene


@dataclass
class FakeExporter(Exporter):
    """Real canonical-JSON export; labelled placeholders for geometry formats."""

    def supports(self, fmt: ExportFormat) -> bool:
        return True

    def export(self, scene: Scene, fmt: ExportFormat, out_path: str) -> ExportResult:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt is ExportFormat.JSON:
            save_scene(scene, str(path))
            return ExportResult(fmt=fmt, paths=[str(path)], note="canonical scene JSON")
        path.write_text(
            f"placeholder {fmt.value} export for scene {scene.id}\n", encoding="utf-8"
        )
        return ExportResult(fmt=fmt, paths=[str(path)], note=f"fake {fmt.value} (placeholder)")
