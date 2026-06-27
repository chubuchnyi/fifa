"""Real exporters (FR-26..28, roadmap M1/M3).

:class:`~pitch3d.adapters.export.gltf.GltfExporter` is **wired now**: real SMPL-X ``.npz`` +
canonical JSON (numpy/stdlib, no extra), glTF/GLB assembly (pure Z-up→Y-up axis conversion, the
actual ``pygltflib`` serialization gated behind ``pitch3d[export]``), and the **dependency-free
three.js web viewer** (``ExportFormat.THREEJS`` → ``index.html`` + ``scene.json``, M3-7) built by
:mod:`pitch3d.adapters.export.web`. USD/FBX/Alembic stay unsupported here (roadmap M3). The dry-run
uses :class:`pitch3d.adapters.fakes.FakeExporter`, whose canonical-JSON path is already real.
"""

from __future__ import annotations

from .gltf import GltfExporter
from .web import build_viewer_payload, write_web_bundle

__all__ = ["GltfExporter", "build_viewer_payload", "write_web_bundle"]
