"""Real exporters (FR-26..28, roadmap M1/M3).

:class:`~pitch3d.adapters.export.gltf.GltfExporter` is **wired now**: real SMPL-X ``.npz`` +
canonical JSON (numpy/stdlib, no extra) and glTF/GLB assembly (pure Z-up→Y-up axis conversion,
the actual ``pygltflib`` serialization gated behind ``pitch3d[export]``). USD/FBX/Alembic and the
three.js bundle stay unsupported here (roadmap M3). The dry-run uses
:class:`pitch3d.adapters.fakes.FakeExporter`, whose canonical-JSON path is already real.
"""

from __future__ import annotations

from .gltf import GltfExporter

__all__ = ["GltfExporter"]
