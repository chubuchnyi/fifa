"""Blender adapter — honest stub for the editor's SceneObserver (ADR-0008, roadmap M1).

The real visual feedback for the LLM loop (and the human UI) comes from Blender: SCENE_3D
snapshots of the resolved proxy from canonical viewpoints, the FRAME_OVERLAY reprojection,
and a UI screenshot. This satisfies :class:`SceneObserver` but raises ``NotImplementedError``
until ``bpy`` wiring lands; :class:`pitch3d.adapters.fakes.FakeSceneObserver` backs the loop
meanwhile. Importable with no ``bpy`` dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...core.ports.observation import (
    ObservationImage,
    SceneObserver,
    ViewpointCamera,
)
from ...core.ports.render import RenderQuality
from ...core.scene.scene import Scene


def _todo(what: str) -> NotImplementedError:
    return NotImplementedError(
        f"Blender {what} is not wired yet (roadmap M1). "
        "Use pitch3d.adapters.fakes.FakeSceneObserver for tests and the dry-run."
    )


@dataclass
class BlenderSceneObserver(SceneObserver):
    """Renders proxy snapshots + reprojection overlay + UI screenshot via ``bpy``."""

    def capture_scene_views(
        self,
        scene: Scene,
        views: Sequence[ViewpointCamera],
        *,
        quality: RenderQuality = RenderQuality.PREVIEW,
    ) -> list[ObservationImage]:
        raise _todo("multi-viewpoint proxy capture")

    def capture_frame_overlay(self, scene: Scene, frame: int) -> ObservationImage:
        raise _todo("reprojection overlay capture")

    def capture_ui(self, scene: Scene | None = None) -> ObservationImage | None:
        raise _todo("UI screenshot capture")


__all__ = ["BlenderSceneObserver"]
