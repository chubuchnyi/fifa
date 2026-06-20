"""SceneObserver — visual + textual feedback for an LLM-in-the-loop editor (ADR-0008).

The product goal is to let an LLM *drive* the editor (via the MCP adapter) and *see* the
consequences of its edits. Seeing means three kinds of snapshot:

* ``SCENE_3D``      the resolved 3D scene rendered from several viewpoints (orbit / cardinal
                    / broadcast / top) so the agent can judge geometry it can't read from
                    numbers — limb crossings, ball height, foot-skate.
* ``FRAME_OVERLAY`` a source frame with the reprojected proposal drawn over it (does the
                    rig line up with the pixels?).
* ``UI``            a screenshot of the editor itself (what a human would see).

This port is the *driven* (outbound) side: the core/app asks for snapshots; an adapter
(Blender/splat renderer + the editor) produces them. The viewpoints themselves are pure
camera math in :mod:`pitch3d.core.agent`. Producing real pixels needs a renderer, so the
core ships only the contract + a fake; the real observer is an adapter (ADR-0001).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from ..scene.camera import CameraTrack
from ..scene.scene import Scene
from .base import Port
from .render import RenderQuality


class ObservationKind(str, Enum):
    """What a snapshot shows."""

    SCENE_3D = "scene_3d"            # resolved scene rendered from a viewpoint
    FRAME_OVERLAY = "frame_overlay"  # source frame + reprojection overlay
    RADAR = "radar"                  # top-down tactical minimap (no camera)
    UI = "ui"                        # editor screenshot


class Viewpoint(str, Enum):
    """Named camera angles for 3D feedback (see :func:`...agent.standard_viewpoints`)."""

    CURRENT = "current"      # the scene's own (broadcast) camera
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"              # tactical / radar (straight down)
    BROADCAST = "broadcast"  # elevated 3/4 TV-style
    ORBIT = "orbit"          # member of an orbit ring


@dataclass
class ViewpointCamera:
    """A named viewpoint bound to a concrete camera (produced by the agent's view math)."""

    viewpoint: Viewpoint
    camera: CameraTrack


@dataclass
class ObservationImage:
    """A single rendered snapshot the LLM can look at — a URI plus what it depicts."""

    kind: ObservationKind
    uri: str
    viewpoint: Viewpoint | None = None
    frame: int | None = None
    width: int | None = None
    height: int | None = None
    note: str | None = None


@dataclass
class Observation:
    """One feedback bundle for the agent: images + a concise textual scene summary."""

    images: list[ObservationImage] = field(default_factory=list)
    summary: str = ""
    scene_id: str | None = None
    frame: int | None = None


class SceneObserver(Port):
    """Produces visual feedback (snapshots) of a scene for the LLM agent loop."""

    @abstractmethod
    def capture_scene_views(
        self,
        scene: Scene,
        views: Sequence[ViewpointCamera],
        *,
        quality: RenderQuality = RenderQuality.PREVIEW,
    ) -> list[ObservationImage]:
        """Render the resolved scene from each viewpoint; return one image per view."""
        raise NotImplementedError

    @abstractmethod
    def capture_frame_overlay(self, scene: Scene, frame: int) -> ObservationImage:
        """Draw the reprojected proposal over the source ``frame`` and snapshot it."""
        raise NotImplementedError

    @abstractmethod
    def capture_ui(self, scene: Scene | None = None) -> ObservationImage | None:
        """Screenshot the editor UI; ``None`` when running headless."""
        raise NotImplementedError

    def capture_radar(self, scene: Scene, frame: int = 0) -> ObservationImage | None:
        """Top-down tactical radar of the resolved scene at ``frame``; ``None`` when unsupported.

        Concrete (default ``None``) so observers *opt in* by overriding — same headless contract
        as :meth:`capture_ui`. The fake observer renders a real, dependency-free minimap; the
        Blender observer delegates to it (the radar is camera-free 2D, so it needs no Blender).
        """
        return None

    def observe(
        self,
        scene: Scene,
        views: Sequence[ViewpointCamera] = (),
        *,
        frame: int | None = None,
        include_ui: bool = False,
        include_radar: bool = False,
        quality: RenderQuality = RenderQuality.PREVIEW,
        summary: str = "",
    ) -> Observation:
        """Compose a feedback bundle (3D views + optional frame overlay + optional radar/UI).

        Concrete so every adapter gets the bundling for free; ``summary`` is supplied by the
        caller (e.g. ``agent.scene_summary(scene)``) to keep this port free of agent logic.
        """
        images: list[ObservationImage] = []
        if views:
            images.extend(self.capture_scene_views(scene, views, quality=quality))
        if frame is not None:
            images.append(self.capture_frame_overlay(scene, frame))
        if include_radar:
            radar = self.capture_radar(scene, frame if frame is not None else 0)
            if radar is not None:
                images.append(radar)
        if include_ui:
            ui = self.capture_ui(scene)
            if ui is not None:
                images.append(ui)
        return Observation(
            images=images, summary=summary, scene_id=getattr(scene, "id", None), frame=frame
        )
