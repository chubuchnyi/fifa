"""Photoreal SceneObserver — Cycles ``SCENE_3D`` snapshots for the LLM loop (M2-10, A-8).

The :class:`~pitch3d.adapters.blender.runner.BlenderSceneObserver` renders the *proxy* (Workbench
armature) for fast feedback; this observer renders the **resolved measured scene** through
Blender/Cycles instead, so the agent sees the same photoreal pixels the final render produces —
LBS-posed textured avatars on the grass pitch under the physical sky, R-6 tint intact. Each named
:class:`ViewpointCamera` (a single-frame :class:`CameraTrack` at the observation frame) is rendered
on its own, reusing the whole :class:`CyclesRenderPass` machinery (posing, env, R-6, OpenCV→Blender
camera maths); the per-view frame is copied out to a stable URI before the next view overwrites it.
The 2D overlay / radar / UI snapshots are camera-free and delegate to a headless fallback, exactly
as the proxy observer does.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ...core.ports.observation import (
    ObservationImage,
    ObservationKind,
    SceneObserver,
    Viewpoint,
    ViewpointCamera,
)
from ...core.ports.render import RenderQuality
from ...core.scene.scene import Scene
from .cycles import CyclesRenderPass


@dataclass
class CyclesSceneObserver(SceneObserver):
    """Renders photoreal ``SCENE_3D`` snapshots via Blender/Cycles for the agent feedback loop.

    ``capture_scene_views`` is the real, Cycles-backed half: each viewpoint is a single-frame
    camera the wrapped :class:`CyclesRenderPass` renders, so what the agent observes is the actual
    measured-photoreal scene from that angle — not a proxy. The 2D ``FRAME_OVERLAY`` reprojection,
    top-down ``RADAR`` and ``UI`` screenshot are camera-free, so they delegate to a headless
    ``fallback`` observer (the fake by default), mirroring the proxy observer's split.
    """

    out_dir: Path = field(default_factory=lambda: Path("out/observations"))
    render: CyclesRenderPass | None = None
    fallback: SceneObserver | None = None

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.render is None:
            self.render = CyclesRenderPass(out_dir=self.out_dir / "_frames")
        if self.fallback is None:
            from ..fakes import FakeSceneObserver

            self.fallback = FakeSceneObserver(out_dir=self.out_dir)

    def capture_scene_views(
        self,
        scene: Scene,
        views: Sequence[ViewpointCamera],
        *,
        quality: RenderQuality = RenderQuality.PREVIEW,
    ) -> list[ObservationImage]:
        if not views:
            return []
        assert self.render is not None  # set in __post_init__
        images: list[ObservationImage] = []
        for idx, view in enumerate(views):
            # Each viewpoint is its own single-frame Cycles render; the pass writes
            # frame_00000.png into a per-(scene,quality) dir, so copy it out to a stable per-view
            # URI before the next view overwrites it.
            result = self.render.render(scene, view.camera, quality)
            produced = Path(result.uri) / "frame_00000.png"
            frame = int(view.camera.frames[0]) if view.camera.frames.shape[0] else None
            dst = self.out_dir / f"{scene.id}_{idx:02d}_{view.viewpoint.value}_{quality.value}.png"
            shutil.copyfile(produced, dst)
            cam = result.camera if result.camera is not None else view.camera
            images.append(
                ObservationImage(
                    kind=ObservationKind.SCENE_3D,
                    uri=str(dst),
                    viewpoint=Viewpoint(view.viewpoint),
                    frame=frame,
                    width=int(cam.intrinsics.width),
                    height=int(cam.intrinsics.height),
                    note=f"cycles photoreal {quality.value}",
                )
            )
        return images

    @property
    def _delegate(self) -> SceneObserver:
        assert self.fallback is not None  # always set in __post_init__
        return self.fallback

    def capture_frame_overlay(self, scene: Scene, frame: int) -> ObservationImage:
        return self._delegate.capture_frame_overlay(scene, frame)

    def capture_ui(self, scene: Scene | None = None) -> ObservationImage | None:
        return self._delegate.capture_ui(scene)

    def capture_radar(self, scene: Scene, frame: int = 0) -> ObservationImage | None:
        return self._delegate.capture_radar(scene, frame)


__all__ = ["CyclesSceneObserver"]
