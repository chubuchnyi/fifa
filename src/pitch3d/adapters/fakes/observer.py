"""FakeSceneObserver — real (tiny) PNG snapshots with no renderer dependency.

Implements :class:`~pitch3d.core.ports.observation.SceneObserver` by writing solid-color
PNGs (one per viewpoint / overlay / UI) using only the stdlib. That keeps the LLM
visual-feedback loop end-to-end testable and lets the dry-run produce inspectable artifacts,
while the photoreal observer (Blender / splat renderer) remains an adapter swap (ADR-0008).
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pitch3d.core.ports.observation import (
    ObservationImage,
    ObservationKind,
    SceneObserver,
    Viewpoint,
    ViewpointCamera,
)
from pitch3d.core.ports.render import RenderQuality
from pitch3d.core.scene.scene import Scene

_VIEW_COLOR: dict[Viewpoint, tuple[int, int, int]] = {
    Viewpoint.CURRENT: (90, 90, 90),
    Viewpoint.FRONT: (200, 80, 80),
    Viewpoint.BACK: (80, 200, 80),
    Viewpoint.LEFT: (80, 80, 200),
    Viewpoint.RIGHT: (200, 200, 80),
    Viewpoint.TOP: (80, 200, 200),
    Viewpoint.BROADCAST: (200, 120, 60),
    Viewpoint.ORBIT: (160, 60, 200),
}


def _png_bytes(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Encode a solid-color RGB PNG (stdlib only — no PIL)."""

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    row = b"\x00" + bytes(rgb) * width  # filter byte 0 + pixels
    idat = zlib.compress(row * height, 9)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


@dataclass
class FakeSceneObserver(SceneObserver):
    """Writes placeholder PNG snapshots under ``out_dir`` (deterministic, dependency-free)."""

    out_dir: Path = field(default_factory=lambda: Path("out/observations"))
    width: int = 64
    height: int = 36
    radar_width: int = 96
    radar_height: int = 64

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _write(
        self, name: str, rgb: tuple[int, int, int], kind: ObservationKind, **meta
    ) -> ObservationImage:
        path = self.out_dir / name
        path.write_bytes(_png_bytes(self.width, self.height, rgb))
        return ObservationImage(
            kind=kind, uri=str(path), width=self.width, height=self.height, **meta
        )

    def capture_scene_views(
        self,
        scene: Scene,
        views: Sequence[ViewpointCamera],
        *,
        quality: RenderQuality = RenderQuality.PREVIEW,
    ) -> list[ObservationImage]:
        out = []
        for i, v in enumerate(views):
            color = _VIEW_COLOR.get(v.viewpoint, (128, 128, 128))
            out.append(
                self._write(
                    f"{scene.id}_view{i:02d}_{v.viewpoint.value}.png",
                    color,
                    ObservationKind.SCENE_3D,
                    viewpoint=v.viewpoint,
                    note=f"fake {quality.value} render",
                )
            )
        return out

    def capture_frame_overlay(self, scene: Scene, frame: int) -> ObservationImage:
        return self._write(
            f"{scene.id}_overlay_{frame:05d}.png", (40, 80, 120),
            ObservationKind.FRAME_OVERLAY, frame=frame,
        )

    def capture_ui(self, scene: Scene | None = None) -> ObservationImage | None:
        sid = scene.id if scene is not None else "none"
        return self._write(f"{sid}_ui.png", (30, 30, 30), ObservationKind.UI)

    def capture_radar(self, scene: Scene, frame: int = 0) -> ObservationImage | None:
        """Real top-down minimap (pure numpy + stdlib PNG) — no renderer dependency."""
        from pitch3d.adapters.render.radar import render_radar

        path = self.out_dir / f"{scene.id}_radar_{frame:05d}.png"
        path.write_bytes(
            render_radar(scene, frame, width=self.radar_width, height=self.radar_height)
        )
        return ObservationImage(
            kind=ObservationKind.RADAR, uri=str(path), viewpoint=Viewpoint.TOP,
            frame=frame, width=self.radar_width, height=self.radar_height, note="tactical radar",
        )
