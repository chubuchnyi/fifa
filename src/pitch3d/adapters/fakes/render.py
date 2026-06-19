"""FakeRenderPass — a dependency-free :class:`RenderPass` over the resolved scene.

Reads only the scene + camera path (never mutates), writes a tiny manifest describing what
*would* be rendered, and returns a :class:`RenderResult` pointing at it. Enough for the
dry-run and the observation loop to exercise the render seam with no splat/avatar/Blender
backend. The real splat/avatar and ViewSynthesizer-seam-A passes are adapter swaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pitch3d.core.ports.render import RenderPass, RenderQuality, RenderResult
from pitch3d.core.scene.camera import CameraTrack
from pitch3d.core.scene.scene import Scene


@dataclass
class FakeRenderPass(RenderPass):
    """Writes a per-render manifest directory and reports it as the output uri."""

    out_dir: Path = field(default_factory=lambda: Path("out/render"))

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        scene: Scene,
        camera_path: CameraTrack,
        quality: RenderQuality = RenderQuality.PREVIEW,
    ) -> RenderResult:
        n = camera_path.n_frames
        target = self.out_dir / f"{scene.id}_{quality.value}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "manifest.txt").write_text(
            f"scene={scene.id} subjects={len(scene.subjects)} "
            f"assets={len(scene.render_assets)} frames={n} quality={quality.value}\n",
            encoding="utf-8",
        )
        return RenderResult(
            uri=str(target),
            n_frames=n,
            quality=quality,
            is_video=False,
            camera=camera_path,
            note=f"fake {quality.value} render",
        )
