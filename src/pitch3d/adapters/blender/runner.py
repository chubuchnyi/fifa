"""Subprocess runner + Blender driving adapters (M1 step 9/10, A-7, ADR-0008).

Blender ships its own Python (3.13), so — unlike the pip-extra adapters — the heavy half can't
be a lazy ``import``; it runs *out of process*. :func:`run_blender` writes the pure
:class:`~pitch3d.adapters.blender.proxy.ProxyPlan` to JSON and invokes
``blender --background --factory-startup --python _script.py -- …``. The binary is located
the same way an optional extra is gated: :func:`locate_blender` checks ``$PITCH3D_BLENDER`` then
``PATH``, and a missing binary raises an actionable :class:`RuntimeError`.

Two adapters sit on top: :class:`BlenderProxyBuilder` writes the editable ``.blend`` (the step-10
editing surface — root/ball as F-curves, β + body pose as channels), and
:class:`BlenderSceneObserver` renders the proxy from canonical viewpoints for the LLM's
``SCENE_3D`` feedback (A-7). The pure plan maths is unit-tested with no Blender; these classes
are exercised when a binary is present.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
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
from .proxy import ProxyPlan, build_proxy_plan, write_plan

_SCRIPT = Path(__file__).with_name("_script.py")
_SUCCESS = "PITCH3D_BLENDER_OK"
_QUALITY_MAX_PX = {RenderQuality.PREVIEW: 480, RenderQuality.FINAL: 1280}


def locate_blender(explicit: str | None = None) -> str | None:
    """Find a Blender binary: explicit arg → ``$PITCH3D_BLENDER`` → ``PATH``; else ``None``."""
    for cand in (explicit, os.environ.get("PITCH3D_BLENDER")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return shutil.which("blender")


def _require_blender(explicit: str | None) -> str:
    found = locate_blender(explicit)
    if found is None:
        raise RuntimeError(
            "Blender not found. Set $PITCH3D_BLENDER to the blender binary "
            "(e.g. /path/to/blender-x.y-linux-x64/blender) or put 'blender' on PATH. "
            "The proxy-plan assembly works without Blender; only building the .blend / "
            "rendering proxy views needs it."
        )
    return found


def run_blender(
    plan: ProxyPlan,
    *,
    out_blend: str | Path | None = None,
    render_dir: str | Path | None = None,
    blender: str | None = None,
    timeout: float = 300.0,
) -> dict:
    """Drive Blender once to build the proxy (and optionally save a ``.blend`` / render views).

    Returns ``{"blend": path|None, "renders": [paths]}`` after verifying every promised artifact
    exists. Raises :class:`RuntimeError` if Blender is absent, exits non-zero, or omits an artifact.
    """
    binary = _require_blender(blender)
    with tempfile.TemporaryDirectory(prefix="pitch3d-blender-") as tmp:
        plan_path = write_plan(plan, Path(tmp) / "plan.json")
        cmd = [binary, "--background", "--factory-startup", "--python", str(_SCRIPT),
               "--", "--plan", str(plan_path)]
        if out_blend is not None:
            Path(out_blend).parent.mkdir(parents=True, exist_ok=True)
            cmd += ["--out-blend", str(out_blend)]
        if render_dir is not None:
            Path(render_dir).mkdir(parents=True, exist_ok=True)
            cmd += ["--render-dir", str(render_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)

    if proc.returncode != 0 or _SUCCESS not in proc.stdout:
        tail = (proc.stdout + "\n" + proc.stderr)[-1500:]
        raise RuntimeError(f"Blender build failed (rc={proc.returncode}). Output tail:\n{tail}")

    renders: list[str] = []
    if render_dir is not None:
        for i, v in enumerate(plan.views):
            path = Path(render_dir) / f"{i:02d}_{v.viewpoint}.png"
            if not path.is_file():
                raise RuntimeError(f"Blender did not produce expected render {path}")
            renders.append(str(path))
    if out_blend is not None and not Path(out_blend).is_file():
        raise RuntimeError(f"Blender did not produce expected .blend {out_blend}")
    return {"blend": str(out_blend) if out_blend is not None else None, "renders": renders}


@dataclass
class BlenderProxyBuilder:
    """Builds the editable SMPL/armature proxy ``.blend`` from a resolved scene (M1 step 10).

    The ``.blend`` is the operator's editing surface: each subject's root and the ball are animated
    with **F-curves** (location + axis-angle), and β / per-joint body pose ride along as channels,
    so a human can scrub and key them in Blender. Edits there map back to :class:`Correction`s
    (the source of truth, ADR-0002); the proxy is a representation, never the canonical data.
    """

    blender: str | None = None
    fps: float = 25.0
    include_pose: bool = True

    def build(self, scene: Scene, out_blend: str | Path) -> Path:
        """Write ``out_blend`` from the scene's resolved proxy plan; return its path."""
        plan = build_proxy_plan(scene, fps=self.fps, include_pose=self.include_pose)
        result = run_blender(plan, out_blend=out_blend, blender=self.blender)
        return Path(result["blend"])


@dataclass
class BlenderSceneObserver(SceneObserver):
    """Renders proxy ``SCENE_3D`` snapshots via Blender for the LLM feedback loop (A-7).

    ``capture_scene_views`` is the real, Blender-backed half: it builds the proxy and renders each
    requested viewpoint (Workbench, CPU — no GPU) to a PNG. The 2D reprojection ``FRAME_OVERLAY``
    (already real in :class:`~pitch3d.adapters.render.ReprojectionOverlayRenderPass`) and the
    GUI ``UI`` screenshot are delegated to a headless ``fallback`` observer until a live editor
    seam lands.
    """

    out_dir: Path = field(default_factory=lambda: Path("out/observations"))
    blender: str | None = None
    fallback: SceneObserver | None = None
    include_pose: bool = False  # a positional proxy render needs no per-joint pose channels

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
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
        max_px = _QUALITY_MAX_PX.get(quality, 480)
        plan = build_proxy_plan(
            scene, views=list(views), include_pose=self.include_pose, preview_max_px=max_px
        )
        render_dir = self.out_dir / f"{scene.id}_blender_{quality.value}"
        result = run_blender(plan, render_dir=render_dir, blender=self.blender)
        images: list[ObservationImage] = []
        for v, path in zip(plan.views, result["renders"], strict=True):
            images.append(
                ObservationImage(
                    kind=ObservationKind.SCENE_3D,
                    uri=path,
                    viewpoint=Viewpoint(v.viewpoint),
                    frame=v.frame,
                    width=v.resolution[0],
                    height=v.resolution[1],
                    note=f"blender proxy {quality.value}",
                )
            )
        return images

    def capture_frame_overlay(self, scene: Scene, frame: int) -> ObservationImage:
        return self.fallback.capture_frame_overlay(scene, frame)

    def capture_ui(self, scene: Scene | None = None) -> ObservationImage | None:
        return self.fallback.capture_ui(scene)


__all__ = [
    "BlenderProxyBuilder",
    "BlenderSceneObserver",
    "locate_blender",
    "run_blender",
]
