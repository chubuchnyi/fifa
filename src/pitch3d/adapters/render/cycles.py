"""Cycles render pass (M2-7, FR-14, ADR-0003) — a real photoreal render of the MEASURED scene.

The first :class:`RenderPass` that produces an actual *photoreal* frame rather than the numpy
splat debug viz. For each avatar :class:`RenderAssetRef` on the resolved scene it reads the
vertex-coloured PLY, places that canonical mesh by the owning subject's *resolved* per-frame root
**rigid** transform, and hands the geometry to Blender/Cycles (out of process, ADR-0003) to render
a lit, shaded PNG per frame. Because the placement reads the *resolved* pose, a ROOT_ORIENTATION or
ROOT_TRANSLATION edit re-projects into the rendered frame with no avatar rebuild (M2-4). R-6 honesty
travels into the picture exactly as in the splat pass: a vertex with ``measured=0`` is drawn in the
distinct *unmeasured* tint, never its fabricated placeholder colour.

This is the editable measured-photoreal path. Honest, deferred limits (so the render is never
oversold): the mesh is placed by its root **rigid** transform only — per-joint limb articulation
(LBS) is M2-8 — and the environment is a single neutral ground plane, not the measured grass/line
material (M2-9). All the OpenCV→Blender camera maths lives in the pure
:mod:`~pitch3d.adapters.blender.cycles_plan`; this orchestrator just resolves assets, writes the
mesh NPZs + plan, and drives the subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ...core.ports.render import RenderPass, RenderQuality, RenderResult
from ...core.scene.assets import RenderAssetKind, RenderAssetRef
from ...core.scene.camera import CameraTrack
from ...core.scene.scene import Scene
from ..blender.cycles_plan import CyclesMeshRef, build_cycles_plan, write_cycles_plan
from ..blender.runner import run_cycles_render
from ..models.avatar import read_vertex_colored_ply
from .avatar_splat import _UNMEASURED_TINT  # one R-6 tint shared across passes

_MESH_AVATAR_KINDS = frozenset({RenderAssetKind.AVATAR_TEXTURED_SMPLX})


@dataclass
class _LoadedAvatar:
    """An avatar asset resolved for Cycles: NPZ-ready geometry + the owning subject track."""

    track_id: int
    name: str
    verts: np.ndarray  # (V, 3) canonical, mesh-local
    faces: np.ndarray  # (F, 3) int
    rgb01: np.ndarray  # (V, 3) float in [0, 1], R-6 tint already applied to unmeasured verts
    n_unmeasured: int  # count of measured=0 verts (drawn in the R-6 tint) — for the manifest/note


@dataclass
class CyclesRenderPass(RenderPass):
    """Render the measured avatar meshes through Blender/Cycles to per-frame PNGs (M2-7).

    Attributes:
        out_dir: Root for the per-render frame directory.
        blender: Explicit Blender binary; ``None`` resolves via ``$PITCH3D_BLENDER`` / PATH.
        samples: Cycles samples per frame (quality/cost lever).
        device: ``"CPU"`` or ``"GPU"`` for Cycles.
        ground_z: World Z (metres) of the neutral ground plane (the measured pitch plane).
        timeout: Per-render Blender subprocess timeout (seconds) — Cycles is CPU-bound.
    """

    out_dir: Path = field(default_factory=lambda: Path("out/render"))
    blender: str | None = None
    samples: int = 48
    device: str = "CPU"
    ground_z: float = 0.0
    timeout: float = 600.0

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        scene: Scene,
        camera_path: CameraTrack,
        quality: RenderQuality = RenderQuality.PREVIEW,
    ) -> RenderResult:
        # Fast low-q preview (M2-6, UX-9): PREVIEW downscales the camera so Cycles touches ~¼ the
        # pixels; FINAL is full-res (scaled(1.0) returns the same camera, identity intact).
        camera = camera_path.scaled(quality.scale)
        k = camera.intrinsics
        width, height = int(k.width), int(k.height)
        target = self.out_dir / f"{scene.id}_{quality.value}"
        target.mkdir(parents=True, exist_ok=True)

        avatars = _load_avatars(scene)
        n_verts = sum(int(a.verts.shape[0]) for a in avatars)
        n_unmeasured = sum(a.n_unmeasured for a in avatars)
        meshes = [CyclesMeshRef(a.name, f"{a.name}.npz", a.track_id) for a in avatars]
        plan = build_cycles_plan(
            scene,
            camera,
            meshes=meshes,
            samples=self.samples,
            device=self.device,
            ground_z=self.ground_z,
        )

        with TemporaryDirectory(prefix="pitch3d-cycles-") as tmp:
            mesh_dir = Path(tmp) / "meshes"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            for avatar in avatars:
                np.savez(
                    mesh_dir / f"{avatar.name}.npz",
                    verts=avatar.verts.astype(np.float64),
                    faces=avatar.faces.astype(np.int64),
                    rgb=avatar.rgb01.astype(np.float64),
                )
            plan_path = write_cycles_plan(plan, Path(tmp) / "plan.json")
            run_cycles_render(
                plan_path,
                mesh_dir=mesh_dir,
                render_dir=target,
                n_frames=camera.n_frames,
                blender=self.blender,
                timeout=self.timeout,
            )

        (target / "manifest.txt").write_text(
            f"scene={scene.id} renderer=cycles avatars={len(avatars)} vertices={n_verts} "
            f"unmeasured={n_unmeasured} samples={self.samples} device={self.device} "
            f"frames={camera.n_frames} size={width}x{height} quality={quality.value}\n",
            encoding="utf-8",
        )
        return RenderResult(
            uri=str(target),
            n_frames=camera.n_frames,
            quality=quality,
            is_video=False,
            camera=camera,
            note=f"cycles {width}x{height} ({quality.value}, {self.samples}spp/{self.device}): "
            f"{len(avatars)} avatar(s), {n_unmeasured}/{n_verts} verts unmeasured (R-6 tinted); "
            f"rigid-root placement only (LBS=M2-8), neutral ground (material=M2-9)",
        )


def _load_avatars(scene: Scene) -> list[_LoadedAvatar]:
    """Read each mesh-avatar PLY; bake the R-6 unmeasured tint into a [0, 1] colour for Cycles."""
    avatars: list[_LoadedAvatar] = []
    for ref in scene.render_assets:
        if not _is_mesh_avatar(ref):
            continue
        verts, faces, rgb, measured = read_vertex_colored_ply(Path(ref.uri))
        tinted = np.where(measured[:, None], rgb, np.array(_UNMEASURED_TINT, dtype=np.uint8))
        avatars.append(
            _LoadedAvatar(
                track_id=int(ref.subject_track_id),
                name=f"avatar_{int(ref.subject_track_id)}",
                verts=verts,
                faces=faces,
                rgb01=tinted.astype(np.float64) / 255.0,
                n_unmeasured=int((~measured).sum()),
            )
        )
    return avatars


def _is_mesh_avatar(ref: RenderAssetRef) -> bool:
    return (
        ref.kind in _MESH_AVATAR_KINDS
        and ref.subject_track_id is not None
        and ref.uri.endswith(".ply")
        and Path(ref.uri).exists()
    )


__all__ = ["CyclesRenderPass"]
