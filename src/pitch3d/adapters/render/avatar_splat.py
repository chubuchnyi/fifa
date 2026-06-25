"""Splat render pass (M2-3/M2-4/M2-1, FR-14/FR-11) — rasterise the MEASURED meshes as splats.

The first :class:`RenderPass` that consumes the M2-2 photoreal assets. For each avatar
:class:`RenderAssetRef` on the resolved scene it reads the vertex-coloured PLY, places that
canonical mesh by the owning subject's *resolved* per-frame root **rigid transform** (orientation
⊕ translation), projects every vertex through the camera, and paints **z-buffered colour splats**
— at a shared pixel the nearer splat wins. Because the placement reads the *resolved* pose, an
edit (ROOT_ORIENTATION or ROOT_TRANSLATION correction) re-projects into the rendered frame with no
avatar rebuild — the M2-4 edit↔render sync (AC-5a). The same pass also splats world-space
**environment** meshes (M2-1, e.g. the measured pitch markings): those are already in world meters,
so they place by identity at every frame and ground the avatars under one shared z-buffer. R-6
honesty travels into the picture: a vertex with ``measured=0`` is drawn in a distinct *unmeasured*
tint, not its fabricated placeholder colour, so the operator/LLM can *see* what was never observed.

Pure numpy + the stdlib PNG encoder (shared with :mod:`~pitch3d.adapters.render.overlay`); no
Blender, no GPU. The pass reads only resolved state and never mutates the scene (RenderPass
contract). Honest limitations, deferred to heavier upgrades: it splats *vertices* (no triangle
fill) and places each mesh by its root *rigid* transform only — per-joint limb articulation
(POSE_BODY_JOINT) and shape (SHAPE_BETA) still need the SMPL-X LBS model and are not faked here.
Even so it wires resolved-scene → avatar-asset → rendered-frame end-to-end with edit↔render sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...core.correction.engine import resolve_subject_motion
from ...core.correction.rotations import axis_angle_to_matrix
from ...core.ports.render import RenderPass, RenderQuality, RenderResult
from ...core.scene.assets import RenderAssetKind, RenderAssetRef
from ...core.scene.camera import CameraTrack
from ...core.scene.motion import PoseSequence
from ...core.scene.projection import project_world_points_with_depth
from ...core.scene.scene import Scene
from ..models.avatar import read_vertex_colored_ply
from .overlay import _BACKGROUND, _encode_png

# Body regions never measured front-facing (R-6) render in this muted blue-grey, NOT their stored
# placeholder colour — the picture admits "no appearance data here" rather than faking one.
_UNMEASURED_TINT = (90, 90, 110)

# Per-subject avatar meshes — posed per frame by the resolved root rigid transform.
_MESH_AVATAR_KINDS = frozenset({RenderAssetKind.AVATAR_TEXTURED_SMPLX})
# World-space environment meshes — already in world coords, placed by identity at every frame.
_MESH_ENV_KINDS = frozenset({RenderAssetKind.ENV_PITCH_MESH})


@dataclass
class _Mesh:
    """A loaded vertex-coloured mesh: canonical verts + colours + R-6 flags.

    Serves both per-subject avatars (``track_id`` set, posed by the resolved root) and world-space
    environment meshes (``track_id is None``, already in world coords, drawn at every frame).
    """

    track_id: int | None
    canonical: np.ndarray  # (V, 3) vertices — mesh-local for avatars, world for env meshes
    rgb: np.ndarray        # (V, 3) uint8 measured colour (placeholder where unmeasured)
    measured: np.ndarray   # (V,) bool — False verts are drawn in the unmeasured tint


@dataclass
class SplatAvatarRenderPass(RenderPass):
    """Splat the measured avatar meshes onto per-frame PNGs (FR-14) — pure numpy + stdlib.

    Attributes:
        out_dir: Root for the per-render frame directory.
        splat_radius: Half-size (px) of the square painted at each projected vertex.
    """

    out_dir: Path = field(default_factory=lambda: Path("out/render"))
    splat_radius: int = 2

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        scene: Scene,
        camera_path: CameraTrack,
        quality: RenderQuality = RenderQuality.PREVIEW,
    ) -> RenderResult:
        # Fast low-q preview (M2-6, UX-9): PREVIEW downscales the camera so every frame rasterises
        # ~¼ the pixels; FINAL is full-res (scaled(1.0) returns the same camera, identity intact).
        camera = camera_path.scaled(quality.scale)
        k = camera.intrinsics
        width, height = int(k.width), int(k.height)
        target = self.out_dir / f"{scene.id}_{quality.value}"
        target.mkdir(parents=True, exist_ok=True)

        avatars = _load_avatars(scene)          # read each PLY once, reuse across all frames
        env_meshes = _load_env_meshes(scene)    # world-space env (pitch); placed by identity
        roots = _resolved_roots(scene)
        n_unmeasured = sum(int((~a.measured).sum()) for a in avatars)
        n_verts = sum(int(a.canonical.shape[0]) for a in avatars)
        n_env_verts = sum(int(m.canonical.shape[0]) for m in env_meshes)
        for i, frame in enumerate(camera.frames.tolist()):
            fb = _compose_frame(
                avatars, env_meshes, roots, camera, int(frame), self.splat_radius
            )
            (target / f"frame_{i:05d}.png").write_bytes(_encode_png(fb))

        (target / "manifest.txt").write_text(
            f"scene={scene.id} avatars={len(avatars)} vertices={n_verts} "
            f"unmeasured={n_unmeasured} env_meshes={len(env_meshes)} env_vertices={n_env_verts} "
            f"frames={camera.n_frames} size={width}x{height} quality={quality.value}\n",
            encoding="utf-8",
        )
        return RenderResult(
            uri=str(target),
            n_frames=camera.n_frames,
            quality=quality,
            is_video=False,
            camera=camera,
            note=f"avatar splat {width}x{height} ({quality.value}): {len(avatars)} avatar(s), "
            f"{len(env_meshes)} env mesh(es) ({n_env_verts} verts), "
            f"{n_unmeasured}/{n_verts} verts unmeasured (R-6 tinted)",
        )


def render_frame_buffer(
    scene: Scene, camera: CameraTrack, frame: int, *, splat_radius: int = 2
) -> np.ndarray:
    """Compose the ``(H, W, 3)`` uint8 framebuffer for one frame — the splat pass's per-frame core.

    Exposed so tests/consumers can inspect pixels without the PNG round-trip. Reloads the avatar
    PLYs each call (fine for a single frame); the :meth:`SplatAvatarRenderPass.render` loop loads
    them once and reuses them across frames.
    """
    return _compose_frame(
        _load_avatars(scene),
        _load_env_meshes(scene),
        _resolved_roots(scene),
        camera,
        int(frame),
        splat_radius,
    )


def _compose_frame(
    avatars: list[_Mesh],
    env_meshes: list[_Mesh],
    roots: dict[int, PoseSequence],
    camera: CameraTrack,
    frame: int,
    radius: int,
) -> np.ndarray:
    k = camera.intrinsics
    width, height = int(k.width), int(k.height)
    fb = np.empty((height, width, 3), dtype=np.uint8)
    fb[:] = _BACKGROUND
    zbuf = np.full((height, width), np.inf)
    # World-space environment meshes (e.g. the measured pitch) — placed by identity every frame,
    # drawn first as the z-buffered backdrop the avatars stand on (M2-0: a leg can't pass it).
    for mesh in env_meshes:
        _splat_mesh(fb, zbuf, camera, frame, mesh.canonical, mesh, radius)
    for avatar in avatars:
        pose = roots.get(avatar.track_id)
        if pose is None:
            continue
        hit = np.nonzero(pose.frames == frame)[0]
        if not hit.size:
            continue  # subject absent this frame — nothing to place (no fabrication)
        world = _place_mesh(avatar.canonical, pose, int(hit[0]))
        _splat_mesh(fb, zbuf, camera, frame, world, avatar, radius)
    return fb


def _place_mesh(canonical: np.ndarray, pose: PoseSequence, row: int) -> np.ndarray:
    """Place the canonical mesh by the resolved root *rigid* transform for frame-row ``row``.

    Applies the resolved root orientation (``global_orient`` axis-angle → matrix) and then the
    root translation. Because both are read from the *resolved* pose, a ROOT_ORIENTATION **or**
    ROOT_TRANSLATION edit re-projects into the rendered frame with no avatar rebuild (M2-4,
    AC-5a). Per-joint body articulation (POSE_BODY_JOINT) and shape (SHAPE_BETA) still need the
    heavy SMPL-X LBS model — an honest deferred limit, not silently faked here.
    """
    rot = axis_angle_to_matrix(pose.global_orient[row])  # (3, 3)
    return canonical @ rot.T + pose.transl[row]


def _resolved_roots(scene: Scene) -> dict[int, PoseSequence]:
    """Per-subject resolved pose (defensive resolve: a no-op on an already-resolved scene)."""
    return {
        s.track_id: resolve_subject_motion(s.proposal, scene.corrections_for(s.track_id)).pose
        for s in scene.subjects
    }


def _load_avatars(scene: Scene) -> list[_Mesh]:
    """Read every mesh-avatar asset on the scene whose PLY exists (skip markers / env / missing)."""
    avatars: list[_Mesh] = []
    for ref in scene.render_assets:
        if not _is_mesh_avatar(ref):
            continue
        verts, _faces, rgb, measured = read_vertex_colored_ply(Path(ref.uri))
        avatars.append(_Mesh(int(ref.subject_track_id), verts, rgb, measured))
    return avatars


def _load_env_meshes(scene: Scene) -> list[_Mesh]:
    """Read every world-space environment mesh asset on the scene whose PLY exists.

    Env meshes (e.g. the M2-1 measured pitch markings) are already in world meters, so they carry
    ``track_id=None`` and are placed by identity at every frame — there is no per-subject root to
    resolve. Their ``measured`` flags ride the same R-6 tint path as the avatars.
    """
    meshes: list[_Mesh] = []
    for ref in scene.render_assets:
        if not _is_env_mesh(ref):
            continue
        verts, _faces, rgb, measured = read_vertex_colored_ply(Path(ref.uri))
        meshes.append(_Mesh(None, verts, rgb, measured))
    return meshes


def _is_mesh_avatar(ref: RenderAssetRef) -> bool:
    return (
        ref.kind in _MESH_AVATAR_KINDS
        and ref.subject_track_id is not None
        and ref.uri.endswith(".ply")
        and Path(ref.uri).exists()
    )


def _is_env_mesh(ref: RenderAssetRef) -> bool:
    return (
        ref.kind in _MESH_ENV_KINDS
        and ref.uri.endswith(".ply")
        and Path(ref.uri).exists()
    )


def _splat_mesh(
    fb: np.ndarray,
    zbuf: np.ndarray,
    camera: CameraTrack,
    frame: int,
    world: np.ndarray,
    mesh: _Mesh,
    radius: int,
) -> None:
    """Project ``world`` verts; paint each visible one as a z-buffered splat (R-6 tint applied)."""
    uv, depth, visible = project_world_points_with_depth(camera, frame, world)
    for idx in np.nonzero(visible)[0]:
        color = tuple(int(c) for c in mesh.rgb[idx]) if mesh.measured[idx] else _UNMEASURED_TINT
        _splat(fb, zbuf, float(uv[idx, 0]), float(uv[idx, 1]), float(depth[idx]), color, radius)


def _splat(
    fb: np.ndarray,
    zbuf: np.ndarray,
    x: float,
    y: float,
    depth: float,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    """Paint a filled square at ``(x, y)`` where ``depth`` beats the z-buffer; clip to bounds."""
    height, width = fb.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return
    region = zbuf[y0:y1, x0:x1]
    nearer = depth < region
    if nearer.any():
        region[nearer] = depth
        fb[y0:y1, x0:x1][nearer] = color
