"""Splat-avatar render pass (M2-3/M2-4, FR-14) — rasterise the MEASURED avatar meshes as splats.

The first :class:`RenderPass` that consumes the M2-2 photoreal assets. For each avatar
:class:`RenderAssetRef` on the resolved scene it reads the vertex-coloured PLY, places that
canonical mesh by the owning subject's *resolved* per-frame root **rigid transform** (orientation
⊕ translation), projects every vertex through the camera, and paints **z-buffered colour splats**
— at a shared pixel the nearer splat wins. Because the placement reads the *resolved* pose, an
edit (ROOT_ORIENTATION or ROOT_TRANSLATION correction) re-projects into the rendered frame with no
avatar rebuild — the M2-4 edit↔render sync (AC-5a). R-6 honesty travels into the picture: a vertex
with ``measured=0`` is drawn in a distinct *unmeasured* tint, not its fabricated placeholder
colour, so the operator/LLM can *see* which body regions were never observed.

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

# Kinds whose payload is a vertex-coloured PLY this pass can splat (others, e.g. env, are skipped).
_MESH_AVATAR_KINDS = frozenset({RenderAssetKind.AVATAR_TEXTURED_SMPLX})


@dataclass
class _Avatar:
    """A loaded avatar mesh ready to be posed per frame: canonical verts + colours + R-6 flags."""

    track_id: int
    canonical: np.ndarray  # (V, 3) rest-pose vertices, mesh-local
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
        k = camera_path.intrinsics
        width, height = int(k.width), int(k.height)
        target = self.out_dir / f"{scene.id}_{quality.value}"
        target.mkdir(parents=True, exist_ok=True)

        avatars = _load_avatars(scene)          # read each PLY once, reuse across all frames
        roots = _resolved_roots(scene)
        n_unmeasured = sum(int((~a.measured).sum()) for a in avatars)
        n_verts = sum(int(a.canonical.shape[0]) for a in avatars)
        for i, frame in enumerate(camera_path.frames.tolist()):
            fb = _compose_frame(avatars, roots, camera_path, int(frame), self.splat_radius)
            (target / f"frame_{i:05d}.png").write_bytes(_encode_png(fb))

        (target / "manifest.txt").write_text(
            f"scene={scene.id} avatars={len(avatars)} vertices={n_verts} "
            f"unmeasured={n_unmeasured} frames={camera_path.n_frames} "
            f"size={width}x{height} quality={quality.value}\n",
            encoding="utf-8",
        )
        return RenderResult(
            uri=str(target),
            n_frames=camera_path.n_frames,
            quality=quality,
            is_video=False,
            camera=camera_path,
            note=f"avatar splat {width}x{height}: {len(avatars)} avatar(s), "
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
        _load_avatars(scene), _resolved_roots(scene), camera, int(frame), splat_radius
    )


def _compose_frame(
    avatars: list[_Avatar],
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
    for avatar in avatars:
        pose = roots.get(avatar.track_id)
        if pose is None:
            continue
        hit = np.nonzero(pose.frames == frame)[0]
        if not hit.size:
            continue  # subject absent this frame — nothing to place (no fabrication)
        world = _place_mesh(avatar.canonical, pose, int(hit[0]))
        _splat_avatar(fb, zbuf, camera, frame, world, avatar, radius)
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


def _load_avatars(scene: Scene) -> list[_Avatar]:
    """Read every mesh-avatar asset on the scene whose PLY exists (skip markers / env / missing)."""
    avatars: list[_Avatar] = []
    for ref in scene.render_assets:
        if not _is_mesh_avatar(ref):
            continue
        verts, _faces, rgb, measured = read_vertex_colored_ply(Path(ref.uri))
        avatars.append(_Avatar(int(ref.subject_track_id), verts, rgb, measured))
    return avatars


def _is_mesh_avatar(ref: RenderAssetRef) -> bool:
    return (
        ref.kind in _MESH_AVATAR_KINDS
        and ref.subject_track_id is not None
        and ref.uri.endswith(".ply")
        and Path(ref.uri).exists()
    )


def _splat_avatar(
    fb: np.ndarray,
    zbuf: np.ndarray,
    camera: CameraTrack,
    frame: int,
    world: np.ndarray,
    avatar: _Avatar,
    radius: int,
) -> None:
    """Project ``world`` verts; paint each visible one as a z-buffered splat (R-6 tint applied)."""
    uv, depth, visible = project_world_points_with_depth(camera, frame, world)
    for idx in np.nonzero(visible)[0]:
        color = tuple(int(c) for c in avatar.rgb[idx]) if avatar.measured[idx] else _UNMEASURED_TINT
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
