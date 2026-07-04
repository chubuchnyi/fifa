"""Measured textured-SMPL-X avatar builder — the first real :class:`AvatarBuilder` (M2-2 #1).

This is M2-0's **primary, measured** realism path made concrete: instead of *generating* an
avatar (an image-to-3D / diffusion model that hallucinates unmeasured detail), it **projects the
player's real broadcast pixels onto the tracked SMPL-X mesh**. Every vertex colour is sampled
from a real frame the player was actually seen in, so the appearance is *measured*, stays in
correspondence with the rigged/animated mesh, and is honest about what it does **not** know:
vertices that were never seen front-facing (the far side, occluded parts) are left **unmeasured**
(``measured=0`` in the asset) rather than fabricated (R-6). Filling those is a *later, flagged*
generative step (M3 seam-B inpaint), never silently here.

Split like every real adapter (ADR-0001), so the model-independent logic runs with **no torch,
no GPU**:

* :class:`TexturedSmplxAvatarBuilder` — the **pure** half. Given an injected backend's per-frame
  *posed* world-space SMPL-X vertices + the camera + the decoded frame, it computes vertex
  normals, projects each vertex, keeps only front-facing + in-frustum + nearest-at-its-pixel
  (a light z-buffer) vertices, samples their colour, and averages across all reference frames
  into one measured per-vertex texture + an observation count. Writes a vertex-coloured PLY
  (geometry + colour + the per-vertex ``measured`` flag). Numpy + stdlib; unit-tested via a stub.
* :class:`SmplxTextureBackend` — the SMPL-X half. Its **geometry** is wired (M2-8a): the subject's
  ``betas`` become the canonical mesh via the pure-numpy SMPL-X forward (:mod:`.smplx_lbs`, no
  torch/GPU), and the render pass poses it per frame for the limbs. Its **texture** is wired
  (M2-8b): given the scene camera + source clip, :meth:`observe` poses the subject at its measured
  pose for a capped set of decoded broadcast frames so the pure half samples the player's real
  pixels. Without a real source it stays geometry-only (every vertex ``measured=0``, honest R-6) —
  never a fabricated look. The (non-commercial) SMPL-X model is the gated asset — its absence raises
  an actionable error.

Swap it in via ``default_ports(avatar="textured")`` (wiring) or inject a vendored backend by
dotted path with ``--avatar-backend pkg.module:Factory`` (ADR-0006) — one fake replaced at a time.
For demos/CI a dependency-free :class:`SyntheticAvatarMeshBackend` drives the real measured path
end-to-end (tiny synthetic mesh, no SMPL-X/GPU) so the dry-run exercises this code, not a marker.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ...core.ports.io import ClipRef, CropRef
from ...core.ports.reconstruction import AvatarBuilder
from ...core.scene.assets import RenderAssetKind, RenderAssetRef, SynthViewRef
from ...core.scene.camera import CameraIntrinsics, CameraTrack
from ...core.scene.projection import (
    camera_center,
    camera_pose,
    project_world_points_with_depth,
)
from ...core.scene.provenance import Backend, ModelInfo
from ...core.scene.subject import Subject
from ..io.frames import iter_clip_frames, resolve_source_path
from .smplx_lbs import SmplxModel, locate_smplx_model

#: Neutral grey written for vertices with no measured observation (the honest "unknown" colour;
#: the per-vertex ``measured`` flag is the real signal — never read this as a true appearance).
_UNMEASURED_RGB = (127, 127, 127)


@dataclass
class FrameObservation:
    """One reference frame's worth of input for texturing: posed vertices + camera + pixels.

    Attributes:
        frame: Source frame index (selects the camera row).
        vertices_world: ``(V, 3)`` SMPL-X vertices for this subject **posed into world metres**
            at ``frame`` (same vertex order as the canonical mesh — colours attach by index).
        camera: The estimated broadcast camera for this frame (world→image).
        image: ``(H, W, 3)`` uint8 decoded frame; ``H/W`` must match ``camera.intrinsics``.
    """

    frame: int
    vertices_world: np.ndarray
    camera: CameraTrack
    image: np.ndarray

    def __post_init__(self) -> None:
        self.vertices_world = np.asarray(self.vertices_world, dtype=float).reshape(-1, 3)
        self.image = np.asarray(self.image)


@dataclass
class AvatarMeshObservations:
    """Everything the pure texturer needs for one subject: canonical mesh + per-frame views.

    Attributes:
        canonical_vertices: ``(V, 3)`` rest-pose vertices written to the asset (geometry).
        faces: ``(F, 3)`` int triangle topology (SMPL-X is fixed; shared by every frame).
        frames: The reference :class:`FrameObservation`\\ s to sample appearance from (may be
            empty — then the avatar is geometry-only, every vertex ``measured=0``, R-6).
    """

    canonical_vertices: np.ndarray
    faces: np.ndarray
    frames: list[FrameObservation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.canonical_vertices = np.asarray(self.canonical_vertices, dtype=float).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=int).reshape(-1, 3)


@runtime_checkable
class AvatarMeshBackend(Protocol):
    """The heavy half: resolve a subject into a canonical mesh + per-frame posed views.

    Behind this protocol so :class:`TexturedSmplxAvatarBuilder`'s projection/sampling/aggregation
    can be tested with a stub returning a tiny synthetic mesh + frame — no SMPL-X model, no GPU.
    """

    def observe(
        self,
        subject: Subject,
        ref_crops: Sequence[CropRef],
        *,
        camera: CameraTrack | None = None,
        clip: ClipRef | None = None,
    ) -> AvatarMeshObservations:
        """Return the canonical mesh + the per-frame posed vertices/camera/pixels to sample.

        ``camera`` + ``clip`` (M2-8b) let the backend sample the subject's real broadcast pixels;
        without them it returns geometry only (``frames=[]``, every vertex ``measured=0``, R-6).
        """
        ...


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals ``(V, 3)`` from triangle ``faces`` (unit length).

    Each face's (un-normalised) cross product is accumulated to its three vertices, so larger
    triangles weigh more, then each vertex normal is normalised. Degenerate (zero-length) normals
    are left as zero — :func:`sample_vertex_colors` treats a zero normal as not-front-facing.
    """
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=int).reshape(-1, 3)
    face_n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for i in range(3):
        np.add.at(vn, f[:, i], face_n)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    return np.divide(vn, norm, out=np.zeros_like(vn), where=norm > 1e-12)


KIT_SKIN, KIT_SHIRT, KIT_SHORTS, KIT_SOCKS, KIT_BOOTS = 0, 1, 2, 3, 4


def smplx_kit_zones(
    weights: np.ndarray, j_regressor: np.ndarray, v_template: np.ndarray
) -> np.ndarray:
    """Football-kit zone per SMPL-X vertex: 0 skin, 1 shirt, 2 shorts, 3 socks, 4 boots.

    The measured per-vertex texture is too dark/noisy at broadcast distance to carry the kit
    LAYOUT (2026-07-04 batch #1: whole bodies rendered as team-colour morphsuits and the
    generative tail kept them that way), so the layout comes from the body model itself: each
    vertex's dominant LBS joint says which garment covers it. Spine/collar/neck/shoulder → shirt
    (a short sleeve covers the deltoid), pelvis/hips → shorts (upper thigh), ankle → sock,
    foot → boot, everything else (head, face, hands, forearms) → skin. Knee-dominated vertices
    straddle the joint — the thigh side is the bare-skin gap below the shorts, the calf side is
    inside the sock — so they split by template height against the knee joint (SMPL-X is Y-up).
    Args are the raw model buffers ``(V, J)``, ``(J, V)``, ``(V, 3)`` so both the pure-numpy
    :class:`~pitch3d.adapters.models.smplx_lbs.SmplxModel` and the torch ``smplx`` model feed it.
    """
    weights = np.asarray(weights, dtype=float)
    arg = weights.argmax(axis=1)
    joints = np.asarray(j_regressor, dtype=float) @ np.asarray(v_template, dtype=float)
    zones = np.full(arg.shape[0], KIT_SKIN, dtype=np.uint8)
    for j in (3, 6, 9, 12, 13, 14, 16, 17):  # spines, neck, collars, shoulders
        zones[arg == j] = KIT_SHIRT
    for j in (0, 1, 2):  # pelvis + hips
        zones[arg == j] = KIT_SHORTS
    for j in (7, 8):  # ankles
        zones[arg == j] = KIT_SOCKS
    for j in (10, 11):  # feet
        zones[arg == j] = KIT_BOOTS
    height = np.asarray(v_template, dtype=float)[:, 1]
    for j in (4, 5):  # knees
        zones[(arg == j) & (height < joints[j, 1])] = KIT_SOCKS
    # Hip-dominated vertices run all the way down the thigh; shorts stop at ~55% hip→knee.
    for hip, knee in ((1, 4), (2, 5)):
        cut = joints[hip, 1] + 0.55 * (joints[knee, 1] - joints[hip, 1])
        zones[(arg == hip) & (height < cut)] = KIT_SKIN
    return zones


def sample_vertex_colors(
    vertices_world: np.ndarray,
    normals_world: np.ndarray,
    camera: CameraTrack,
    frame_index: int,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one frame's measured colour per vertex; flag which vertices were actually seen.

    A vertex contributes a colour only if it is **front-facing** (its normal points toward the
    optical centre), **in-frustum** (projects in front of the camera and onto the image), and the
    **nearest** vertex landing on its pixel (a light per-pixel z-buffer, so a back-of-body vertex
    that bleeds through to the same pixel as a front one is rejected). Returns
    ``(colors (V, 3) uint8, observed (V,) bool)`` — ``colors`` is 0 where ``observed`` is False.
    """
    verts = np.asarray(vertices_world, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals_world, dtype=float).reshape(-1, 3)
    img = np.asarray(image)
    n = verts.shape[0]
    colors = np.zeros((n, 3), dtype=np.uint8)
    observed = np.zeros(n, dtype=bool)
    if n == 0:
        return colors, observed

    uv, depth, visible = project_world_points_with_depth(camera, frame_index, verts)
    view = camera_center(camera, frame_index)[None, :] - verts
    facing = np.einsum("ij,ij->i", normals, view) > 0.0
    candidate = visible & facing
    if not candidate.any():
        return colors, observed

    px = np.round(uv).astype(int)
    height, width = img.shape[:2]
    nearest: dict[tuple[int, int], tuple[float, int]] = {}
    for vid in np.nonzero(candidate)[0]:
        x, y = int(px[vid, 0]), int(px[vid, 1])
        if not (0 <= x < width and 0 <= y < height):
            continue
        prev = nearest.get((x, y))
        if prev is None or depth[vid] < prev[0]:
            nearest[(x, y)] = (float(depth[vid]), int(vid))
    for (x, y), (_d, vid) in nearest.items():
        colors[vid] = img[y, x][:3]
        observed[vid] = True
    return colors, observed


def aggregate_observations(
    colors_per_frame: Sequence[np.ndarray],
    observed_per_frame: Sequence[np.ndarray],
    n_vertices: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average each vertex's colour over the frames that saw it; count the observations.

    Returns ``(rgb (V, 3) uint8, count (V,) int)``. A vertex seen in no frame keeps ``count=0``
    and an ``_UNMEASURED_RGB`` placeholder — the count, not the colour, is the honest signal (R-6).
    """
    acc = np.zeros((n_vertices, 3), dtype=float)
    count = np.zeros(n_vertices, dtype=int)
    for colors, observed in zip(colors_per_frame, observed_per_frame, strict=True):
        obs = np.asarray(observed, dtype=bool)
        acc[obs] += np.asarray(colors, dtype=float)[obs]
        count[obs] += 1
    measured = count > 0
    rgb = np.zeros((n_vertices, 3), dtype=np.uint8)
    rgb[~measured] = _UNMEASURED_RGB
    rgb[measured] = np.round(acc[measured] / count[measured, None]).astype(np.uint8)
    return rgb, count


def measured_vertex_texture(obs: AvatarMeshObservations) -> tuple[np.ndarray, np.ndarray]:
    """Project + sample every reference frame into one measured ``(rgb (V,3), count (V,))``.

    The measured-appearance core shared by the textured-mesh and Gaussian avatar builders:
    per-frame front-facing/z-buffer visibility (:func:`sample_vertex_colors`) averaged across frames
    (:func:`aggregate_observations`). A vertex seen in no frame keeps ``count=0`` — the honest R-6
    'unmeasured' signal, never fabricated.
    """
    n = obs.canonical_vertices.shape[0]
    colors_per_frame: list[np.ndarray] = []
    observed_per_frame: list[np.ndarray] = []
    for fo in obs.frames:
        normals = vertex_normals(fo.vertices_world, obs.faces)
        colors, observed = sample_vertex_colors(
            fo.vertices_world, normals, fo.camera, fo.frame, fo.image
        )
        colors_per_frame.append(colors)
        observed_per_frame.append(observed)
    return aggregate_observations(colors_per_frame, observed_per_frame, n)


def bake_body_vertex_texture(
    verts_per_frame: np.ndarray,
    faces: np.ndarray,
    camera: CameraTrack,
    frame_indices: Sequence[int],
    images: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Average a measured per-vertex body colour over (posed verts, frame, image) triples.

    The image-level core of :func:`measured_texture_from_clip` (no video decode, so it unit-tests
    with a synthetic mesh + frames). ``verts_per_frame (R, V, 3)`` are the subject's world vertices
    posed at each reference frame, ``frame_indices (R,)`` the matching camera rows, ``images`` the
    ``R`` decoded ``(H, W, 3)`` uint8 RGB frames already in the camera's pixel convention. Returns
    ``(vcolor (V, 3) float in [0,1], measured (V,) bool)``; ``vcolor`` is 0 where ``measured`` is
    False (the vertex was never seen front-facing) so the caller fills it with the kit colour (R-6).
    """
    verts_per_frame = np.asarray(verts_per_frame, dtype=float)
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    canonical = verts_per_frame[0] if verts_per_frame.shape[0] else np.zeros((0, 3))
    obs = AvatarMeshObservations(
        canonical_vertices=canonical,
        faces=faces,
        frames=[
            FrameObservation(frame=int(fi), vertices_world=vw, camera=camera, image=img)
            for fi, vw, img in zip(frame_indices, verts_per_frame, images, strict=True)
        ],
    )
    rgb, count = measured_vertex_texture(obs)
    measured = count > 0
    vcolor = rgb.astype(np.float32) / 255.0
    vcolor[~measured] = 0.0
    return vcolor, measured


def measured_texture_from_clip(
    verts_per_frame: np.ndarray,
    faces: np.ndarray,
    camera: CameraTrack,
    frames: Sequence[int],
    video_uri: str,
    *,
    max_frames: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Bake a measured per-vertex body texture by sampling the source clip at the subject's poses.

    Picks up to ``max_frames`` evenly-spread reference frames from ``frames`` (the per-row source
    indices of ``verts_per_frame (T, V, 3)``), decodes each — resized to the calibration resolution
    and rotated 180 deg into the solved camera's rolled pixel convention (the same roll the stadium
    bake uses, see :mod:`pitch3d.adapters.render.stadium_backdrop`) — and reuses the avatar samplers
    (front-facing / z-buffer / cross-frame average) to colour each vertex from real pixels. Returns
    ``(vcolor (V, 3) float in [0,1], measured (V,) bool)``; unseen vertices come back
    ``measured=False`` with ``vcolor=0`` — the caller fills those with the kit colour (R-6).
    """
    import cv2

    verts_per_frame = np.asarray(verts_per_frame, dtype=float)
    frame_arr = np.asarray(frames, dtype=int)
    k = camera.intrinsics
    w, h = int(k.width), int(k.height)
    v = int(verts_per_frame.shape[1]) if verts_per_frame.ndim == 3 else 0
    rows = _even_subset(np.arange(verts_per_frame.shape[0]), max_frames)
    if rows.size == 0:
        return np.zeros((v, 3), np.float32), np.zeros(v, bool)

    rot0, _ = camera_pose(camera, int(frame_arr[rows[0]]))
    upside_down = float(-rot0[1, 2]) < 0.0
    want = {int(frame_arr[r]): int(r) for r in rows}

    sel_verts: list[np.ndarray] = []
    sel_frames: list[int] = []
    sel_images: list[np.ndarray] = []
    for idx, bgr in iter_clip_frames(video_uri, sorted(want.keys())):
        if bgr.shape[1] != w or bgr.shape[0] != h:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        if upside_down:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        sel_images.append(np.ascontiguousarray(bgr[:, :, ::-1]))  # BGR -> RGB uint8
        sel_frames.append(int(idx))
        sel_verts.append(verts_per_frame[want[int(idx)]])
    if not sel_images:
        return np.zeros((v, 3), np.float32), np.zeros(v, bool)
    return bake_body_vertex_texture(np.asarray(sel_verts), faces, camera, sel_frames, sel_images)


def write_vertex_colored_ply(
    path: Path, vertices: np.ndarray, faces: np.ndarray, rgb: np.ndarray, measured: np.ndarray
) -> str:
    """Write an ASCII PLY: geometry + per-vertex RGB + a ``measured`` flag (0/1). Returns the path.

    The ``measured`` property is the R-6 honesty channel carried *in the asset*: a downstream
    renderer can grey-out / hatch / request inpaint for ``measured=0`` vertices instead of
    trusting a fabricated colour. Pure stdlib text (no trimesh/PIL).
    """
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=int).reshape(-1, 3)
    c = np.asarray(rgb, dtype=int).reshape(-1, 3)
    m = np.asarray(measured).reshape(-1).astype(int)
    lines = [
        "ply", "format ascii 1.0", f"element vertex {v.shape[0]}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "property uchar measured",
        f"element face {f.shape[0]}", "property list uchar int vertex_indices", "end_header",
    ]
    for i in range(v.shape[0]):
        lines.append(
            f"{v[i, 0]:.6f} {v[i, 1]:.6f} {v[i, 2]:.6f} "
            f"{c[i, 0]} {c[i, 1]} {c[i, 2]} {m[i]}"
        )
    for tri in f:
        lines.append(f"3 {tri[0]} {tri[1]} {tri[2]}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def read_vertex_colored_ply(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`write_vertex_colored_ply` — parse our ASCII PLY back into arrays.

    Returns ``(vertices (V, 3) float, faces (F, 3) int, rgb (V, 3) uint8, measured (V,) bool)`` so a
    RenderPass can consume the measured avatar asset — including the R-6 ``measured`` flag — without
    trimesh/PIL. Assumes our own fixed header (``x y z red green blue measured``).
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    nv = next(int(ln.split()[-1]) for ln in lines if ln.startswith("element vertex"))
    nf = next(int(ln.split()[-1]) for ln in lines if ln.startswith("element face"))
    body = lines[lines.index("end_header") + 1:]
    verts = np.empty((nv, 3), dtype=float)
    rgb = np.empty((nv, 3), dtype=np.uint8)
    measured = np.empty(nv, dtype=bool)
    for i in range(nv):
        p = body[i].split()
        verts[i] = (float(p[0]), float(p[1]), float(p[2]))
        rgb[i] = (int(p[3]), int(p[4]), int(p[5]))
        measured[i] = bool(int(p[6]))
    faces = np.empty((nf, 3), dtype=int)
    for j in range(nf):
        p = body[nv + j].split()
        faces[j] = (int(p[1]), int(p[2]), int(p[3]))
    return verts, faces, rgb, measured


@dataclass
class TexturedSmplxAvatarBuilder(AvatarBuilder):
    """Measured textured-SMPL-X avatar (M2-2 #1) — pure projection/sampling over a backend.

    Attributes:
        backend: Resolves a subject into a canonical mesh + per-frame posed views. If ``None`` a
            real :class:`SmplxTextureBackend` is built lazily on first use (needs the ``avatar``
            extra + the SMPL-X model + scene context).
        out_dir: Where the per-subject ``.ply`` assets are written.
        device: Inference device for the default backend (provenance + forwarded).
    """

    backend: AvatarMeshBackend | None = None
    out_dir: Path = field(default_factory=lambda: Path("out/assets"))
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="TexturedSmplxAvatar",
            backend=Backend.LOCAL,
            license="measured (broadcast pixels) on non-commercial SMPL-X topology",
            params={"device": self.device},
        )

    def build(
        self,
        subject: Subject,
        ref_crops: Sequence[CropRef],
        synth_views: Sequence[SynthViewRef] | None = None,
        *,
        camera: CameraTrack | None = None,
        clip: ClipRef | None = None,
    ) -> RenderAssetRef:
        obs = self._backend().observe(subject, ref_crops, camera=camera, clip=clip)
        n = obs.canonical_vertices.shape[0]
        rgb, count = measured_vertex_texture(obs)
        measured = count > 0
        uri = write_vertex_colored_ply(
            self.out_dir / f"avatar_{subject.track_id}.ply",
            obs.canonical_vertices, obs.faces, rgb, measured,
        )
        coverage = float(measured.mean()) if n else 0.0
        return RenderAssetRef(
            id=f"avatar-{subject.track_id}",
            kind=RenderAssetKind.AVATAR_TEXTURED_SMPLX,
            uri=uri,
            model=self.info(),
            subject_track_id=subject.track_id,
            extra={
                "coverage": coverage,                 # measured fraction of the body surface (R-6)
                "n_vertices": int(n),
                "n_measured": int(measured.sum()),
                "frames_used": len(obs.frames),
                "synth_views": 0 if synth_views is None else len(synth_views),
            },
        )

    def _backend(self) -> AvatarMeshBackend:
        return self.backend or SmplxTextureBackend(device=self.device)


def _even_subset(values: np.ndarray, k: int) -> np.ndarray:
    """Up to ``k`` evenly-spaced elements of sorted ``values`` (all of them if ``len <= k``).

    Spreading the reference frames across the track gives the texturer more *distinct* viewing
    angles per decode than a contiguous burst would — better measured coverage for the same cost.
    """
    if k <= 0:
        return values[:0]
    if values.size <= k:
        return values
    idx = np.unique(np.linspace(0, values.size - 1, k).round().astype(int))
    return values[idx]


@dataclass
class SmplxTextureBackend:
    """Real SMPL-X meshing (geometry, M2-8a) + measured pixel texturing (M2-8b).

    :meth:`observe` shapes the subject's ``betas`` into the canonical mesh with the pure-numpy
    SMPL-X forward (no torch/GPU). When the scene ``camera`` + source ``clip`` are supplied it also
    poses the subject at its **measured** (proposal) pose for a capped, evenly-spread set of
    reference frames, decodes those frames, and returns them so the pure texturer projects + samples
    the player's real pixels (M2-8b). Without a real source (no camera/clip, or a synthetic URI with
    no pixels) it returns geometry-only — every vertex ``measured=0`` (honest R-6, never faked). The
    model ``.npz`` is loaded lazily, located like every gated asset.

    Attributes:
        device: Inference device (provenance only; the forward is numpy).
        max_ref_frames: Cap on reference frames decoded per subject (cost vs. coverage lever).
    """

    device: str = "cuda"
    max_ref_frames: int = 8
    _model: SmplxModel | None = None

    def observe(
        self,
        subject: Subject,
        ref_crops: Sequence[CropRef],
        *,
        camera: CameraTrack | None = None,
        clip: ClipRef | None = None,
    ) -> AvatarMeshObservations:
        """Canonical SMPL-X mesh + measured reference frames for the subject (M2-8a + M2-8b).

        The subject's ``betas`` give the rest/canonical mesh (the render pass re-poses it per frame
        for display). When ``camera`` + ``clip`` resolve to real pixels, the subject's measured pose
        is projected into a handful of decoded frames so the pure texturer samples its real
        appearance; otherwise no frames are returned (every vertex ``measured=0``, honest R-6). The
        model is the gated asset — its absence raises an actionable error, never a fabricated mesh.
        """
        model = self._smplx_model()
        return AvatarMeshObservations(
            canonical_vertices=model.shaped(subject.proposal.shape.betas),
            faces=model.faces,
            frames=self._sample_frames(subject, model, camera, clip),
        )

    def _sample_frames(
        self,
        subject: Subject,
        model: SmplxModel,
        camera: CameraTrack | None,
        clip: ClipRef | None,
    ) -> list[FrameObservation]:
        """Decode a capped set of reference frames with the subject posed into each (M2-8b).

        Returns ``[]`` (geometry-only, R-6) when there is nothing measured to sample: no camera or
        clip, no real source pixels behind the clip URI, or no frame common to pose/camera/clip.
        The pose used is the **measured** (proposal) pose so vertices land on the pixels the player
        was actually seen in; the resulting per-vertex colour is pose-invariant (canonical ids).
        """
        if camera is None or clip is None:
            return []
        if not os.path.exists(resolve_source_path(clip.uri)):
            return []
        pose = subject.proposal.pose
        common = np.intersect1d(np.intersect1d(pose.frames, camera.frames), clip.frames)
        picks = _even_subset(common, self.max_ref_frames)
        if picks.size == 0:
            return []
        row_of = {int(f): i for i, f in enumerate(pose.frames.tolist())}
        rows = [row_of[int(f)] for f in picks]
        posed = model.pose_sequence(
            subject.proposal.shape.betas,
            pose.global_orient[rows],
            pose.body_pose[rows],
            pose.transl[rows],
        )
        images = dict(iter_clip_frames(clip.uri, [int(f) for f in picks]))
        frames: list[FrameObservation] = []
        for i, f in enumerate(picks):
            rgb = np.ascontiguousarray(images[int(f)][:, :, :3][:, :, ::-1])  # cv2 BGR -> RGB
            frames.append(
                FrameObservation(frame=int(f), vertices_world=posed[i], camera=camera, image=rgb)
            )
        return frames

    def _smplx_model(self) -> SmplxModel:
        """Load (and cache) the pure-numpy SMPL-X model, or raise an actionable locate error."""
        if self._model is None:
            path = locate_smplx_model()
            if path is None:
                raise RuntimeError(
                    "SMPL-X meshing needs the (non-commercial) SMPL-X model .npz. Point "
                    "$PITCH3D_SMPLX_MODELS at the models dir (or $PITCH3D_SMPLX_MODEL at the "
                    "file), inject an AvatarMeshBackend (--avatar-backend pkg.module:Factory), "
                    "or keep the fake (--avatar fake)."
                )
            self._model = SmplxModel.load(path)
        return self._model


@dataclass
class SyntheticAvatarMeshBackend:
    """A dependency-free, injectable :class:`AvatarMeshBackend` for demos/CI — no SMPL-X, no GPU.

    Yields a tiny synthetic mesh + one camera-facing frame so the *real* measured path (vertex
    normals -> front-facing/z-buffer visibility -> per-vertex colour averaging -> measured PLY) runs
    end-to-end. By construction one of the three vertices projects off-frame and stays
    ``measured=0`` (R-6 partial coverage, 2/3). It deliberately ignores the subject's real pose and
    uses no broadcast pixels — it exercises the pipeline, it does **not** claim a real appearance;
    that is :class:`SmplxTextureBackend`'s job once wired. Inject with
    ``--avatar-backend pitch3d.adapters.models.avatar:SyntheticAvatarMeshBackend``.
    """

    def observe(
        self,
        subject: Subject,
        ref_crops: Sequence[CropRef],
        *,
        camera: CameraTrack | None = None,
        clip: ClipRef | None = None,
    ) -> AvatarMeshObservations:
        intr = CameraIntrinsics(fx=50.0, fy=50.0, cx=32.0, cy=24.0, width=64, height=48)
        camera = CameraTrack.identity(intr, 1)
        # One triangle facing the camera (face normal -> -Z so all three verts are front-facing);
        # v0 lands on the lone coloured pixel, v2 on an in-frame black pixel, v1 projects off the
        # right edge and is never measured -> genuine 2/3 coverage through the real projection path.
        verts_world = np.array([[0.0, 0.0, 5.0], [5.0, 0.0, 5.0], [0.0, 0.5, 5.0]])
        canonical = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 2, 1]])
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[24, 32] = (200, 100, 50)
        frame = FrameObservation(frame=0, vertices_world=verts_world, camera=camera, image=image)
        return AvatarMeshObservations(canonical_vertices=canonical, faces=faces, frames=[frame])
