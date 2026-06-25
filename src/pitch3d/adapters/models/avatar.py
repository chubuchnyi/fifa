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
* :class:`SmplxTextureBackend` — the **heavy** half: *not wired yet* (roadmap M2). Turning a
  subject's resolved SMPL-X params into posed world vertices needs the SMPL-X body model
  (non-commercial), and sampling needs the decoded source frames + the scene camera, so it is
  built with that scene context and gated behind the ``avatar`` extra; :meth:`observe` raises an
  actionable ``NotImplementedError``. The pure half above is complete and tested.

Swap it in via ``default_ports(avatar="textured")`` (wiring) or inject a vendored backend by
dotted path with ``--avatar-backend pkg.module:Factory`` (ADR-0006) — one fake replaced at a time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ...core.ports.io import CropRef
from ...core.ports.reconstruction import AvatarBuilder
from ...core.scene.assets import RenderAssetKind, RenderAssetRef, SynthViewRef
from ...core.scene.camera import CameraTrack
from ...core.scene.projection import camera_center, project_world_points_with_depth
from ...core.scene.provenance import Backend, ModelInfo
from ...core.scene.subject import Subject

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
        self, subject: Subject, ref_crops: Sequence[CropRef]
    ) -> AvatarMeshObservations:
        """Return the canonical mesh + the per-frame posed vertices/camera/pixels to sample."""
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
    ) -> RenderAssetRef:
        obs = self._backend().observe(subject, ref_crops)
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
        rgb, count = aggregate_observations(colors_per_frame, observed_per_frame, n)
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


@dataclass
class SmplxTextureBackend:
    """Real SMPL-X meshing + frame sampling: lazy torch/smplx imports, no import cost.

    Posing a subject into world vertices needs the SMPL-X body model and sampling needs the
    decoded source frames + the scene camera, so a real instance is constructed with that scene
    context. Imported only when :meth:`observe` is first called, so this module stays import-safe
    without the ``avatar`` extra.
    """

    device: str = "cuda"
    _model: object = None

    def observe(  # pragma: no cover - heavy path
        self, subject: Subject, ref_crops: Sequence[CropRef]
    ) -> AvatarMeshObservations:
        self._load()
        raise NotImplementedError(
            "measured SMPL-X texturing is not wired yet (roadmap M2). The pure half "
            "(projection + front-facing/z-buffer visibility + per-vertex colour averaging + "
            "the measured/PLY writer) is complete and tested; the heavy half needs (a) the "
            "SMPL-X body model (non-commercial) to pose the subject's resolved params into "
            "world vertices, (b) the decoded source frames, and (c) the scene camera — it is "
            "built with that scene context. Inject an AvatarMeshBackend (--avatar-backend "
            "pkg.module:Factory) that yields a canonical mesh + per-frame posed "
            "vertices/camera/pixels, or keep the fake (--avatar fake)."
        )

    def _load(self) -> object:  # pragma: no cover - exercised only without the extra
        if self._model is None:
            try:
                import torch  # noqa: F401  (stand-in for the SMPL-X body-model stack)
            except ImportError as exc:
                raise RuntimeError(
                    "SMPL-X texturing needs the avatar extra. Install it "
                    "(`pip install 'pitch3d[avatar]'`) or inject an AvatarMeshBackend."
                ) from exc
            self._model = object()
        return self._model
