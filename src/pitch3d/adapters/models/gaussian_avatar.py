"""Per-subject Gaussian (3DGS) avatar builder — strategy #3 (M3-1).

The third avatar strategy from the brief, built the same **measured-over-generative** way the
textured-mesh builder (#1) is (M2-0): instead of *generating* a Gaussian avatar with a feed-forward
LRM that hallucinates the whole body from one crop, this **anchors one 3D Gaussian on every measured
SMPL-X vertex** — position from the canonical mesh, colour from the player's real broadcast pixels
(the exact measured sampling :mod:`.avatar` already does), scale from the local surface spacing. A
vertex the cameras never saw front-facing keeps its honest ``measured=0`` flag and a *faint* opacity
rather than a fabricated splat (R-6). So the pure half is a real, deterministic, GPU-free 3DGS init:

* :class:`GaussianAvatarBuilder` — the **pure** half. Reuses :func:`measured_vertex_texture` for the
  per-vertex measured colour, then :func:`mesh_to_gaussians` lays down the anchored Gaussians and
  :func:`write_gaussian_splat_ply` writes a standard 3DGS ``.ply`` (with our extra per-Gaussian
  ``measured`` flag — the R-6 honesty channel carried in the asset). Numpy + stdlib; unit-tested via
  the same synthetic mesh backend the textured builder uses.
* :class:`FeedForwardGaussianRefiner` — the **heavy/generative** half (IDOL / LHM / PF-LHM for the
  bulk, per-subject GART / GaussianAvatar for hero shots). It would densify the measured init and
  inpaint the unseen sides; it is importable now (no torch at import) but every call raises an
  actionable :class:`NotImplementedError` pointing at the ``avatar`` extra (R-8). The builder runs
  **without** it (measured init only); a refiner is an optional, injected enhancement.

Select with ``default_ports(avatar="gaussian")`` / ``--avatar gaussian``; inject a real mesh backend
by dotted path with ``--avatar-backend pkg.module:Factory`` (ADR-0006), exactly like ``textured``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ...core.ports.io import ClipRef, CropRef
from ...core.ports.reconstruction import AvatarBuilder
from ...core.scene.assets import RenderAssetKind, RenderAssetRef, SynthViewRef
from ...core.scene.camera import CameraTrack
from ...core.scene.provenance import Backend, ModelInfo
from ...core.scene.subject import Subject
from .avatar import (
    AvatarMeshBackend,
    SmplxTextureBackend,
    measured_vertex_texture,
    vertex_normals,
)

#: Spherical-harmonics band-0 normalisation (INRIA 3DGS convention) — maps RGB ↔ the ``f_dc_*``
#: diffuse colour the standard splat ``.ply`` stores, so the asset loads in off-the-shelf viewers.
_SH_C0 = 0.28209479177387814
#: Opacity (pre-sigmoid logit is written) for measured vs. unmeasured Gaussians. Unmeasured stays
#: *faint*, not invisible: present enough to inspect/inpaint, never a confident fabricated surface.
_MEASURED_OPACITY = 0.99
_UNMEASURED_OPACITY = 0.1


def _rgb_to_f_dc(rgb: np.ndarray) -> np.ndarray:
    """RGB ``[0,255]`` → SH band-0 ``f_dc`` coefficients (inverse of :func:`_f_dc_to_rgb`)."""
    return (np.asarray(rgb, dtype=float) / 255.0 - 0.5) / _SH_C0


def _f_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    """SH band-0 ``f_dc`` coefficients → RGB ``[0,255]`` uint8 (inverse of :func:`_rgb_to_f_dc`)."""
    c = np.asarray(f_dc, dtype=float) * _SH_C0 + 0.5
    return np.clip(np.round(c * 255.0), 0, 255).astype(np.uint8)


def _logit(p: np.ndarray | float) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(q / (1.0 - q))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _vertex_edge_scale(
    vertices: np.ndarray, faces: np.ndarray, *, factor: float = 0.5, default: float = 0.01
) -> np.ndarray:
    """Per-vertex isotropic Gaussian scale ``(V,)`` from the mean length of its incident edges.

    Sizing each splat to its local surface spacing makes the Gaussians tile the body without big
    gaps or gross overlap. A vertex with no incident edge (isolated) falls back to ``default``.
    """
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=int).reshape(-1, 3)
    acc = np.zeros(v.shape[0], dtype=float)
    cnt = np.zeros(v.shape[0], dtype=int)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ia, ib = f[:, a], f[:, b]
        d = np.linalg.norm(v[ia] - v[ib], axis=1)
        np.add.at(acc, ia, d)
        np.add.at(cnt, ia, 1)
        np.add.at(acc, ib, d)
        np.add.at(cnt, ib, 1)
    scale = np.where(cnt > 0, acc / np.maximum(cnt, 1), default) * factor
    return np.maximum(scale, 1e-6)


@dataclass
class GaussianAvatar:
    """One subject's anchored 3D Gaussians (linear/intuitive units; the ``.ply`` stores log/logit).

    Attributes:
        positions: ``(V, 3)`` Gaussian centres in world metres (the mesh vertices).
        normals: ``(V, 3)`` unit vertex normals (written for viewers; not used for shading here).
        colors_rgb: ``(V, 3)`` uint8 measured diffuse colour (``_UNMEASURED_RGB`` where unseen).
        scales: ``(V, 3)`` linear standard deviations (isotropic per vertex).
        opacities: ``(V,)`` in ``[0, 1]`` — high for measured, faint for unmeasured (R-6).
        rotations: ``(V, 4)`` ``wxyz`` unit quaternions (identity here).
        measured: ``(V,)`` bool — the honesty channel: was this Gaussian's colour ever observed.
    """

    positions: np.ndarray
    normals: np.ndarray
    colors_rgb: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    rotations: np.ndarray
    measured: np.ndarray

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        self.normals = np.asarray(self.normals, dtype=float).reshape(-1, 3)
        self.colors_rgb = np.asarray(self.colors_rgb, dtype=np.uint8).reshape(-1, 3)
        self.scales = np.asarray(self.scales, dtype=float).reshape(-1, 3)
        self.opacities = np.asarray(self.opacities, dtype=float).reshape(-1)
        self.rotations = np.asarray(self.rotations, dtype=float).reshape(-1, 4)
        self.measured = np.asarray(self.measured, dtype=bool).reshape(-1)

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])

    @property
    def coverage(self) -> float:
        """Fraction of Gaussians whose colour is measured (R-6)."""
        return float(self.measured.mean()) if self.n else 0.0


def mesh_to_gaussians(
    vertices: np.ndarray,
    faces: np.ndarray,
    rgb: np.ndarray,
    measured: np.ndarray,
    *,
    scale_factor: float = 0.5,
) -> GaussianAvatar:
    """Lay one anchored 3D Gaussian on each mesh vertex from its measured colour + local spacing.

    Pure geometry: centre = vertex, colour = measured RGB, scale = local edge length ×
    ``scale_factor`` (isotropic), rotation = identity, opacity high if measured else faint (R-6).
    No optimisation, no GPU — the honest measured *init* a generative refiner would densify/inpaint.
    """
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=int).reshape(-1, 3)
    m = np.asarray(measured, dtype=bool).reshape(-1)
    normals = vertex_normals(v, f)
    scale = _vertex_edge_scale(v, f, factor=scale_factor)
    scales = np.repeat(scale[:, None], 3, axis=1)
    opacities = np.where(m, _MEASURED_OPACITY, _UNMEASURED_OPACITY)
    rotations = np.zeros((v.shape[0], 4), dtype=float)
    rotations[:, 0] = 1.0  # identity quaternion (w, x, y, z)
    return GaussianAvatar(
        positions=v, normals=normals, colors_rgb=rgb, scales=scales,
        opacities=opacities, rotations=rotations, measured=m,
    )


_PLY_PROPS = [
    "property float x", "property float y", "property float z",
    "property float nx", "property float ny", "property float nz",
    "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
    "property float opacity",
    "property float scale_0", "property float scale_1", "property float scale_2",
    "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3",
    "property uchar measured",
]


def write_gaussian_splat_ply(path: Path, ga: GaussianAvatar) -> str:
    """Write the standard 3DGS ``.ply`` (SH band-0 colour, log scale, logit opacity) + ``measured``.

    Uses the INRIA-3DGS property names so off-the-shelf splat viewers load it, plus our extra
    per-Gaussian ``measured`` uchar so a renderer can grey-out / request inpaint for the unseen
    surface instead of trusting a fabricated splat (R-6). ASCII (stdlib only — no plyfile/torch).
    """
    f_dc = _rgb_to_f_dc(ga.colors_rgb)
    opacity = _logit(ga.opacities)
    scale_log = np.log(np.maximum(ga.scales, 1e-9))
    meas = ga.measured.astype(int)
    lines = ["ply", "format ascii 1.0", f"element vertex {ga.n}", *_PLY_PROPS, "end_header"]
    for i in range(ga.n):
        p, nrm, r = ga.positions[i], ga.normals[i], ga.rotations[i]
        lines.append(
            f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
            f"{nrm[0]:.6f} {nrm[1]:.6f} {nrm[2]:.6f} "
            f"{f_dc[i, 0]:.6f} {f_dc[i, 1]:.6f} {f_dc[i, 2]:.6f} "
            f"{opacity[i]:.6f} "
            f"{scale_log[i, 0]:.6f} {scale_log[i, 1]:.6f} {scale_log[i, 2]:.6f} "
            f"{r[0]:.6f} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {meas[i]}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def read_gaussian_splat_ply(path: Path) -> GaussianAvatar:
    """Inverse of :func:`write_gaussian_splat_ply` — parse our 3DGS ``.ply`` back to a Gaussian set.

    Recovers linear units (RGB from ``f_dc``, opacity from its logit, scale from its log) so a
    RenderPass or a test can round-trip the asset — including the R-6 ``measured`` flag — with no
    plyfile/torch. Assumes our own fixed property order.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    nv = next(int(ln.split()[-1]) for ln in lines if ln.startswith("element vertex"))
    body = lines[lines.index("end_header") + 1:]
    cols = np.array([[float(x) for x in body[i].split()[:17]] for i in range(nv)], dtype=float)
    meas = np.array([int(body[i].split()[17]) for i in range(nv)], dtype=bool)
    return GaussianAvatar(
        positions=cols[:, 0:3],
        normals=cols[:, 3:6],
        colors_rgb=_f_dc_to_rgb(cols[:, 6:9]),
        scales=np.exp(cols[:, 10:13]),
        opacities=_sigmoid(cols[:, 9]),
        rotations=cols[:, 13:17],
        measured=meas,
    )


@runtime_checkable
class GaussianRefiner(Protocol):
    """The heavy/generative half: densify the measured init + inpaint unseen sides (R-8 gated)."""

    def refine(
        self,
        avatar: GaussianAvatar,
        subject: Subject,
        ref_crops: Sequence[CropRef],
        *,
        camera: CameraTrack | None = None,
        clip: ClipRef | None = None,
    ) -> GaussianAvatar:
        """Return a refined GaussianAvatar (densified + unseen sides inpainted)."""
        ...


@dataclass
class FeedForwardGaussianRefiner:
    """IDOL / LHM / PF-LHM / GART-class feed-forward Gaussian refinement (FR-12 #3, gated, R-8).

    Importable with no torch; :meth:`refine` raises an actionable error. The measured init
    (:class:`GaussianAvatarBuilder` with ``refiner=None``) is real and runs without it.
    """

    name: str = "IDOL"

    def refine(
        self,
        avatar: GaussianAvatar,
        subject: Subject,
        ref_crops: Sequence[CropRef],
        *,
        camera: CameraTrack | None = None,
        clip: ClipRef | None = None,
    ) -> GaussianAvatar:
        raise NotImplementedError(
            "generative Gaussian-avatar refinement (IDOL/LHM/PF-LHM/GART) is not wired yet — "
            "install the `avatar` extra (roadmap M3-1). The measured Gaussian init "
            "(GaussianAvatarBuilder with no refiner) is real and runs without it; the unseen "
            "surface stays flagged measured=0 (R-6), never fabricated."
        )


@dataclass
class GaussianAvatarBuilder(AvatarBuilder):
    """Measured per-subject 3DGS avatar (strategy #3, M3-1) — anchored Gaussians over a backend.

    Attributes:
        mesh_backend: Resolves a subject into a canonical mesh + per-frame posed views (shared with
            the textured builder). If ``None`` a real :class:`SmplxTextureBackend` is built lazily
            (needs the ``avatar`` extra + the SMPL-X model + scene context).
        refiner: Optional generative densify/inpaint (gated, R-8). ``None`` ⇒ measured init only.
        out_dir: Where the per-subject ``.ply`` splat assets are written.
        device: Inference device for the default backend (provenance + forwarded).
        scale_factor: Gaussian size as a fraction of local edge length.
    """

    mesh_backend: AvatarMeshBackend | None = None
    refiner: GaussianRefiner | None = None
    out_dir: Path = field(default_factory=lambda: Path("out/assets"))
    device: str = "cuda"
    scale_factor: float = 0.5

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="GaussianAvatar",
            backend=Backend.LOCAL,
            license="measured 3DGS init on non-commercial SMPL-X topology; generative refine gated",
            params={
                "device": self.device,
                "refiner": type(self.refiner).__name__ if self.refiner is not None else "none",
            },
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
        obs = self._mesh_backend().observe(subject, ref_crops, camera=camera, clip=clip)
        n = obs.canonical_vertices.shape[0]
        rgb, count = measured_vertex_texture(obs)
        measured = count > 0
        ga = mesh_to_gaussians(
            obs.canonical_vertices, obs.faces, rgb, measured, scale_factor=self.scale_factor
        )
        refined = False
        if self.refiner is not None:
            ga = self.refiner.refine(ga, subject, ref_crops, camera=camera, clip=clip)
            refined = True
        uri = write_gaussian_splat_ply(
            self.out_dir / f"avatar_{subject.track_id}_gaussian.ply", ga
        )
        coverage = float(measured.mean()) if n else 0.0
        return RenderAssetRef(
            id=f"avatar-{subject.track_id}",
            kind=RenderAssetKind.AVATAR_GAUSSIAN,
            uri=uri,
            model=self.info(),
            subject_track_id=subject.track_id,
            extra={
                "coverage": coverage,                 # measured fraction of the body surface (R-6)
                "n_vertices": int(n),
                "n_gaussians": int(ga.n),
                "n_measured": int(measured.sum()),
                "frames_used": len(obs.frames),
                "refined": refined,
                "synth_views": 0 if synth_views is None else len(synth_views),
            },
        )

    def _mesh_backend(self) -> AvatarMeshBackend:
        return self.mesh_backend or SmplxTextureBackend(device=self.device)


__all__ = [
    "FeedForwardGaussianRefiner",
    "GaussianAvatar",
    "GaussianAvatarBuilder",
    "GaussianRefiner",
    "mesh_to_gaussians",
    "read_gaussian_splat_ply",
    "write_gaussian_splat_ply",
]
