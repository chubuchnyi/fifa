"""Real interchange export — SMPL-X ``.npz`` + glTF/GLB (FR-26), canonical JSON (FR-28).

Two halves, the same split the model adapters use. **Real & dependency-free (numpy + stdlib):**
the canonical scene JSON (delegates to the core serializer), the SMPL-X ``.npz`` animation
(resolved per-subject betas/pose/translation), and the *assembly* of the glTF scene graph — the
Z-up→Y-up axis conversion plus per-track translation samples (:func:`build_gltf_scene`), which is
pure numpy and fully unit-tested. **Gated heavy half:** turning that scene graph into a real
``.gltf``/``.glb`` needs :mod:`pygltflib`, so :meth:`GltfExporter._write_gltf` lazy-imports it and
raises an actionable error pointing at ``pip install 'pitch3d[export]'``. USD/FBX/Alembic and the
three.js bundle stay unsupported here (roadmap M3); the export adapter is the only place axis/scale
conversion happens (never the core).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...core.correction.engine import resolve_ball, resolve_subject_motion
from ...core.ports.export import Exporter, ExportFormat, ExportResult
from ...core.scene.scene import Scene
from ...core.scene.serialization import save_scene

#: Formats this exporter actually writes; everything else → ``supports()`` is False (honest scope).
_SUPPORTED_FORMATS = frozenset(
    {ExportFormat.JSON, ExportFormat.SMPLX_NPZ, ExportFormat.GLTF, ExportFormat.GLB}
)

#: Z-up (world) → Y-up (glTF) basis: ``(x, y, z) → (x, z, -y)`` (a −90° turn about X).
_ZUP_TO_YUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def zup_to_yup(points: np.ndarray) -> np.ndarray:
    """Convert ``(N, 3)`` Z-up world points into glTF's Y-up convention."""
    return np.asarray(points, dtype=float).reshape(-1, 3) @ _ZUP_TO_YUP.T


@dataclass
class GltfNode:
    """One animated node: a named track with per-sample time + Y-up translation."""

    name: str
    times: np.ndarray         # (T,) seconds
    translations: np.ndarray  # (T, 3) Y-up meters


@dataclass
class GltfScene:
    """The assembled, axis-converted scene graph — the testable glTF intermediate."""

    nodes: list[GltfNode] = field(default_factory=list)
    up_axis: str = "Y"


def build_gltf_scene(scene: Scene, *, fps: float = 25.0) -> GltfScene:
    """Resolve every subject root + the ball into Y-up translation tracks (pure numpy)."""
    nodes: list[GltfNode] = []
    for subj in scene.subjects:
        motion = resolve_subject_motion(subj.proposal, scene.corrections_for(subj.track_id))
        nodes.append(
            GltfNode(
                name=f"subject_{subj.track_id}",
                times=np.asarray(motion.pose.frames, dtype=float) / fps,
                translations=zup_to_yup(motion.pose.transl),
            )
        )
    if scene.ball is not None:
        ball = resolve_ball(scene.ball, scene.corrections_for(None))
        nodes.append(
            GltfNode(
                name="ball",
                times=np.asarray(ball.frames, dtype=float) / fps,
                translations=zup_to_yup(ball.positions_3d),
            )
        )
    return GltfScene(nodes=nodes)


def _export_npz(scene: Scene, out_dir: Path) -> list[str]:
    """Write one resolved SMPL-X ``.npz`` per subject (shape + pose + root), return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for subj in scene.subjects:
        motion = resolve_subject_motion(subj.proposal, scene.corrections_for(subj.track_id))
        path = out_dir / f"subject_{subj.track_id}.npz"
        np.savez(
            path,
            frames=motion.pose.frames,
            betas=motion.shape.betas,
            global_orient=motion.pose.global_orient,
            body_pose=motion.pose.body_pose,
            transl=motion.pose.transl,
            body_model=motion.shape.body_model.value,
        )
        paths.append(str(path))
    return paths


@dataclass
class GltfExporter(Exporter):
    """Real SMPL-X ``.npz`` + glTF/GLB + canonical-JSON exporter (axis-converted here)."""

    fps: float = 25.0

    def supports(self, fmt: ExportFormat) -> bool:
        return fmt in _SUPPORTED_FORMATS

    def export(self, scene: Scene, fmt: ExportFormat, out_path: str) -> ExportResult:
        path = Path(out_path)
        if fmt is ExportFormat.JSON:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_scene(scene, str(path))
            return ExportResult(fmt=fmt, paths=[str(path)], note="canonical scene JSON")
        if fmt is ExportFormat.SMPLX_NPZ:
            paths = _export_npz(scene, path)
            return ExportResult(fmt=fmt, paths=paths, note=f"{len(paths)} SMPL-X npz")
        if fmt in (ExportFormat.GLTF, ExportFormat.GLB):
            gscene = build_gltf_scene(scene, fps=self.fps)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write_gltf(gscene, path, binary=fmt is ExportFormat.GLB)
            return ExportResult(
                fmt=fmt, paths=[str(path)], note=f"{len(gscene.nodes)} node {fmt.value}"
            )
        raise ValueError(
            f"{fmt.value} export is unsupported by GltfExporter (handles json/smplx_npz/gltf/glb)"
        )

    def _write_gltf(  # pragma: no cover - needs the export extra (pygltflib)
        self, gscene: GltfScene, out_path: Path, *, binary: bool
    ) -> None:
        try:
            import pygltflib
        except ImportError as exc:
            raise RuntimeError(
                "glTF/GLB export needs the 'export' extra: pip install 'pitch3d[export]'"
            ) from exc

        blob = bytearray()
        accessors: list = []
        buffer_views: list = []
        samplers: list = []
        channels: list = []
        nodes: list = []

        def _accessor(arr: np.ndarray, kind: str) -> int:
            data = np.asarray(arr, dtype=np.float32)
            raw = data.tobytes()
            offset = len(blob)
            blob.extend(raw)
            while len(blob) % 4:  # accessors must start on a 4-byte boundary
                blob.append(0)
            buffer_views.append(
                pygltflib.BufferView(buffer=0, byteOffset=offset, byteLength=len(raw))
            )
            flat = data.reshape(data.shape[0], -1)
            accessors.append(
                pygltflib.Accessor(
                    bufferView=len(buffer_views) - 1,
                    componentType=pygltflib.FLOAT,
                    count=int(data.shape[0]),
                    type=kind,
                    min=flat.min(axis=0).tolist(),
                    max=flat.max(axis=0).tolist(),
                )
            )
            return len(accessors) - 1

        for node in gscene.nodes:
            if node.times.size == 0:
                nodes.append(pygltflib.Node(name=node.name))
                continue
            t_in = _accessor(node.times.reshape(-1, 1), "SCALAR")
            t_out = _accessor(node.translations, "VEC3")
            samplers.append(pygltflib.AnimationSampler(input=t_in, output=t_out))
            channels.append(
                pygltflib.AnimationChannel(
                    sampler=len(samplers) - 1,
                    target=pygltflib.AnimationChannelTarget(node=len(nodes), path="translation"),
                )
            )
            nodes.append(pygltflib.Node(name=node.name))

        gltf = pygltflib.GLTF2(
            asset=pygltflib.Asset(generator="pitch3d"),
            scene=0,
            scenes=[pygltflib.Scene(nodes=list(range(len(nodes))))],
            nodes=nodes,
            accessors=accessors,
            bufferViews=buffer_views,
            buffers=[pygltflib.Buffer(byteLength=len(blob))],
            animations=[pygltflib.Animation(samplers=samplers, channels=channels)]
            if channels
            else [],
        )
        gltf.set_binary_blob(bytes(blob))
        if binary:
            gltf.save_binary(str(out_path))
        else:
            gltf.convert_buffers(pygltflib.BufferFormat.DATAURI)
            gltf.save_json(str(out_path))


__all__ = ["GltfExporter", "GltfNode", "GltfScene", "build_gltf_scene", "zup_to_yup"]
