"""Pure Cycles render-plan assembly (M2-7, ADR-0003) — no bpy, fully unit-tested.

The dependency-free half of the Cycles render adapter, the same split :mod:`.proxy` uses. It turns
the *resolved* scene's avatar meshes + the estimated camera into a serializable :class:`CyclesPlan`:
per output frame the Blender camera ``matrix_world`` (OpenCV→Blender converted) plus each *present*
subject's root **rigid** placement matrix, and the camera intrinsics mapped to Blender lens / sensor
/ shift. :mod:`._cycles_script` consumes the JSON *inside* Blender and renders with Cycles;
:class:`~pitch3d.adapters.render.cycles.CyclesRenderPass` orchestrates (loads the PLY assets, writes
the mesh NPZs, drives the subprocess).

Two conventions meet here and only the camera optical frame needs a flip:

* **World frame** is Z-up metres (ADR-0003) — Blender's native frame — so world points pass through
  unchanged (no glTF-style axis swap).
* **Camera extrinsics** are world→camera OpenCV optical (``X_c = R X_w + t``; +Z forward, +x right,
  +y down). Blender's camera looks down its local **-Z** with **+Y up**, and ``matrix_world`` is
  camera→world. ``diag(1, -1, -1)`` maps the OpenCV optical axes onto Blender's.

Two avatar geometry layouts are supported. A **rigid** mesh (M2-7) is one canonical mesh placed by
its resolved root transform — identical to the splat pass. A **posed** mesh (M2-8) carries per-frame
LBS vertices (the resolved root *and* per-joint limb articulation, baked on the pure side via
:mod:`...models.smplx_lbs`); the plan then carries identity placements + a per-frame ``vert_index``
and the script swaps the geometry each frame, so the limbs follow the resolved pose. The environment
is still a single neutral ground plane; the measured grass/line material is M2-9.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...core.correction.engine import resolve_subject_motion
from ...core.correction.rotations import axis_angle_to_matrix
from ...core.scene.camera import CameraIntrinsics, CameraTrack
from ...core.scene.projection import camera_pose
from ...core.scene.scene import Scene

# OpenCV optical → Blender camera axis flip: x stays, y/z negate (down→up, forward→backward).
_CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0])
_SENSOR_WIDTH_MM = 36.0  # Blender's default 35mm-equivalent sensor; lens is derived from fx


def cv_to_blender_camera_matrix(rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    """OpenCV world→camera ``(R, t)`` → Blender camera ``matrix_world`` (4x4 camera→world).

    The camera centre is ``C = -Rᵀ t`` and the camera→world rotation is ``Rᵀ`` re-expressed in
    Blender's optical frame via the ``diag(1, -1, -1)`` flip (OpenCV +Z-forward/+y-down → Blender
    -Z-forward/+y-up). Set this straight onto ``camera.matrix_world`` and the Cycles projection
    matches our pinhole :func:`~pitch3d.core.scene.projection.project_world_points_with_depth`.
    """
    rot = np.asarray(rot, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)
    rot_c2w = rot.T
    matrix = np.eye(4)
    matrix[:3, :3] = rot_c2w @ _CV_TO_BLENDER
    matrix[:3, 3] = -rot_c2w @ t
    return matrix


def root_object_matrix(global_orient_aa: np.ndarray, transl: np.ndarray) -> np.ndarray:
    """Resolved root **rigid** transform → a 4x4 ``matrix_world`` (orientation + translation).

    Mirrors the splat pass's ``canonical @ rot.T + transl`` exactly: with the canonical mesh as the
    object's local data, ``matrix_world @ [v; 1]`` yields ``rot @ v + transl`` — a ROOT_ORIENTATION
    or ROOT_TRANSLATION edit re-projects into Cycles with no mesh rebuild (M2-4). Per-joint body
    pose (LBS) is **not** applied — an honest deferred limit (M2-8), not silently faked.
    """
    matrix = np.eye(4)
    matrix[:3, :3] = axis_angle_to_matrix(np.asarray(global_orient_aa, dtype=float).reshape(3))
    matrix[:3, 3] = np.asarray(transl, dtype=float).reshape(3)
    return matrix


def blender_lens_params(
    intrinsics: CameraIntrinsics, sensor_width_mm: float = _SENSOR_WIDTH_MM
) -> dict[str, Any]:
    """Map pinhole intrinsics ``K`` to Blender camera lens / sensor / shift / pixel-aspect.

    Follows the well-tested BlenderProc ``K``→camera mapping: an ``fx≠fy`` asymmetry rides the
    render pixel aspect; the principal-point offset rides the camera lens *shift* (a fraction of
    the fitted sensor dimension); the focal length sets ``lens`` for the ``sensor_fit``. Broadcast
    soccer is landscape (``width ≥ height``) → HORIZONTAL fit, where ``lens = fx·sensor/width``. A
    sub-pixel ``(w-1)/2`` principal-centre convention is used (BlenderProc's); any residual ≤1px
    offset versus our splat projection is a known qualitative gap to confirm in E2E.
    """
    fx, fy = float(intrinsics.fx), float(intrinsics.fy)
    cx, cy = float(intrinsics.cx), float(intrinsics.cy)
    width, height = int(intrinsics.width), int(intrinsics.height)
    if fx >= fy:
        pixel_aspect_x, pixel_aspect_y = 1.0, fx / fy
    else:
        pixel_aspect_x, pixel_aspect_y = fy / fx, 1.0
    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x
    if width >= height * pixel_aspect_ratio:
        sensor_fit, view_fac_in_px = "HORIZONTAL", float(width)
    else:
        sensor_fit, view_fac_in_px = "VERTICAL", pixel_aspect_ratio * float(height)
    return {
        "sensor_fit": sensor_fit,
        "sensor_width_mm": float(sensor_width_mm),
        "lens_mm": fx / view_fac_in_px * float(sensor_width_mm),
        "pixel_aspect_x": float(pixel_aspect_x),
        "pixel_aspect_y": float(pixel_aspect_y),
        "shift_x": -(cx - (width - 1) / 2.0) / view_fac_in_px,
        "shift_y": (cy - (height - 1) / 2.0) / view_fac_in_px * pixel_aspect_ratio,
    }


@dataclass
class CyclesMeshRef:
    """A per-subject avatar mesh the script loads once from ``<mesh_dir>/<npz>`` and reuses.

    ``posed=False`` (M2-7) the NPZ holds one canonical ``(V, 3)`` mesh placed rigidly by each
    frame's root ``matrix_world``. ``posed=True`` (M2-8) the NPZ holds per-pose-frame LBS vertices
    ``(T, V, 3)`` (root *and* limbs already baked on the pure side); placements are identity and the
    script swaps in row ``vert_index`` each output frame, so the limbs follow the resolved pose.
    """

    name: str
    npz: str
    track_id: int
    posed: bool = False


@dataclass
class CyclesPlacement:
    """One avatar's placement on one output frame: its ``matrix_world`` and render visibility.

    ``visible=False`` (subject absent this frame) hides the object for that render rather than
    fabricating a position — the same "no placement, no fabrication" rule the splat pass follows.
    ``vert_index >= 0`` (posed mesh) selects the LBS-vertex row to swap in this frame; ``-1`` (the
    default) is a rigid mesh placed by ``matrix_world`` alone.
    """

    name: str
    matrix_world: np.ndarray  # (4, 4) object→world
    visible: bool
    vert_index: int = -1


@dataclass
class CyclesFrame:
    """One output frame: the Blender camera ``matrix_world`` + every avatar's placement."""

    index: int
    camera_matrix_world: np.ndarray  # (4, 4) camera→world
    placements: list[CyclesPlacement] = field(default_factory=list)


@dataclass
class CyclesPlan:
    """The full, serializable description the in-Blender Cycles script builds and renders."""

    scene_id: str
    width: int
    height: int
    camera_intrinsics: dict[str, Any]
    meshes: list[CyclesMeshRef] = field(default_factory=list)
    frames: list[CyclesFrame] = field(default_factory=list)
    samples: int = 48
    device: str = "CPU"
    ground_z: float = 0.0
    ground_size: float = 140.0  # metres — covers a 105x68 m pitch + margin
    up_axis: str = "Z"


def _resolved_roots(scene: Scene, track_ids: set[int]) -> dict[int, Any]:
    """Per-subject resolved pose for the requested tracks (proposal ⊕ corrections; copy-safe)."""
    return {
        subj.track_id: resolve_subject_motion(
            subj.proposal, scene.corrections_for(subj.track_id)
        ).pose
        for subj in scene.subjects
        if subj.track_id in track_ids
    }


def build_cycles_plan(
    scene: Scene,
    camera: CameraTrack,
    *,
    meshes: list[CyclesMeshRef],
    samples: int = 48,
    device: str = "CPU",
    ground_z: float = 0.0,
    ground_size: float = 140.0,
    sensor_width_mm: float = _SENSOR_WIDTH_MM,
) -> CyclesPlan:
    """Assemble the Cycles plan from a resolved scene + its estimated camera.

    ``meshes`` are the avatar assets already resolved to NPZ refs (one per owning subject). For each
    camera frame the plan carries the Blender camera matrix and each avatar's resolved-root rigid
    placement (or ``visible=False`` where the subject is absent that frame). The placement reads the
    *resolved* pose, so edits re-project with no mesh rebuild (M2-4).
    """
    intr = blender_lens_params(camera.intrinsics, sensor_width_mm)
    roots = _resolved_roots(scene, {m.track_id for m in meshes})
    frames: list[CyclesFrame] = []
    for i, frame in enumerate(camera.frames.tolist()):
        rot, t = camera_pose(camera, int(frame))
        placements: list[CyclesPlacement] = []
        for mesh in meshes:
            pose = roots.get(mesh.track_id)
            hit = np.nonzero(pose.frames == int(frame))[0] if pose is not None else np.empty(0)
            if pose is None or not hit.size:
                placements.append(CyclesPlacement(mesh.name, np.eye(4), visible=False))
                continue
            row = int(hit[0])
            if mesh.posed:
                # Root + limbs are baked into the per-frame LBS vertices; place identity and let the
                # script swap in row ``row``. (transl is already in the baked world vertices.)
                placements.append(
                    CyclesPlacement(mesh.name, np.eye(4), visible=True, vert_index=row)
                )
            else:
                placements.append(
                    CyclesPlacement(
                        mesh.name,
                        root_object_matrix(pose.global_orient[row], pose.transl[row]),
                        visible=True,
                    )
                )
        frames.append(
            CyclesFrame(
                index=i,
                camera_matrix_world=cv_to_blender_camera_matrix(rot, t),
                placements=placements,
            )
        )
    return CyclesPlan(
        scene_id=scene.id,
        width=int(camera.intrinsics.width),
        height=int(camera.intrinsics.height),
        camera_intrinsics=intr,
        meshes=list(meshes),
        frames=frames,
        samples=samples,
        device=device,
        ground_z=ground_z,
        ground_size=ground_size,
    )


# --- JSON round-trip (the subprocess boundary) --------------------------------------


def plan_to_json(plan: CyclesPlan) -> str:
    """Serialize a :class:`CyclesPlan` to JSON for the in-Blender script to consume."""
    return json.dumps(_plan_dict(plan), default=_json_default)


def write_cycles_plan(plan: CyclesPlan, path: str | Path) -> Path:
    """Write the plan JSON to ``path`` and return it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(plan_to_json(plan), encoding="utf-8")
    return p


def load_cycles_plan(path: str | Path) -> CyclesPlan:
    """Reload a plan from JSON — symmetric with :func:`write_cycles_plan` (used by tests)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    meshes = [
        CyclesMeshRef(m["name"], m["npz"], int(m["track_id"]), bool(m.get("posed", False)))
        for m in data["meshes"]
    ]
    frames = [
        CyclesFrame(
            index=int(f["index"]),
            camera_matrix_world=np.asarray(f["camera_matrix_world"], dtype=float).reshape(4, 4),
            placements=[
                CyclesPlacement(
                    p["name"],
                    np.asarray(p["matrix_world"], dtype=float).reshape(4, 4),
                    bool(p["visible"]),
                    int(p.get("vert_index", -1)),
                )
                for p in f["placements"]
            ],
        )
        for f in data["frames"]
    ]
    return CyclesPlan(
        scene_id=data["scene_id"],
        width=int(data["width"]),
        height=int(data["height"]),
        camera_intrinsics=data["camera_intrinsics"],
        meshes=meshes,
        frames=frames,
        samples=int(data["samples"]),
        device=data["device"],
        ground_z=float(data["ground_z"]),
        ground_size=float(data["ground_size"]),
        up_axis=data.get("up_axis", "Z"),
    )


def _plan_dict(plan: CyclesPlan) -> dict:
    return {
        "scene_id": plan.scene_id,
        "width": plan.width,
        "height": plan.height,
        "samples": plan.samples,
        "device": plan.device,
        "ground_z": plan.ground_z,
        "ground_size": plan.ground_size,
        "up_axis": plan.up_axis,
        "camera_intrinsics": plan.camera_intrinsics,
        "meshes": [
            {"name": m.name, "npz": m.npz, "track_id": m.track_id, "posed": m.posed}
            for m in plan.meshes
        ],
        "frames": [
            {
                "index": f.index,
                "camera_matrix_world": f.camera_matrix_world,
                "placements": [
                    {
                        "name": p.name,
                        "matrix_world": p.matrix_world,
                        "visible": p.visible,
                        "vert_index": p.vert_index,
                    }
                    for p in f.placements
                ],
            }
            for f in plan.frames
        ],
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)!r}")


__all__ = [
    "CyclesFrame",
    "CyclesMeshRef",
    "CyclesPlacement",
    "CyclesPlan",
    "blender_lens_params",
    "build_cycles_plan",
    "cv_to_blender_camera_matrix",
    "load_cycles_plan",
    "plan_to_json",
    "root_object_matrix",
    "write_cycles_plan",
]
