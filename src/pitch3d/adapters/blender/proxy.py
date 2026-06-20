"""Pure proxy-plan assembly for the Blender editing surface (M1 step 9/10, ADR-0008).

This is the dependency-free half of the Blender adapter — the same split the model/export
adapters use. It turns a (resolved) :class:`Scene` into a serializable :class:`ProxyPlan`: per
subject a root controller with **root translation** and **root orientation** animation channels
(the "ball/root as F-curves" of step 10), the β shape, and optionally the per-joint body pose;
plus the ball trajectory and the camera placements for any requested viewpoints. It is pure numpy
+ stdlib and fully unit-tested; :mod:`._script` (runs *inside* Blender) consumes the JSON and
builds the actual armature/empties + F-curves, and :mod:`.runner` drives that subprocess.

The world frame is already **Z-up meters** — Blender's native convention — so, unlike the glTF
exporter, there is *no* axis conversion here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...core.correction.engine import resolve_ball, resolve_subject_motion
from ...core.ports.observation import ViewpointCamera
from ...core.scene.scene import Scene
from ...core.scene.subject import Role, Subject, Team

_SENSOR_WIDTH_MM = 36.0  # Blender default 35mm-equivalent sensor; lens is derived from fx
_REFEREE_RGB = (1.0, 0.31, 0.78)   # pink — referees carry no team colour
_PLAYER_RGB = (0.24, 0.67, 1.0)    # blue — default when a team has no colour set
_BALL_RGB = (1.0, 0.84, 0.0)       # gold


@dataclass
class ProxyObject:
    """One animated proxy object: a named controller with per-frame transform channels.

    ``location`` is the root/ball world position (meters, Z-up). ``rotation_aa`` is the root
    orientation as an axis-angle 3-vector per frame (``None`` for the ball). ``betas`` /
    ``body_pose`` carry the SMPL-X identity + per-joint pose so they round-trip as editable
    channels even before a rigged mesh exists (the mesh awaits the avatar model, M2).
    """

    name: str
    kind: str                       # "subject" | "ball"
    frames: np.ndarray              # (T,)
    location: np.ndarray            # (T, 3) world meters
    color_rgb: tuple[float, float, float]
    rotation_aa: np.ndarray | None = None   # (T, 3) axis-angle, subjects only
    betas: np.ndarray | None = None          # (n_betas,) static, subjects only
    body_pose: np.ndarray | None = None      # (T, J, 3) axis-angle, subjects only


@dataclass
class ProxyView:
    """A camera placement to render the proxy from (derived from a ``ViewpointCamera``)."""

    viewpoint: str
    eye: np.ndarray              # (3,) world position of the camera
    target: np.ndarray           # (3,) world point it looks at
    lens_mm: float               # focal length for Blender, derived from fx
    resolution: tuple[int, int]  # (width, height) px to render
    frame: int = 0               # scene frame to pose the proxy at for this render


@dataclass
class ProxyPlan:
    """The full, serializable description Blender builds: objects (+ optional render views)."""

    scene_id: str
    fps: float = 25.0
    objects: list[ProxyObject] = field(default_factory=list)
    views: list[ProxyView] = field(default_factory=list)
    up_axis: str = "Z"


def _subject_rgb(subject: Subject, teams: list[Team]) -> tuple[float, float, float]:
    if subject.role == Role.REFEREE:
        return _REFEREE_RGB
    team = next((t for t in teams if t.id == subject.team_id), None)
    if team is not None and team.color_rgb is not None:
        return tuple(float(np.clip(c, 0.0, 1.0)) for c in team.color_rgb)
    return _PLAYER_RGB


def _quat_to_world_cam_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """World→camera rotation ``(3, 3)`` from a (w, x, y, z) quaternion (rows = cam axes)."""
    q = np.asarray(quat_wxyz, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]],
        dtype=float,
    )


def camera_eye_target(view: ViewpointCamera) -> tuple[np.ndarray, np.ndarray]:
    """Recover the world-space camera eye and a look-at target from a feedback camera.

    The feedback cameras are world→camera (``X_c = R X_w + t``, OpenCV +Z-forward optical
    frame). The eye is the camera centre ``-Rᵀt`` and the look direction is the camera's
    forward axis in world coords (row 2 of ``R``); the target is one metre along it. This lets
    Blender aim with ``Vector.to_track_quat`` instead of re-deriving the rotation convention.
    """
    cam = view.camera
    rot = _quat_to_world_cam_matrix(cam.rotation_quat[0])
    t = np.asarray(cam.translation[0], dtype=float).reshape(3)
    eye = -rot.T @ t
    target = eye + rot[2]  # camera +Z (forward) expressed in world
    return eye, target


def _view_from_camera(view: ViewpointCamera, *, max_px: int | None) -> ProxyView:
    eye, target = camera_eye_target(view)
    k = view.camera.intrinsics
    width, height = int(k.width), int(k.height)
    if max_px is not None and max(width, height) > max_px:
        scale = max_px / max(width, height)
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
    lens_mm = float(k.fx) * _SENSOR_WIDTH_MM / float(k.width)
    frame = int(view.camera.frames[0]) if view.camera.frames.shape[0] else 0
    return ProxyView(
        viewpoint=view.viewpoint.value, eye=eye, target=target,
        lens_mm=lens_mm, resolution=(width, height), frame=frame,
    )


def build_proxy_plan(
    scene: Scene,
    *,
    fps: float = 25.0,
    views: list[ViewpointCamera] | None = None,
    include_pose: bool = True,
    preview_max_px: int | None = 480,
) -> ProxyPlan:
    """Assemble the Blender build plan from a scene (proposal ⊕ corrections; copy-safe).

    Pass ``views`` (e.g. from ``standard_viewpoints``) to also place render cameras.
    ``include_pose`` carries the per-joint body pose as channels (skip it for a positions-only
    render); already-resolved scenes (empty correction stack) pass through unchanged.
    """
    objects: list[ProxyObject] = []
    for subj in scene.subjects:
        motion = resolve_subject_motion(subj.proposal, scene.corrections_for(subj.track_id))
        objects.append(
            ProxyObject(
                name=f"subject_{subj.track_id}",
                kind="subject",
                frames=np.asarray(motion.pose.frames, dtype=int),
                location=np.asarray(motion.pose.transl, dtype=float),
                color_rgb=_subject_rgb(subj, scene.teams),
                rotation_aa=np.asarray(motion.pose.global_orient, dtype=float),
                betas=np.asarray(motion.shape.betas, dtype=float),
                body_pose=np.asarray(motion.pose.body_pose, dtype=float) if include_pose else None,
            )
        )
    if scene.ball is not None:
        ball = resolve_ball(scene.ball, scene.corrections_for(None))
        objects.append(
            ProxyObject(
                name="ball",
                kind="ball",
                frames=np.asarray(ball.frames, dtype=int),
                location=np.asarray(ball.positions_3d, dtype=float),
                color_rgb=_BALL_RGB,
            )
        )
    view_plans = [_view_from_camera(v, max_px=preview_max_px) for v in (views or [])]
    return ProxyPlan(scene_id=scene.id, fps=float(fps), objects=objects, views=view_plans)


# --- JSON round-trip (the subprocess boundary) --------------------------------------


def plan_to_json(plan: ProxyPlan) -> str:
    """Serialize a :class:`ProxyPlan` to JSON for the in-Blender script to consume."""
    return json.dumps(_plan_dict(plan), default=_json_default)


def write_plan(plan: ProxyPlan, path: str | Path) -> Path:
    """Write the plan JSON to ``path`` and return it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(plan_to_json(plan), encoding="utf-8")
    return p


def load_plan(path: str | Path) -> ProxyPlan:
    """Reload a plan from JSON — symmetric with :func:`write_plan`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    objects = [
        ProxyObject(
            name=o["name"],
            kind=o["kind"],
            frames=np.asarray(o["frames"], dtype=int),
            location=np.asarray(o["location"], dtype=float).reshape(-1, 3),
            color_rgb=tuple(o["color_rgb"]),
            rotation_aa=None if o.get("rotation_aa") is None
            else np.asarray(o["rotation_aa"], dtype=float).reshape(-1, 3),
            betas=None if o.get("betas") is None else np.asarray(o["betas"], dtype=float),
            body_pose=None if o.get("body_pose") is None
            else np.asarray(o["body_pose"], dtype=float),
        )
        for o in data["objects"]
    ]
    views = [
        ProxyView(
            viewpoint=v["viewpoint"],
            eye=np.asarray(v["eye"], dtype=float).reshape(3),
            target=np.asarray(v["target"], dtype=float).reshape(3),
            lens_mm=float(v["lens_mm"]),
            resolution=(int(v["resolution"][0]), int(v["resolution"][1])),
            frame=int(v.get("frame", 0)),
        )
        for v in data.get("views", [])
    ]
    return ProxyPlan(
        scene_id=data["scene_id"], fps=float(data["fps"]), objects=objects,
        views=views, up_axis=data.get("up_axis", "Z"),
    )


def _plan_dict(plan: ProxyPlan) -> dict:
    return {
        "scene_id": plan.scene_id,
        "fps": plan.fps,
        "up_axis": plan.up_axis,
        "objects": [
            {
                "name": o.name,
                "kind": o.kind,
                "frames": o.frames,
                "location": o.location,
                "color_rgb": list(o.color_rgb),
                "rotation_aa": o.rotation_aa,
                "betas": o.betas,
                "body_pose": o.body_pose,
            }
            for o in plan.objects
        ],
        "views": [
            {
                "viewpoint": v.viewpoint,
                "eye": v.eye,
                "target": v.target,
                "lens_mm": v.lens_mm,
                "resolution": list(v.resolution),
                "frame": v.frame,
            }
            for v in plan.views
        ],
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)!r}")


__all__ = [
    "ProxyObject",
    "ProxyPlan",
    "ProxyView",
    "build_proxy_plan",
    "camera_eye_target",
    "load_plan",
    "plan_to_json",
    "write_plan",
]
