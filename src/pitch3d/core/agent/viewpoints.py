"""Pure camera math for multi-view LLM feedback (ADR-0008).

Generates the named :class:`~pitch3d.core.ports.observation.Viewpoint` cameras an observer
renders so the agent sees the resolved scene "from different angles". All world-frame math
(Z-up, meters, right-handed); no rendering here — these are just :class:`CameraTrack`s a
:class:`~pitch3d.core.ports.observation.SceneObserver` consumes.
"""

from __future__ import annotations

import numpy as np

from ..correction.rotations import axis_angle_to_quat, matrix_to_axis_angle
from ..ports.observation import Viewpoint, ViewpointCamera
from ..scene.camera import CameraIntrinsics, CameraTrack
from ..scene.scene import Scene

_EPS = 1e-9


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """World→camera ``(quat_wxyz, translation)`` for a camera at ``eye`` facing ``target``.

    OpenCV-style optical frame: +Z forward (toward the target), +X right, +Y down. Robust to
    a degenerate ``up`` (e.g. a straight-down TOP camera) by swapping to an orthogonal up.
    """
    eye = np.asarray(eye, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)
    up = np.asarray(up, dtype=float).reshape(3)

    z = target - eye
    nz = np.linalg.norm(z)
    z = np.array([0.0, 1.0, 0.0]) if nz < _EPS else z / nz
    if abs(float(np.dot(up / max(np.linalg.norm(up), _EPS), z))) > 0.999:
        up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x = x / max(np.linalg.norm(x), _EPS)
    y = np.cross(z, x)
    rot = np.stack([x, y, z], axis=0)  # rows are camera axes in world ⇒ world→camera
    quat = axis_angle_to_quat(matrix_to_axis_angle(rot))
    transl = -rot @ eye
    return quat, transl


def default_intrinsics(width: int = 1280, height: int = 720, vfov_deg: float = 50.0) -> CameraIntrinsics:
    """Square-pixel pinhole intrinsics for a synthetic feedback camera."""
    f = 0.5 * height / np.tan(np.radians(vfov_deg) / 2.0)
    return CameraIntrinsics(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0, width=width, height=height)


def camera_at(
    eye: np.ndarray,
    target: np.ndarray,
    *,
    frame: int,
    intrinsics: CameraIntrinsics,
    up: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> CameraTrack:
    """A static single-frame :class:`CameraTrack` looking from ``eye`` at ``target``."""
    quat, transl = look_at(eye, target, up)
    return CameraTrack(
        intrinsics=intrinsics,
        frames=np.array([int(frame)]),
        rotation_quat=quat[None, :],
        translation=transl[None, :],
        estimated=False,
    )


def action_centroid(scene: Scene, frame: int | None = None) -> np.ndarray:
    """World point the feedback cameras aim at: mean subject root, else field center."""
    pts = []
    for s in scene.subjects:
        transl = s.proposal.pose.transl
        if transl.shape[0] == 0:
            continue
        if frame is not None and frame in set(s.proposal.pose.frames.tolist()):
            pts.append(transl[s.proposal.pose.frame_pos(frame)])
        else:
            pts.append(transl.mean(axis=0))
    if pts:
        c = np.mean(pts, axis=0)
        return np.array([c[0], c[1], 1.0])  # aim ~1 m up (torso height)
    plane_z = scene.field.plane_z if scene.field else 0.0
    return np.array([0.0, 0.0, plane_z])


def _default_frame(scene: Scene) -> int:
    for s in scene.subjects:
        if s.proposal.pose.frames.shape[0]:
            return int(s.proposal.pose.frames[0])
    return 0


def standard_viewpoints(
    scene: Scene,
    *,
    frame: int | None = None,
    which: list[Viewpoint] | None = None,
    n_orbit: int = 0,
    width: int = 1280,
    height: int = 720,
    vfov_deg: float = 50.0,
) -> list[ViewpointCamera]:
    """Build the named feedback cameras around the action centroid.

    Default set is FRONT, LEFT, TOP, BROADCAST; pass ``n_orbit > 0`` to add an evenly spaced
    orbit ring, or ``which`` to pick a subset. Radius scales with the pitch so the action
    stays framed.
    """
    if frame is None:
        frame = _default_frame(scene)
    target = action_centroid(scene, frame)
    intr = default_intrinsics(width, height, vfov_deg)
    dims = scene.field.dimensions if scene.field else None
    radius = 0.6 * max(dims.length, dims.width) if dims else 30.0

    placements: dict[Viewpoint, np.ndarray] = {
        Viewpoint.FRONT: target + np.array([0.0, -radius, 0.15 * radius]),
        Viewpoint.BACK: target + np.array([0.0, radius, 0.15 * radius]),
        Viewpoint.LEFT: target + np.array([-radius, 0.0, 0.15 * radius]),
        Viewpoint.RIGHT: target + np.array([radius, 0.0, 0.15 * radius]),
        Viewpoint.TOP: target + np.array([0.0, 0.0, radius]),
        Viewpoint.BROADCAST: target + np.array([0.0, -0.9 * radius, 0.55 * radius]),
    }
    selected = which or [Viewpoint.FRONT, Viewpoint.LEFT, Viewpoint.TOP, Viewpoint.BROADCAST]

    out: list[ViewpointCamera] = []
    for vp in selected:
        eye = placements.get(vp)
        if eye is None:
            continue
        out.append(ViewpointCamera(vp, camera_at(eye, target, frame=frame, intrinsics=intr)))

    for k in range(max(n_orbit, 0)):
        ang = 2.0 * np.pi * k / n_orbit
        eye = target + np.array(
            [radius * np.cos(ang), radius * np.sin(ang), 0.3 * radius]
        )
        out.append(ViewpointCamera(Viewpoint.ORBIT, camera_at(eye, target, frame=frame, intrinsics=intr)))
    return out


# Seam A is a *moderate* re-aim, never a free-viewpoint orbit (R-15): cap the sweep hard.
_ORBIT_MAX_DEVIATION_DEG = 45.0


def bounded_orbit_camera(
    scene: Scene,
    frames: np.ndarray,
    *,
    max_deviation_deg: float = 20.0,
    width: int = 1280,
    height: int = 720,
    vfov_deg: float = 50.0,
) -> CameraTrack:
    """A prescribed limited-orbit camera for ViewSynthesizer seam A (R-14, ADR-0007).

    Pans the BROADCAST framing by a *bounded* azimuth arc around the action centroid, one pose
    per source ``frames`` entry. Azimuth runs linearly from ``−dev`` to ``+dev`` (passing the
    source framing at the midpoint), keeping the broadcast camera's ground radius and height so
    the action stays framed — an orbit, not a dolly. The sweep is hard-capped at
    ``±45°`` (seam A is a moderate re-aim, never free-viewpoint) and the track is
    ``estimated=False`` (a *prescribed* trajectory, not a measurement). The synthesizer that
    consumes this returns a non-editable video whose ``frustum_overlap`` falls as ``dev`` grows,
    so callers can still gate application (R-15).
    """
    frames = np.asarray(frames).reshape(-1).astype(int)
    t = frames.shape[0]
    intr = default_intrinsics(width, height, vfov_deg)
    if t == 0:
        return CameraTrack(
            intrinsics=intr, frames=frames,
            rotation_quat=np.zeros((0, 4)), translation=np.zeros((0, 3)),
            estimated=False,
        )
    target = action_centroid(scene, int(frames[0]))
    dims = scene.field.dimensions if scene.field else None
    radius = 0.6 * max(dims.length, dims.width) if dims else 30.0
    # Anchor on the BROADCAST placement (see standard_viewpoints): behind (−Y) and above.
    offset = np.array([0.0, -0.9 * radius, 0.55 * radius])
    ground_r = float(np.hypot(offset[0], offset[1]))
    base_ang = float(np.arctan2(offset[1], offset[0]))
    z = float(offset[2])
    dev = np.radians(min(abs(float(max_deviation_deg)), _ORBIT_MAX_DEVIATION_DEG))
    sweep = np.linspace(-dev, dev, t) if t > 1 else np.array([0.0])

    quats = np.empty((t, 4))
    transl = np.empty((t, 3))
    for k in range(t):
        ang = base_ang + float(sweep[k])
        eye = target + np.array([ground_r * np.cos(ang), ground_r * np.sin(ang), z])
        quats[k], transl[k] = look_at(eye, target)
    return CameraTrack(
        intrinsics=intr, frames=frames,
        rotation_quat=quats, translation=transl, estimated=False,
    )
