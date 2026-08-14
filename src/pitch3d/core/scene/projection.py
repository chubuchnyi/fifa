"""Pinhole world→image projection through a :class:`CameraTrack` — pure camera geometry.

This is core geometry, not an adapter concern: both the reprojection-overlay render pass
(M1) and the measured-texture avatar builder (M2) project resolved world points through the
estimated camera. Keeping it here gives them **one** projection implementation to share
(``X_c = R @ X_w + t`` then the pinhole intrinsics), so they cannot drift apart. Numpy only.
"""

from __future__ import annotations

import numpy as np

from .camera import CameraTrack


def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """World→camera rotation ``(3, 3)`` from a (w, x, y, z) quaternion (normalised first)."""
    q = np.asarray(quat, dtype=float).reshape(4)
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


def frame_row(frames: np.ndarray, frame_index: int) -> int:
    """Nearest camera row for a frame index (exact match when present)."""
    i = int(np.searchsorted(frames, frame_index))
    return min(max(i, 0), frames.shape[0] - 1)


def camera_pose(camera: CameraTrack, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame world→camera pose ``(R (3, 3), t (3,))`` (the nearest stored row)."""
    row = frame_row(camera.frames, frame_index)
    return quat_to_rotation_matrix(camera.rotation_quat[row]), camera.translation[row]


def camera_center(camera: CameraTrack, frame_index: int) -> np.ndarray:
    """World-space optical centre at a frame: ``C = -Rᵀ t`` (where ``X_c = R X_w + t``)."""
    rot, t = camera_pose(camera, frame_index)
    return -rot.T @ t


def project_world_points_with_depth(
    camera: CameraTrack, frame_index: int, world_xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to pixels **and** return camera-space depth + visibility.

    Applies the per-frame world→camera pose (``X_c = R @ X_w + t``) then the pinhole intrinsics.
    A point is visible only if it is in front of the camera (``Z_c > 0``) *and* lands inside the
    image rectangle. Returns ``(uv (N, 2), depth (N,), visible (N,) bool)`` — ``depth`` is the
    camera-space Z (metres), the value a consumer needs for an occlusion/z-buffer test.
    """
    pts = np.asarray(world_xyz, dtype=float).reshape(-1, 3)
    rot, t = camera_pose(camera, frame_index)
    cam = pts @ rot.T + t
    z = cam[:, 2]
    in_front = z > 1e-6
    safe_z = np.where(in_front, z, 1.0)
    # Per-frame, not the track's shared intrinsics: on a clip that zooms, one focal costs
    # 1.65 -> 4.56 px against the paint (camlab, `fan`, 1.59x). `intrinsics_at` returns the shared
    # ones unchanged when the track has a single focal, so this is free where there is no zoom.
    k = camera.intrinsics_at(frame_index)
    u = k.fx * cam[:, 0] / safe_z + k.cx
    v = k.fy * cam[:, 1] / safe_z + k.cy
    on_image = (u >= 0) & (u < k.width) & (v >= 0) & (v < k.height)
    return np.column_stack([u, v]), z, in_front & on_image


def project_world_points(
    camera: CameraTrack, frame_index: int, world_xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to image pixels at a frame; flag which are actually visible.

    Thin wrapper over :func:`project_world_points_with_depth` that drops the depth channel —
    the signature the reprojection overlay has always used. Returns ``(uv (N, 2), visible (N,))``.
    """
    uv, _depth, visible = project_world_points_with_depth(camera, frame_index, world_xyz)
    return uv, visible
