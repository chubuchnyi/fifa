"""Project the broadcast clip onto the stadium bowl — the *measured* half of the hybrid backdrop.

The bowl geometry (:mod:`pitch3d.core.scene.stadium`) is a plausible shell; this adapter gives it
real appearance by sampling the broadcast frames through the **solved camera**. For every bowl
vertex visible in a frame we read the pixel it projects to (the core
:func:`~pitch3d.core.scene.projection.project_world_points_with_depth` is the single projection the
overlay and this share), then take the per-vertex **median** across all frames it was visible in —
median rejects a player or the ball briefly crossing a low front-row vertex. Vertices the camera
never saw (its own near stand, end gaps) come back ``covered=False`` for
:func:`~pitch3d.core.scene.stadium.fill_holes_by_copy` to fill. Lazy ``cv2``; numpy otherwise.
"""

from __future__ import annotations

import numpy as np

from ...core.scene.camera import CameraTrack
from ...core.scene.projection import camera_pose, project_world_points_with_depth
from ..io.frames import iter_clip_frames


def _bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinearly sample ``img (H, W, 3)`` at float pixel coords ``uv (N, 2)`` → ``(N, 3)``."""
    h, w = img.shape[:2]
    u = np.clip(uv[:, 0], 0, w - 1.001)
    v = np.clip(uv[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    return (
        img[y0, x0] * (1 - fx) * (1 - fy)
        + img[y0, x1] * fx * (1 - fy)
        + img[y1, x0] * (1 - fx) * fy
        + img[y1, x1] * fx * fy
    )


def bake_backdrop_colors(
    camera: CameraTrack, verts: np.ndarray, video_uri: str
) -> tuple[np.ndarray, np.ndarray]:
    """Sample per-vertex RGB for the bowl from the clip: ``(colors (M, 3) in [0,1], covered (M,))``.

    Reads each frame of ``camera.frames`` at the calibration resolution (the intrinsics' width ×
    height — the clip is decoded then resized to match, so projected pixel coords index it
    directly), projects every vertex, and accumulates the colour where visible. ``colors`` is the
    per-vertex median over the frames it appeared in (black where never seen); ``covered`` the rest.
    """
    import cv2

    verts = np.asarray(verts, dtype=float)
    m = verts.shape[0]
    k = camera.intrinsics
    w, h = int(k.width), int(k.height)
    # (M, F, 3) sample store; NaN = "not visible in this frame" so nanmedian ignores it.
    frames = np.asarray(camera.frames, dtype=int)
    samples = np.full((m, frames.shape[0], 3), np.nan, dtype=np.float32)

    # The solved broadcast camera for our clip is rolled 180° relative to the *raw decoded* video:
    # the pitch model, the SMPL-X bodies and this bowl all project onto the frame turned upside
    # down, not the frame as decoded (verified by overlaying all three). Its image axes therefore
    # read +u = world −X and +v = world +up, so image-up (−R[1]) points *down* in world (−Z). When
    # that happens we rotate each decoded frame 180° into the camera's pixel convention before
    # sampling — else every stand vertex reads the pitch grass on the opposite side of the frame.
    rot0, _ = camera_pose(camera, int(frames[0]))
    upside_down = float(-rot0[1, 2]) < 0.0

    for col, (idx, bgr) in enumerate(iter_clip_frames(video_uri, frames.tolist())):
        if bgr.shape[1] != w or bgr.shape[0] != h:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        if upside_down:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
        uv, _depth, visible = project_world_points_with_depth(camera, int(idx), verts)
        if visible.any():
            samples[visible, col] = _bilinear(rgb, uv[visible])

    covered = np.asarray(np.isfinite(samples).any(axis=(1, 2)), dtype=bool)
    colors = np.zeros((m, 3), dtype=np.float32)
    if covered.any():
        colors[covered] = np.nanmedian(samples[covered], axis=1)
    return colors, covered
