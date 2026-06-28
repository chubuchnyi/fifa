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


def _busiest_window(img: np.ndarray, h_frac: float, w_frac: float) -> np.ndarray:
    """Crop the busiest (most-textured) ``h_frac × w_frac`` sub-rectangle of ``img``.

    Packed crowd is densely *textured* — thousands of small edges from individual spectators; the
    LED ad boards, FIFA signage rings and scoreboard pillars that creep into the band box are broad
    flat panels with only sparse text. We score by edge *density* (fraction of pixels whose gradient
    beats the box median), not edge energy: a few high-contrast signage letters have big energy but
    density, so density keeps the window on the crowd. Rows and columns are scored separately (the
    contamination is roughly axis-aligned: a flat column for a scoreboard pillar, a flat row strip
    for a signage ring).
    """
    h, w = img.shape[:2]
    luma = img.mean(axis=2)
    grad = np.zeros_like(luma)
    grad[:, :-1] += np.abs(np.diff(luma, axis=1))
    grad[:-1, :] += np.abs(np.diff(luma, axis=0))
    dens = (grad > float(np.median(grad)) + 1e-6).astype(np.float32)
    hw = int(np.clip(round(h * h_frac), 2, h))
    ww = int(np.clip(round(w * w_frac), 2, w))
    r0 = int(np.argmax(np.convolve(dens.mean(axis=1), np.ones(hw), "valid"))) if hw < h else 0
    c0 = int(np.argmax(np.convolve(dens.mean(axis=0), np.ones(ww), "valid"))) if ww < w else 0
    return img[r0 : r0 + hw, c0 : c0 + ww].copy()


def extract_crowd_tile(
    camera: CameraTrack,
    verts: np.ndarray,
    param: np.ndarray,
    covered: np.ndarray,
    video_uri: str,
    *,
    height_band: tuple[float, float] = (0.45, 0.78),
    inset: float = 0.1,
    tile_frac: tuple[float, float] = (0.7, 0.45),
) -> np.ndarray:
    """Cut one clean crowd patch from the clip to tile over the bowl: ``(h, w, 3)`` RGB ``[0,1]``.

    The per-vertex bake stretches a single median pixel across the bowl, so the crowd reads blurry;
    the mosaic backdrop instead repeats a *real* image of spectators. We take the camera-seen
    (``covered``) vertices in a mid-height band — high enough to clear the LED boards and grass
    margin the lowest rows sample — project them into the frame that sees the most, and crop the
    robust (percentile) bounding box of those pixels. A final :func:`_busiest_window` pass shrinks
    that box to its most-textured ``tile_frac`` sub-rectangle, dropping any scoreboard / ad-board
    that still intrudes (else it would tile into a row of repeating billboards around the bowl).

    Same rolled-camera convention as :func:`bake_backdrop_colors`: the frame is rotated 180° before
    cropping so the projected pixel coords index it directly.
    """
    import cv2

    verts = np.asarray(verts, dtype=float)
    param = np.asarray(param, dtype=float)
    covered = np.asarray(covered, dtype=bool)
    k = camera.intrinsics
    w, h = int(k.width), int(k.height)
    frames = np.asarray(camera.frames, dtype=int)

    lo, hi = height_band
    band = covered & (param[:, 1] >= lo) & (param[:, 1] <= hi)
    if not band.any():
        band = covered
    band_verts = verts[band]

    # Projection is numpy-only — scan every frame for the one that sees the most band vertices
    # before paying to decode a single image.
    best_idx, best_uv, best_n = int(frames[0]), np.empty((0, 2)), -1
    for idx in frames:
        uv, _depth, visible = project_world_points_with_depth(camera, int(idx), band_verts)
        n = int(visible.sum())
        if n > best_n:
            best_idx, best_uv, best_n = int(idx), uv[visible], n
    if best_uv.shape[0] < 4:
        raise ValueError("no crowd-band vertex is visible in any frame; cannot cut a tile")

    rot0, _ = camera_pose(camera, int(frames[0]))
    upside_down = float(-rot0[1, 2]) < 0.0
    bgr = None
    for _idx, frame_bgr in iter_clip_frames(video_uri, [best_idx]):
        bgr = frame_bgr
    if bgr is None:
        raise ValueError(f"could not decode frame {best_idx} from {video_uri}")
    if bgr.shape[1] != w or bgr.shape[0] != h:
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    if upside_down:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0

    # Percentile box (not min/max) so a few stray verts grazing the roofline or pitch can't drag the
    # crop off the crowd; the same trim doubles as the inset.
    pct = inset * 100.0
    x0, x1 = np.percentile(best_uv[:, 0], [pct, 100.0 - pct])
    y0, y1 = np.percentile(best_uv[:, 1], [pct, 100.0 - pct])
    xa, xb = int(np.clip(round(x0), 0, w - 2)), int(np.clip(round(x1), 1, w - 1))
    ya, yb = int(np.clip(round(y0), 0, h - 2)), int(np.clip(round(y1), 1, h - 1))
    if xb - xa < 2 or yb - ya < 2:
        raise ValueError("crowd-band box collapsed; widen height_band or lower inset")
    return _busiest_window(rgb[ya : yb + 1, xa : xb + 1], tile_frac[0], tile_frac[1])
