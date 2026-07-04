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


def apply_stand_structure(
    quilt: np.ndarray,
    *,
    tier_v: float = 0.65,
    tier_halfwidth_v: float = 0.022,
    aisle_period_u: float = 0.024,
    aisle_width_u: float = 0.0035,
    fade_start_v: float = 0.20,
    top_fade: float = 0.50,
) -> np.ndarray:
    """Overlay stand ARCHITECTURE onto the crowd quilt so it reads as stadium tiers, not TV
    static: a dark concourse walkway with a bright railing lip between lower and upper tier
    (``tier_v`` up the rake), dark aisle stairways every ``aisle_period_u`` around the bowl,
    and a fade toward the top rows. All pure luma multipliers on the measured pixels — the
    crowd texture itself stays real.

    Calibrated on the night clip (2026-07-04): stair aisles read 3–5 px dark at 1080p (≈1.3 m
    wide, sub-metre ones vanish after projection+filtering), the stand's vertical luma profile
    dims steadily from bottom (~65–75) to roofline (~33–41) — a whole-rake fade to ~0.55, not a
    mid-rake-only one. IMPORTANT: from a broadcast camera the ads boards occlude the lower HALF
    of the far-stand rake (measured: a black band at bowl-v 0.2–0.5 is invisible; the on-screen
    stand is v≈0.5–1.0), so the tier break must sit in the visible upper half — v 0.65 puts it
    in the lower third of what the camera actually sees, matching the clip.

    v axis: 0 = bottom row of the bowl, 1 = top. The quilt is stored screen-style (numpy row 0
    = top of the stand — the renderer flips it to UV space), so bowl-v = 1 - row_frac. Defaults
    in bowl metres: walkway ≈1.2 m tall, aisles ≈1.3 m wide every ≈9 m (perimeter ≈375 m,
    rake ≈24 m).
    """
    q = np.asarray(quilt, dtype=np.float32).copy()
    h, w = q.shape[:2]
    v = 1.0 - (np.arange(h, dtype=np.float32) + 0.5) / h
    u = (np.arange(w, dtype=np.float32) + 0.5) / w

    gain_v = np.ones(h, dtype=np.float32)
    walk = np.abs(v - tier_v) < tier_halfwidth_v
    gain_v[walk] = 0.28
    # The clip's tier break reads mostly as the continuous BRIGHT railing line, so the lip is
    # wide/strong enough to survive grazing-angle texture filtering (a 1-px line would smear away).
    rail = (v - (tier_v + tier_halfwidth_v) >= 0.0) & (v - (tier_v + tier_halfwidth_v) < 4.0 / h)
    gain_v[rail] = 1.6
    upper = v > fade_start_v
    gain_v[upper] *= 1.0 + (top_fade - 1.0) * (v[upper] - fade_start_v) / max(
        1e-6, 1.0 - fade_start_v
    )

    gain = np.repeat(gain_v[:, None], w, axis=1)
    # Stairways stop at the concourse: the upper tier's run offset half a period so the two
    # tiers don't line up into a see-through grid (they don't in the clip either).
    aisle_lo = np.mod(u, aisle_period_u) < aisle_width_u
    aisle_hi = np.mod(u + aisle_period_u / 2.0, aisle_period_u) < aisle_width_u
    lower = v < tier_v
    gain[np.ix_(lower, aisle_lo)] *= 0.50
    gain[np.ix_(~lower, aisle_hi)] *= 0.50

    return np.clip(q * gain[..., None], 0.0, 1.0)


def assemble_crowd_quilt(
    tile: np.ndarray,
    *,
    width: int = 8192,
    height: int = 512,
    seed: int = 0,
    fan_scale: float = 1.0,
) -> np.ndarray:
    """Stitch one large NON-repeating crowd texture from random crops of the measured tile.

    Repeating the small tile over the bowl (the 40×4 mirror mosaic) reads as a kaleidoscope the
    moment the video is sharpened — the tell is the *periodicity*, not the tile. The quilt keeps
    every pixel measured but kills the period: a canvas the size of the whole unwrapped bowl is
    covered by ~a hundred crops of the tile at random offsets, half of them flipped left-right,
    each with a small brightness jitter, and blended in under a Hann window so patch seams feather
    away. Placement wraps in ``x`` (the around-the-bowl axis), so with the continuous 0–1 unwrap
    and REPEAT extension the wrap seam is blended like any interior seam. The default 16:1 canvas
    matches the bowl's perimeter:rake aspect (~375 m : 24 m at default geometry).

    Deterministic for a given ``seed`` — the manual dial next to ``--crowd-mode`` (auto default =
    quilt, seed 0). The exporter overlays :func:`apply_stand_structure` on top (its
    ``--crowd-structure`` flag). Returns ``(height, width, 3)`` float32 RGB in ``[0, 1]``.

    ``fan_scale`` is quilt-px per tile-px: at the default 1.0 the tile is stitched at NATIVE
    resolution, so the quilt's per-fan grain equals the measured clip grain no matter the canvas
    size. (The first cut sized patches as ``height // 2`` and UPSAMPLED the tile to fit — the
    canvas resolution then cancelled out of the on-screen grain, measured 2026-07-04: doubling
    the canvas left the stand marbled at 7→9 px while the clip grain is ~2.7 px.)
    """
    import cv2

    tile = np.asarray(tile, dtype=np.float32)
    rng = np.random.default_rng(seed)

    sh = max(3, int(round(tile.shape[0] * fan_scale)))
    sw = max(3, int(round(tile.shape[1] * fan_scale)))
    if (sh, sw) != tile.shape[:2]:
        interp = cv2.INTER_AREA if sh < tile.shape[0] else cv2.INTER_LINEAR
        scaled = cv2.resize(tile, (sw, sh), interpolation=interp)
    else:
        scaled = tile
    # Patch < scaled tile leaves vertical slack so crops differ in spectator-row phase.
    ph = min(height, max(2, int(round(sh / 1.4))))
    pw = min(width, sw, max(2, int(round(ph * tile.shape[1] / tile.shape[0]))))

    win = (np.hanning(ph)[:, None] * np.hanning(pw)[None, :]).astype(np.float32) + 1e-4
    acc = np.zeros((height, width, 3), dtype=np.float32)
    wsum = np.zeros((height, width), dtype=np.float32)
    step_y = max(1, ph - ph // 3)
    step_x = max(1, pw - pw // 3)
    ys = list(range(0, height - ph + 1, step_y))
    if ys[-1] != height - ph:
        ys.append(height - ph)
    for y in ys:
        for x in range(0, width, step_x):
            y0 = int(rng.integers(0, sh - ph + 1))
            x0 = int(rng.integers(0, sw - pw + 1))
            patch = scaled[y0 : y0 + ph, x0 : x0 + pw]
            if rng.random() < 0.5:
                patch = patch[:, ::-1]
            patch = patch * rng.uniform(0.92, 1.08)
            cols = (x + np.arange(pw)) % width  # wrap around the bowl
            acc[y : y + ph, cols] += patch * win[..., None]
            wsum[y : y + ph, cols] += win
    return np.clip(acc / np.maximum(wsum, 1e-8)[..., None], 0.0, 1.0).astype(np.float32)
