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

from collections.abc import Sequence

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


def _robust_quadfit(
    t: np.ndarray, y: np.ndarray, valid: np.ndarray, *, floor_px: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic fit y(t) with MAD outlier rejection: (coef, kept mask). Raises if too sparse."""
    keep = valid.copy()
    if int(keep.sum()) < 8:
        raise ValueError("too few valid columns for a robust band fit")
    coef = np.polyfit(t[keep], y[keep], 2)
    for _ in range(3):
        resid = y - np.polyval(coef, t)
        mad = max(floor_px, float(np.median(np.abs(resid[keep] - np.median(resid[keep])))))
        new_keep = valid & (np.abs(resid) < 6.0 * mad)
        if int(new_keep.sum()) < 8 or bool(np.all(new_keep == keep)):
            break
        keep = new_keep
        coef = np.polyfit(t[keep], y[keep], 2)
    return coef, keep


def strip_emission(
    strip: np.ndarray, *, target: float = 1.05, lo: float = 1.0, hi: float = 4.0, q: float = 90.0
) -> float:
    """Emission strength that puts the strip's ``q``-quantile V at ``target``.

    Defaults calibrate an LED band — saturate its own bright content and nothing else. A fixed
    strength inverts dark ads: x4 pushed 0.3 panels to 1.2 — clipped to PNG white right
    alongside the glowing text, so the whole band rendered as one featureless bright stripe
    (measured 2026-07-05). Scale so the strip's p90 V lands just past white; everything darker
    keeps its measured level. ``PITCH3D_BOARD_EMISSION`` stays the manual override.
    The fascia window is NOT a glowing element — the same rule at p90→1.05 rendered the whole
    band 1.6x brighter than the clip's dark zone (t13). It calibrates with ``q=50``,
    ``target=0.40``: the walkway-validated emitted level that survives the night grade at the
    clip's V≈0.2, allowed to DIM below x1 when the window catches bright content.
    """
    v = float(np.percentile(np.asarray(strip, dtype=np.float32).max(axis=2), q))
    return float(np.clip(target / max(v, 1e-6), lo, hi))


def dominant_strip_index(strips: Sequence[np.ndarray]) -> int:
    """Index of the strip showing the window's *dominant* ad appearance.

    LED boards rotate sponsor panels within the render window (this clip: dark FIFA panels most
    of the time, a white BANK OF AMERICA moment); appearances separate cleanly by panel level
    (median V — text is a minority of pixels), so pick the strip whose level sits closest to the
    median level across candidates.
    """
    levels = np.array(
        [float(np.median(np.asarray(s, dtype=np.float32).max(axis=2))) for s in strips]
    )
    return int(np.argmin(np.abs(levels - np.median(levels))))


def extract_board_strip(
    camera: CameraTrack,
    board_verts: np.ndarray,
    video_uri: str,
    *,
    strip_height: int = 48,
    frame_override: int | None = None,
    gap_rel: float = 4.6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Cut the LED ad-board run from the clip to wrap around the ring: ``(strip_height, W, 3)``.

    The geometric ring renders as a flat white band, but the clip's boards are LED panels with
    sponsor text ("BANK OF AMERICA") — that rhythm is the realism tell at the grass boundary.
    Three measurements (2026-07-04, night clip) plus the ad-rotation fix (2026-07-05) shape the
    recipe:

    - **Raw-frame sampling.** The solved camera projects onto the 180°-rolled frame; cropping the
      rotated image (the crowd-tile trick) would bake the text in upside-down. Projected coords
      are point-reflected into RAW-frame coords instead, and the strip is cut from the raw frame
      at its native decode resolution — letters come out upright, in reading order, and as sharp
      as the source.
    - **The ring prior only picks the run.** The scene's CameraTrack is an aggregated constant
      camera: its far-touchline projection came out exactly horizontal and ~200 px (≈19 board
      heights) above the real band, which slants ~40 px across the frame — projection can place
      NOTHING here. It still answers WHAT to cut: the farthest visible ring straight (max mean
      depth, span tie-break) at its widest frame — the run facing the broadcast camera — plus the
      on-screen scale of one board height. Selecting verts by raw depth had mixed straights and
      corner arcs into one x-sorted zigzag polyline landing in the crowd.
    - **The band is anchored to the measured grass boundary.** Prior-relative LED search kept
      locking onto the stand's white fascia rail (same bright-unsaturated signature) because the
      real band sat outside any sane prior window. What IS reliable image-side: the boards stand
      immediately above the floodlit pitch. Per column we find the topmost solid run of bright
      saturated green (hedges and dark seats fail the brightness gate), robust-fit that boundary,
      then take the row maximising the box-filtered LED signature — val·(1−sat) — in the thin
      zone just above it. MAD-rejected quadratic fits ride over goalposts and players crossing
      the boards; the band height is the median contiguous high-score run at the fitted centre.
    - **The frame is chosen by TIME, not span.** LED ads rotate within the render window (this
      clip: dark FIFA panels most of it, one white BANK OF AMERICA stretch); the widest-span
      frame is time-blind and cut the minority white ad. Up to 9 evenly-spaced candidate frames
      are cut and the one whose panel level (median V) sits at the candidates' median wins — the
      dominant ad. ``frame_override`` (env ``PITCH3D_BOARD_FRAME`` at the exporter) pins an
      exact clip frame instead.
    - **The fascia band above rides the same cut.** The clip reads boards → dark walkway/fascia
      sandwich (FIFA/GUADALAJARA panels) → crowd, and the ring's walkway band was a flat grey
      (measured 2026-07-05: our crowd mosaic ran almost straight into the boards). The window
      the walkway band physically occupies — ``gap_rel`` board heights starting half a band
      above the fitted LED centre — is cut from the same rectified run, whatever the clip has
      there, and returned as a second strip for the walkway band to wear.

    ``board_verts`` is the ring's board band as built by ``adboard_ring_geometry`` (bottom/top
    interleaved per loop point); ``gap_rel`` is the walkway band height in board heights
    (``gap / height`` of the ring). Returns ``(strip, fascia, frame_index)``: image-style rows
    (row 0 = strip top) RGB ``[0, 1]`` plus the clip frame the cuts came from.
    """
    pairs = np.asarray(board_verts, dtype=float).reshape(-1, 2, 3)  # (n, [bottom, top], 3)
    k = camera.intrinsics
    w, h = int(k.width), int(k.height)
    frames = np.asarray(camera.frames, dtype=int)

    bx, by = pairs[:, 0, 0], pairs[:, 0, 1]
    hx, hy = float(np.abs(bx).max()), float(np.abs(by).max())
    sides = [
        np.abs(by - hy) < 1e-3,
        np.abs(by + hy) < 1e-3,
        np.abs(bx - hx) < 1e-3,
        np.abs(bx + hx) < 1e-3,
    ]

    best = None  # ((depth bin, span px), side index)
    cand: list[list[tuple[float, int, np.ndarray, np.ndarray]]] = [[] for _ in sides]
    for idx in frames:
        uv_all_b, d_b, vis_b = project_world_points_with_depth(camera, int(idx), pairs[:, 0])
        uv_all_t, _d_t, vis_t = project_world_points_with_depth(camera, int(idx), pairs[:, 1])
        ok = vis_b & vis_t
        for si, side in enumerate(sides):
            m = ok & side
            if int(m.sum()) < 8:
                continue
            span = float(uv_all_b[m, 0].max() - uv_all_b[m, 0].min())
            if span < 32.0:
                continue
            # Depth quantised to 5 m bins: the far straight beats the near one outright, while
            # frames of the SAME straight compete on visible span.
            key = (round(float(d_b[m].mean()) / 5.0), span)
            cand[si].append((span, int(idx), uv_all_b[m], uv_all_t[m]))
            if best is None or key > best[0]:
                best = (key, si)
    if best is None:
        raise ValueError("no ad-board straight is visible in any frame; cannot cut a strip")
    cands = sorted(cand[best[1]], key=lambda c: c[1])
    best_span = max(c[0] for c in cands)
    cands = [c for c in cands if c[0] >= 0.6 * best_span]

    if frame_override is not None:
        picks = [c for c in cands if c[1] == int(frame_override)]
        if not picks:
            have = [c[1] for c in cands]
            raise ValueError(
                f"frame_override={frame_override} does not see the chosen run wide enough; "
                f"usable frames: {have[0]}..{have[-1]} ({len(have)} candidates)"
            )
    else:
        step = max(1, len(cands) // 9)
        picks = cands[::step][:9]

    rot0, _ = camera_pose(camera, int(frames[0]))
    reflect = float(-rot0[1, 2]) < 0.0  # rolled camera: reflect coords, don't rotate the image

    strips: list[np.ndarray] = []
    fascias: list[np.ndarray] = []
    kept: list[int] = []
    by_idx = {c[1]: c for c in picks}
    for idx, bgr in iter_clip_frames(video_uri, sorted(by_idx)):
        _span, _i, uv_b, uv_t = by_idx[int(idx)]
        if reflect:
            uv_b = np.float32([w - 1, h - 1]) - uv_b
            uv_t = np.float32([w - 1, h - 1]) - uv_t
        try:
            strip, fascia = _cut_run_strip(bgr, uv_b, uv_t, w, h, strip_height, gap_rel)
            strips.append(strip)
            fascias.append(fascia)
            kept.append(int(idx))
        except ValueError:
            if len(by_idx) == 1:  # a pinned frame must cut or fail loudly
                raise
    if not strips:
        raise ValueError("LED band fit failed on every candidate frame")
    j = dominant_strip_index(strips)
    return strips[j], fascias[j], kept[j]


def _cut_run_strip(
    bgr: np.ndarray, uv_b: np.ndarray, uv_t: np.ndarray, w: int, h: int, strip_height: int,
    gap_rel: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One frame's cut: rectify the run, anchor to the grass boundary, sample the LED band
    plus the walkway/fascia window sitting ``gap_rel`` board heights above it."""
    import cv2

    dh, dw = bgr.shape[:2]
    scale = np.float32([dw / w, dh / h])  # cut at native decode resolution, not intrinsics grid
    uv_b = uv_b.astype(np.float32) * scale
    uv_t = uv_t.astype(np.float32) * scale
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0

    order = np.argsort(uv_b[:, 0])
    xb, yb = uv_b[order, 0], uv_b[order, 1]
    ot = np.argsort(uv_t[:, 0])
    xt, yt = uv_t[ot, 0], uv_t[ot, 1]
    width = int(round(float(xb[-1] - xb[0])))
    if width < 48:
        raise ValueError("ad-board run projects too small; check ring geometry vs the camera")
    xs = np.linspace(float(xb[0]), float(xb[-1]), width, dtype=np.float32)
    y_mid = 0.5 * (np.interp(xs, xb, yb) + np.interp(xs, xt, yt)).astype(np.float32)
    h_prior = max(4.0, float(np.median(np.interp(xs, xb, yb) - np.interp(xs, xt, yt))))

    # Window: headroom above the prior line, then all the way down to the frame bottom — the
    # grass boundary is found inside it, wherever the aggregated camera's residual pushed it.
    # The fascia cut reaches gap_rel band heights above the fitted centre, so the headroom
    # grows with it (the prior usually sits above the real band, but don't rely on that).
    up = int(round(max(6.0, gap_rel + 1.5) * h_prior))
    down = int(np.ceil(float(dh - 1 - y_mid.min())))
    offs = np.arange(-up, max(down, up) + 1, dtype=np.float32)  # 1 px row spacing
    map_y = y_mid[None, :] + offs[:, None]
    map_x = np.broadcast_to(xs[None, :], map_y.shape).copy()
    rect = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    n_rows = rect.shape[0]

    val = rect.max(axis=2)
    sat = (val - rect.min(axis=2)) / np.maximum(val, 1e-6)
    led = val * (1.0 - sat)
    kk = max(3, int(round(h_prior)) | 1)
    sm = cv2.boxFilter(led, -1, (1, kk), borderType=cv2.BORDER_REPLICATE)

    # Topmost solid run of floodlit grass per column -> pitch boundary. val > 0.3 keeps night
    # hedges and green seats out; k2 (~2 board heights) demands a run, not a stray green pixel.
    hsv = cv2.cvtColor(rect, cv2.COLOR_RGB2HSV)  # float32 RGB in -> H in [0, 360)
    grass = (
        (hsv[..., 0] > 70.0) & (hsv[..., 0] < 170.0) & (hsv[..., 1] > 0.25) & (val > 0.3)
    ).astype(np.float32)
    k2 = max(3, int(round(2.0 * h_prior)) | 1)
    g_sm = cv2.boxFilter(grass, -1, (1, k2), borderType=cv2.BORDER_REPLICATE)
    solid = g_sm > 0.6
    have = solid.any(axis=0)
    y_edge = np.argmax(solid, axis=0).astype(np.float32) - 0.1 * k2  # crossing -> run top
    t = np.arange(width, dtype=np.float32) / max(1.0, width - 1.0)
    g_coef, g_keep = _robust_quadfit(t, y_edge, have)
    if int(g_keep.sum()) < 0.3 * width:
        raise ValueError("grass boundary not found under the ad-board run")
    g_fit = np.polyval(g_coef, t).astype(np.float32)

    # LED band centre in the thin zone just above the boundary.
    rr = np.arange(n_rows, dtype=np.float32)[:, None]
    zone = (rr >= (g_fit - 3.2 * h_prior)[None, :]) & (rr <= (g_fit + 0.2 * h_prior)[None, :])
    r_hat = np.argmax(np.where(zone, sm, -1.0), axis=0).astype(np.float32)
    coef, keep = _robust_quadfit(t, r_hat, zone.any(axis=0))
    if int(keep.sum()) < 0.3 * width:
        raise ValueError("LED band fit failed; boards not found in the frame")
    r_fit = np.polyval(coef, t).astype(np.float32)

    ri = np.clip(np.round(r_fit).astype(int), 0, rect.shape[0] - 1)
    hmax = int(round(3.0 * h_prior))
    heights = []
    for c in range(0, width, max(1, width // 200)):
        prof = sm[:, c]
        r = int(ri[c])
        thr = 0.5 * (float(prof[r]) + float(np.median(prof)))
        a = b = r
        while a > 0 and r - a < hmax and prof[a - 1] >= thr:
            a -= 1
        while b < len(prof) - 1 and b - r < hmax and prof[b + 1] >= thr:
            b += 1
        if b > a:
            heights.append(b - a + 1)
    h_band = float(np.median(heights)) if heights else h_prior
    h_band = float(np.clip(h_band, 0.5 * h_prior, 3.0 * h_prior))

    tt = (np.arange(strip_height, dtype=np.float32) + 0.5) / strip_height
    smap_y = (r_fit - 0.5 * h_band)[None, :] + (tt * h_band)[:, None]
    smap_x = np.broadcast_to(np.arange(width, dtype=np.float32)[None, :], smap_y.shape).copy()
    strip = cv2.remap(rect, smap_x, smap_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # The walkway band's own window: gap_rel board heights ending at the board top edge. Same
    # px-per-metre as the fitted band, so the cut lands exactly where the ring geometry renders.
    fh = max(8, int(round(strip_height * gap_rel)))
    ft = (np.arange(fh, dtype=np.float32) + 0.5) / fh
    fmap_y = (r_fit - (0.5 + gap_rel) * h_band)[None, :] + (ft * (gap_rel * h_band))[:, None]
    fmap_x = np.broadcast_to(np.arange(width, dtype=np.float32)[None, :], fmap_y.shape).copy()
    fascia = cv2.remap(rect, fmap_x, fmap_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return strip.astype(np.float32), fascia.astype(np.float32)


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
    contrast: float = 1.0,
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

    ``contrast`` scales per-pixel luma deviation about the quilt's median (chroma preserved),
    applied AFTER the blend: the Hann overlap-averaging measurably eats the bright-fan tail
    (tile frac(V>med+.2) .117 → quilt .089 on this clip, 2026-07-05), and the stands tone pin
    downstream is a *median* pin — level changes cancel, only distribution SHAPE survives to
    the final. 1.0 = off (default; auto), >1 restores/boosts fans-vs-background contrast.
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
    quilt = acc / np.maximum(wsum, 1e-8)[..., None]
    if contrast != 1.0:
        luma = quilt.mean(axis=2)
        med = float(np.median(luma))
        factor = (med + (luma - med) * contrast) / np.maximum(luma, 1e-6)
        quilt = quilt * factor[..., None]
    return np.clip(quilt, 0.0, 1.0).astype(np.float32)
