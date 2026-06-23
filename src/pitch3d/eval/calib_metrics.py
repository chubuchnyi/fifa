"""Calibration reprojection metrics — score a homography against SoccerNet pitch-line GT.

The companion to :mod:`pitch3d.eval.datasets_soccernet`: given a predicted image→world homography
``H`` (pixels → field metres, the convention of
:class:`pitch3d.core.scene.field.FieldCalibration`) and a frame's annotated straight pitch lines,
how well does ``H`` place those image points onto their known world lines? Two complementary units,
both the README's metric ("L2 distance between an annotated point and the line it belongs to"):

* **world metres** — project each GT image point through ``H`` and measure its distance to the
  line's world segment. Physically interpretable (how many metres off on the pitch).
* **image pixels** — project the world segment back through ``H⁻¹`` and measure the GT image
  point's distance to that predicted image segment. Comparable in spirit to SoccerNet's pixel
  threshold, on whatever resolution ``H`` was fitted in.

From the pixel errors we derive a per-line accuracy at a pixel threshold (a line counts as correct
iff *all* its annotated points reproject within the threshold). This is a **homography-plane proxy**
for SoccerNet's official ``Completeness × JaC@5`` — we deliberately do **not** call it JaC@5: the
official metric scores full camera parameters (with lens distortion, the three circles, and the
left/right tactical-ambiguity flip), whereas this scores a planar homography over straight lines
only. It is an honest internal quality signal, not a leaderboard number (R-6).

Pure / numpy-only — no decode, no model — so it is unit-testable against a known homography.
"""

from __future__ import annotations

import numpy as np

from .datasets_soccernet import CalibFrameGT


def _apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project ``pts`` ``(N, 2)`` through homography ``h`` ``(3, 3)`` → ``(N, 2)``."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))]) @ h.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return hom[:, :2] / hom[:, 2:3]


def _point_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points ``p`` ``(N, 2)`` to segment ``a``–``b``, clamped to the segment."""
    p = np.asarray(p, dtype=float).reshape(-1, 2)
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:  # degenerate world segment → fall back to point distance
        return np.linalg.norm(p - a, axis=-1)
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=-1)


def frame_world_errors(h: np.ndarray, gt: CalibFrameGT) -> np.ndarray:
    """Per-annotated-point reprojection error in **world metres** for one frame.

    Projects each line's image points through ``h`` and measures distance to that line's world
    segment. Returns a flat ``(n_points,)`` array (empty if the frame has no straight lines).
    """
    errs: list[np.ndarray] = []
    for line in gt.lines:
        world = _apply_h(h, line.image_uv)
        errs.append(_point_segment_dist(world, line.world_a, line.world_b))
    return np.concatenate(errs) if errs else np.empty(0, dtype=float)


def frame_pixel_errors(h: np.ndarray, gt: CalibFrameGT) -> dict[str, np.ndarray]:
    """Per-line reprojection error in **image pixels** for one frame → ``{class: errors(K,)}``.

    Inverts ``h`` (world → image), projects each line's world segment into the image, and measures
    each GT image point's distance to that predicted segment. If ``h`` is singular the frame has no
    usable prediction and every point is scored as ``inf`` (an honest miss, not a crash).
    """
    try:
        h_inv = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return {line.name: np.full(line.image_uv.shape[0], np.inf) for line in gt.lines}
    out: dict[str, np.ndarray] = {}
    for line in gt.lines:
        seg = _apply_h(h_inv, np.stack([line.world_a, line.world_b]))
        d = _point_segment_dist(line.image_uv, seg[0], seg[1])
        out[line.name] = np.where(np.isfinite(d), d, np.inf)
    return out


def frame_metrics(
    h: np.ndarray, gt: CalibFrameGT, thresholds_px: tuple[float, ...] = (5.0, 10.0)
) -> dict[str, float]:
    """Score one frame's homography → world/pixel error summaries + per-line accuracy@threshold.

    ``lines_ok@{t}px`` counts lines whose *every* annotated point reprojects within ``t`` px (the
    line-level accuracy proxy described in the module docstring); divide by ``n_lines`` for the
    fraction. World/pixel RMS/median are over all annotated points in the frame.
    """
    world = frame_world_errors(h, gt)
    px = frame_pixel_errors(h, gt)
    px_all = np.concatenate([v for v in px.values()]) if px else np.empty(0)
    finite_px = px_all[np.isfinite(px_all)]
    out: dict[str, float] = {
        "n_lines": float(gt.n_lines),
        "n_points": float(gt.n_points),
        "rms_m": float(np.sqrt((world**2).mean())) if world.size else float("nan"),
        "median_m": float(np.median(world)) if world.size else float("nan"),
        "rms_px": float(np.sqrt((finite_px**2).mean())) if finite_px.size else float("nan"),
        "median_px": float(np.median(finite_px)) if finite_px.size else float("nan"),
    }
    for t in thresholds_px:
        out[f"lines_ok@{t:g}px"] = float(
            sum(bool(np.all(e <= t)) for e in px.values())
        )
    return out


def _pool_summary(
    world_pool: list[np.ndarray],
    px_pool: list[np.ndarray],
    ok_lines: dict[float, int],
    total_lines: int,
    thresholds_px: tuple[float, ...],
) -> dict[str, float]:
    """Reduce pooled per-point world/pixel errors + per-threshold line-ok counts to a stats dict.

    Shared by the all-frames grid and the ``on_completed`` sub-grid so both report *identical*
    statistics over their respective frame pools (only the pool membership differs).
    """
    world = np.concatenate(world_pool) if world_pool else np.empty(0)
    px_all = np.concatenate(px_pool) if px_pool else np.empty(0)
    finite_px = px_all[np.isfinite(px_all)]
    out: dict[str, float] = {
        "reproj_rms_m": float(np.sqrt((world**2).mean())) if world.size else float("nan"),
        "reproj_median_m": float(np.median(world)) if world.size else float("nan"),
        "reproj_p95_m": float(np.percentile(world, 95)) if world.size else float("nan"),
        "reproj_rms_px": float(np.sqrt((finite_px**2).mean())) if finite_px.size else float("nan"),
        "reproj_median_px": float(np.median(finite_px)) if finite_px.size else float("nan"),
    }
    for t in thresholds_px:
        out[f"line_acc@{t:g}px"] = (ok_lines[t] / total_lines) if total_lines else float("nan")
    return out


def evaluate_calibration(
    frames: list[CalibFrameGT],
    homographies: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    thresholds_px: tuple[float, ...] = (5.0, 10.0),
) -> dict[str, object]:
    """Aggregate calibration quality over a set of SoccerNet frames → a JSON-able summary grid.

    ``homographies`` is ``(T, 3, 3)`` image→world, positionally aligned with ``frames`` (as produced
    by a :class:`FieldCalibrator` over a clip of these frames, in order). World/pixel RMS and median
    are pooled over **all** annotated points across frames (no per-frame cherry-picking); the
    ``line_acc@{t}px`` figures are total correct lines over total GT lines. ``completeness`` is the
    fraction of frames the calibrator was confident about (``confidence > 0``) when a confidence
    vector is supplied — surfacing carried/under-detected frames honestly rather than hiding them.

    When ``confidence`` is given the grid also carries ``n_completed`` and an ``on_completed``
    sub-grid: the same reprojection stats pooled over **only** the confident frames. This matters
    because failed frames hold a degenerate/identity ``H`` that projects pixel coordinates as
    "metres", so their errors are huge-but-finite and dominate the all-frames ``reproj_rms_m`` /
    ``reproj_p95_m``. ``on_completed`` reports the accuracy *where the calibrator actually locked
    on*, kept separate from ``completeness`` (how often it locks on) so neither hides the other.
    """
    n = min(len(frames), homographies.shape[0])
    conf = np.asarray(confidence, dtype=float).reshape(-1)[:n] if confidence is not None else None
    world_pool: list[np.ndarray] = []
    px_pool: list[np.ndarray] = []
    total_lines = 0
    ok_lines = {t: 0 for t in thresholds_px}
    per_frame_lines: list[int] = []
    # Completed-only pools (confidence > 0); populated only when a confidence vector is supplied.
    world_pool_c: list[np.ndarray] = []
    px_pool_c: list[np.ndarray] = []
    total_lines_c = 0
    ok_lines_c = {t: 0 for t in thresholds_px}
    for i in range(n):
        gt = frames[i]
        h = homographies[i]
        world_err = frame_world_errors(h, gt)
        px = frame_pixel_errors(h, gt)
        line_ok = {t: int(sum(bool(np.all(e <= t)) for e in px.values())) for t in thresholds_px}
        world_pool.append(world_err)
        px_pool.extend(px.values())
        total_lines += gt.n_lines
        per_frame_lines.append(gt.n_lines)
        for t in thresholds_px:
            ok_lines[t] += line_ok[t]
        if conf is not None and i < conf.size and conf[i] > 0:
            world_pool_c.append(world_err)
            px_pool_c.extend(px.values())
            total_lines_c += gt.n_lines
            for t in thresholds_px:
                ok_lines_c[t] += line_ok[t]

    grid: dict[str, object] = {
        "n_frames": int(n),
        "total_lines": int(total_lines),
        "mean_lines_per_frame": float(np.mean(per_frame_lines)) if per_frame_lines else 0.0,
    }
    grid.update(_pool_summary(world_pool, px_pool, ok_lines, total_lines, thresholds_px))
    if conf is not None:
        grid["completeness"] = float((conf > 0).mean()) if conf.size else float("nan")
        grid["n_completed"] = int((conf > 0).sum())
        grid["on_completed"] = _pool_summary(
            world_pool_c, px_pool_c, ok_lines_c, total_lines_c, thresholds_px
        )
    return grid
