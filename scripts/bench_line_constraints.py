#!/usr/bin/env python3
"""Do PnLCalib's discarded line detections actually improve the DLT solve? (R3, #95)

R3's original form — fitting the two *edges* of each painted line — was rejected on measurement:
``scripts/measure_pitch_line_width.py`` found pitch paint is 2 px wide (raw FWHM 1-2 px), a
single-lobe PSF from which two independent edge positions cannot be recovered. See ADR-0012.

What survived is cheaper and does not need the paint resolved at all. PnLCalib runs a **line head**
on every frame regardless — it is what completes occluded keypoints — and the DLT path then threw
its output away. A detected line gives image points known to lie *somewhere* on a known world line:
one linear DLT row each (``lᵀ·H·x = 0``) against a correspondence's two, in the same 9 unknowns.

This measures whether that is worth having, locally, with no GPU and no downloaded dataset::

    PYTHONPATH=src .venv/bin/python scripts/bench_line_constraints.py

The regime under test is the one that actually bites. On the target clip PnLCalib returns 10-11
keypoints per frame at confidence ~0.61 — keypoints are line *intersections*, so a tight broadcast
zoom that shows plenty of paint can still show few corners. So the sweep is over **keypoint count**,
holding the line evidence at what the same detector would have already produced.

Scored with the project's own :func:`~pitch3d.eval.calib_metrics.evaluate_calibration`, so the
numbers are directly comparable to the B1 SoccerNet figure (0.236 m).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np

from pitch3d.adapters.models.calibration import _apply_homography, solve_homography_ransac
from pitch3d.core.scene.pitch import world_line_from_segment
from pitch3d.eval.calib_metrics import evaluate_calibration
from pitch3d.eval.datasets_soccernet import synthetic_calib_frames

N_FRAMES = 24
TRIALS = 12
#: Keypoint counts to sweep. The target clip sits at 10-11; 4 is the bare DLT minimum.
KP_COUNTS = (4, 5, 6, 8, 10, 14)
#: Detector localisation noise (px). PnLCalib's heatmap decode is at 960x540 then upscaled, so
#: even a perfect argmax lands ~2 px out at 1920x1080.
NOISE_PX = 2.0
#: Points sampled per detected line. PnLCalib's line head emits the two extremities.
PTS_PER_LINE = 2


@dataclass(frozen=True)
class LineNoise:
    """How good the line head is assumed to be. The gap between the two presets is the finding.

    Attributes:
        label: Name for the printed table.
        lines_per_frame: Straight lines detected per frame. The synthetic template makes all 17
            visible, which real broadcast framing never does — a tight shot shows a touchline, the
            halfway line and part of a box.
        bias_px: Error **shared by every point on one line**. A stripe detector localises the whole
            painted line slightly off; that offset cannot be averaged out by sampling more points
            along it, so it is the term that actually limits what lines can buy.
        jitter_px: Independent per-point error on top. This one *does* average away.
    """

    label: str
    lines_per_frame: int
    bias_px: float
    jitter_px: float


#: Modelling line error as iid per point, with every line visible, is the flattering assumption —
#: kept here because the gap to REALISTIC is what the whole benchmark turns on.
OPTIMISTIC = LineNoise("optimistic (all lines, iid noise)", 17, 0.0, 2.0)
REALISTIC = LineNoise("realistic (5 lines, correlated)", 5, 2.0, 1.0)


def _visible_intersections(
    frame, h_true: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Where the frame's annotated lines cross → the image↔world correspondences a kp head finds.

    PnLCalib keypoints *are* pitch-line intersections, so deriving them this way keeps the two
    evidence types honestly coupled: a frame showing few corners shows few keypoints, exactly the
    coupling that makes the sweep below meaningful.
    """
    w2i = np.linalg.inv(h_true)
    lines = [(ln.name, world_line_from_segment(ln.world_a, ln.world_b)) for ln in frame.lines]
    uv, world = [], []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a, b = lines[i][1], lines[j][1]
            det = a[0] * b[1] - a[1] * b[0]
            if abs(det) < 1e-9:  # parallel
                continue
            p = np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2]]) / det
            if abs(p[0]) > 53.0 or abs(p[1]) > 35.0:  # off the pitch: not a real landmark
                continue
            img = _apply_homography(w2i, p[None])[0]
            if 0 <= img[0] < width and 0 <= img[1] < height:
                uv.append(img)
                world.append(p)
    return np.asarray(uv, dtype=float).reshape(-1, 2), np.asarray(world, dtype=float).reshape(-1, 2)


def _line_observations(
    frame, rng: np.random.Generator, noise: LineNoise
) -> tuple[np.ndarray, np.ndarray]:
    """The line head's contribution: noisy image points on detected lines + each line's world eqn.

    Only ``noise.lines_per_frame`` of the visible lines are detected, and each carries one shared
    positional bias plus small per-point jitter.
    """
    usable = [ln for ln in frame.lines if ln.image_uv.shape[0] >= 2]
    if not usable:
        return np.empty((0, 2)), np.empty((0, 3))
    n = min(noise.lines_per_frame, len(usable))
    chosen = rng.choice(len(usable), size=n, replace=False)
    uv, abc = [], []
    for i in chosen:
        ln = usable[i]
        line = world_line_from_segment(ln.world_a, ln.world_b)
        bias = rng.normal(0, noise.bias_px, 2)
        pick = rng.choice(ln.image_uv.shape[0], size=min(PTS_PER_LINE, ln.image_uv.shape[0]),
                          replace=False)
        for k in pick:
            uv.append(ln.image_uv[k] + bias + rng.normal(0, noise.jitter_px, 2))
            abc.append(line)
    return np.asarray(uv, dtype=float).reshape(-1, 2), np.asarray(abc, dtype=float).reshape(-1, 3)


def _sweep(noise: LineNoise) -> list[dict]:
    frames, true_h = synthetic_calib_frames(n_frames=N_FRAMES, seed=0)
    width, height = frames[0].width, frames[0].height
    rows: list[dict] = []

    for n_kp in KP_COUNTS:
        pts_only, with_lines, n_solved, n_lines_used = [], [], 0, []
        for trial in range(TRIALS):
            rng = np.random.default_rng(1000 + trial)
            h_pts, h_lin, keep = [], [], []
            for f, frame in enumerate(frames):
                uv, world = _visible_intersections(frame, true_h[f], width, height)
                if uv.shape[0] < n_kp:
                    continue
                idx = rng.choice(uv.shape[0], size=n_kp, replace=False)
                uv_n = uv[idx] + rng.normal(0, NOISE_PX, (n_kp, 2))
                l_uv, l_abc = _line_observations(frame, rng, noise)
                try:
                    a, _ = solve_homography_ransac(uv_n, world[idx], threshold=1.0)
                    b, _ = solve_homography_ransac(
                        uv_n, world[idx], threshold=1.0, line_uv=l_uv, line_abc=l_abc
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue
                h_pts.append(a)
                h_lin.append(b)
                keep.append(frame)
                n_lines_used.append(l_uv.shape[0])
            if not keep:
                continue
            n_solved = len(keep)
            for store, hs in ((pts_only, h_pts), (with_lines, h_lin)):
                grid = evaluate_calibration(keep, np.stack(hs), thresholds_px=(5.0,))
                store.append((grid["reproj_median_m"], grid["reproj_p95_m"], grid["line_acc@5px"]))

        if not pts_only:
            continue
        p, w = np.asarray(pts_only, dtype=float), np.asarray(with_lines, dtype=float)
        rows.append({
            "n_kp": n_kp,
            "frames_solved": n_solved,
            "mean_line_obs": round(float(np.mean(n_lines_used)), 1),
            "median_m_points_only": round(float(np.median(p[:, 0])), 4),
            "median_m_with_lines": round(float(np.median(w[:, 0])), 4),
            "p95_m_points_only": round(float(np.median(p[:, 1])), 4),
            "p95_m_with_lines": round(float(np.median(w[:, 1])), 4),
            "line_acc5px_points_only": round(float(np.median(p[:, 2])), 4),
            "line_acc5px_with_lines": round(float(np.median(w[:, 2])), 4),
        })
    return rows


def _under_determined(noise: LineNoise) -> dict:
    """The frames points alone cannot solve at all: 2-3 keypoints, which lines rescue."""
    frames, true_h = synthetic_calib_frames(n_frames=N_FRAMES, seed=0)
    width, height = frames[0].width, frames[0].height
    rng = np.random.default_rng(7)
    hs, keep = [], []
    for f, frame in enumerate(frames):
        uv, world = _visible_intersections(frame, true_h[f], width, height)
        if uv.shape[0] < 3:
            continue
        idx = rng.choice(uv.shape[0], size=3, replace=False)
        uv_n = uv[idx] + rng.normal(0, NOISE_PX, (3, 2))
        l_uv, l_abc = _line_observations(frame, rng, noise)
        try:
            h, _ = solve_homography_ransac(
                uv_n, world[idx], threshold=1.0, line_uv=l_uv, line_abc=l_abc
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        hs.append(h)
        keep.append(frame)
    if not keep:
        return {"note": "no frame had 3 visible intersections"}
    grid = evaluate_calibration(keep, np.stack(hs), thresholds_px=(5.0,))
    return {
        "n_kp": 3,
        "frames_solved": len(keep),
        "points_only": "UNSOLVABLE (3 points = 6 DLT rows, need 8)",
        "median_m_with_lines": round(float(grid["reproj_median_m"]), 4),
        "p95_m_with_lines": round(float(grid["reproj_p95_m"]), 4),
        "line_acc5px_with_lines": round(float(grid["line_acc@5px"]), 4),
    }


def _report(noise: LineNoise) -> list[dict]:
    rows = _sweep(noise)
    print(f"\n=== {noise.label} ===")
    print(f"{noise.lines_per_frame} lines/frame x {PTS_PER_LINE} pts, per-line bias "
          f"{noise.bias_px} px + {noise.jitter_px} px jitter\n")
    print(f"{'n_kp':>5} {'lines':>6} | {'median m':>19} | {'p95 m':>19} | {'line_acc@5px':>19}")
    print(f"{'':>5} {'obs':>6} | {'points':>9} {'+lines':>9} | "
          f"{'points':>9} {'+lines':>9} | {'points':>9} {'+lines':>9}")
    print("-" * 78)
    for r in rows:
        gain = (r["median_m_points_only"] - r["median_m_with_lines"]) / max(
            r["median_m_points_only"], 1e-9)
        print(f"{r['n_kp']:>5} {r['mean_line_obs']:>6} | "
              f"{r['median_m_points_only']:>9.4f} {r['median_m_with_lines']:>9.4f} | "
              f"{r['p95_m_points_only']:>9.4f} {r['p95_m_with_lines']:>9.4f} | "
              f"{r['line_acc5px_points_only']:>9.3f} {r['line_acc5px_with_lines']:>9.3f}"
              f"   ({gain:+.0%})")
    print("\nframes points alone cannot solve at all:")
    print(json.dumps(_under_determined(noise), indent=2))
    return rows


def main() -> None:
    print(f"synthetic broadcast frames: {N_FRAMES}, trials: {TRIALS}, "
          f"keypoint noise {NOISE_PX} px iid")
    out = {n.label: _report(n) for n in (OPTIMISTIC, REALISTIC)}
    print("\nQuote the REALISTIC row: a line detector's error is shared along the line, and the "
          "gap between the two tables is how much a benchmark can flatter itself by forgetting it.")
    json.dump(out, sys.stdout)
    print()


if __name__ == "__main__":
    main()
