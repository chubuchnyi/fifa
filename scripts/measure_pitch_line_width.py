#!/usr/bin/env python3
"""Measure how many pixels wide a painted pitch line actually is in a broadcast frame.

This is the measurement that decided R3 (#95). The research brief proposed fitting the two
**edges** of each painted line rather than its centreline: IFAB caps line width at 0.12 m, so each
line would supply two parallel constraints at a known separation. That is only reachable if the two
edges are separately resolvable in the image.

Three independent estimates, because a mask threshold can invent or erase a pixel::

    .venv/bin/python scripts/measure_pitch_line_width.py [VIDEO] [FRAME]

1. **Distance transform** over a paint mask restricted to grass-surrounded, thin, elongated blobs.
   Thickness at a ridge pixel is ``2 * distance-to-edge`` — rotation-invariant, so it does not care
   which way the line runs.
2. **Raw intensity FWHM** across the nearest lines, sampled straight from the greyscale image with
   no mask involved at all. This is the check on (1).
3. **Crops** written next to the numbers so the result can be judged by eye at native resolution.

Outputs land in ``out/r3_lines/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path("out/r3_lines")
DEFAULT_VIDEO = "samples/video/Colombia-1-0-Congo-DR1080p.mp4"
DEFAULT_FRAME = 60

#: Rows above this are stands and advertising boards on our target clip, never pitch.
SKY_ROWS = 180


def line_mask(bgr: np.ndarray) -> np.ndarray:
    """Paint pixels that sit on the field and belong to a thin, elongated blob."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = (hsv[:, :, i].astype(int) for i in range(3))
    grass = ((h > 30) & (h < 95) & (s > 60) & (v > 40)).astype(np.float32)
    onfield = cv2.blur(grass, (61, 61)) > 0.45
    paint = (((v > 140) & (s < 80)) & onfield).astype(np.uint8)
    paint[:SKY_ROWS, :] = 0

    n, lab, stats, _ = cv2.connectedComponentsWithStats(paint, 8)
    keep = np.zeros_like(paint)
    for i in range(1, n):
        _, _, bw, bh, area = stats[i]
        if area < 40:
            continue
        elongation = max(bw, bh) / max(1, min(bw, bh))
        # A painted line is long, thin, and does not fill its bounding box; a player does.
        if elongation >= 3.0 and area / float(bw * bh) < 0.55:
            keep[lab == i] = 1
    return keep


def thickness_by_distance_transform(keep: np.ndarray) -> None:
    dist = cv2.distanceTransform(keep, cv2.DIST_L2, 5)
    peak = cv2.dilate(dist, np.ones((5, 5), np.float32))
    ridge = (keep > 0) & (dist >= peak - 1e-3) & (dist > 0.9)
    ys, xs = np.nonzero(ridge)
    if ys.size == 0:
        print("  no ridge pixels found")
        return
    thick = 2.0 * dist[ys, xs]
    print(f"  ridge samples: {ys.size}")
    print(f"  overall: median {np.median(thick):.2f} px   "
          f"p10 {np.percentile(thick, 10):.2f}   p90 {np.percentile(thick, 90):.2f}")
    print("  by image row (larger row = nearer the camera):")
    for lo, hi in [(180, 300), (300, 450), (450, 600), (600, 750), (750, 900), (900, 1080)]:
        m = (ys >= lo) & (ys < hi)
        if m.sum() < 25:
            print(f"    rows {lo:>4}-{hi:<4} n={int(m.sum()):>5}  (too few)")
            continue
        t = thick[m]
        print(f"    rows {lo:>4}-{hi:<4} n={int(m.sum()):>5}  "
              f"median {np.median(t):>5.2f} px   p90 {np.percentile(t, 90):>5.2f} px")


def fwhm_of_nearest_lines(bgr: np.ndarray, keep: np.ndarray, n_samples: int = 8) -> None:
    """Raw greyscale profiles across the nearest line pixels — no mask in the measurement."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float)
    ys, xs = np.nonzero(keep)
    if ys.size == 0:
        return
    picked: list[tuple[int, int]] = []
    seen: set[int] = set()
    for i in np.argsort(-ys):
        x, y = int(xs[i]), int(ys[i])
        if x in seen or not (30 <= y < gray.shape[0] - 30):
            continue
        seen.add(x)
        picked.append((x, y))
        if len(picked) >= n_samples:
            break
    for x, y in picked:
        col = gray[y - 12:y + 13, x]
        base = np.median(np.concatenate([col[:6], col[-6:]]))
        half = (base + col.max()) / 2.0
        print(f"    x={x:>4} y={y:>4}  grass~{base:>5.1f} peak~{col.max():>5.1f}  "
              f"FWHM={int((col >= half).sum())} px   profile={np.round(col[8:17]).astype(int)}")


def main() -> None:
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FRAME
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {frame_idx} of {video}")

    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"frame{frame_idx:03d}.png"), bgr)
    keep = line_mask(bgr)
    cv2.imwrite(str(OUT / "line_mask.png"), keep * 255)
    print(f"{video} frame {frame_idx}, {bgr.shape[1]}x{bgr.shape[0]}; "
          f"line-like pixels: {int(keep.sum())}\n")

    print("[1] thickness via distance transform")
    thickness_by_distance_transform(keep)
    print("\n[2] raw intensity FWHM across the nearest lines (no mask)")
    fwhm_of_nearest_lines(bgr, keep)

    ys, xs = np.nonzero(keep)
    if ys.size:
        cy, cx = int(ys.max()), int(xs[ys.argmax()])
        x0, y0 = max(0, cx - 45), max(0, cy - 25)
        crop = bgr[y0:y0 + 50, x0:x0 + 90]
        cv2.imwrite(str(OUT / "near_line.png"),
                    cv2.resize(crop, (crop.shape[1] * 8, crop.shape[0] * 8),
                               interpolation=cv2.INTER_NEAREST))
        print(f"\n[3] wrote {OUT / 'near_line.png'} — nearest line at (x={cx}, y={cy}), 8x nearest")


if __name__ == "__main__":
    main()
