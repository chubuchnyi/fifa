#!/usr/bin/env python
"""Player contact-shadow pin: paint a soft elliptical shadow beneath each detected player.

t23 measured gap (2026-07-06, f28 of t21_pinned8): clip has tight elliptical contact
shadows under players (V drop -.029 under feet, -.022 in the fade zone below); our v2v
pass produces smeared blobs with NO distinct contact shadow (contact zone is dark from
player smear, but the shadow signature is missing — flanks even read BRIGHTER than grass
because of v2v spill). Fix at screen scale after everything: detect players by shirt
color (Colombia yellow + Congo azure + white ref/GK), paint a soft-alpha ellipse under
the foot line, multiply BGR by (1 - alpha*strength) so V drops ~.03 on-grass.

Runs LAST in the batch (after stands red-scatter pin). No temporal smoothing needed —
the pin is per-frame deterministic, players' detected boxes shift smoothly.

usage (pod, after stands_red_pin):
  python scripts/player_shadow_pin.py --video pinned7.mp4 --out pinned8.mp4
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np


def player_boxes(bgr: np.ndarray, y_band: tuple[float, float] = (0.40, 0.88)
                 ) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) of each player blob (shirt-color HSV threshold + morphology)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV_FULL)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = (H > 25) & (H < 45) & (S > 100) & (V > 100)      # Colombia
    azure = (H > 130) & (H < 175) & (S > 80) & (V > 60)       # Congo DR
    white = (S < 40) & (V > 180)                              # ref / GK
    mask = (yellow | azure | white).astype(np.uint8) * 255

    h, w = mask.shape
    ymin, ymax = int(y_band[0] * h), int(y_band[1] * h)
    mask[:ymin] = 0
    mask[ymax:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if 40 < area < 4000 and 4 < ww < 60 and 8 < hh < 90:
            boxes.append((int(x), int(y), int(ww), int(hh)))
    return boxes


def grass_mask(bgr: np.ndarray) -> np.ndarray:
    """Float32 mask: 1 where the pixel is UNSHADED grass (green hue, mid-V), 0 elsewhere."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV_FULL)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # HSV_FULL: green ~ 60..100 (out of 256); grass V > .35 (skip v2v halo)
    is_grass = (H > 50) & (H < 110) & (S > 40) & (V > 90)
    return is_grass.astype(np.float32)


def paint_shadows(bgr: np.ndarray, boxes, strength: float = 0.20,
                  ax_w: float = 0.75, ax_h: float = 0.18,
                  feather: int = 5, grass_only: bool = True) -> np.ndarray:
    """Composite soft elliptical shadow beneath each box; darkens BGR multiplicatively.

    grass_only=True: gate the shadow by a green/mid-V grass mask so the pin doesn't
    stack on v2v player-halo pixels (which already read dark).
    """
    h, w = bgr.shape[:2]
    alpha = np.zeros((h, w), np.float32)
    for x, y, ww, hh in boxes:
        cx = x + ww // 2
        cy = y + hh                        # foot line
        rx = max(3, int(ww * ax_w))
        ry = max(2, int(hh * ax_h))
        cv2.ellipse(alpha, (cx, cy), (rx, ry), 0, 0, 360, 1.0, thickness=-1)
    if feather:
        k = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    if grass_only:
        gm = grass_mask(bgr)
        # soften grass gate a touch so the shadow feathers into halo edges
        gm = cv2.GaussianBlur(gm, (5, 5), 0)
        alpha = alpha * gm
    factor = 1.0 - strength * np.clip(alpha, 0.0, 1.0)
    out = bgr.astype(np.float32) * factor[..., None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def shadow_stats(bgr: np.ndarray, boxes) -> str:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV_FULL)
    V = hsv[..., 2].astype(np.float32) / 255.0
    h, w = V.shape
    contact = []
    grass = []
    for x, y, ww, hh in boxes:
        foot = y + hh
        y0, y1 = foot, min(h, foot + 2)
        x0, x1 = max(0, x), min(w, x + ww)
        if y1 > y0 and x1 > x0:
            contact.append(float(np.median(V[y0:y1, x0:x1])))
        gx0 = min(w, x + 4 * ww)
        gx1 = min(w, gx0 + ww)
        if gx1 <= gx0 + 2:
            gx1 = max(0, x - 3 * ww); gx0 = max(0, gx1 - ww)
        gy0, gy1 = min(h, foot - 4), min(h, foot + 4)
        if gy1 > gy0 and gx1 > gx0:
            grass.append(float(np.median(V[gy0:gy1, gx0:gx1])))
    if not contact:
        return "n=0"
    g = float(np.median(grass))
    c = float(np.median(contact))
    return f"n={len(contact)} grass V {g:.3f}  contact V {c:.3f}  delta {c - g:+.3f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--strength", type=float, default=0.20,
                   help="peak darkening at ellipse center (0..1); .20 lands clip's -.029 V drop")
    p.add_argument("--ax-w", type=float, default=0.75, help="ellipse rx / box width")
    p.add_argument("--ax-h", type=float, default=0.18, help="ellipse ry / box height")
    p.add_argument("--feather", type=int, default=5, help="gaussian blur half-kernel px")
    p.add_argument("--y-band", nargs=2, type=float, default=[0.40, 0.88],
                   metavar=("Y0", "Y1"), help="pitch band (skip stands/boards)")
    p.add_argument("--no-grass-only", action="store_true",
                   help="disable grass gating (default: shadow only on green/mid-V pixels)")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    done = 0
    last = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        boxes = player_boxes(bgr, tuple(args.y_band))
        out = paint_shadows(bgr, boxes, args.strength, args.ax_w, args.ax_h,
                            args.feather, grass_only=not args.no_grass_only)
        writer.write(out)
        last = (bgr, out, boxes)
        done += 1
    writer.release()
    cap.release()
    if last is not None:
        bgr, out, boxes = last
        print(f"before: {shadow_stats(bgr, boxes)}", flush=True)
        print(f"after : {shadow_stats(out, boxes)}", flush=True)
    print(f"PLAYER_SHADOW_OK {done}f strength={args.strength} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
