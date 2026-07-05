#!/usr/bin/env python
"""Stands red-scatter pin: re-seed the crowd's scattered red fans at SCREEN scale.

The clip's stands are 3.6% strict-red scattered fans (median blob 3.2 px @1080p — sub-pixel
"pepper" at our novel-view scale, where the stands read ~4x smaller). Texture-space red dies
before the screen (measured twice, t19 2026-07-05): quilt clusters at an honest anisotropic
screen-scale sim = 5.4% red, yet the Cycles render + denoise lands 0.001 at beauty, and the
Wan prompt only adds ~0.8% back — every upstream gate (render minification, denoiser, v2v
repaint, tone pins) eats it. So the recolour happens HERE, after all of them: deterministic
luma-preserving chroma swap of scattered fan-sized specks inside the stands band, positions
static across frames (the deliverable camera is fixed; fans don't move). Auto: measure the
band's current strict-red on a mid frame, scatter the shortfall to --target (clip-measured
0.036); manual --frac overrides.

usage (pod, after the panel-row pin — the tone pins would re-amber the red):
  python scripts/stands_red_pin.py --video pinned6.mp4 --out pinned7.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pitch3d.adapters.render.stadium_backdrop import scatter_fan_recolor  # noqa: E402


def strict_red_frac(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV_FULL).astype(np.float32)
    hue = hsv[..., 0] * (360.0 / 255.0)
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    red = ((hue > 330.0) | (hue < 10.0)) & (sat > 0.40) & (val > 0.10)
    return float(red.mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--roi", nargs=4, type=float, default=[0.08, 0.32, 0.0, 1.0],
                   metavar=("Y0", "Y1", "X0", "X1"),
                   help="stands band, fractions of the frame (matches the stands tone pin)")
    p.add_argument("--target", type=float, default=0.036,
                   help="strict-red fraction to land in the band (clip f28 measured 0.036)")
    p.add_argument("--frac", type=float, default=-1.0,
                   help="manual scatter fraction; <0 = auto (target - measured current)")
    p.add_argument("--diam", nargs=2, type=int, default=[2, 5],
                   help="speck diameter range px (clip: median 3.2 px @1080p scales to ~1-3)")
    p.add_argument("--rgb", nargs=3, type=float, default=[0.50, 0.08, 0.08])
    p.add_argument("--luma-cap", type=float, default=0.35,
                   help="max preserved luma for a swapped speck (clip red V p75 = 0.31; a "
                        "dark-red shirt cannot glow at a bright fan's V); <=0 disables")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    y0, y1, x0, x1 = args.roi
    ry0, ry1 = int(y0 * h), int(y1 * h)
    rx0, rx1 = int(x0 * w), int(x1 * w)

    frac = args.frac
    if frac < 0.0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
        ok, mid = cap.read()
        if not ok:
            sys.exit("cannot read mid frame for auto measure")
        current = strict_red_frac(mid[ry0:ry1, rx0:rx1])
        frac = max(0.0, args.target - current)
        print(f"stands_red_pin auto: band strict-red {current:.3f} -> target "
              f"{args.target:.3f}, scatter frac {frac:.3f}", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    done = 0
    last = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if frac > 0.0:
            band = bgr[ry0:ry1, rx0:rx1, ::-1].astype(np.float32) / 255.0
            # same seed every frame -> identical speck positions (static camera, seated fans)
            band = scatter_fan_recolor(
                band, frac=frac, rgb=tuple(args.rgb), seed=args.seed,
                diam_range=tuple(args.diam),
                luma_cap=args.luma_cap if args.luma_cap > 0.0 else None)
            bgr[ry0:ry1, rx0:rx1] = (band[:, :, ::-1] * 255.0).astype(np.uint8)
        writer.write(bgr)
        last = bgr
        done += 1
    writer.release()
    cap.release()
    if last is not None:
        print(f"stands_red_pin: band strict-red now "
              f"{strict_red_frac(last[ry0:ry1, rx0:rx1]):.3f}", flush=True)
    print(f"STANDS_RED_PIN_OK {done}f frac={frac:.3f} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
