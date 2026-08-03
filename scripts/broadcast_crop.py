#!/usr/bin/env python3
"""Crop a phone/fan clip down to the broadcast framing the calibrator was built for.

The calibrator wants a frame that is mostly pitch. A vertical fan clip is mostly stand, sky and
scoreboard, with the pitch as a band across part of it — PnLCalib then finds too few keypoints and
the #125 gate refuses the run. Measured on ``14604731_1080_1920_30fps.mp4`` (1080x1920, 355 f):
grass covers 27.8% of the frame and only starts at y=1276. Raw, the pipeline solved 0/8 calibration
frames; cropped to the grass band it solved 8/8 at confidence 0.524.

The band is *measured*, not assumed — a fan can hold the phone any way — and the measurement is
overridable, because a clip whose pitch is half-occluded by a crowd will mis-measure and the
operator's eye should win:

    python scripts/broadcast_crop.py --clip in.mp4                     # measure and print only
    python scripts/broadcast_crop.py --clip in.mp4 --out crop.mp4      # measure and encode
    python scripts/broadcast_crop.py --clip in.mp4 --out crop.mp4 --rect 1080:608:0:1312

This is a *pre*-processing step: it changes which pixels the pipeline sees, so the calibration it
produces belongs to the cropped clip. Feed the same cropped file to every downstream stage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import cv2
import numpy as np

#: Broadcast-green in HSV. Wide on hue because floodlights push the grass cool and the mown
#: stripes split it in two; the saturation floor is what keeps grey stand out of the mask.
GRASS_HSV_LO = (30, 40, 40)
GRASS_HSV_HI = (95, 255, 255)


def grass_row_cover(clip: str, n_samples: int = 11) -> tuple[np.ndarray, int, int, float]:
    """Per-row grass fraction averaged over ``n_samples`` frames, plus (w, h, overall fraction)."""
    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows, taken = np.zeros(h, dtype=float), 0
    for i in np.linspace(0, max(total - 1, 0), n_samples, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(GRASS_HSV_LO), np.array(GRASS_HSV_HI))
        rows += mask.mean(axis=1) / 255.0
        taken += 1
    cap.release()
    if not taken:
        raise SystemExit(f"read no frames from {clip}")
    rows /= taken
    return rows, w, h, float(rows.mean())


def longest_band(cover: np.ndarray, min_cover: float) -> tuple[int, int]:
    """The longest run of rows whose grass fraction clears ``min_cover`` — [y0, y1)."""
    hot = cover >= min_cover
    if not hot.any():
        raise SystemExit(
            f"no row reaches {min_cover:.0%} grass (max {cover.max():.1%}) — "
            "lower --min-cover or pass --rect"
        )
    best = cur = (0, 0)
    for y, on in enumerate(hot):
        cur = (cur[0], y + 1) if on else (y + 1, y + 1)
        if cur[1] - cur[0] > best[1] - best[0]:
            best = cur
    return best


def crop_rect(w: int, h: int, band: tuple[int, int], aspect: float) -> tuple[int, int, int, int]:
    """The widest ``aspect`` window that fits the frame, centred on the grass band."""
    y0, y1 = band
    cw = w
    ch = min(int(round(cw / aspect)) // 2 * 2, h)
    cw = min(int(round(ch * aspect)) // 2 * 2, w)
    y = int(round((y0 + y1) / 2 - ch / 2))
    return cw, ch, (w - cw) // 2, max(0, min(y, h - ch))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="", help="write the cropped mp4 here (default: measure only)")
    ap.add_argument("--rect", default="", help="override the measurement: w:h:x:y in source pixels")
    ap.add_argument("--aspect", type=float, default=16 / 9)
    ap.add_argument("--scale", default="1920:1080", help="output size, or 'none' to keep the crop")
    ap.add_argument("--min-cover", type=float, default=0.25, help="grass fraction that marks pitch")
    ap.add_argument("--samples", type=int, default=11)
    args = ap.parse_args()

    cover, w, h, overall = grass_row_cover(args.clip, args.samples)
    y0, y1 = longest_band(cover, args.min_cover)
    print(f"clip      {args.clip}  {w}x{h}")
    print(f"grass     {overall:.1%} of frame · band y={y0}..{y1} ({y1 - y0} px, "
          f"{y0 / h:.0%}..{y1 / h:.0%} down)")

    if args.rect:
        cw, ch, cx, cy = (int(v) for v in args.rect.split(":"))
        print(f"crop      {cw}x{ch}+{cx}+{cy}  (manual --rect, measurement overridden)")
    else:
        cw, ch, cx, cy = crop_rect(w, h, (y0, y1), args.aspect)
        print(f"crop      {cw}x{ch}+{cx}+{cy}  (measured)")
    kept = float(cover[cy:cy + ch].mean())
    print(f"kept      {kept:.1%} grass inside the crop (was {overall:.1%} over the whole frame)")

    vf = f"crop={cw}:{ch}:{cx}:{cy}"
    if args.scale != "none":
        vf += f",scale={args.scale}:flags=lanczos"
    if not args.out:
        print(f"\nffmpeg -i {args.clip} -vf {vf} -c:v libx264 -crf 16 -preset medium -an out.mp4")
        return 0
    cmd = ["ffmpeg", "-y", "-i", args.clip, "-vf", vf,
           "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-an", args.out]
    print("\n$ " + " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
