#!/usr/bin/env python
"""Stands grain-soften pin: land the crowd band's TEXTURE statistics on the clip's.

t20 measured gap (2026-07-05, f28): our stands band reads as saturated "lego confetti" —
luma local-contrast 2.4x the clip's (med .030 vs .013), chroma grain 2.6x, saturated mass
frac(S>.5) .71 vs .43. The tone pins (stages 10/12) land MEDIANS but not the distribution
shape; no stage touches micro-contrast. Real crowd at bowl distance = optics + sensor blur;
ours = crisp quilt blocks re-sharpened by v2v+SeedVR2. Fix at SCREEN scale, after all gates:

  1) blur-blend  out = G9(band) + keep*(band - G9)   -> luma lc med .0316 -> .0144 (clip .0134)
  2) S quantile map to the clip band's S distribution -> S med/p90/frac>.5 all land on clip
     (a knee can't: matching frac(S>.5) undershoots p90 — the clip has its own saturated tail)

The LUT is computed ONCE (mid frame of ours x a clip reference frame) and applied per frame:
static camera, seated fans -> temporally stable. ROI edges feather vertically (no seam).
Runs BEFORE the red-scatter pin so the re-seeded specks stay crisp.

usage (pod, after the panel-row pin, before stands_red_pin):
  python scripts/stands_soften_pin.py --video pinned6.mp4 \
      --ref-video /workspace/Colombia-1-0-Congo-DR1080p.mp4 --out pinned6b.mp4
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np


def soften_grain(band: np.ndarray, ksize: int = 9, keep: float = 0.25) -> np.ndarray:
    """Blur-blend a float32 0-1 band: gaussian base + keep fraction of the detail."""
    blur = cv2.GaussianBlur(band, (ksize, ksize), 0)
    return np.clip(blur + keep * (band - blur), 0.0, 1.0).astype(np.float32)


def sat_of(band: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S, V>0.1 mask) of a float32 0-1 BGR band via cv2 HSV_FULL."""
    hsv = cv2.cvtColor((band * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV_FULL)
    return hsv[..., 1].astype(np.float32) / 255.0, hsv[..., 2] > 25


def sat_match_lut(src_s: np.ndarray, ref_s: np.ndarray, n: int = 256) -> np.ndarray:
    """Monotone quantile-mapping LUT (2,n): source S quantiles -> reference S quantiles."""
    qs = np.linspace(0.0, 1.0, n)
    return np.stack([np.quantile(src_s, qs), np.quantile(ref_s, qs)]).astype(np.float32)


def apply_sat_lut(band: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Remap the band's saturation through the LUT; hue and V (luma) untouched."""
    hsv = cv2.cvtColor((band * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV_FULL).astype(np.float32)
    s = hsv[..., 1] / 255.0
    hsv[..., 1] = np.clip(np.interp(s, lut[0], lut[1]) * 255.0, 0.0, 255.0)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR_FULL)
    return out.astype(np.float32) / 255.0


def band_stats(band: np.ndarray) -> str:
    luma = band.mean(axis=2)
    lc = np.abs(luma - cv2.blur(luma, (5, 5)))
    s, m = sat_of(band)
    return (f"lc med {np.median(lc):.4f} | S med {np.median(s[m]):.3f} "
            f"p90 {np.percentile(s[m], 90):.3f} frac>0.5 {(s[m] > 0.5).mean():.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ref-image", help="clip frame (png) for the target S distribution")
    p.add_argument("--ref-video", help="clip video to grab the reference frame from")
    p.add_argument("--ref-frame", type=int, default=28,
                   help="frame index in --ref-video (default 28 = t20 measured frame)")
    p.add_argument("--roi", nargs=4, type=float, default=[0.08, 0.32, 0.0, 1.0],
                   metavar=("Y0", "Y1", "X0", "X1"),
                   help="stands band on the video (matches the stands tone/red pins)")
    p.add_argument("--ref-roi", nargs=4, type=float, default=None,
                   metavar=("Y0", "Y1", "X0", "X1"),
                   help="stands band on --ref-image (defaults to --roi; its framing may differ)")
    p.add_argument("--ksize", type=int, default=9, help="gaussian kernel (odd px)")
    p.add_argument("--keep", type=float, default=0.25,
                   help="detail fraction kept after the blur (t20 tuned: lc lands on clip)")
    p.add_argument("--no-sat-map", action="store_true", help="soften only, keep saturation")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    y0, y1 = int(args.roi[0] * h), int(args.roi[1] * h)
    x0, x1 = int(args.roi[2] * w), int(args.roi[3] * w)

    if args.ref_image:
        ref = cv2.imread(args.ref_image, cv2.IMREAD_COLOR)
        if ref is None:
            sys.exit(f"cannot read --ref-image {args.ref_image}")
    elif args.ref_video:
        rcap = cv2.VideoCapture(args.ref_video)
        rcap.set(cv2.CAP_PROP_POS_FRAMES, args.ref_frame)
        ok, ref = rcap.read()
        rcap.release()
        if not ok:
            sys.exit(f"cannot read frame {args.ref_frame} of --ref-video {args.ref_video}")
    else:
        sys.exit("need --ref-image or --ref-video")
    rroi = args.ref_roi if args.ref_roi else args.roi
    rh, rw = ref.shape[:2]
    ref_band = ref[int(rroi[0] * rh):int(rroi[1] * rh),
                   int(rroi[2] * rw):int(rroi[3] * rw)].astype(np.float32) / 255.0

    lut = None
    if not args.no_sat_map:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
        ok, mid = cap.read()
        if not ok:
            sys.exit("cannot read mid frame for the LUT")
        mid_band = soften_grain(mid[y0:y1, x0:x1].astype(np.float32) / 255.0,
                                args.ksize, args.keep)
        ms, mm = sat_of(mid_band)
        rs, rm = sat_of(ref_band)
        lut = sat_match_lut(ms[mm], rs[rm])
        print(f"soften LUT: src S med {np.median(ms[mm]):.3f} -> "
              f"ref S med {np.median(rs[rm]):.3f}", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # vertical feather inside the band: full effect in the middle, fades at the edges
    fh = max(4, int(0.15 * (y1 - y0)))
    fmask = np.ones(y1 - y0, dtype=np.float32)
    fmask[:fh] = np.linspace(0.0, 1.0, fh)
    fmask[-fh:] = np.linspace(1.0, 0.0, fh)
    fmask = fmask[:, None, None]

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    done = 0
    last = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        band = bgr[y0:y1, x0:x1].astype(np.float32) / 255.0
        out = soften_grain(band, args.ksize, args.keep)
        if lut is not None:
            out = apply_sat_lut(out, lut)
        out = band + fmask * (out - band)
        bgr[y0:y1, x0:x1] = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        writer.write(bgr)
        last = bgr
        done += 1
    writer.release()
    cap.release()
    if last is not None:
        print(f"stands_soften after: {band_stats(last[y0:y1, x0:x1].astype(np.float32) / 255.0)}",
              flush=True)
        print(f"clip ref            : {band_stats(ref_band)}", flush=True)
    print(f"STANDS_SOFTEN_OK {done}f ksize={args.ksize} keep={args.keep} -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
