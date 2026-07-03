#!/usr/bin/env python
"""Re-pin a team's kit hue after generative finishing (the v2v/upscaler colour drift).

Wan-VACE (and SeedVR2 after it) re-interpret the measured azure team-B kit as the prior "blue
football kit": measured 2026-07-03 on the Colombia clip, blue-pixel median hue goes render 191°
→ v2v 231° → SeedVR2 250°, while the clip truth is ~195°. The finisher is structure-locked, so
the renderer's team-mask pass (`blender_animate.py --team-mask 1` — A=red, B=green, other=blue on
black) still says WHERE the kit is in the finished frame. This tool rotates hue back to a target
only inside (dilated mask) ∩ (hue-band ∩ saturation gate), so grass, crowd and the other team are
untouched even where the dilated mask spills.

One frame per call (stills validation); the video loop lands with the pod v2v integration.

usage:
  python scripts/hue_pin.py --image IN.png --mask MASK.png --out OUT.png --target-hue 191 \
      [--channel g] [--dilate 13] [--hue-band 170 290] [--sat-min 0.15]
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

CHANNEL_TO_BGR_INDEX = {"b": 0, "g": 1, "r": 2}


def pin_hue(
    img_bgr: np.ndarray,
    mask_gray: np.ndarray,
    *,
    target_hue: float,
    dilate: int = 13,
    hue_band: tuple[float, float] = (170.0, 290.0),
    sat_min: float = 0.15,
) -> tuple[np.ndarray, dict]:
    """Rotate hue inside (dilated ``mask_gray``>127) ∩ gate so its median lands on ``target_hue``.

    The rotation is one constant per frame (the drift is global, per-pixel correction would eat
    real shading variation). Returns the corrected BGR image and a stats dict.
    """
    h, w = img_bgr.shape[:2]
    if mask_gray.shape[:2] != (h, w):
        mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask_gray > 127).astype(np.uint8)
    if dilate > 1:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        mask = cv2.dilate(mask, kern)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV_FULL).astype(np.float32)
    hue = hsv[..., 0] * (360.0 / 255.0)
    sat = hsv[..., 1] / 255.0
    lo, hi = hue_band
    gate = mask.astype(bool) & (hue >= lo) & (hue <= hi) & (sat >= sat_min)
    n = int(gate.sum())
    if n < 30:
        return img_bgr, {"n": n, "hue_before": None, "hue_after": None, "delta": 0.0}

    before = float(np.median(hue[gate]))
    delta = float(target_hue) - before
    hue[gate] = np.mod(hue[gate] + delta, 360.0)
    hsv[..., 0] = hue * (255.0 / 360.0)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR_FULL)
    hue_out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV_FULL)[..., 0].astype(np.float32)
    after = float(np.median(hue_out[gate] * (360.0 / 255.0)))
    return out, {"n": n, "hue_before": before, "hue_after": after, "delta": delta}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True, help="team-mask frame from blender_animate --team-mask")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--target-hue", type=float, required=True, help="degrees; e.g. the render's kit hue"
    )
    p.add_argument("--channel", choices=("r", "g", "b"), default="g", help="mask channel: A=r, B=g")
    p.add_argument("--dilate", type=int, default=13)
    p.add_argument("--hue-band", type=float, nargs=2, default=(170.0, 290.0))
    p.add_argument("--sat-min", type=float, default=0.15)
    args = p.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    mask_img = cv2.imread(args.mask, cv2.IMREAD_COLOR)
    if img is None or mask_img is None:
        raise SystemExit(f"cannot read --image {args.image} or --mask {args.mask}")
    mask_gray = mask_img[..., CHANNEL_TO_BGR_INDEX[args.channel]]

    out, st = pin_hue(
        img,
        mask_gray,
        target_hue=args.target_hue,
        dilate=args.dilate,
        hue_band=tuple(args.hue_band),
        sat_min=args.sat_min,
    )
    cv2.imwrite(args.out, out)
    if st["hue_before"] is None:
        print(f"HUE_PIN_NOOP n={st['n']} (<30 gated pixels) -> {args.out}")
    else:
        print(
            f"HUE_PIN_OK n={st['n']} hue {st['hue_before']:.1f} -> {st['hue_after']:.1f} "
            f"(target {args.target_hue:.1f}, delta {st['delta']:+.1f}) -> {args.out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
