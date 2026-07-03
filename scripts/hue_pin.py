#!/usr/bin/env python
"""Re-pin a team's kit hue after generative finishing (the v2v/upscaler colour drift).

Wan-VACE (and SeedVR2 after it) re-interpret the measured azure team-B kit as the prior "blue
football kit": measured 2026-07-03 on the Colombia clip, blue-pixel median hue goes render 191°
→ v2v 231° → SeedVR2 250°, while the clip truth is ~195°. The finisher is structure-locked, so
the renderer's team-mask pass (`blender_animate.py --team-mask 1` — A=red, B=green, other=blue on
black) still says WHERE the kit is in the finished frame. This tool rotates hue back to a target
only inside (dilated mask) ∩ (hue-band ∩ saturation gate), so grass, crowd and the other team are
untouched even where the dilated mask spills.

usage:
  python scripts/hue_pin.py --image IN.png --mask MASK.png --out OUT.png --target-hue 183.5
  python scripts/hue_pin.py --video IN.mp4 --mask-dir DIR --out OUT.mp4 --target-hue 183.5
      [--channel g] [--dilate 13] [--hue-band 170 290] [--sat-min 0.15]
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

CHANNEL_TO_BGR_INDEX = {"b": 0, "g": 1, "r": 2}


def _gate(
    img_bgr: np.ndarray,
    mask_gray: np.ndarray,
    dilate: int,
    hue_band: tuple[float, float],
    sat_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(hsv float32, gate bool): dilated mask ∩ hue-band ∩ saturation, in the image's resolution."""
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
    return hsv, gate


def pin_hue(
    img_bgr: np.ndarray,
    mask_gray: np.ndarray,
    *,
    target_hue: float,
    dilate: int = 13,
    hue_band: tuple[float, float] = (170.0, 290.0),
    sat_min: float = 0.15,
    delta_override: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Rotate hue inside (dilated ``mask_gray``>127) ∩ gate so its median lands on ``target_hue``.

    The rotation is one constant (the drift is global, per-pixel correction would eat real shading
    variation): per frame when ``delta_override`` is None, or the given constant — video mode
    measures ONE delta over the whole clip so the correction cannot flicker. Returns the corrected
    BGR image and a stats dict.
    """
    hsv, gate = _gate(img_bgr, mask_gray, dilate, hue_band, sat_min)
    n = int(gate.sum())
    if n < 30 and delta_override is None:
        return img_bgr, {"n": n, "hue_before": None, "hue_after": None, "delta": 0.0}

    hue = hsv[..., 0] * (360.0 / 255.0)
    before = float(np.median(hue[gate])) if n else None
    delta = float(delta_override) if delta_override is not None else float(target_hue) - before
    hue[gate] = np.mod(hue[gate] + delta, 360.0)
    hsv[..., 0] = hue * (255.0 / 360.0)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR_FULL)
    after = None
    if n:
        hue_out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV_FULL)[..., 0].astype(np.float32)
        after = float(np.median(hue_out[gate] * (360.0 / 255.0)))
    return out, {"n": n, "hue_before": before, "hue_after": after, "delta": delta}


def pin_video(
    video_in: str,
    mask_dir: str,
    video_out: str,
    *,
    target_hue: float,
    channel: str = "g",
    dilate: int = 13,
    hue_band: tuple[float, float] = (170.0, 290.0),
    sat_min: float = 0.15,
) -> dict:
    """Pin every frame of ``video_in`` against ``mask_dir/frame_%04d.png`` with ONE global delta.

    Pass 1 gathers the gated hues of every frame to fix a single clip-wide rotation (per-frame
    medians would flicker); pass 2 applies it. Mask index = video frame index (the mask pass
    renders the same frame list the beauty pass fed the finisher).
    """
    ch = CHANNEL_TO_BGR_INDEX[channel]

    def _frames():
        cap = cv2.VideoCapture(video_in)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mp = os.path.join(mask_dir, f"frame_{i:04d}.png")
            mask_img = cv2.imread(mp, cv2.IMREAD_COLOR)
            yield i, frame, None if mask_img is None else mask_img[..., ch]
            i += 1
        cap.release()

    hues: list[np.ndarray] = []
    n_frames = 0
    w = h = 0
    fps = 25.0
    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    for _i, frame, mask_gray in _frames():
        n_frames += 1
        h, w = frame.shape[:2]
        if mask_gray is None:
            continue
        hsv, gate = _gate(frame, mask_gray, dilate, hue_band, sat_min)
        if gate.any():
            hues.append(hsv[..., 0][gate] * (360.0 / 255.0))
    if not hues:
        raise SystemExit(f"no gated pixels in any frame (masks in {mask_dir}?)")
    before = float(np.median(np.concatenate(hues)))
    delta = target_hue - before

    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    pinned = 0
    for _i, frame, mask_gray in _frames():
        if mask_gray is not None:
            frame, st = pin_hue(
                frame,
                mask_gray,
                target_hue=target_hue,
                dilate=dilate,
                hue_band=hue_band,
                sat_min=sat_min,
                delta_override=delta,
            )
            pinned += 1 if st["n"] else 0
        writer.write(frame)
    writer.release()
    return {"frames": n_frames, "pinned": pinned, "hue_before": before, "delta": delta}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", help="single-frame mode: input still")
    p.add_argument("--mask", help="single-frame mode: team-mask frame")
    p.add_argument("--video", help="video mode: input mp4")
    p.add_argument("--mask-dir", help="video mode: dir of frame_%%04d.png team masks")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--target-hue", type=float, required=True, help="degrees; e.g. the render's kit hue"
    )
    p.add_argument("--channel", choices=("r", "g", "b"), default="g", help="mask channel: A=r, B=g")
    p.add_argument("--dilate", type=int, default=13)
    p.add_argument("--hue-band", type=float, nargs=2, default=(170.0, 290.0))
    p.add_argument("--sat-min", type=float, default=0.15)
    args = p.parse_args()

    if args.video:
        if not args.mask_dir:
            raise SystemExit("--video needs --mask-dir")
        st = pin_video(
            args.video,
            args.mask_dir,
            args.out,
            target_hue=args.target_hue,
            channel=args.channel,
            dilate=args.dilate,
            hue_band=tuple(args.hue_band),
            sat_min=args.sat_min,
        )
        print(
            f"HUE_PIN_VIDEO_OK frames={st['frames']} pinned={st['pinned']} "
            f"hue {st['hue_before']:.1f} -> target {args.target_hue:.1f} "
            f"(delta {st['delta']:+.1f}) -> {args.out}"
        )
        return 0

    if not args.image or not args.mask:
        raise SystemExit("either --image+--mask or --video+--mask-dir")
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
