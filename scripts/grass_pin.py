#!/usr/bin/env python
"""Region tone pin: land a colour-band region of a finished video on the clip's measured tone.

Default gate = the grass band (the original use); --roi/--target-roi/--pin-val turn the same
ONE-global-delta machinery on any banded region (e.g. the stands: warm band 15-80 in the top
third, clip crowd is darker AND yellower than the finisher's amber).

Prompt wording cannot land it (measured 2026-07-05, three A/Bs: clip grass H 78.8 S 0.67;
"muted yellow-green" -> H ~68 S ~.88, dropping "yellow-" -> H ~71.7 (1/3 of the gap, S
unchanged), "dull green" -> olive H 68.9 — intensity words carry their own hue prior and S
never left 0.85+). So the correction is deterministic post: ONE global hue delta + ONE global
saturation scale over the grass band, targets auto-measured from a clip frame (manual
--target-hue/--target-sat override). Team-mask exclusion is REQUIRED whenever masks exist:
pin A parks the yellow kit at H≈70, inside the grass band — without the mask gate the grass
pin would drag the shirts green.

--flatten-val-x BINS (stands): kills hot-edge lighting — our floodlit bowl renders the LEFT
stands ~1.9x brighter than mid (t9/t10 measured .51 vs .27; the clip's own edge is the DIM
end, .22-.26). The clip's x-profile wanders with the pan (.22-.35), so the framing-independent
target is FLAT: per-column-bin gated V medians -> gain = band-median / bin-median (clamped,
smoothed), applied to ALL pixels in the ROI band (the blowout is partly desaturated, i.e.
outside the gate) except kit masks, feathered vertically so no seam.

usage (pod, after the kit pins):
  python scripts/grass_pin.py --video pinned2.mp4 --mask-dir MASKS \
      --target-from-image out/v2v/ref_night.png --out pinned3.mp4
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def _gate(
    img_bgr: np.ndarray,
    kit_mask: np.ndarray | None,
    hue_band: tuple[float, float],
    sat_min: float,
    val_min: float,
    roi: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV_FULL).astype(np.float32)
    hue = hsv[..., 0] * (360.0 / 255.0)
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    lo, hi = hue_band
    gate = (hue >= lo) & (hue <= hi) & (sat >= sat_min) & (val >= val_min)
    y0, y1, x0, x1 = roi
    if (y0, y1, x0, x1) != (0.0, 1.0, 0.0, 1.0):
        h, w = img_bgr.shape[:2]
        sp = np.zeros((h, w), dtype=bool)
        sp[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = True
        gate &= sp
    if kit_mask is not None:
        gate &= ~kit_mask
    return hsv, gate


def _kit_mask(mask_path: str, shape: tuple[int, int], dilate: int) -> np.ndarray | None:
    """Any-channel team mask (A=r, B=g, other=b), dilated, at the video's resolution."""
    m = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if m is None:
        return None
    if m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    any_kit = (m.max(axis=2) > 127).astype(np.uint8)
    if dilate > 1:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        any_kit = cv2.dilate(any_kit, kern)
    return any_kit.astype(bool)


def measure_image(path: str, hue_band, sat_min, val_min, roi) -> tuple[float, float, float]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read --target-from-image {path}")
    hsv, gate = _gate(img, None, hue_band, sat_min, val_min, roi)
    if gate.sum() < 500:
        raise SystemExit(f"target-from-image: only {int(gate.sum())} gated pixels in {path}")
    return (
        float(np.median(hsv[..., 0][gate]) * (360.0 / 255.0)),
        float(np.median(hsv[..., 1][gate]) / 255.0),
        float(np.median(hsv[..., 2][gate]) / 255.0),
    )


def xflat_gains(bin_medians: np.ndarray, lo: float = 0.55, hi: float = 1.3) -> np.ndarray:
    """Per-bin V gains that flatten the band's x-profile to its own global median."""
    med = float(np.median(bin_medians))
    g = np.clip(med / np.maximum(bin_medians, 1e-6), lo, hi)
    return np.convolve(np.pad(g, 1, mode="edge"), np.ones(3) / 3.0, "valid")


def xflat_field(
    shape: tuple[int, int], roi: tuple[float, float, float, float], gains: np.ndarray
) -> np.ndarray:
    """Full-frame multiplicative V field: gains interpolated across the ROI x-span, feathered
    vertically inside the ROI band (no horizontal seam), 1.0 everywhere else."""
    h, w = shape
    y0, y1, x0, x1 = roi
    ya, yb, xa, xb = int(y0 * h), int(y1 * h), int(x0 * w), int(x1 * w)
    nb = len(gains)
    centers = xa + (np.arange(nb) + 0.5) * (xb - xa) / nb
    gx = np.interp(np.arange(w), centers, gains).astype(np.float32)
    gx[:xa] = 1.0
    gx[xb:] = 1.0
    vmask = np.zeros(h, dtype=np.float32)
    vmask[ya:yb] = 1.0
    f = max(4, int(0.15 * (yb - ya)))
    vmask[ya : ya + f] = np.linspace(0.0, 1.0, f)
    vmask[yb - f : yb] = np.linspace(1.0, 0.0, f)
    return 1.0 + vmask[:, None] * (gx[None, :] - 1.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mask-dir", help="team-mask frames; omit ONLY for maskless smoke tests")
    p.add_argument("--target-from-image", help="clip frame to auto-measure the target from")
    p.add_argument("--target-hue", type=float, help="manual target override, degrees")
    p.add_argument("--target-sat", type=float, help="manual target override, 0-1")
    p.add_argument("--target-val", type=float, help="manual target override, 0-1 (with --pin-val)")
    p.add_argument("--hue-band", type=float, nargs=2, default=(55.0, 140.0))
    p.add_argument("--sat-min", type=float, default=0.25)
    p.add_argument("--val-min", type=float, default=0.10)
    p.add_argument("--dilate", type=int, default=15)
    p.add_argument(
        "--roi", type=float, nargs=4, default=(0.0, 1.0, 0.0, 1.0), metavar=("Y0", "Y1", "X0", "X1"),
        help="fractional spatial gate on the VIDEO (e.g. stands band 0.08 0.35 0.02 0.98)",
    )
    p.add_argument(
        "--target-roi", type=float, nargs=4, default=None, metavar=("Y0", "Y1", "X0", "X1"),
        help="fractional spatial gate on --target-from-image (its framing differs from the video)",
    )
    p.add_argument(
        "--pin-val", action="store_true",
        help="also scale V to the target (stands: the clip crowd is darker than the finisher's)",
    )
    p.add_argument(
        "--flatten-val-x", type=int, default=0, metavar="BINS",
        help="flatten the band's V x-profile to its own median over BINS columns "
        "(hot-edge fix; all ROI pixels except kit masks, vertical feather)",
    )
    args = p.parse_args()

    band = tuple(args.hue_band)
    roi = tuple(args.roi)
    t_hue, t_sat, t_val = args.target_hue, args.target_sat, args.target_val
    if args.target_from_image and (t_hue is None or t_sat is None or (args.pin_val and t_val is None)):
        troi = tuple(args.target_roi) if args.target_roi else roi
        mh, ms, mv = measure_image(args.target_from_image, band, args.sat_min, args.val_min, troi)
        t_hue = mh if t_hue is None else t_hue
        t_sat = ms if t_sat is None else t_sat
        t_val = mv if t_val is None else t_val
        print(
            f"TONE_PIN_TARGET H={t_hue:.1f} S={t_sat:.2f} V={t_val:.2f} "
            f"from {args.target_from_image}"
        )
    if t_hue is None or t_sat is None or (args.pin_val and t_val is None):
        raise SystemExit("need --target-from-image or explicit --target-hue/--target-sat(/--target-val)")

    def frames():
        cap = cv2.VideoCapture(args.video)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            km = None
            if args.mask_dir:
                km = _kit_mask(
                    os.path.join(args.mask_dir, f"frame_{i:04d}.png"),
                    frame.shape[:2],
                    args.dilate,
                )
            yield frame, km
            i += 1
        cap.release()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    hs, ss, vs = [], [], []
    nb = args.flatten_val_x
    xbin_vals: list[list[np.ndarray]] = [[] for _ in range(nb)]
    n_frames = 0
    w = h = 0
    for frame, km in frames():
        n_frames += 1
        h, w = frame.shape[:2]
        hsv, gate = _gate(frame, km, band, args.sat_min, args.val_min, roi)
        if gate.any():
            hs.append(hsv[..., 0][gate] * (360.0 / 255.0))
            ss.append(hsv[..., 1][gate] / 255.0)
            vs.append(hsv[..., 2][gate] / 255.0)
            if nb:
                xs = np.nonzero(gate)[1]
                xa, xb = int(roi[2] * w), int(roi[3] * w)
                bins = np.clip((xs - xa) * nb // max(xb - xa, 1), 0, nb - 1)
                for k in np.unique(bins):
                    xbin_vals[k].append(vs[-1][bins == k])
    if not hs:
        raise SystemExit("no gated pixels in any frame")
    before_h = float(np.median(np.concatenate(hs)))
    before_s = float(np.median(np.concatenate(ss)))
    before_v = float(np.median(np.concatenate(vs)))
    delta_h = t_hue - before_h
    scale_s = t_sat / max(before_s, 1e-6)
    scale_v = (t_val / max(before_v, 1e-6)) if args.pin_val else 1.0

    gain_field = None
    if nb:
        meds = np.array([
            float(np.median(np.concatenate(v))) if v else before_v for v in xbin_vals
        ])
        gains = xflat_gains(meds)
        gain_field = xflat_field((h, w), roi, gains)
        print(
            f"XFLAT bins={nb} gains {gains.min():.2f}..{gains.max():.2f} "
            f"profile {' '.join(f'{m:.2f}' for m in meds)}"
        )

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame, km in frames():
        hsv, gate = _gate(frame, km, band, args.sat_min, args.val_min, roi)
        hue = hsv[..., 0] * (360.0 / 255.0)
        hue[gate] = np.mod(hue[gate] + delta_h, 360.0)
        hsv[..., 0] = hue * (255.0 / 360.0)
        hsv[..., 1][gate] = np.clip(hsv[..., 1][gate] * scale_s, 0.0, 255.0)
        if scale_v != 1.0:
            hsv[..., 2][gate] = np.clip(hsv[..., 2][gate] * scale_v, 0.0, 255.0)
        if gain_field is not None:
            gf = gain_field if km is None else np.where(km, 1.0, gain_field)
            hsv[..., 2] = np.clip(hsv[..., 2] * gf, 0.0, 255.0)
        writer.write(cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR_FULL))
    writer.release()
    print(
        f"TONE_PIN_OK frames={n_frames} H {before_h:.1f} -> {t_hue:.1f} (delta {delta_h:+.1f}) "
        f"S {before_s:.2f} -> {t_sat:.2f} (x{scale_s:.2f}) "
        f"V {before_v:.2f} -> x{scale_v:.2f} masks={'yes' if args.mask_dir else 'NO'} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
