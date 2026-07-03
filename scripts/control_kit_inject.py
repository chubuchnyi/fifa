#!/usr/bin/env python
"""Re-inject team kit colours into night-graded control frames (the cluster-smear fix).

Diagnosed 2026-07-03 (out/clusters/control_vs_output_f10.png): grade3 night-grading keeps
clean separable geometry in tight player clusters but all but erases kit colour there —
every player becomes the same dark-teal mannequin, so the Wan-VACE control carries shape
without identity and the model hallucinates kits (white/orange shirts) or merges players.
This tool pushes H and S toward each team's kit colour inside the (eroded) team-mask AOV
(`blender_animate.py --team-mask 1`: A=red, B=green channel) while keeping V untouched, so
shading and limb boundaries survive and team identity returns to the control signal.

Auto-target: measured from the frames themselves — the masked pixels whose saturation
survived the grade (isolated players) vote for the team colour. Manual --team-a-hsv /
--team-b-hsv override per the auto+manual rule.

usage:
  python scripts/control_kit_inject.py --frames NIGHT_DIR --masks MASK_DIR --out OUT_DIR \
      [--alpha 0.8] [--erode 3] [--team-a-hsv 45 0.7] [--team-b-hsv 190 0.6] \
      [--measure-from DIR]
"""

from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

TEAM_TO_BGR_INDEX = {"a": 2, "b": 1}  # mask AOV: A=red, B=green (hue_pin convention)


def _gate_from_mask(mask_bgr: np.ndarray, ch: int, hw: tuple[int, int], erode: int) -> np.ndarray:
    h, w = hw
    m = mask_bgr[..., ch]
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8)
    if erode > 1:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
        m = cv2.erode(m, kern)
    return m.astype(bool)


def measure_team_hsv(
    frame_paths: list[str],
    mask_dir: str,
    team: str,
    *,
    erode: int = 3,
    sat_pct: float = 60.0,
    sat_floor: float = 0.10,
) -> tuple[float, float, int] | None:
    """(target hue deg, target sat 0..1, n voting px) from the grade-surviving masked pixels.

    Dominant-mode measurement: histogram peak + circular mean of its ±25° neighbourhood.
    A plain circular mean is wrong here — grade3 splits team-A yellow into a yellow/olive
    bimodal (measured 2026-07-03: beauty 60-80° unimodal, night 140-160°+60-70°) and the
    mean lands in no-man's land between the modes.
    """
    ch = TEAM_TO_BGR_INDEX[team]
    hs: list[np.ndarray] = []
    ss: list[np.ndarray] = []
    for fp in frame_paths:
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(mask_dir, os.path.basename(fp)), cv2.IMREAD_COLOR)
        if img is None or mask is None:
            continue
        gate = _gate_from_mask(mask, ch, img.shape[:2], erode)
        if not gate.any():
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL).astype(np.float32)
        hs.append(hsv[..., 0][gate] * (360.0 / 255.0))
        ss.append(hsv[..., 1][gate] / 255.0)
    if not hs:
        return None
    h_all = np.concatenate(hs)
    s_all = np.concatenate(ss)
    thr = max(float(np.percentile(s_all, sat_pct)), sat_floor)
    keep = s_all >= thr
    if not keep.any():
        return None
    h_keep = h_all[keep]
    hist, edges = np.histogram(h_keep, bins=72, range=(0.0, 360.0))
    peak = float(edges[int(np.argmax(hist))]) + 2.5
    near = np.abs(np.mod(h_keep - peak + 180.0, 360.0) - 180.0) <= 25.0
    ang = np.deg2rad(h_keep[near])
    hue = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360.0)
    sat = float(np.percentile(s_all[keep][near], 75.0))
    return hue, sat, int(near.sum())


def inject_frame(
    img_bgr: np.ndarray,
    targets: list[tuple[np.ndarray, float, float]],
    alpha: float,
) -> tuple[np.ndarray, int]:
    """Blend H circularly and push S up toward each (gate, hue, sat) target; V untouched."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV_FULL).astype(np.float32)
    hue = hsv[..., 0] * (360.0 / 255.0)
    sat = hsv[..., 1] / 255.0
    n = 0
    for gate, t_hue, t_sat in targets:
        if not gate.any():
            continue
        d = np.mod(t_hue - hue[gate] + 180.0, 360.0) - 180.0
        hue[gate] = np.mod(hue[gate] + alpha * d, 360.0)
        sat[gate] = np.maximum(sat[gate], sat[gate] + alpha * (t_sat - sat[gate]))
        n += int(gate.sum())
    hsv[..., 0] = hue * (255.0 / 360.0)
    hsv[..., 1] = np.clip(sat, 0.0, 1.0) * 255.0
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR_FULL)
    return out, n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--frames", required=True, help="control frames to inject (frame_*.png)")
    p.add_argument("--masks", required=True, help="team-mask AOV dir (shared basenames)")
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=0.8, help="H/S pull strength toward the kit")
    p.add_argument("--erode", type=int, default=3, help="mask erosion kernel (px, after resize)")
    p.add_argument("--team-a-hsv", type=float, nargs=2, metavar=("HUE", "SAT"),
                   help="manual team-A target (deg, 0..1); default auto-measured")
    p.add_argument("--team-b-hsv", type=float, nargs=2, metavar=("HUE", "SAT"),
                   help="manual team-B target (deg, 0..1); default auto-measured")
    p.add_argument("--measure-from", metavar="DIR",
                   help="measure auto targets from these frames instead (e.g. beauty render)")
    p.add_argument("--sat-pct", type=float, default=60.0,
                   help="saturation percentile a pixel must beat to vote in the measurement")
    p.add_argument("--sat-floor", type=float, default=0.10)
    args = p.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frames, "frame_*.png")))
    if not frame_paths:
        raise SystemExit(f"no frame_*.png in {args.frames}")
    measure_paths = frame_paths
    if args.measure_from:
        measure_paths = sorted(glob.glob(os.path.join(args.measure_from, "frame_*.png")))
        if not measure_paths:
            raise SystemExit(f"no frame_*.png in {args.measure_from}")

    manual = {"a": args.team_a_hsv, "b": args.team_b_hsv}
    targets: dict[str, tuple[float, float]] = {}
    for team in ("a", "b"):
        if manual[team] is not None:
            targets[team] = (manual[team][0] % 360.0, manual[team][1])
            print(f"KIT_INJECT_TARGET team={team} hue={targets[team][0]:.1f} "
                  f"sat={targets[team][1]:.2f} (manual)")
            continue
        m = measure_team_hsv(measure_paths, args.masks, team,
                             erode=args.erode, sat_pct=args.sat_pct, sat_floor=args.sat_floor)
        if m is None:
            print(f"KIT_INJECT_SKIP team={team} (no voting pixels)")
            continue
        targets[team] = (m[0], m[1])
        print(f"KIT_INJECT_TARGET team={team} hue={m[0]:.1f} sat={m[1]:.2f} "
              f"(auto, {m[2]} voting px)")
    if not targets:
        raise SystemExit("no team targets — pass --team-a-hsv/--team-b-hsv")

    os.makedirs(args.out, exist_ok=True)
    total_px = 0
    for fp in frame_paths:
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(args.masks, os.path.basename(fp)), cv2.IMREAD_COLOR)
        if img is None or mask is None:
            raise SystemExit(f"cannot read {fp} or its mask")
        frame_targets = [
            (_gate_from_mask(mask, TEAM_TO_BGR_INDEX[t], img.shape[:2], args.erode), h, s)
            for t, (h, s) in targets.items()
        ]
        out, n = inject_frame(img, frame_targets, args.alpha)
        total_px += n
        cv2.imwrite(os.path.join(args.out, os.path.basename(fp)), out)
    print(f"KIT_INJECT_OK frames={len(frame_paths)} teams={sorted(targets)} "
          f"px/frame={total_px // max(len(frame_paths), 1)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
