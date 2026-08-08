#!/usr/bin/env python3
"""What our one-focal camera model cannot express, measured against real broadcast GT.

`docs/pipeline-io-proposed.md` proposes three candidate gaps in the camera model — zoom,
camera translation and lens distortion — and orders the work by guessing which matters.
WorldPose settles it by measurement: 89 clips of real World Cup broadcast, each with a
per-frame GT camera (`K`, `R`, `t`, 5 distortion coefficients).

Our model is one focal + one camera centre + per-frame rotation + no distortion
(`calib/<clip>.npz`, pinned by tests/e2e/test_golden_real_camera.py). This script measures
how far that is from what a real broadcast camera does, and converts the focal half into
metres of player-position error — which is the number the goal actually cares about.

    PYTHONPATH=src python scripts/bench_camera_model_gap.py

Reads only WorldPose/cameras/*.npz (GT cameras, ~89 files, gitignored). Prints four tables
and exits 0. No GPU, no pipeline, no network.

Written 2026-08-08. Findings: docs/findings/camera-model-gap-2026-08-08.md
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

CAMERA_GLOB = "WorldPose/cameras/*.npz"

# Our fit, from calib/Colombia-1-0-Congo-DR1080p.npz — the numbers the golden test pins.
OUR_FOCAL_PX = 4169.32
OUR_HEIGHT_M = 17.22
OUR_FRAMES = 60
FRAME_W, FRAME_H = 1920, 1080


def _pct(values: list[float], q: float) -> float:
    v = sorted(values)
    return v[min(int(len(v) * q), len(v) - 1)]


def _row(label: str, values: list[float], fmt: str = "{:8.2f}") -> None:
    print(
        f"{label:<40s}"
        f"{fmt.format(_pct(values, 0.5)):>10s}"
        f"{fmt.format(_pct(values, 0.9)):>10s}"
        f"{fmt.format(_pct(values, 1.0)):>10s}"
    )


def load_cameras() -> list[dict]:
    paths = sorted(glob.glob(CAMERA_GLOB))
    if not paths:
        print(f"No GT cameras at {CAMERA_GLOB}.", file=sys.stderr)
        print("WorldPose is gitignored; see docs/models-dir.md.", file=sys.stderr)
        raise SystemExit(1)
    out = []
    for p in paths:
        d = np.load(p)
        K, R, t = d["K"], d["R"], d["t"]
        # camera centre C = -R^T t, per frame
        centre = -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)
        out.append(
            {
                "clip": os.path.basename(p)[:-4],
                "fx": K[:, 0, 0],
                "cx": K[:, 0, 2],
                "cy": K[:, 1, 2],
                "k1": d["k"][:, 0],
                "centre": centre,
            }
        )
    return out


def table_whole_clip(cams: list[dict]) -> None:
    print(f"\n== 1. Whole clip: what a real broadcast camera does ({len(cams)} GT clips) ==")
    print(f"{'':40s}{'median':>10s}{'p90':>10s}{'max':>10s}")
    _row("focal fx (px)", [c["fx"].mean() for c in cams], "{:8.0f}")
    _row(
        "focal drift within clip (% of mean)",
        [(c["fx"].max() - c["fx"].min()) / c["fx"].mean() * 100 for c in cams],
    )
    _row(
        "principal point wander (px)",
        [float(np.abs(c["cx"] - c["cx"][0]).max()) for c in cams],
    )
    _row("|k1| max in clip", [float(np.abs(c["k1"]).max()) for c in cams], "{:8.4f}")
    _row(
        "camera translation (m, max from mean)",
        [float(np.linalg.norm(c["centre"] - c["centre"].mean(0), axis=1).max()) for c in cams],
        "{:8.3f}",
    )
    moves = sum((c["fx"].max() - c["fx"].min()) / c["fx"].mean() > 0.05 for c in cams)
    shifts = sum(
        np.linalg.norm(c["centre"] - c["centre"].mean(0), axis=1).max() > 0.10 for c in cams
    )
    print(f"\n  clips whose focal moves >5%          : {moves}/{len(cams)}")
    print(f"  clips whose camera moves >10 cm      : {shifts}/{len(cams)}")


def table_distortion(cams: list[dict]) -> None:
    """Distortion in the unit that matters: pixels of displacement at the frame corner."""
    print("\n== 2. Distortion, expressed at the frame corner ==")
    r_corner = float(np.hypot(FRAME_W / 2, FRAME_H / 2))
    f_med = float(np.median([c["fx"].mean() for c in cams]))
    k1s = [float(np.abs(c["k1"]).max()) for c in cams]
    print(f"  reference: {FRAME_W}x{FRAME_H}, r_corner={r_corner:.0f} px, f={f_med:.0f} px")
    for label, q in (("median", 0.5), ("p90", 0.9), ("max", 1.0)):
        k1 = _pct(k1s, q)
        r_n = r_corner / f_med
        print(f"  {label:>6s} |k1|={k1:.4f}  ->  {abs(k1 * r_n**3) * f_med:6.1f} px at the corner")
    print("  (our CameraIntrinsics.distortion is None on every solve we produce)")


def table_windows(cams: list[dict]) -> None:
    """Drift at OUR window size — a 60-frame fit is not a 1000-frame clip."""
    print(f"\n== 3. Focal drift inside a sliding window (our fit is {OUR_FRAMES} frames) ==")
    print(f"{'window (frames)':<40s}{'median':>10s}{'p90':>10s}{'max':>10s}{'>2%':>8s}")
    for w in (30, 60, 120, 240, 480):
        drifts, over = [], 0
        for c in cams:
            fx = c["fx"]
            for s in range(0, len(fx) - w, w):  # non-overlapping
                seg = fx[s : s + w]
                d = (seg.max() - seg.min()) / seg.mean() * 100
                drifts.append(d)
                over += d > 2.0
        if not drifts:
            continue
        mark = "  <- ours" if w == OUR_FRAMES else ""
        print(
            f"{w:<40d}{_pct(drifts, 0.5):9.1f}%{_pct(drifts, 0.9):9.1f}%"
            f"{_pct(drifts, 1.0):9.1f}%{over / len(drifts) * 100:7.0f}%{mark}"
        )


def table_sensitivity() -> None:
    """Convert a focal error into the unit the goal is stated in: metres on the pitch.

    A ground point at distance d projects to v = f*h/d below the principal point. Perturb f
    by eps and invert: the same pixel now reads as a different ground distance.
    """
    print("\n== 4. What a focal error costs, as player-position error ==")
    print(f"  our camera: f={OUR_FOCAL_PX:.0f} px, height={OUR_HEIGHT_M:.1f} m")
    print(f"\n{'ground distance':<20s}" + "".join(f"{e:>10s}" for e in ("1%", "2%", "5%", "9%")))
    for d in (20, 40, 60, 80):
        v = OUR_FOCAL_PX * OUR_HEIGHT_M / d
        cells = []
        for eps in (0.01, 0.02, 0.05, 0.09):
            d_err = abs(d - OUR_HEIGHT_M * OUR_FOCAL_PX / (v + eps * v))
            cells.append(f"{d_err:9.2f}m")
        print(f"{str(d) + ' m':<20s}" + "".join(cells))
    print("\n  Read with table 3: a 60-frame fit sits at ~2% median drift, a 240-frame one")
    print("  at ~9%. That is the cost of one focal per clip, in metres, at pitch distances.")


def main() -> int:
    cams = load_cameras()
    print("Camera-model gap — our one-focal model vs real broadcast GT (WorldPose)")
    table_whole_clip(cams)
    table_distortion(cams)
    table_windows(cams)
    table_sensitivity()
    print("\nWrite-up: docs/findings/camera-model-gap-2026-08-08.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
