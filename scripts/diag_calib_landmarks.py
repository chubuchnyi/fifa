"""B1 calibration-completeness diagnostic — decompose *why* SoccerNet frames fail to lock.

The B1 benchmark scores PnLCalib on the open SoccerNet ``calibration-2023`` test split. The DLT
(points-only) path locks ~149/200 frames; the full camera module (points **and** lines) locks
~148/200 — so in the measured A/B, adding lines did **not** raise completeness (#112). This script
explains the *mechanism* behind that ~26% drop so #122 ("line-only fusion fallback") can be decided
on data, not assertion.

For every benchmark frame it runs the shared PnLCalib front half (:meth:`_infer_frame`) and records:

* ``n_kp``    — keypoints **after** ``complete_keypoints`` (so line-intersections are *already*
  folded in); this is exactly the DLT input, which needs ``>= 4`` to fit a planar homography.
* ``n_lines`` — straight pitch lines the line head detected.

The benchmark population itself (``load_calib_dir(min_lines=4)``) is GT-marking-rich (every frame
has >= 4 annotated line classes), so a frame that fails is **not** a markingless close-up — it is a
view where PnLCalib's *keypoint* head under-detected. The decisive question is therefore:

    Among frames where ``n_kp < 4`` (DLT cannot lock), how many lines did the line head find?

* Failing frames also **line-poor** (``n_lines`` low) → the line head fails on the *same* hard views
  (zoom / motion blur / occlusion); line-only fusion has nothing extra to fuse → the drop is a
  detector-quality / data ceiling, and #122 is a won't-fix (the real lever is a stronger detector or
  temporal propagation across the clip).
* Failing frames **line-rich** (``n_lines`` high) → the line head sees structure the kp head misses
  → a line-only homography fallback could rescue them → #122 is worth implementing.

This is the granular complement to the #112 camera-module A/B: that measured the *outcome*
(completeness stayed flat); this measures the *cause* (per-frame kp vs line supply).

Run on the GPU box (PnLCalib weights live there, never vendored — see the backend module)::

    PYTHONPATH=src \\
    PNLCALIB_REPO=/workspace/repos/PnLCalib \\
    PNLCALIB_WEIGHTS_KP=/workspace/weights/pnlcalib/SV_kp \\
    PNLCALIB_WEIGHTS_LINES=/workspace/weights/pnlcalib/SV_lines \\
    PNLCALIB_DEVICE=cuda:0 \\
    python scripts/diag_calib_landmarks.py \\
        --frames-dir /workspace/SoccerNet/calibration-2023/test --limit 200
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from pitch3d.adapters.models.pnlcalib_backend import make
from pitch3d.eval.datasets_soccernet import as_clip, load_calib_dir


def _hist(values: list[int]) -> dict[int, int]:
    """Ascending ``{count: n_frames}`` histogram (small ints, so a plain dict reads fine)."""
    return dict(sorted(Counter(values).items()))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--frames-dir", required=True, help="SoccerNet calibration split directory")
    ap.add_argument("--min-lines", type=int, default=4,
                    help="GT straight-line floor (mirrors the benchmark eval set; default 4)")
    ap.add_argument("--limit", type=int, default=None, help="cap frames (quick smoke)")
    ap.add_argument("--lock-kp", type=int, default=4,
                    help="detected keypoints the DLT needs to lock a homography (default 4)")
    ap.add_argument("--line-rich", type=int, default=2,
                    help="detected lines for a failing frame to count as 'line-rich' (default 2)")
    ap.add_argument("--json-out", default=None, help="optional path to dump the per-frame table")
    args = ap.parse_args(argv)

    frames = load_calib_dir(args.frames_dir, min_lines=args.min_lines, limit=args.limit)
    if not frames:
        raise SystemExit(
            f"no annotated frames under {args.frames_dir} (min_lines={args.min_lines})"
        )
    clip = as_clip(frames, args.frames_dir)

    backend = make()
    s = backend._load()  # heavy: builds both HRNet heads on the box GPU

    rows: list[dict] = []
    for idx, bgr in s["iter_frames"](clip):
        kp_dict, lines_dict, _w, _h = backend._infer_frame(s, bgr)
        rows.append({"frame": int(idx), "n_kp": len(kp_dict), "n_lines": len(lines_dict)})

    n = len(rows)
    locked = [r for r in rows if r["n_kp"] >= args.lock_kp]
    failed = [r for r in rows if r["n_kp"] < args.lock_kp]
    f_rich = [r for r in failed if r["n_lines"] >= args.line_rich]
    f_poor = [r for r in failed if r["n_lines"] < args.line_rich]

    print(f"frames                 : {n}")
    print(f"locked  (n_kp >= {args.lock_kp})    : {len(locked):4d}  ({len(locked) / n:.3f})")
    print(f"failed  (n_kp <  {args.lock_kp})    : {len(failed):4d}  ({len(failed) / n:.3f})")
    if failed:
        denom = len(failed)
        print(f"  among failed, line-rich (n_lines >= {args.line_rich}): "
              f"{len(f_rich):4d}  ({len(f_rich) / denom:.3f})   <- line-fusion could rescue")
        print(f"  among failed, line-poor (n_lines <  {args.line_rich}): "
              f"{len(f_poor):4d}  ({len(f_poor) / denom:.3f})   <- detector/data ceiling")
        print(f"  failed-frame n_lines histogram : {_hist([r['n_lines'] for r in failed])}")
    print(f"n_kp    : min={min(r['n_kp'] for r in rows):3d}  "
          f"median={int(np.median([r['n_kp'] for r in rows])):3d}  "
          f"max={max(r['n_kp'] for r in rows):3d}")
    print(f"n_lines : min={min(r['n_lines'] for r in rows):3d}  "
          f"median={int(np.median([r['n_lines'] for r in rows])):3d}  "
          f"max={max(r['n_lines'] for r in rows):3d}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(rows, fh)
        print(f"wrote per-frame table -> {args.json_out}")


if __name__ == "__main__":
    main()
