"""Do the markings the pixels contain agree with the markings the camera projects?

`scripts/detect_markings.py` reads paint and never sees the camera. The calibration in
`calib/*.npz` draws markings and never sees the paint. They are two independent answers to
the same question, so disagreeing is informative in a way that neither is alone -- and
`bench_overlay_residual.py`'s header has promised this comparison since it was written
without ever implementing it.

Two numbers, and they fail differently:

  recall     of the model markings that are actually visible in the frame, what fraction did
             the detector find? A miss is the detector's fault: paint the eye can see and the
             CV cannot.
  precision  of the markings the detector found, what fraction lands on a projected model
             marking? A detected line with no model line near it is either a false positive
             (a board edge, a net) or a real marking the camera has put somewhere else.

Precision is the interesting one. The detector has no way to invent a straight 60 px line on
the playing surface out of nothing, so a *confident* detection with no model marking under it
is evidence against the camera, not against the detector.

Run:
    PYTHONPATH=src .venv/bin/python scripts/bench_markings_vs_camera.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poseannot.pitch_evidence import _masks  # noqa: E402
from poseannot.video import read_frame  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_markings import detect  # noqa: E402

#: A detected segment is explained by the model if its midpoint and both ends sit this close
#: to a projected marking. Wide enough to absorb the 12 cm paint width plus the fit's own
#: ~1.4 px paint residual, tight enough that a board edge 20 px off the touchline still fails.
EXPLAIN_PX = 12.0

#: A projected marking counts as "visible" only where it lands on the playing surface. A
#: marking projected into the crowd is unmeasurable, not missed -- the #113/#114 rule that
#: fit_rigid_camera.paint_error already applies to its own metric.
#:
#: The model side is rasterised and distance-transformed rather than compared point to point.
#: That is not a performance choice. pitch_polylines samples every 0.5 m, which at this
#: camera is tens of pixels apart in the near field, so "distance to the nearest sampled
#: model point" charges a detected line up to half a sample spacing for lying exactly on the
#: model. Measured: it reported precision 2.3 % against recall 92 % -- an impossible pair,
#: and the tell that the metric was wrong rather than the camera.


def project(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    q = np.column_stack([xy, np.ones(len(xy))]) @ h.T
    w = np.where(np.abs(q[:, 2]) > 1e-9, q[:, 2], 1e-9)
    return q[:, :2] / w[:, None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib", default="calib/Colombia-1-0-Congo-DR1080p.npz")
    ap.add_argument("--clip", default="samples/video/Colombia-1-0-Congo-DR1080p.mp4")
    ap.add_argument("--limit", type=int, default=0, help="0 = every calibrated frame")
    args = ap.parse_args()

    from pitch3d.core.scene.pitch import pitch_polylines

    cal = np.load(args.calib, allow_pickle=True)
    w2i, frames = cal["world_to_image"], cal["frames"]
    if args.limit:
        w2i, frames = w2i[: args.limit], frames[: args.limit]

    model = [p[:, :2] for p in pitch_polylines() if len(p) > 1]
    lsd = cv2.createLineSegmentDetector()
    rec, prec, unexplained = [], [], []

    for h, fi in zip(w2i, frames, strict=True):
        bgr = read_frame(args.clip, int(fi))
        hgt, wid = bgr.shape[:2]
        # detect() and not a copy of its body: a bench that re-implements the thing it scores
        # silently stops scoring it. This one did -- it went on reading 97.8/100.0 after the
        # goal-structure filter landed, because the filter was only in detect().
        dist, surface = _masks(bgr)
        segs = detect(bgr, lsd)["segments"]

        # --- where the model says markings are: rasterised, then distance-transformed ---
        canvas = np.zeros((hgt, wid), np.uint8)
        vis = []
        for poly in model:
            uv = project(h, poly)
            ok = (uv[:, 0] > 1) & (uv[:, 0] < wid - 2) & (uv[:, 1] > 1) & (uv[:, 1] < hgt - 2)
            if ok.sum() >= 2:
                cv2.polylines(canvas, [np.rint(uv[ok]).astype(np.int32)], False, 255, 1)
                sub = uv[ok]
                on = surface[np.rint(sub[:, 1]).astype(int), np.rint(sub[:, 0]).astype(int)] > 0
                vis.append(sub[on])
        pts = np.vstack(vis) if vis else np.zeros((0, 2))
        if not len(pts) or not len(segs):
            continue
        to_model = cv2.distanceTransform((canvas == 0).astype(np.uint8), cv2.DIST_L2, 5)

        # recall: model points with paint under them (the detector's own mask answers this)
        d_paint = dist[np.rint(pts[:, 1]).astype(int), np.rint(pts[:, 0]).astype(int)]
        rec.append(float((d_paint <= EXPLAIN_PX).mean()))

        # precision: detected segments whose whole extent lies on a projected marking
        hit = 0
        for x1, y1, x2, y2 in segs:
            t = np.linspace(0, 1, 24)[:, None]
            along = np.array([x1, y1]) * (1 - t) + np.array([x2, y2]) * t
            uu = np.clip(np.rint(along[:, 0]).astype(int), 0, wid - 1)
            vv = np.clip(np.rint(along[:, 1]).astype(int), 0, hgt - 1)
            near = float(np.median(to_model[vv, uu]))
            if near <= EXPLAIN_PX:
                hit += 1
            else:
                unexplained.append((int(fi), near, float(np.hypot(x2 - x1, y2 - y1))))
        prec.append(hit / len(segs))

    r, p = np.array(rec), np.array(prec)
    print(f"\n{len(r)} calibrated frames, tolerance {EXPLAIN_PX:.0f} px\n")
    print(f"  recall     model markings the detector found : {r.mean():6.1%}"
          f"   (p10 {np.percentile(r, 10):.1%}, p90 {np.percentile(r, 90):.1%})")
    print(f"  precision  detections the camera explains    : {p.mean():6.1%}"
          f"   (p10 {np.percentile(p, 10):.1%}, p90 {np.percentile(p, 90):.1%})")
    if unexplained:
        u = np.array([x[1] for x in unexplained])
        ln = np.array([x[2] for x in unexplained])
        print(f"\n  {len(unexplained)} unexplained detections: median miss {np.median(u):.0f} px, "
              f"median length {np.median(ln):.0f} px")
        worst = sorted(unexplained, key=lambda x: -x[2])[:8]
        print("  longest (these are the ones worth looking at by eye):")
        for fi, miss, length in worst:
            print(f"    f{fi:3d}  {length:5.0f} px long, {miss:5.0f} px from any model marking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
