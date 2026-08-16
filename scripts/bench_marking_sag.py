"""Do this clip's straight markings bow, and does the bow grow with radius? — the lens question.

**Neither repo models lens distortion.** pitch3d has a `CameraIntrinsics.distortion` field that is
`None` on every solve it produces (`bench_camera_model_gap.py` prints exactly that), and camlab has
no distortion term at all. camlab measured whether it needed one and answered no — *"the bending is
real and it is 0.37 px, the residual it is offered to explain is 14–24 px, and its direction is
random, which a lens cannot produce"*. That verdict was measured on **its** two clips. It does not
transfer to a clip shot on a different lens, and `MOR_POR_181952` is the case that asks.

This is camlab's own measurement, run from here so the numbers are comparable to its table:

* connected runs of painted **centreline** — no line detector, because a straight-line finder
  cannot be asked whether lines are straight;
* keep runs longer than ``MIN_SPAN_PX`` that a parabola fits better than ``MAX_FIT_PX``, so what is
  measured is one smooth marking and not the junction of two;
* **sag** is the parabola's departure from its own chord at mid-span;
* and the two things that tell a lens from everything else: does the bow grow with radius from the
  optical axis, and does it bow consistently one way. A lens bends every line the same way and
  harder further out. Nothing else does.

Run::

    .venv/bin/python scripts/bench_marking_sag.py --clip MOR_POR_181952 --frame 35
"""

from __future__ import annotations

import argparse
import urllib.request

import cv2
import numpy as np

#: camlab's own thresholds, so the output can be read against its table.
MIN_SPAN_PX = 120.0
MAX_FIT_PX = 1.2


def paint_mask(server: str, clip: str, frame: int) -> np.ndarray:
    url = f"{server}/api/run/{clip}/paint/{frame}.png"
    with urllib.request.urlopen(url, timeout=600) as r:  # noqa: S310 - configured host
        buf = np.frombuffer(r.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"{clip} f{frame}: no paint mask")
    return (img > 127).astype(np.uint8)


def runs(mask: np.ndarray) -> list[np.ndarray]:
    """Connected runs of the thinned centreline, as point arrays."""
    from skimage.morphology import skeletonize

    thin = skeletonize(mask.astype(bool)).astype(np.uint8)
    n, lab = cv2.connectedComponents(thin, 8)
    out = []
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        if len(xs) >= 40:
            out.append(np.column_stack([xs, ys]).astype(float))
    return out


def sag(points: np.ndarray) -> tuple[float, float, float] | None:
    """``(span, sag, fit_error)`` for one run, in its own along/across frame, or None if not clean.

    Sign convention on the sag is the caller's business — it needs the optical axis to say whether
    a bow is *toward* it, and that is what separates a lens from a wobbly line.
    """
    centre = points.mean(0)
    _u, _s, vt = np.linalg.svd(points - centre)
    along, across = vt[0], np.array([-vt[0][1], vt[0][0]])
    t = (points - centre) @ along
    d = (points - centre) @ across
    span = float(t.max() - t.min())
    if span < MIN_SPAN_PX:
        return None
    coeffs = np.polyfit(t, d, 2)
    fit = float(np.abs(np.polyval(coeffs, t) - d).max())
    if fit > MAX_FIT_PX:
        return None
    # Departure of the parabola from its own chord at mid-span: with d(t) = a t² + b t + c the
    # chord between the ends is linear, so the gap at the midpoint is a·(span/2)² in size.
    return span, float(coeffs[0] * (span / 2.0) ** 2), fit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", default="0-59", help="frame range, e.g. 0-59, or one number")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--server", default="http://127.0.0.1:8899")
    args = ap.parse_args()

    # Over MANY frames, because one frame yields one or two clean markings and camlab's own table
    # rests on 514 and 441. A median of two is not a median.
    lo, _, hi = args.frames.partition("-")
    frames = range(int(lo), int(hi or lo) + 1, args.stride)

    spans, sags, radii, toward = [], [], [], []
    w = h = 0
    for f in frames:
        try:
            mask = paint_mask(args.server, args.clip, f)
        except Exception:  # noqa: BLE001 - a frame that will not decode is not the question here
            continue
        h, w = mask.shape
        axis = np.array([w / 2.0, h / 2.0])
        for pts in runs(mask):
            got = sag(pts)
            if got is None:
                continue
            span, s, _fit = got
            centre = pts.mean(0)
            to_axis = axis - centre
            _u, _sv, vt = np.linalg.svd(pts - centre)
            across = np.array([-vt[0][1], vt[0][0]])
            spans.append(span)
            sags.append(abs(s))
            radii.append(float(np.hypot(*to_axis)))
            toward.append(bool(np.sign(s) == np.sign(across @ to_axis)))

    if not sags:
        print(f"{args.clip}: no clean marking longer than {MIN_SPAN_PX:.0f} px — "
              f"nothing to measure. That is a finding about the paint, not about the lens.")
        return 1

    sags_a, radii_a = np.asarray(sags), np.asarray(radii)
    inner = sags_a[radii_a <= np.median(radii_a)]
    outer = sags_a[radii_a > np.median(radii_a)]
    print(f"{args.clip} frames {args.frames}   ({w}x{h}, optical axis assumed at the centre)")
    print(f"  clean markings      {len(sags)}")
    print(f"  median span         {np.median(spans):.0f} px")
    print(f"  median |sag|        {np.median(sags_a):.2f} px")
    print(f"  90th percentile     {np.percentile(sags_a, 90):.2f} px")
    print(f"  bows toward axis    {100 * np.mean(toward):.0f} %   (a lens is consistent; "
          f"~50 % is noise)")
    fmt = lambda a: f"{np.median(a):.2f}" if len(a) else "  n/a"  # noqa: E731
    print(f"  |sag| inner half    {fmt(inner)} px")
    print(f"  |sag| outer half    {fmt(outer)} px   (a lens bends harder further out)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
