"""Is the principal point actually at the image centre? (#140 follow-up, 2026-08-10)

``fit_rigid_camera.kmat()`` hardcodes ``cx, cy = W/2, H/2`` and never fits them, and
``apply_rigid_camera.py:138`` writes that assumption into every scene's ``CameraIntrinsics``
beside a genuinely fitted focal. PnLCalib *returns* a ``principal_point``; the fit that consumes
its output drops it. So two of the four numbers in our K are assumed, and nothing has ever checked
the assumption.

It matters because #140 leaves open: *"remaining residual grows 6.2 -> 15.7 px centre-to-edge,
which is where distortion becomes testable."* A displaced principal point produces **exactly that
signature** — small error near the optical axis, growing away from it — at 2 free parameters
against a distortion model's 2-5. Fitting distortion first, to an error that is really a shifted
centre, would absorb it into the wrong term and still look like it worked.

Three stages, cheapest first:

1. **The radial profile at the shipped fit.** Bin the paint residual by distance from the assumed
   centre. If it is flat, the assumption is not costing anything measurable and stages 2-3 are
   unnecessary. This needs one fit and no search.
2. **A grid over (cx, cy).** Refit *everything else* at each candidate centre and report the paint
   residual. Improvement alone is weak evidence — two more parameters over ~19 000 residuals should
   buy something — so the reported verdict is the change in **radial slope**, which overfitting two
   parameters does not flatten.
3. **The pan instrument, which is the one that can actually see it.** On a plane, a principal-point
   shift is nearly degenerate with a small rotation, so the paint may be unable to separate them.
   ``K R Rt K^-1`` is *exact at the principal point for every focal* (``fit_rigid_camera``'s own
   note at :data:`PAN_GRID`), so the measured image->image motion is sensitive to the principal
   point in a way the plane homography is not. This is the discriminator, not the paint.

    PYTHONPATH=src .venv/bin/python scripts/bench_principal_point.py --line --span 900 --steps 7
    PYTHONPATH=src .venv/bin/python scripts/bench_principal_point.py --pan --span 240 --steps 5

CPU. ~11 s to read the paint, then ~9 s per candidate centre (~50 s with ``--pan``).

**Verdict, 2026-08-10: the principal point is NOT the cause, and it is not identifiable here.**

* ``cy`` is **flat** over ±900 px — paint 1.415 to 1.443, a 2 % spread across a sweep wider than
  the image is tall. The data carries no information about it at all.
* ``cx`` has a shallow minimum at **+600 px**, i.e. 81 % of the way across a 1920-wide frame. That
  is not a principal point, and the focal walks with it monotonically (4318 → 4099 px over the
  sweep): it is a valley in *(cx, focal)*, not a measurement.
* And the growth it was invoked to explain **is not radial**. Binned separately, ``|v-cy|`` reads
  1.12 / **2.54** / 1.26 / 1.08 px — it peaks near the middle and *falls* toward the edge. No lens
  does that. The apparent "centre-to-edge" rise is one localised band of bad paint, which radial
  binning smears into a slope.

So the hypothesis this script was written to test is refuted by it. Kept because the refutation is
the useful part: it removes the principal point from #140's candidate list and, with the paint
measured at 2.97 px where the overlay reads 15.7, points at grounding rather than at optics.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.fit_rigid_camera as frc  # noqa: E402

#: The principal point every call to the patched ``kmat`` will use. Module-level because
#: ``fit_rigid_camera`` threads the focal through a parameter vector and the centre through a
#: closure; adding two columns to that vector would mean re-indexing every ``q[4 + 3 * j]`` in the
#: file, which is a live file in another workstream. Patching the one function that reads them
#: leaves the fit itself byte-identical.
PRINCIPAL = [frc.WIDTH / 2.0, frc.HEIGHT / 2.0]


def _kmat(focal: float) -> np.ndarray:
    return np.array([[focal, 0, PRINCIPAL[0]], [0, focal, PRINCIPAL[1]], [0, 0, 1.0]])


frc.kmat = _kmat


def paint_radial(
    p: np.ndarray, frames: list[int], evidence: dict, xy1: np.ndarray, bins: int = 5
) -> tuple[np.ndarray, np.ndarray, float]:
    """Median paint residual binned by radius from the ASSUMED centre. ``(edges, medians, slope)``.

    Radius is measured from ``PRINCIPAL`` rather than from the image centre, so that when stage 2
    moves the candidate the profile follows it — the question is always "does the error grow away
    from the optical axis", and the optical axis is what is being varied.
    """
    rad, err = [], []
    for j, i in enumerate(frames):
        tree, _spine, surface = evidence[i]
        h = frc.plane_h(p[0], p[4 + 3 * j : 7 + 3 * j], p[1:4])
        uv = frc.project(h, xy1)
        ok = ((uv[:, 0] > 1) & (uv[:, 0] < frc.WIDTH - 2)
              & (uv[:, 1] > 1) & (uv[:, 1] < frc.HEIGHT - 2))
        sub = uv[ok]
        if not len(sub):
            continue
        on = surface[np.rint(sub[:, 1]).astype(int), np.rint(sub[:, 0]).astype(int)] > 0
        if not on.any():
            continue
        sub = sub[on]
        rad.append(np.linalg.norm(sub - np.asarray(PRINCIPAL), axis=1))
        err.append(tree.query(sub)[0])
    r, e = np.concatenate(rad), np.concatenate(err)
    edges = np.linspace(0.0, np.percentile(r, 98), bins + 1)
    med = np.array([
        np.median(e[(r >= a) & (r < b)]) if ((r >= a) & (r < b)).any() else np.nan
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ])
    # px of residual gained per 100 px of radius — the number a displaced centre inflates and a
    # correct one flattens. Fitted on the bin medians so one noisy far bin cannot dominate.
    mid = (edges[:-1] + edges[1:]) / 2.0
    good = np.isfinite(med)
    if good.sum() < 2:
        return edges, med, float("nan")
    return edges, med, float(np.polyfit(mid[good], med[good], 1)[0] * 100.0)


def axis_profiles(p: np.ndarray, frames: list[int], evidence: dict, xy1: np.ndarray) -> str:
    """Growth along |u-cx| vs |v-cy| — the control that says whether "radial" is really radial.

    A lens grows its error symmetrically about the optical axis. The *pitch* does something that
    looks similar and is not optical: the far field sits near the horizon, high in the frame, where
    the markings compress and the ridge filter's centreline is worth less. That is growth in **v**
    alone. If the two profiles disagree, the effect is perspective and no intrinsic will fix it.
    """
    du, dv, err = [], [], []
    for j, i in enumerate(frames):
        tree, _spine, surface = evidence[i]
        h = frc.plane_h(p[0], p[4 + 3 * j : 7 + 3 * j], p[1:4])
        uv = frc.project(h, xy1)
        ok = ((uv[:, 0] > 1) & (uv[:, 0] < frc.WIDTH - 2)
              & (uv[:, 1] > 1) & (uv[:, 1] < frc.HEIGHT - 2))
        sub = uv[ok]
        if not len(sub):
            continue
        on = surface[np.rint(sub[:, 1]).astype(int), np.rint(sub[:, 0]).astype(int)] > 0
        if not on.any():
            continue
        sub = sub[on]
        du.append(np.abs(sub[:, 0] - PRINCIPAL[0]))
        dv.append(np.abs(sub[:, 1] - PRINCIPAL[1]))
        err.append(tree.query(sub)[0])
    e = np.concatenate(err)
    out = []
    for name, d in (("|u-cx|", np.concatenate(du)), ("|v-cy|", np.concatenate(dv))):
        edges = np.linspace(0.0, np.percentile(d, 98), 5)
        med = [np.median(e[(d >= a) & (d < b)]) if ((d >= a) & (d < b)).any() else np.nan
               for a, b in zip(edges[:-1], edges[1:], strict=True)]
        cells = "  ".join(f"{a:4.0f}-{b:4.0f}:{m:5.2f}"
                          for a, b, m in zip(edges[:-1], edges[1:], med, strict=True))
        out.append(f"    {name}  {cells}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--seed-focal", type=float, default=4179.0)
    ap.add_argument("--span", type=float, default=80.0, help="px either side of the image centre")
    ap.add_argument("--steps", type=int, default=5, help="grid points per axis (odd keeps the "
                                                         "assumed centre in the grid)")
    ap.add_argument("--line", action="store_true",
                    help="sweep cx alone then cy alone instead of the full grid — the cheap way to "
                         "ask the only question that matters, which is whether the optimum is "
                         "INTERIOR. A grid whose best cell is a corner has not found a minimum, it "
                         "has found a direction the fit can walk down.")
    ap.add_argument("--pan", action="store_true", help="also fit and report the measured pixel "
                                                       "motion — the instrument that can actually "
                                                       "separate a centre shift from a rotation")
    ap.add_argument("--pan-cache", type=Path, default=Path("/tmp/pan_pairs.npz"))
    args = ap.parse_args()

    frames = list(range(args.frames))
    w2i = frc.load_world_to_image(frc.SCENE)
    xy1 = frc.marking_samples()
    print(f"{len(frames)} frames, {len(xy1)} marking samples", flush=True)
    t0 = time.time()
    evidence = frc.paint_trees(frames)
    print(f"  paint read in {time.time() - t0:.0f} s", flush=True)

    pan = None
    if args.pan:
        pairs, measured = frc.pixel_motion(frames, frc.PAN_GAPS, args.pan_cache)
        pan = (pairs, frc.pan_seen(measured))

    def run() -> tuple[np.ndarray, float, float, float]:
        p, med = frc.fit(frames, evidence, xy1, args.seed_focal, w2i, pan=pan)
        _e, _m, slope = paint_radial(p, frames, evidence, xy1)
        turn = (float(np.median(frc.pan_error(frc.pan_maps(p, pan[0]), pan[1])))
                if pan is not None else float("nan"))
        return p, med, slope, turn

    # ---------------------------------------------------------------- 1. is there anything to fix?
    print("\n== 1. radial profile of the paint residual at the ASSUMED centre ==")
    p0, med0, slope0, turn0 = run()
    edges, prof, _ = paint_radial(p0, frames, evidence, xy1)
    print(f"  f={p0[0]:.1f}  paint median {med0:.3f} px" + (f"  pan {turn0:.2f} px" if pan else ""))
    for a, b, m in zip(edges[:-1], edges[1:], prof, strict=True):
        print(f"    radius {a:6.0f}-{b:6.0f} px : {m:5.2f} px")
    print(f"  slope {slope0:+.3f} px per 100 px of radius")
    print("  A flat profile means the assumed centre costs nothing measurable on the paint.")
    print("  Control — is the growth really radial, or only vertical (the far field at the")
    print("  horizon, which no intrinsic can fix)?")
    print(axis_profiles(p0, frames, evidence, xy1))

    # ---------------------------------------------------------------- 2. does moving it help?
    off = np.linspace(-args.span, args.span, args.steps)
    if args.line:
        cells = ([(d, 0.0) for d in off] + [(0.0, d) for d in off if d])
        print(f"\n== 2. cx alone, then cy alone, +-{args.span:.0f} px ==")
    else:
        cells = [(dx, dy) for dy in off for dx in off]
        print(f"\n== 2. grid over (cx, cy), everything else refitted at each point "
              f"({args.steps}x{args.steps}) ==")
    print(f"{'cx':>8} {'cy':>8} {'paint':>8} {'slope/100px':>12} {'focal':>8}"
          + (f" {'pan':>7}" if pan else ""))
    best = (med0, list(PRINCIPAL), slope0, turn0)
    for dx, dy in cells:
        PRINCIPAL[0] = frc.WIDTH / 2.0 + dx
        PRINCIPAL[1] = frc.HEIGHT / 2.0 + dy
        p, med, slope, turn = run()
        mark = ""
        if med < best[0]:
            best = (med, list(PRINCIPAL), slope, turn)
            mark = "  <-"
        print(f"{PRINCIPAL[0]:8.1f} {PRINCIPAL[1]:8.1f} {med:8.3f} {slope:+12.3f} "
              f"{p[0]:8.1f}" + (f" {turn:7.2f}" if pan else "") + mark, flush=True)

    PRINCIPAL[0], PRINCIPAL[1] = best[1]
    print(f"\nbest (cx, cy) = ({best[1][0]:.1f}, {best[1][1]:.1f}) — "
          f"offset ({best[1][0] - frc.WIDTH / 2:+.1f}, {best[1][1] - frc.HEIGHT / 2:+.1f}) px")
    print(f"  paint {med0:.3f} -> {best[0]:.3f} px   ({100 * (best[0] - med0) / med0:+.1f} %)")
    print(f"  radial slope {slope0:+.3f} -> {best[2]:+.3f} px per 100 px")
    if pan:
        print(f"  pan   {turn0:.2f} -> {best[3]:.2f} px  <- the discriminator: on a plane a centre")
        print("        shift is near-degenerate with a rotation, so believe this column over paint")
    print("\nRead the shape of the sweep, not the best cell. Two extra parameters over ~19 000")
    print("residuals lower a median whatever the truth is, so a best cell at the grid EDGE means")
    print("no minimum was found — the fit walked down a valley. And the slope column is only")
    print("comparable within one candidate: radius is measured from the candidate centre, so")
    print("moving the centre re-bins the samples and flattens a profile for free.")


if __name__ == "__main__":
    main()
