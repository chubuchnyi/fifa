"""M-1 — can ONE camera centre explain a handheld clip, or does the phone really translate?

This is the measurement that decides whether `docs/camlab-spec.md` gets built. It exists because
[`architecture-brief-2026-08-09.md`](../docs/findings/architecture-brief-2026-08-09.md:57)
concluded that for handheld footage *"novel view does not exist"*, on the grounds that a phone
translates every frame — and that half was never measured. WorldPose, where the 0.000 m in 89/89
comes from, contains no phone clips.

**The confound this script exists to remove.** The fan clip zooms **1.66×** over f0–130. Fitting
#119's model — one focal, one centre — to it would fail from the zoom alone, and the failure would
look exactly like translation. So a single fit answers nothing. Three numbers do:

| | model | DOF | what it isolates |
|---|---|---|---|
| **seed** | the stored per-frame homographies | 8 F | how good the paint evidence is at all |
| **B** | per-frame focal + rotation, **ONE centre** | 3 + 4 F | the cost of fixing the centre |
| **A** | one focal, one centre, rotation per frame | 4 + 3 F | the extra cost of fixing the focal |

B is the diagnostic. It hands the model **every** advantage except a moving centre: the focal is
free per frame and unpenalised, so anything the zoom can explain, it will explain. If B still lands
far from the seed, what is left is translation, and the 2026-08-09 conclusion stands.

Read it as:

* **B ≈ seed** → one centre is defensible. The storm is surplus DOF, not physics. Build `camlab`.
* **B ≫ seed, A ≈ B** → the centre cannot be fixed. Novel view really does not exist for this
  clip class, and a day here saved a month.
* **B ≈ seed, A ≫ B** → the centre is fine and the *zoom* is what one-focal models cannot hold.
  That is the `f_t` curve in the spec, and it is a design input, not a refutation.

**What this is not.** It scores paint only — no SIFT/MAGSAC pan term, no jitter budget. #119's own
docstring records that a paint-only fit jitters worse than the free homographies it replaces, so
these parameters are a *feasibility verdict*, not a camera to ship. Producing one is M2.

Run:

    PYTHONPATH=src .venv/bin/python scripts/probe_handheld_centre.py            # fan clip
    PYTHONPATH=src .venv/bin/python scripts/probe_handheld_centre.py --control  # tripod control
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: The handheld case. Its homographies live in the AUTO-CROP rect, `1080x608+0+1294`, not in the
#: full portrait frame — `--crop auto` sets `ClipRef.crop` and every adapter decodes through it
#: (`adapters/io/frames.py:iter_clip_frames`), so PnLCalib only ever saw 1080x608.
#:
#: Measured 2026-08-10 against the paint, because guessing costs a whole run. Projecting the pitch
#: markings through the stored homographies and scoring against the painted lines:
#:
#:      space                    rot180 | paint median      n
#:      crop 1080x608+0+1294     False  |       9.47 px  1271   <- wins on BOTH
#:      crop 1080x608+0+1294     True   |      16.95 px  1148
#:      full 1080x1920           True   |      17.32 px  1168
#:      full 1080x1920           False  |      30.59 px    36
#:
#: The sample count is half the evidence: a wrong space projects most markings off-surface, where
#: they are unscored, so it can post a flattering median on a handful of survivors. The crop wins
#: on the median AND on how much of the pitch it manages to place at all.
#:
#: NB this clip has **no 180 roll** — the roll in `landmines.md` is a property of the solved
#: CameraTrack, not of every calibration.
FAN = dict(
    scene=ROOT / "out/fan_auto/scene_fan_auto.json",
    video=ROOT / "samples/video/14604731_1080_1920_30fps.mp4",
    width=1080, height=608, crop=(1080, 608, 0, 1294), n_frames=120,
)

#: The tripod control. Same code, a clip where the answer is known: #119 lands at 1.4 px here.
CONTROL = dict(
    scene=ROOT / "out/carry_off/export/scene.json",
    video=ROOT / "samples/video/Colombia-1-0-Congo-DR1080p.mp4",
    width=1920, height=1080, crop=None, n_frames=60,
)


def _patch_globals(width: int, height: int) -> None:
    """Point the two benchmark modules at this clip's image space.

    `fit_rigid_camera` and `bench_frame_preprocessing` both hold `WIDTH, HEIGHT = 1920, 1080` at
    module level and read them inside `kmat`, `decompose`, `paint_error` and the pan grid. They
    were written for one clip. Rather than thread a parameter through 544 lines that are pinned by
    a golden test, this rebinds them — which is honest for a probe and is exactly the parameter
    `camlab` has to introduce properly (spec §6.2).
    """
    import scripts.bench_frame_preprocessing as bfp
    import scripts.fit_rigid_camera as frc

    for mod in (frc, bfp):
        mod.WIDTH, mod.HEIGHT = width, height
    frc.PAN_GRID = np.column_stack([
        g.ravel() for g in np.meshgrid(
            np.linspace(0.1 * width, 0.9 * width, 9), np.linspace(0.1 * height, 0.9 * height, 5)
        )
    ] + [np.ones(45)])


def load_w2i(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """World→image per frame in the honest right-handed Z-up world (#118), plus a suspect mask.

    `fit_rigid_camera.load_world_to_image` asserts the whole clip agrees on handedness. On the fan
    clip it does not: frames 115 and 117 measure mirrored and the other 118 do not. They are not a
    mid-clip frame change — they are near-degenerate homographies, `|det| = 1.0e-6` and `5.3e-8`
    against a clip median of `3.4e-3`, whose plane has collapsed far enough toward a line that the
    orientation test reads the wrong sign. Both pass `plane_camera._SINGULAR_DET = 1e-12` by six
    orders of magnitude and both carry an unremarkable confidence (0.48, 0.39) — so neither
    existing guard sees them.

    Majority vote on the handedness, and hand back which frames to leave out.
    """
    import json

    from poseannot.camera import plane_orientation

    blob = json.loads(path.read_text())
    cal = blob["fields"]["field"]["fields"]["calibration"]["fields"]
    i2w = np.asarray(cal["homographies"]["__ndarray__"]["data"], dtype=float)
    w2i = np.stack([np.linalg.inv(h) for h in i2w])

    det = np.abs(np.linalg.det(i2w))
    #: Relative, not absolute: what matters is a frame collapsing against ITS OWN clip, and the
    #: absolute scale of |det| depends on the image size and the world units.
    rank_poor = det < 1e-3 * np.median(det)

    mirrored = np.array([plane_orientation(h, width, height) < 0 for h in w2i])
    if mirrored.mean() > 0.5:
        w2i = w2i @ np.diag([1.0, -1.0, 1.0])
        odd = ~mirrored
    else:
        odd = mirrored
    suspect = odd | rank_poor
    if suspect.any():
        print(f"    !! {int(suspect.sum())} suspect frame(s) excluded: "
              f"{np.flatnonzero(suspect).tolist()}")
        print(f"       handedness outlier {np.flatnonzero(odd).tolist()}  "
              f"rank-poor {np.flatnonzero(rank_poor).tolist()}  "
              f"(|det| median {np.median(det):.3g})")
    return w2i, suspect


def paint_trees(video: Path, frames: list[int], rot180: bool, crop=None) -> dict:
    """Per frame: KD-tree of painted centreline px, their coords, the playing-surface mask.

    `crop` must be the SAME rect the calibrator decoded through, or every number below is measured
    in a space the homographies do not live in.
    """
    from poseannot.pitch_evidence import _masks
    from poseannot.video import read_frame

    from pitch3d.adapters.io.frames import apply_crop

    out = {}
    for n, i in enumerate(frames):
        bgr = apply_crop(read_frame(str(video), i), crop)
        if rot180:
            bgr = np.ascontiguousarray(bgr[::-1, ::-1])
        dist, surface = _masks(bgr)
        spine = np.argwhere(dist == 0)[:, ::-1].astype(float)  # (u, v)
        out[i] = (cKDTree(spine), spine, surface)
        if n % 4 == 0:
            print(f"      frame {i:3d}: {len(spine):6d} painted centreline px", flush=True)
    return out


# --------------------------------------------------------------------------------------------
# Variant B: per-frame focal, per-frame rotation, ONE centre.
#
# `fit_rigid_camera._solve` spends exactly one focal on the clip (`p[0]`), which is the whole point
# of #119 and the wrong instrument here. This is that solver with the focal moved inside the
# per-frame block, and with NO smoothness penalty on it: the question is whether a fixed centre can
# survive when the zoom is given free rein, so tying the zoom down would beg it.
# --------------------------------------------------------------------------------------------

def plane_h_b(focal: float, rvec: np.ndarray, centre: np.ndarray, w: int, h: int) -> np.ndarray:
    from scripts.fit_rigid_camera import rodrigues

    rot = rodrigues(rvec)
    k = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]])
    return k @ np.column_stack([rot[:, 0], rot[:, 1], -rot @ centre])


def _pack_b(centre: np.ndarray, focals: np.ndarray, rvecs: np.ndarray) -> np.ndarray:
    return np.concatenate([centre, np.column_stack([focals, rvecs]).ravel()])


def _unpack_b(q: np.ndarray, j: int) -> tuple[float, np.ndarray]:
    return float(q[3 + 4 * j]), q[4 + 4 * j : 7 + 4 * j]


def _solve_b(q, src, dst, owner, n_f, w, h, smooth: float = 0.0):
    """One least-squares step of the one-centre / free-focal camera.

    Two guards that the first version of this function did without, and paid for. With a fixed
    centre, focal and distance-to-pitch trade off almost exactly, so a per-frame focal left both
    unbounded and unpenalised is free to collapse: the first run reached **5 px** of focal on a
    1080 px wide image, projected the markings into a smear, and scored a *better-looking* median
    on the leftovers. "Give the model every advantage" is not the same as "let it be degenerate".

    * **Bounds.** `FOCAL_BOUNDS` — the same (300, 20000) px bracket `plane_camera` searches, wide
      enough for a phone wide angle and a broadcast long lens both.
    * **`smooth`.** Optional penalty on `f_{j+1} - f_j`, normalised by the mean focal so the weight
      does not depend on the lens. A real zoom ramp is smooth; 0 keeps it off so it can be shown
      the answer does not come from this term.
    """
    from scripts.fit_rigid_camera import project

    from pitch3d.core.scene.plane_camera import FOCAL_BOUNDS

    def residuals(x):
        out = np.empty_like(dst)
        for j in range(n_f):
            m = owner == j
            if m.any():
                f, rv = _unpack_b(x, j)
                out[m] = project(plane_h_b(f, rv, x[0:3], w, h), src[m])
        paint = (out - dst).ravel()
        if smooth <= 0.0 or n_f < 2:
            return paint
        f = x[3::4]
        return np.concatenate([paint, smooth * np.diff(f) / max(float(f.mean()), 1e-9)])

    n_pen = 0 if (smooth <= 0.0 or n_f < 2) else n_f - 1
    spar = np.zeros((2 * len(owner) + n_pen, len(q)), dtype=np.uint8)
    spar[:2 * len(owner), :3] = 1                      # the centre touches every paint row
    for j in range(n_f):
        hit = np.flatnonzero(owner == j)
        rows = np.concatenate([2 * hit, 2 * hit + 1])
        spar[np.ix_(rows, list(range(3 + 4 * j, 7 + 4 * j)))] = 1
    for j in range(n_pen):
        spar[2 * len(owner) + j, [3 + 4 * j, 3 + 4 * (j + 1)]] = 1
    lo = np.full(len(q), -np.inf)
    hi = np.full(len(q), np.inf)
    lo[3::4], hi[3::4] = FOCAL_BOUNDS[0], FOCAL_BOUNDS[1]
    q = np.clip(q, lo, hi)
    return least_squares(residuals, q, jac_sparsity=spar, x_scale="jac", loss="soft_l1",
                         f_scale=3.0, bounds=(lo, hi), max_nfev=400, verbose=0).x


def fit_b(frames, evidence, xy1, seed_focal, w2i, w, h, rounds=4, smooth=0.0):
    """ICP a one-centre / free-focal camera onto the paint. Returns (params, median px)."""
    from scripts.bench_frame_preprocessing import decompose
    from scripts.fit_rigid_camera import MATCH_PX, paint_error, project

    n_f = len(frames)
    poses = [decompose(w2i[i], seed_focal) for i in frames]
    q = _pack_b(
        np.median([c for _r, c in poses], axis=0),
        np.full(n_f, seed_focal),
        np.stack([_unrod(r) for r, _c in poses]),
    )

    # Coarse stage, same reasoning as #119: make the pose self-consistent against the homographies'
    # own lawn mapping before showing it any paint, or ICP matches half the samples to a
    # neighbouring line and converges there confidently.
    gx, gy = np.meshgrid(np.linspace(-52.5, 52.5, 60), np.linspace(-34.0, 34.0, 40))
    lawn = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    seen, want = [], []
    for i in frames:
        uv = project(w2i[i], lawn)
        keep = (uv[:, 0] > 0) & (uv[:, 0] < w) & (uv[:, 1] > 0) & (uv[:, 1] < h)
        seen.append(lawn[keep])
        want.append(uv[keep])
    own = np.concatenate([np.full(len(s), j) for j, s in enumerate(seen)])
    q = _solve_b(q, np.concatenate(seen), np.concatenate(want), own, n_f, w, h, smooth)

    for _round in range(rounds):
        src, dst, owner = [], [], []
        for j, i in enumerate(frames):
            tree, spine, surface = evidence[i]
            f, rv = _unpack_b(q, j)
            uv = project(plane_h_b(f, rv, q[0:3], w, h), xy1)
            ok = (uv[:, 0] > 1) & (uv[:, 0] < w - 2) & (uv[:, 1] > 1) & (uv[:, 1] < h - 2)
            idx = np.flatnonzero(ok)
            if not len(idx):
                continue
            sub = uv[idx]
            on = surface[np.rint(sub[:, 1]).astype(int), np.rint(sub[:, 0]).astype(int)] > 0
            idx, sub = idx[on], sub[on]
            if not len(idx):
                continue
            d, nn = tree.query(sub, distance_upper_bound=MATCH_PX)
            hit = np.isfinite(d)
            src.append(xy1[idx[hit]])
            dst.append(spine[nn[hit]])
            owner.append(np.full(int(hit.sum()), j))
        if not src:
            return q, float("nan")
        q = _solve_b(q, np.concatenate(src), np.concatenate(dst), np.concatenate(owner),
                     n_f, w, h, smooth)

    err = np.concatenate([
        paint_error(plane_h_b(*_unpack_b(q, j), q[0:3], w, h), *evidence[i][::2], xy1)
        for j, i in enumerate(frames)
    ])
    return q, float(np.median(err)) if err.size else float("nan")


def _unrod(rot):
    from scripts.fit_rigid_camera import unrodrigues
    return unrodrigues(rot)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", action="store_true", help="run the tripod clip instead")
    ap.add_argument("--frames", type=int, default=12, help="evenly spaced frames to score on")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=None, help="default: the clip's solved span")
    ap.add_argument("--focal-smooth", type=float, default=0.0, metavar="W",
                    help="penalty weight on frame-to-frame focal change in variant B "
                         "(0 = off, so the answer cannot come from this term)")
    ap.add_argument("--rot180", choices=["auto", "yes", "no"], default="auto",
                    help="the solved camera is 180-rolled vs raw video on some clips; auto picks "
                         "whichever orientation gives the smaller SEED paint error")
    args = ap.parse_args()

    cfg = CONTROL if args.control else FAN
    w, h = cfg["width"], cfg["height"]
    _patch_globals(w, h)

    from scripts.fit_rigid_camera import fit, marking_samples, paint_error

    tag = "TRIPOD CONTROL" if args.control else "HANDHELD (fan clip)"
    print(f"\n=== M-1  {tag} ===")
    print(f"    scene {cfg['scene'].relative_to(ROOT)}")
    print(f"    video {cfg['video'].name}   image space {w}x{h}")

    w2i, suspect = load_w2i(cfg["scene"], w, h)
    last = args.last if args.last is not None else cfg["n_frames"] - 1
    want = np.linspace(args.first, min(last, len(w2i) - 1), args.frames)
    ok = np.flatnonzero(~suspect)
    # Snap each wanted frame to the nearest usable one rather than dropping it: the fit needs its
    # frames spread over the span, and losing one to a degenerate homography should cost a
    # neighbour, not a hole in the coverage.
    frames = sorted({int(ok[np.argmin(np.abs(ok - x))]) for x in want})
    xy1 = marking_samples()
    print(f"    frames  {frames}")
    print(f"    marking samples: {len(xy1)}")

    # --- orientation: measure it, do not assume it -------------------------------------------
    def seed_err(ev):
        e = [paint_error(w2i[i], *ev[i][::2], xy1) for i in frames]
        e = [x for x in e if x.size]
        return np.concatenate(e) if e else np.array([])

    if args.rot180 == "auto":
        print("\n--- orientation check (raw vs 180-rolled) ---")
        cand = {}
        for flag in (False, True):
            ev = paint_trees(cfg["video"], frames, flag, cfg["crop"])
            e = seed_err(ev)
            cand[flag] = (ev, e)
            med = np.median(e) if e.size else float("nan")
            print(f"    rot180={str(flag):5s}: seed paint median {med:8.2f} px  (n={e.size})")
        # Rank on the median only among orientations that place a comparable amount of the pitch.
        # A wrong orientation projects most markings off-surface, where they go unscored, so it can
        # post a flattering median on a handful of survivors — which is exactly how the first run
        # of this probe produced a confident wrong verdict.
        best_n = max(c[1].size for c in cand.values())
        live = {k: v for k, v in cand.items() if v[1].size >= 0.5 * best_n} or cand
        rot = min(live, key=lambda k: np.median(live[k][1]) if live[k][1].size else np.inf)
        evidence, seed = cand[rot]
        print(f"    -> using rot180={rot}  (n floor {0.5 * best_n:.0f} of {best_n})")
    else:
        rot = args.rot180 == "yes"
        evidence = paint_trees(cfg["video"], frames, rot, cfg["crop"])
        seed = seed_err(evidence)

    if not seed.size:
        print("\n!! the stored homographies project NO marking onto painted surface. "
              "The paint evidence cannot judge this clip; stop here and read the frames by eye.")
        return

    seed_med = float(np.median(seed))
    print(f"\n[seed]  stored per-frame homographies, 8 DOF x {len(frames)} frames")
    print(f"        paint {seed_med:7.2f} px   (p90 {np.percentile(seed, 90):7.2f})")

    focal0 = float(np.median([_focal_guess(w2i[i], w, h) for i in frames]))
    print(f"        seed focal for the fits: {focal0:.1f} px")

    t0 = time.time()
    print(f"\n[B]     ONE centre, focal FREE per frame, rotation per frame  "
          f"({3 + 4 * len(frames)} params)")
    qb, _ = fit_b(frames, evidence, xy1, focal0, w2i, w, h, smooth=args.focal_smooth)
    mb = [plane_h_b(*_unpack_b(qb, j), qb[0:3], w, h) for j in range(len(frames))]
    errb, nb, p90b = _paint_stats(mb, frames, evidence, xy1)
    fb = qb[3::4]
    print(f"        paint {errb:7.2f} px (p90 {p90b:7.2f})  n={nb}")
    print(f"        centre ({qb[0]:7.2f},{qb[1]:7.2f},{qb[2]:6.2f}) m   "
          f"focal {fb.min():.0f} -> {fb.max():.0f} px (x{fb.max() / max(fb.min(), 1e-9):.2f})")
    print(f"        focal curve: {' '.join(f'{v:.0f}' for v in fb)}")
    print(f"        {time.time() - t0:.0f}s")

    t0 = time.time()
    print(f"\n[A]     ONE centre, ONE focal, rotation per frame  ({4 + 3 * len(frames)} params) "
          f"— the #119 model")
    pa, _ = fit(frames, evidence, xy1, focal0, w2i, rounds=4)
    from scripts.fit_rigid_camera import plane_h
    ma = [plane_h(pa[0], pa[4 + 3 * j:7 + 3 * j], pa[1:4]) for j in range(len(frames))]
    erra, na, p90a = _paint_stats(ma, frames, evidence, xy1)
    print(f"        paint {erra:7.2f} px (p90 {p90a:7.2f})  n={na}")
    print(f"        centre ({pa[1]:7.2f},{pa[2]:7.2f},{pa[3]:6.2f}) m   focal {pa[0]:.0f} px")
    print(f"        {time.time() - t0:.0f}s")

    # --- verdict, with the coverage guard that the first version of this probe lacked ----------
    # `paint_error` scores only markings that land inside the image AND on the playing surface. A
    # camera that has run away to a degenerate focal projects almost everything off-surface, where
    # it goes unscored, and posts a flattering median on the handful of survivors. Comparing
    # medians without comparing n is how this script first reported "ONE CENTRE HOLDS" off a fit
    # whose focal was 87 px on a 1080 px image.
    print("\n--- verdict ---")
    print(f"    {'model':<26} {'paint px':>9} {'n':>7} {'vs seed':>9}")
    print(f"    {'seed (8 DOF per frame)':<26} {seed_med:9.2f} {seed.size:7d} {'—':>9}")
    print(f"    {'B (centre fixed)':<26} {errb:9.2f} {nb:7d} {errb / seed_med:8.2f}x")
    print(f"    {'A (centre + focal fixed)':<26} {erra:9.2f} {na:7d} {erra / seed_med:8.2f}x")

    floor = 0.6 * seed.size
    lost = [n for n, c in (("B", nb), ("A", na)) if c < floor]
    physical = 0.2 * w < fb.min() and fb.max() < 30 * w and 0.2 * w < pa[0] < 30 * w
    print()
    if lost:
        print(f"    VERDICT INVALID — {', '.join(lost)} scored on under 60% of the seed's samples")
        print(f"    ({floor:.0f}). Those markings did not get better, they left the frame. Fix the")
        print("    fit before reading the medians.")
    elif not physical:
        print(f"    VERDICT INVALID — a fitted focal is unphysical for a {w} px wide image "
              f"(B {fb.min():.0f}-{fb.max():.0f}, A {pa[0]:.0f} px).")
        print("    The optimiser ran away; bound the focal and re-run.")
    elif errb <= max(1.6 * seed_med, seed_med + 2.0):
        print("    ONE CENTRE HOLDS. Fixing the camera position costs little against free")
        print("    per-frame homographies, so the storm is surplus DOF, not translation.")
        print("    -> the 2026-08-09 'novel view does not exist' conclusion does not survive this.")
    else:
        print("    ONE CENTRE DOES NOT HOLD. With the focal free per frame and unpenalised, a")
        print("    fixed position still cannot reproduce the paint.")
        print("    -> the 2026-08-09 conclusion stands for this clip class.")
    if np.isfinite(erra) and np.isfinite(errb) and erra > 1.6 * errb and not lost:
        print("    Separately: one focal costs much more than one centre — that is the 1.66x zoom,")
        print("    and it is why the spec's f_t curve is not optional.")


def _paint_stats(maps, frames, evidence, xy1) -> tuple[float, int, float]:
    """(median px, sample count, p90) — the count is not decoration, see the verdict guard."""
    from scripts.fit_rigid_camera import paint_error

    e = [paint_error(h, *evidence[i][::2], xy1) for h, i in zip(maps, frames, strict=True)]
    e = [x for x in e if x.size]
    if not e:
        return float("nan"), 0, float("nan")
    a = np.concatenate(e)
    return float(np.median(a)), int(a.size), float(np.percentile(a, 90))


def _focal_guess(h_w2i: np.ndarray, w: int, h: int) -> float:
    """The focal that best makes this one homography come from a real rotation (Zhang)."""
    from pitch3d.core.scene.plane_camera import FOCAL_BOUNDS, _k_inv, _orthonormality

    grid = np.geomspace(*FOCAL_BOUNDS, 160)
    cost = [_orthonormality(h_w2i, _k_inv(float(f), w / 2, h / 2)) for f in grid]
    return float(grid[int(np.argmin(cost))])


if __name__ == "__main__":
    main()
