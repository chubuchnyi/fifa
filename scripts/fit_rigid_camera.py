"""#119 — solve the clip as ONE camera, fitted to the paint instead of to 60 homographies.

The clip is currently 60 independent 8-parameter homographies: 480 free parameters with nothing
tying them together. #117 measured what that costs — 5.3 px of pitch-point wobble against a smooth
pan (62% of the real inter-frame motion is estimation noise), a camera centre that wanders 7-11 m
for a rig bolted to a gantry, and a focal that reads 2700, 3903 or 4277 depending on which
functional of the homographies you ask.

``bench_rigid_camera`` already answered the *re-reading* question and answered it no: the best
single pinhole reproduces those 60 homographies only to 11.5 px. That is the right answer to the
wrong question. The homographies are not the evidence — they are 60 noisy fits *to* the evidence,
and a rigid camera is being asked to reproduce their noise. This asks the question the other way
round: fit one camera **straight to the painted lines**, which is what #114 measured the current
solve at 1.4-1.7 px against, and let the 60 homographies fall out of it.

Two things make that tractable where scoring against a capped distance map did not:

* **Explicit correspondences, ICP-style.** The distance map is a scalar with a cap, so a sample
  that is 20 px out has no gradient at all. Its zero set is the paint's *centreline*, though, so a
  KD-tree over those pixels turns every sample into a real 2D correspondence with a real Jacobian.
  Re-matched each outer round, so the fit can move further than one match radius.
* **A sparse Jacobian.** Each frame's rotation touches only its own residuals, so 184 parameters
  cost ~8 function evaluations per Jacobian instead of 184.

The model is deliberately the strict one — one focal, one centre, one rotation per frame, no
temporal smoothing. Smoothing would *impose* the property being measured; leaving it out means the
jitter this reports is an honest output. Nothing here is fitted to a per-frame free parameter, so
if the number comes out good, it came from the physics.

The paint alone does not settle it. It pins where the camera points and how far away it is, but
its focal minimum is shallow (``--sweep``: 4.85 px at f=2700 down to 1.40 px at f=4600, with
f/distance near-constant throughout) and it says nothing at all about *time* — a fit to the paint
alone still jitters 6.4 px, worse than the 60 free homographies it replaces. So there is a second
instrument here, and it is a genuinely independent one: image→image homographies measured straight
from the pixels by SIFT + MAGSAC, with no pitch model, no calibration and no focal anywhere in
their derivation. #117 measured them at 0.16-0.77 px against a rotation-only model, 10-30x better
than the solve's own relative motion. A camera turning about a fixed centre satisfies
``H(i→j) = K Rⱼ Rᵢᵀ K⁻¹`` whatever the scene is, so those maps are a direct measurement of both
the frame-to-frame motion and — through the same equation — the focal. They disagree with the
paint about the focal (2700 vs 4179), and this script is the place that disagreement gets settled
or gets reported.

Run: ``PYTHONPATH=src .venv/bin/python scripts/fit_rigid_camera.py --pan``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCENE = ROOT / "out/carry_off/export/scene.json"
VIDEO = ROOT / "samples/video/Colombia-1-0-Congo-DR1080p.mp4"
WIDTH, HEIGHT = 1920, 1080

#: A sample matches the paint it is nearest to only if it is this close. Wide enough to let the
#: seed's 3-12 px error find its own line, narrow enough that a marking cannot capture its
#: neighbour: the tightest pair on the pitch (goal-area line and goal line, 5.5 m apart) is still
#: ~40 px apart in the near field where both are visible.
MATCH_PX = 14.0

#: Marking samples per metre of polyline. The model draws centrelines, so this only needs to be
#: dense enough that every visible marking contributes; 2/m puts ~1400 samples on the pitch.
SAMPLES_PER_M = 2.0

#: Frame gaps the pixel motion is measured over. Gap 1 is the one that pins the jitter, and it is
#: the only gap that can: it is the very quantity that regressed. The long gaps are what make the
#: focal identifiable at all — ``K R Rᵀ K⁻¹`` is degenerate in f below a few degrees of turn (#117
#: §C), so a fit given only consecutive pairs would read the focal off nothing.
PAN_GAPS = (1, 10, 30, 59)

#: Where the measured and predicted image→image maps are compared. Any point set is legitimate —
#: the residual is a disagreement between two *maps*, not a claim about the scene — but the spread
#: matters: ``K R Rᵀ K⁻¹`` is exact at the principal point for every focal, so a grid huddled near
#: the centre would be blind to the one parameter the pan is here to measure. Inset 10% because
#: that is where the measured homography has features on both sides of the pair.
PAN_GRID = np.column_stack([
    g.ravel() for g in np.meshgrid(
        np.linspace(0.1 * WIDTH, 0.9 * WIDTH, 9), np.linspace(0.1 * HEIGHT, 0.9 * HEIGHT, 5)
    )
] + [np.ones(45)])


def rodrigues(r: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)


def unrodrigues(rot: np.ndarray) -> np.ndarray:
    theta = float(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    if theta < 1e-9:
        return np.zeros(3)
    v = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    return v * (theta / (2 * np.sin(theta)))


def kmat(focal: float) -> np.ndarray:
    return np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1.0]])


def plane_h(focal: float, rvec: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """World ``(X, Y, 1)`` → image, for the pitch plane ``Z = 0``."""
    rot = rodrigues(rvec)
    return kmat(focal) @ np.column_stack([rot[:, 0], rot[:, 1], -rot @ centre])


def load_world_to_image(path: Path) -> np.ndarray:
    """The exported calibration as world→image, in the honest right-handed Z-up world (#118).

    The frame is *measured* rather than assumed, so this reads both a legacy export (solved in
    PnLCalib's mirrored top-down template) and anything solved after the #118 fix.
    """
    from poseannot.camera import plane_orientation

    blob = json.loads(path.read_text())
    cal = blob["fields"]["field"]["fields"]["calibration"]["fields"]
    i2w = np.asarray(cal["homographies"]["__ndarray__"]["data"], dtype=float)
    w2i = np.stack([np.linalg.inv(h) for h in i2w])
    mirrored = np.array([plane_orientation(h, WIDTH, HEIGHT) < 0 for h in w2i])
    if mirrored.any():
        assert mirrored.all(), f"the clip changes frame mid-way: {int(mirrored.sum())}/{len(w2i)}"
        w2i = w2i @ np.diag([1.0, -1.0, 1.0])
    return w2i


def marking_samples() -> np.ndarray:
    """Every painted marking, resampled evenly, as world ``(X, Y, 1)``."""
    from pitch3d.core.scene.pitch import pitch_polylines

    out = []
    for poly in pitch_polylines():
        seg = np.linalg.norm(np.diff(poly[:, :2], axis=0), axis=1)
        run = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(2, int(run[-1] * SAMPLES_PER_M))
        want = np.linspace(0.0, run[-1], n)
        out.append(np.column_stack([np.interp(want, run, poly[:, i]) for i in (0, 1)]))
    xy = np.concatenate(out)
    return np.column_stack([xy, np.ones(len(xy))])


def paint_trees(frames: list[int]) -> dict[int, tuple[cKDTree, np.ndarray, np.ndarray]]:
    """Per frame: a KD-tree of painted centreline pixels, their coords, and the surface mask."""
    from poseannot.pitch_evidence import _masks
    from poseannot.video import read_frame

    out = {}
    for n, i in enumerate(frames):
        dist, surface = _masks(read_frame(str(VIDEO), i))
        spine = np.argwhere(dist == 0)[:, ::-1].astype(float)  # (u, v)
        out[i] = (cKDTree(spine), spine, surface)
        if n % 10 == 0:
            print(f"    frame {i}: {len(spine)} painted centreline px", flush=True)
    return out


def project(h: np.ndarray, xy1: np.ndarray) -> np.ndarray:
    q = xy1 @ h.T
    w = np.where(np.abs(q[:, 2]) > 1e-9, q[:, 2], 1e-9)
    return q[:, :2] / w[:, None]


def paint_error(h: np.ndarray, tree: cKDTree, surface: np.ndarray, xy1: np.ndarray) -> np.ndarray:
    """Distance from this map's markings to the nearest paint, for samples with evidence.

    Only samples that land on the playing surface count: a marking projected into the crowd is
    unmeasurable, not wrong, and letting it score would make an overlay look worse for pointing at
    something the frame cannot judge (the #113/#114 rule, applied to the fit's own metric).
    """
    uv = project(h, xy1)
    good = (uv[:, 0] > 1) & (uv[:, 0] < WIDTH - 2) & (uv[:, 1] > 1) & (uv[:, 1] < HEIGHT - 2)
    sub = uv[good]
    if not len(sub):
        return np.array([])
    on = surface[np.rint(sub[:, 1]).astype(int), np.rint(sub[:, 0]).astype(int)] > 0
    return tree.query(sub[on])[0] if on.any() else np.array([])


def pixel_motion(
    frames: list[int], gaps: tuple[int, ...], cache: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Measured image→image homographies, from the pixels alone. ``(pairs, H)``.

    ``pairs`` indexes into ``frames``; ``H[n]`` maps image ``pairs[n][0]`` to image ``pairs[n][1]``.
    Nothing in here has seen the pitch model, the solved calibration or a focal, which is the whole
    point — it is the only evidence in this repo that is independent of the thing being fitted.
    """
    pairs = np.array([(a, a + g) for g in gaps if g < len(frames) for a in range(len(frames) - g)])
    named = np.array([(frames[a], frames[b]) for a, b in pairs])
    if cache is not None and cache.exists():
        blob = np.load(cache)
        if blob["pairs"].shape == named.shape and (blob["pairs"] == named).all():
            print(f"  {len(pairs)} pairs from cache {cache}", flush=True)
            return pairs, blob["h"]

    from scripts.bench_frame_preprocessing import pixel_homographies

    t0 = time.time()
    got = pixel_homographies([(int(a), int(b)) for a, b in named])
    assert all(h is not None for h in got), "MAGSAC failed on a pair — too few matches"
    h = np.stack(got)
    print(f"  {len(pairs)} pairs in {time.time() - t0:.0f} s", flush=True)
    if cache is not None:
        np.savez(cache, pairs=named, h=h)
    return pairs, h


def pan_maps(q: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """The rigid model's own image→image motion: ``K Rⱼ Rᵢᵀ K⁻¹``, one per pair.

    True for *any* scene, not just the plane — which is why it can be compared against homographies
    fitted to crowd and stand features as well as grass, and why it carries no centre term.
    """
    k = kmat(q[0])
    ki = np.linalg.inv(k)
    rot = [rodrigues(q[4 + 3 * j : 7 + 3 * j]) for j in range((len(q) - 4) // 3)]
    return np.stack([k @ rot[b] @ rot[a].T @ ki for a, b in pairs])


def pan_self_consistency(pairs: np.ndarray, h: np.ndarray) -> str:
    """Does the measured motion agree with itself? Chained gap-1 maps vs the direct long map.

    A check on the *instrument*, before anything is fitted to it. Both sides are measurements, so a
    disagreement convicts the pixels, not the camera model — and how it grows says which way: noise
    accumulating over n hops grows like √n, a systematic per-frame bias grows like n.
    """
    at = {(int(a), int(b)): i for i, (a, b) in enumerate(pairs)}
    out = []
    for gap in sorted({int(b - a) for a, b in pairs} - {1}):
        errs = []
        for a, b in ((a, a + gap) for a in range(max(pairs[:, 1]) + 1)):
            if (a, b) not in at or any((k, k + 1) not in at for k in range(a, b)):
                continue
            chain = np.eye(3)
            for k in range(a, b):
                chain = h[at[(k, k + 1)]] @ chain
            errs.append(np.median(np.linalg.norm(
                project(chain, PAN_GRID) - project(h[at[(a, b)]], PAN_GRID), axis=1
            )))
        if errs:
            out.append(f"{gap}→{np.median(errs):.2f} px ({np.median(errs) / gap:.3f}/hop)")
    return "  ".join(out)


def pan_seen(maps: np.ndarray) -> np.ndarray:
    """Where each image→image map sends :data:`PAN_GRID`. ``(pairs, grid, 2)``."""
    return np.stack([project(h, PAN_GRID) for h in maps])


def pan_error(maps: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """Per-point px disagreement between a model's image→image motion and the measured motion."""
    return np.linalg.norm(pan_seen(maps) - moved, axis=2).ravel()


def _solve(
    p: np.ndarray, src: np.ndarray, dst: np.ndarray, owner: np.ndarray, n_f: int,
    pan: tuple[np.ndarray, np.ndarray] | None = None, pan_weight: float = 1.0,
    lock_focal: bool = False,
) -> np.ndarray:
    """One least-squares step of the rigid camera onto fixed 2D correspondences."""
    pairs = pan[0] if pan is not None else np.empty((0, 2), int)
    n_paint, n_grid = 2 * len(owner), PAN_GRID.shape[0]

    def residuals(q: np.ndarray) -> np.ndarray:
        out = np.empty_like(dst)
        for j in range(n_f):
            m = owner == j
            if m.any():
                out[m] = project(plane_h(q[0], q[4 + 3 * j : 7 + 3 * j], q[1:4]), src[m])
        paint = (out - dst).ravel()
        if pan is None:
            return paint
        turned = pan_seen(pan_maps(q, pairs)) - pan[1]
        return np.concatenate([paint, pan_weight * turned.ravel()])

    # Frame j's rotation touches only frame j's rows; the focal and the centre touch all of them.
    # Declaring that turns a 184-column numerical Jacobian into ~8 evaluations.
    spar = np.zeros((n_paint + 2 * n_grid * len(pairs), len(p)), dtype=np.uint8)
    spar[:n_paint, :4] = 1
    for j in range(n_f):
        hit = np.flatnonzero(owner == j)
        rows = np.concatenate([2 * hit, 2 * hit + 1])
        spar[np.ix_(rows, [4 + 3 * j, 5 + 3 * j, 6 + 3 * j])] = 1
    # A pan row sees the focal and the two rotations it relates — and, deliberately, not the centre:
    # a pure rotation's image→image map has no translation in it to constrain.
    for n, (a, b) in enumerate(pairs):
        rows = np.arange(n_paint + 2 * n_grid * n, n_paint + 2 * n_grid * (n + 1))
        cols = [0, 4 + 3 * a, 5 + 3 * a, 6 + 3 * a, 4 + 3 * b, 5 + 3 * b, 6 + 3 * b]
        spar[np.ix_(rows, cols)] = 1

    # ``x_scale="jac"`` is not a tuning knob, it is required: the focal is ~4000 and a rotation
    # component is ~1, so on a common scale the optimiser cannot see the focal at all — it hands
    # back the number it was seeded with, to the decimal.
    lo = np.full(len(p), -np.inf)
    hi = np.full(len(p), np.inf)
    lo[0], hi[0] = (p[0] - 1e-6, p[0] + 1e-6) if lock_focal else (1.0, np.inf)
    return least_squares(
        residuals, p, jac_sparsity=spar, x_scale="jac", loss="soft_l1", f_scale=3.0,
        bounds=(lo, hi), max_nfev=400, verbose=0,
    ).x


def fit(
    frames: list[int],
    evidence: dict[int, tuple[cKDTree, np.ndarray, np.ndarray]],
    xy1: np.ndarray,
    seed_focal: float,
    w2i: np.ndarray,
    rounds: int = 4,
    report_coarse: bool = False,
    lock_focal: bool = False,
    pan: tuple[np.ndarray, np.ndarray] | None = None,
    pan_weight: float = 1.0,
) -> tuple[np.ndarray, float]:
    """ICP the rigid camera onto the paint. Returns ``(params, median px)``.

    ``params`` is ``[focal, Cx, Cy, Cz, rvec per frame]`` — 4 + 3F against the 8F a free per-frame
    solve spends, and the only thing that varies between frames is where the camera is pointed.
    """
    from scripts.bench_frame_preprocessing import decompose

    n_f = len(frames)
    poses = [decompose(w2i[i], seed_focal) for i in frames]
    p = np.concatenate([
        [seed_focal],
        np.median([c for _r, c in poses], axis=0),
        np.concatenate([unrodrigues(r) for r, _c in poses]),
    ])

    # Coarse stage — make the pose self-consistent before showing it any paint. Each frame's
    # rotation was decomposed against its *own* centre, and those centres wander 7-11 m (#117 §B),
    # so pinning one centre and keeping the rotations leaves frames tens of px out. ICP would then
    # match half the samples to the wrong line and confidently converge there. Fitting to the
    # homographies' own lawn mapping first costs nothing and is the one thing they are good for.
    lawn = np.column_stack([
        np.meshgrid(np.linspace(-52.5, 52.5, 60), np.linspace(-34.0, 34.0, 40))[0].ravel(),
        np.meshgrid(np.linspace(-52.5, 52.5, 60), np.linspace(-34.0, 34.0, 40))[1].ravel(),
    ])
    lawn = np.column_stack([lawn, np.ones(len(lawn))])
    seen, want = [], []
    for i in frames:
        uv = project(w2i[i], lawn)
        keep = (uv[:, 0] > 0) & (uv[:, 0] < WIDTH) & (uv[:, 1] > 0) & (uv[:, 1] < HEIGHT)
        seen.append(lawn[keep])
        want.append(uv[keep])
    own_lawn = np.concatenate([np.full(len(s), j) for j, s in enumerate(seen)])
    src_l, dst_l = np.concatenate(seen), np.concatenate(want)
    p = _solve(p, src_l, dst_l, own_lawn, n_f, pan, pan_weight, lock_focal)
    if report_coarse:
        coarse = np.concatenate([
            paint_error(plane_h(p[0], p[4 + 3 * j : 7 + 3 * j], p[1:4]), *evidence[i][::2], xy1)
            for j, i in enumerate(frames)
        ])
        print(f"    coarse (homographies only, no paint): f={p[0]:7.1f} px, "
              f"paint {np.median(coarse):5.2f} px", flush=True)

    for _round in range(rounds):
        # Re-match against the paint at the current pose, then hold the correspondences fixed for
        # one least-squares solve. Samples with no paint within MATCH_PX are dropped from this
        # round rather than dragged to a wrong line.
        src, dst, owner = [], [], []
        for j, i in enumerate(frames):
            tree, spine, surface = evidence[i]
            h = plane_h(p[0], p[4 + 3 * j : 7 + 3 * j], p[1:4])
            uv = project(h, xy1)
            ok = (uv[:, 0] > 1) & (uv[:, 0] < WIDTH - 2) & (uv[:, 1] > 1) & (uv[:, 1] < HEIGHT - 2)
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
        p = _solve(
            p, np.concatenate(src), np.concatenate(dst), np.concatenate(owner), n_f,
            pan, pan_weight, lock_focal,
        )

    err = np.concatenate([
        paint_error(plane_h(p[0], p[4 + 3 * j : 7 + 3 * j], p[1:4]), *evidence[i][::2], xy1)
        for j, i in enumerate(frames)
    ])
    return p, float(np.median(err))


def score(
    p: np.ndarray,
    frames: list[int],
    evidence: dict[int, tuple[cKDTree, np.ndarray, np.ndarray]],
    xy1: np.ndarray,
    probe: np.ndarray,
    pan: tuple[np.ndarray, np.ndarray] | None,
) -> str:
    """Every number #119 is judged on, for one camera: paint, pixel motion, jitter.

    Reported together deliberately. The failure this issue is about is a fit that improves one of
    them by quietly spending the others — a rigid camera fitted to the paint alone beat the 480
    free parameters on paint *and* came out jitterier than what it replaced.
    """
    from scripts.bench_frame_preprocessing import smooth_residual, smooth_residual_domain

    maps = [plane_h(p[0], p[4 + 3 * j : 7 + 3 * j], p[1:4]) for j in range(len(frames))]
    err = np.concatenate([
        paint_error(h, *evidence[i][::2], xy1) for h, i in zip(maps, frames, strict=True)
    ])
    turn = np.median(pan_error(pan_maps(p, pan[0]), pan[1])) if pan is not None else float("nan")
    tracks = np.stack([project(h, probe) for h in maps])
    jitter = np.median(smooth_residual(tracks))
    warn = smooth_residual_domain(tracks.shape[0])
    return (f"f={p[0]:7.1f}  C=({p[1]:6.1f},{p[2]:6.1f},{p[3]:5.1f}) m  "
            f"paint {np.median(err):5.2f} (p90 {np.percentile(err, 90):5.2f})  "
            f"pan {turn:6.2f}  jitter {jitter:5.2f}"
            + (f"  <<{warn}>>" if warn else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=SCENE,
                    help="scene whose per-frame homographies the fit is measured against. The "
                         "default is the 60-frame carry_off export the shipped calib npz came "
                         "from; point it at a longer scene to test whether one camera still "
                         "reduces past 60 frames (#140).")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--stride", type=int, default=1, help="subsample; a short run must still span "
                                                          "the clip or there is no pan baseline")
    ap.add_argument("--seeds", type=float, nargs="+", default=[2700.0, 3400.0, 4277.0, 5200.0])
    ap.add_argument("--out", type=Path, default=None, help="write the winning camera as .npz")
    ap.add_argument("--sweep", action="store_true", help="hold the focal and refit everything "
                                                        "else, to show how sharp the minimum is")
    ap.add_argument("--pan", action="store_true", help="also fit the measured pixel motion, and "
                                                      "report it either way")
    ap.add_argument("--pan-weight", type=float, default=1.0, help="px of pan residual worth one px "
                                                                 "of paint residual; 0 measures "
                                                                 "and reports it without letting "
                                                                 "it move the fit")
    ap.add_argument("--pan-cache", type=Path, default=Path("/tmp/pan_pairs.npz"))
    args = ap.parse_args()

    frames = list(range(0, args.frames, args.stride))
    w2i = load_world_to_image(args.scene)
    xy1 = marking_samples()
    print(f"{len(frames)} frames, {len(xy1)} marking samples", flush=True)

    print("reading the paint (decode + ridge filter per frame) ...", flush=True)
    t0 = time.time()
    evidence = paint_trees(frames)
    print(f"  {time.time() - t0:.0f} s", flush=True)

    pan: tuple[np.ndarray, np.ndarray] | None = None
    if args.pan:
        print("measuring the pixel motion (SIFT + MAGSAC, no pitch and no calibration) ...")
        pairs, measured = pixel_motion(frames, PAN_GAPS, args.pan_cache)
        pan = (pairs, pan_seen(measured))
        print(f"  chained gap-1 vs direct: {pan_self_consistency(pairs, measured)}", flush=True)

    from scripts.bench_frame_preprocessing import probe_points, smooth_residual

    xy = probe_points(w2i)
    probe = np.column_stack([xy, np.ones(len(xy))])
    free = np.stack([project(w2i[i], probe) for i in frames])
    base = np.concatenate([paint_error(w2i[i], *evidence[i][::2], xy1) for i in frames])
    base_pan = float("nan")
    if pan is not None:
        # What the current solve says the pixels did, scored against what they actually did. The
        # free homographies have no shared camera, so their relative motion is whatever the
        # per-frame noise leaves — this is the number a rigid camera has to beat to be worth having.
        base_pan = np.median(pan_error(
            np.stack([w2i[frames[b]] @ np.linalg.inv(w2i[frames[a]]) for a, b in pan[0]]), pan[1]
        ))
    motion = np.median(np.linalg.norm(np.diff(free, axis=0), axis=2))
    print(f"\nbaseline — {len(frames)} free homographies, {8 * len(frames)} parameters:")
    print(f"  paint {np.median(base):5.2f} (p90 {np.percentile(base, 90):5.2f})  "
          f"pan {base_pan:6.2f}  jitter {np.median(smooth_residual(free)):5.2f}  "
          f"[real inter-frame motion {motion:.2f} px, n={len(base)}]\n", flush=True)

    if args.sweep:
        # Does the paint actually *pick* a focal, or only a focal-over-distance? A pinhole further
        # away with a longer lens draws almost the same lawn, so a flat sweep would mean the number
        # the fit converges on is an artefact of the seed path, not a measurement. With --pan the
        # same sweep puts both instruments on one axis, which is the only way to see whether their
        # disagreement about the focal is real or an artefact of how each one is being asked.
        print(f"focal held, everything else refitted — {3 + 3 * len(frames)} free parameters:")
        for held in (2700.0, 3000.0, 3400.0, 3900.0, 4179.0, 4600.0, 5200.0):
            p, _ = fit(frames, evidence, xy1, held, w2i, lock_focal=True,
                       pan=pan, pan_weight=args.pan_weight)
            print(f"  f={held:6.0f} held  {score(p, frames, evidence, xy1, probe, pan)}  "
                  f"f/dist {held / np.linalg.norm(p[1:4]):.2f}", flush=True)
        return

    print(f"rigid camera — {4 + 3 * len(frames)} parameters, fitted to "
          f"{'paint + pan' if pan is not None and args.pan_weight else 'paint only'}:")
    best: tuple[float, np.ndarray] | None = None
    for seed_focal in args.seeds:
        t0 = time.time()
        p, med = fit(frames, evidence, xy1, seed_focal, w2i, pan=pan, pan_weight=args.pan_weight)
        print(f"  seed {seed_focal:6.0f} -> {score(p, frames, evidence, xy1, probe, pan)} "
              f"[{time.time() - t0:.0f} s]", flush=True)
        if best is None or med < best[0]:
            best = (med, p.copy())
    assert best is not None
    _med, p = best
    print(f"\nbest:            {score(p, frames, evidence, xy1, probe, pan)}")
    print("  centre wander 0.0 m by construction (free per-frame: see #117 §B, 7-11 m)")

    if pan is not None:
        # Does the fixed centre survive a long baseline? Over one frame a small translation and a
        # small rotation are indistinguishable, so gap 1 cannot test it; over 59 they are not, and a
        # residual that climbs with the gap is the rig drifting off its pivot. This is the one
        # assumption #119 makes that the paint has no way to check.
        gap = pan[0][:, 1] - pan[0][:, 0]
        by_pair = pan_error(pan_maps(p, pan[0]), pan[1]).reshape(len(pan[0]), -1)
        print("  pan by frame gap: " + "   ".join(
            f"{g}→{np.median(by_pair[gap == g]):.2f} px" for g in sorted(set(gap.tolist()))
        ))

    if args.out:
        h = np.stack([plane_h(p[0], p[4 + 3 * j : 7 + 3 * j], p[1:4]) for j in range(len(frames))])
        # width/height travel with the fit: a focal is meaningless without the pixel space it was
        # measured in, and the principal point the consumer rebuilds K from lives here too.
        np.savez(args.out, focal=p[0], centre=p[1:4], rvecs=p[4:].reshape(-1, 3),
                 frames=np.array(frames), world_to_image=h, width=WIDTH, height=HEIGHT)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
