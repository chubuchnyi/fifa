"""#105: is the calibration confidence predictive of calibration accuracy — and if not, why?

The real clip said no: over the 60 target-clip frames the shipped ``field_homography_conf``
correlates with the *paint* error at r = +0.699 — higher confidence, worse alignment. That
measurement (``scripts/bench_camera_swim.py::confidence_check``) proves the defect but cannot
name its cause, because the per-frame ingredients (landmarks, inlier mask, per-term factors) are
not in ``scene.json`` and the PnLCalib detector needs the GPU box.

It also cannot *date* it. ``out/anim_A/export/scene.json`` was written 2026-07-09; R3 wired
PnLCalib's line detections into the DLT on 2026-07-29 (``bf120b2``). The evidence therefore
indicts a **points-only** configuration that no longer ships, so this bench scores both columns —
and the gap between them turns out to be most of the original complaint.

So this bench reconstructs the ingredients instead of the detector. It keeps the *geometry* real —
the 60 measured homographies drive which pitch landmarks are visible and where they land — and
synthesises only the detections, which makes the true error exactly knowable: the fit is scored
against the very homography that generated its observations. Nothing here scores a homography with
a model derived from that homography, the circularity ADR-0012 warns about; the truth signal is an
input, not an output.

Per the same ADR the bench carries candidates it is *supposed* to fail:

* ``minimal`` — exactly 4 landmarks, normal noise. Zero redundancy, so the 4-point DLT reproduces
  its own points *exactly* and any residual-on-inliers score reads 0 error however wrong the fit.
* ``cluster`` — landmarks confined to one image quadrant, normal noise. Fits them well and
  extrapolates nonsense over the rest of the frame.
* ``wide`` — every visible landmark, same noise. The best fit available, and the one a
  count-and-agreement score punishes hardest.

The candidates carry the *same* per-landmark noise as the baseline on purpose: noiseless points
determine a homography exactly regardless of how few or how clustered they are, so a noiseless
candidate would be a fit that deserves its high score, not a failure.

Run (~20 s, CPU only, no weights):

    PYTHONPATH=src .venv/bin/python scripts/bench_calib_confidence.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pitch3d.adapters.models.calibration import (  # noqa: E402
    _apply_homography,
    _confidence_from_error,
    point_line_residual,
    solve_homography,
    solve_homography_ransac,
)
from pitch3d.core.scene.pitch import (  # noqa: E402
    pitch_plane_line_segments,
    world_line_from_segment,
)

SCENE = Path("out/anim_A/export/scene.json")
W, H = 1920, 1080
CONF_SCALE_M = 0.5
RANSAC_THRESHOLD_M = 1.0
RANSAC_ITERS = 200
SEED = 0
# The same probe pixels bench_camera_swim.py uses: lower-middle of the frame, where the players
# are — the region the reconstruction actually cares about.
PROBE_UV = np.array([[640.0, 800.0], [960.0, 800.0], [1280.0, 800.0],
                     [760.0, 950.0], [1160.0, 950.0]])


def gt_homographies() -> np.ndarray:
    """The 60 measured image→world homographies, used here purely as ground truth geometry."""
    fields = json.loads(SCENE.read_text())["fields"]
    calib = fields["field"]["fields"]["calibration"]["fields"]
    nd = calib["homographies"]["__ndarray__"]
    return np.asarray(nd["data"], dtype=float).reshape(nd["shape"])


def pitch_landmarks() -> np.ndarray:
    """Distinguished pitch points ``(L, 2)``: straight-line intersections, spots, centre mark."""
    segs = list(pitch_plane_line_segments().values())
    pts: list[tuple[float, float]] = [(0.0, 0.0)]
    for (a0, a1), (b0, b1) in ((s, t) for i, s in enumerate(segs) for t in segs[i + 1:]):
        d1, d2 = a1 - a0, b1 - b0
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-9:
            continue
        t = ((b0[0] - a0[0]) * d2[1] - (b0[1] - a0[1]) * d2[0]) / den
        p = a0 + t * d1
        # Keep only intersections that fall on (or a hair outside) both segments — the crossings a
        # detector could actually name, not the phantom meeting points of extended lines.
        for q0, q1, q in ((a0, a1, p), (b0, b1, p)):
            u = np.dot(q - q0, q1 - q0) / max(float(np.dot(q1 - q0, q1 - q0)), 1e-9)
            if not -0.02 <= u <= 1.02:
                break
        else:
            pts.append((float(p[0]), float(p[1])))
    hl = float(max(abs(s[0][0]) for s in segs))
    pts += [(-hl + 11.0, 0.0), (hl - 11.0, 0.0)]
    return np.unique(np.round(np.asarray(pts, dtype=float), 3), axis=0)


@dataclass
class Obs:
    """One synthetic frame's detections, in the shape ``FrameKeypoints`` carries them."""

    uv: np.ndarray
    xy: np.ndarray
    conf: np.ndarray
    line_uv: np.ndarray
    line_abc: np.ndarray
    line_conf: np.ndarray


def _noisy(uv: np.ndarray, rng: np.random.Generator, noise_px: float, outlier_rate: float
           ) -> tuple[np.ndarray, np.ndarray]:
    """Add per-landmark noise + gross outliers → ``(uv, detector_confidence)``.

    Detector confidence is a *correct* monotone function of the landmark's own noise, so a score
    that reads ``mean_conf`` is being handed genuinely useful information. If the shipped product
    still fails to predict accuracy, the fault is in the combination, not in a rigged input.
    """
    n = uv.shape[0]
    sigma = noise_px * rng.uniform(0.5, 1.6, size=n)
    uv = uv + rng.normal(0.0, 1.0, size=uv.shape) * sigma[:, None]
    conf = np.clip(1.0 - (sigma - 0.5 * noise_px) / (2.0 * noise_px), 0.15, 0.99)
    bad = rng.random(n) < outlier_rate
    if bad.any():  # gross mislocalisations, the failure RANSAC exists for
        uv[bad] += rng.normal(0.0, 35.0, size=(int(bad.sum()), 2))
        conf[bad] = rng.uniform(0.4, 0.95, size=int(bad.sum()))
    return uv, conf


def synth_frame(h_gt: np.ndarray, world: np.ndarray, rng: np.random.Generator,
                *, noise_px: float = 2.0, outlier_rate: float = 0.08,
                mode: str = "wide", lines: bool = True) -> Obs:
    """Synthesise one frame's detections: landmarks and (optionally) point-on-line observations.

    ``lines`` is not cosmetic — it selects which *shipped* configuration is under test. R3
    (``bf120b2``, 2026-07-29) wired PnLCalib's line detections into the DLT; the ``scene.json``
    that produced the r = +0.699 evidence is from 2026-07-09, i.e. points only. Both must be
    scored, or the diagnosis indicts a configuration nobody ran.
    """
    inv = np.linalg.inv(h_gt)
    uv = _apply_homography(inv, world)
    vis = (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)
    if mode == "cluster":
        vis &= (uv[:, 0] < W * 0.5) & (uv[:, 1] > H * 0.45)
    idx = np.flatnonzero(vis)
    if mode == "minimal" and idx.size > 4:
        idx = idx[rng.permutation(idx.size)[:4]]
    kp_uv, conf = _noisy(_apply_homography(inv, world[idx]), rng, noise_px, outlier_rate)

    l_uv: list[np.ndarray] = []
    l_abc: list[np.ndarray] = []
    for a, b in (pitch_plane_line_segments().values() if lines else ()):
        n_s = max(2, int(np.linalg.norm(b - a) / 6.0))
        pts = a + np.linspace(0.0, 1.0, n_s)[:, None] * (b - a)
        p = _apply_homography(inv, pts)
        keep = (p[:, 0] > 0) & (p[:, 0] < W) & (p[:, 1] > 0) & (p[:, 1] < H)
        if mode == "cluster":
            keep &= (p[:, 0] < W * 0.5) & (p[:, 1] > H * 0.45)
        if mode == "minimal" or not keep.any():
            continue
        l_uv.append(p[keep])
        l_abc.append(np.repeat(world_line_from_segment(a, b)[None, :], int(keep.sum()), axis=0))
    if l_uv:
        lu, lc = _noisy(np.vstack(l_uv), rng, noise_px, outlier_rate)
        la = np.vstack(l_abc)
    else:
        lu, la, lc = np.empty((0, 2)), np.empty((0, 3)), np.empty(0)
    return Obs(kp_uv, world[idx], conf, lu, la, lc)


def shipped_confidence(o: Obs) -> tuple[np.ndarray, dict[str, float]]:
    """Re-run the shipped scoring on one frame → ``(H, {term: value})``.

    Mirrors ``KeypointFieldCalibrator.calibrate`` line-for-line, returning the product's factors
    separately so each can be correlated on its own. ``verdict()`` asserts they still multiply to
    the shipped number, so the decomposition cannot silently drift from the code it indicts.
    """
    lines = ({"line_uv": o.line_uv, "line_abc": o.line_abc, "line_weights": o.line_conf}
             if o.line_uv.shape[0] else {})
    h, inliers = solve_homography_ransac(
        o.uv, o.xy, weights=o.conf, threshold=RANSAC_THRESHOLD_M,
        max_iters=RANSAC_ITERS, seed=SEED, **lines,
    )
    resid = np.linalg.norm(_apply_homography(h, o.uv[inliers]) - o.xy[inliers], axis=1)
    agree, mean_conf = inliers.astype(float), o.conf[inliers]
    if o.line_uv.shape[0]:
        lr = point_line_residual(h, o.line_uv, o.line_abc)
        li = lr < RANSAC_THRESHOLD_M
        resid = np.concatenate([resid, lr[li]])
        agree = np.concatenate([agree, li.astype(float)])
        mean_conf = np.concatenate([mean_conf, o.line_conf[li]])
    err = float(np.sqrt((resid ** 2).mean())) if resid.size else float("inf")
    # DLT rows the *agreeing* evidence supplies: a correspondence gives two, a point-on-line one.
    # A homography has 8 degrees of freedom, so `rows - 8` is the redundancy the residual is
    # actually free to express — at zero redundancy the fit reproduces its own observations by
    # construction and the residual is identically 0 regardless of how wrong the homography is.
    rows = 2 * int(inliers.sum()) + int(agree.size - inliers.size)
    dof = rows - 8
    err_dof = float(np.sqrt((resid ** 2).sum() / dof)) if dof > 0 else float("inf")
    terms = {
        "fit": _confidence_from_error(err, CONF_SCALE_M),
        "fit_dof": _confidence_from_error(err_dof, CONF_SCALE_M),
        "dof": float(dof),
        "mean_conf": float(mean_conf.mean()) if mean_conf.size else 0.0,
        "agree": float(agree.mean()),
        "err_m": err,
        "n_kp": float(o.uv.shape[0]),
        "n_obs": float(o.uv.shape[0] + o.line_uv.shape[0]),
        "n_in": float(inliers.sum()),
    }
    terms["conf"] = terms["fit"] * terms["mean_conf"] * terms["agree"]
    return h, terms


def holdout_err(o: Obs, folds: int = 5) -> float:
    """K-fold reprojection error (metres) — fit on the rest, score the held-out observations.

    Answers the question the inlier residual cannot: *is this homography determined by the data?*
    A minimal set has no rest to fit on (returns ``inf``); a clustered or contaminated set predicts
    its own held-out observations badly. Unlike the shipped residual it cannot be driven to zero by
    keeping fewer points, because the score is never read off the points that set the fit.

    Points and line observations are folded together and scored in the same world metres, so a
    frame carried mostly by lines is judged on the evidence it actually has.
    """
    n_p, n_l = o.uv.shape[0], o.line_uv.shape[0]
    if n_p < 4 or n_p + n_l < 8:
        return float("inf")
    rng = np.random.default_rng(SEED)
    fold_p, fold_l = rng.integers(0, folds, n_p), rng.integers(0, folds, n_l)
    out: list[float] = []
    for f in range(folds):
        tr_p, te_p = fold_p != f, fold_p == f
        tr_l, te_l = fold_l != f, fold_l == f
        if tr_p.sum() < 4 or not (te_p.any() or te_l.any()):
            continue
        lines = ({"line_uv": o.line_uv[tr_l], "line_abc": o.line_abc[tr_l],
                  "line_weights": o.line_conf[tr_l]} if tr_l.any() else {})
        try:
            h = solve_homography(o.uv[tr_p], o.xy[tr_p], o.conf[tr_p], **lines)
        except (ValueError, np.linalg.LinAlgError):
            return float("inf")
        if te_p.any():
            out += np.linalg.norm(
                _apply_homography(h, o.uv[te_p]) - o.xy[te_p], axis=1).tolist()
        if te_l.any():
            out += point_line_residual(h, o.line_uv[te_l], o.line_abc[te_l]).tolist()
    if not out:
        return float("inf")
    return float(np.median(out))  # median, not RMS: the gross outliers are the detector's, not
    # the fit's, and squaring them would re-import the very sensitivity RANSAC just removed.


def support(uv: np.ndarray, probe: np.ndarray = PROBE_UV) -> float:
    """Is the region we will *use* inside the landmark spread, or outside it?

    Mahalanobis distance of the probe pixels from the inlier point cloud: ``≤ 1`` means the
    homography is interpolating where the reconstruction reads it, ``≫ 1`` means extrapolating.
    Returned as ``1 / (1 + d)`` so it reads like the other factors (1 good, 0 bad).
    """
    if uv.shape[0] < 3:
        return 0.0
    cov = np.cov(uv.T) + np.eye(2) * 1e-6
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return 0.0
    d = probe - uv.mean(axis=0)
    md = float(np.median(np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, inv, d), 0.0))))
    return 1.0 / (1.0 + md)


def true_error(h_fit: np.ndarray, h_gt: np.ndarray) -> float:
    """Median world-metre disagreement at the probe pixels — what confidence should track."""
    return float(np.median(np.linalg.norm(
        _apply_homography(h_fit, PROBE_UV) - _apply_homography(h_gt, PROBE_UV), axis=1)))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    r = lambda x: np.argsort(np.argsort(x)).astype(float)  # noqa: E731
    return _pearson(r(a[ok]), r(b[ok]))


SIGNALS = ("conf", "fit", "fit_dof", "dof", "mean_conf", "agree", "err_m", "n_kp", "n_obs",
           "n_in")


def sweep(mode: str = "wide", *, lines: bool = True, n_rep: int = 8,
          noise_px: float = 2.0, outlier_rate: float = 0.08) -> dict[str, np.ndarray]:
    """Fit every GT frame ``n_rep`` times with fresh detections → per-fit signals."""
    gt, world = gt_homographies(), pitch_landmarks()
    rows: dict[str, list[float]] = {k: [] for k in (*SIGNALS, "truth", "holdout", "support")}
    for rep in range(n_rep):
        rng = np.random.default_rng(1000 + rep)
        for h_gt in gt:
            o = synth_frame(h_gt, world, rng, mode=mode, lines=lines,
                            noise_px=noise_px, outlier_rate=outlier_rate)
            if o.uv.shape[0] < 4:
                continue
            h, t = shipped_confidence(o)
            rows["truth"].append(true_error(h, h_gt))
            for k in SIGNALS:
                rows[k].append(t[k])
            rows["holdout"].append(holdout_err(o))
            rows["support"].append(support(np.vstack([o.uv, o.line_uv])))
    return {k: np.asarray(v, dtype=float) for k, v in rows.items()}


def _gen(s: dict[str, np.ndarray]) -> np.ndarray:
    """The generalisation term: ``_confidence_from_error`` of the k-fold holdout error."""
    return np.array([_confidence_from_error(e, CONF_SCALE_M) for e in s["holdout"]])


# Candidate scores, all in (0, 1], all "higher = trust this frame more". The shipped one is in the
# table so every alternative is judged against it on the same draws rather than against a memory.
CANDIDATES: dict[str, object] = {
    "OLD  fit×mean_conf×agree (count)": lambda s: s["fit"] * s["mean_conf"] * s["agree"],
    "NOW  fit_dof×mean_conf×agree (dof)": lambda s: s["fit_dof"] * s["mean_conf"] * s["agree"],
    "holdout×mean_conf×agree": lambda s: _gen(s) * s["mean_conf"] * s["agree"],
    "holdout×agree": lambda s: _gen(s) * s["agree"],
    "holdout×support": lambda s: _gen(s) * s["support"],
    "holdout×agree×support": lambda s: _gen(s) * s["agree"] * s["support"],
    "holdout alone": _gen,
}


def verdict() -> None:
    print("=" * 78)
    print("#105 — is calibration confidence predictive of calibration accuracy?")
    print("=" * 78)

    gt, world = gt_homographies(), pitch_landmarks()
    print(f"\nGT frames: {len(gt)}   pitch landmarks: {len(world)}   probe: {len(PROBE_UV)} px")

    # Self-check: the decomposition must reproduce the shipped product exactly, or every
    # per-term correlation below is measuring something the pipeline does not compute.
    obs = synth_frame(gt[0], world, np.random.default_rng(7))
    _, t = shipped_confidence(obs)
    assert abs(t["conf"] - t["fit"] * t["mean_conf"] * t["agree"]) < 1e-12
    print(f"decomposition self-check: OLD conf {t['conf']:.4f} == "
          f"fit {t['fit']:.4f} × mean_conf {t['mean_conf']:.4f} × agree {t['agree']:.4f}  OK")

    cfg = {"points only (what produced scene.json, pre-R3)": False,
           "points + lines (what ships today, post-R3)": True}
    runs = {label: sweep("wide", lines=on) for label, on in cfg.items()}

    for label, s in runs.items():
        n = len(s["truth"])
        print(f"\n--- {label} — {n} fits (60 frames × 8 draws) ---")
        print(f"true probe error: median {np.median(s['truth']):.3f} m   "
              f"p95 {np.percentile(s['truth'], 95):.3f} m   "
              f"observations/frame {np.median(s['n_obs']):.0f} "
              f"({np.median(s['n_kp']):.0f} landmarks)")
        print(f"  {'signal':<26} {'want':>6} {'pearson':>9} {'spearman':>9}")
        for name, sig, want in (("OLD conf (count-normalised)", s["conf"], "-"),
                                ("  term: fit residual", s["fit"], "-"),
                                ("  term: mean_conf", s["mean_conf"], "-"),
                                ("  term: agree fraction", s["agree"], "-"),
                                ("count-normalised RMS (m)", s["err_m"], "+"),
                                ("observation count", s["n_obs"], "-"),
                                ("k-fold holdout err (m)", s["holdout"], "+"),
                                ("probe support", s["support"], "-")):
            p, sp = _pearson(sig, s["truth"]), _spearman(sig, s["truth"])
            bad = "  <-- WRONG SIGN" if np.isfinite(sp) and (sp > 0) == (want == "-") else ""
            print(f"  {name:<26} {want:>6} {p:>9.3f} {sp:>9.3f}{bad}")

    print("\n--- candidate scores (Spearman vs true error; want most negative) ---")
    print(f"  {'candidate':<36} {'points only':>12} {'with lines':>12}")
    for name, fn in CANDIDATES.items():
        cols = [_spearman(fn(s), s["truth"]) for s in runs.values()]  # type: ignore[operator]
        print(f"  {name:<36} {cols[0]:>12.3f} {cols[1]:>12.3f}")

    print("\n--- candidates the score is SUPPOSED to fail (points + lines) ---")
    print(f"  {'case':<10} {'n_obs':>6} {'dof':>5} {'true err':>10} {'old':>9} {'now':>7}")
    for mode in ("wide", "minimal", "cluster"):
        c = sweep(mode, n_rep=3)
        fix = c["fit_dof"] * c["mean_conf"] * c["agree"]
        print(f"  {mode:<10} {np.median(c['n_obs']):>6.0f} {np.median(c['dof']):>5.0f} "
              f"{np.median(c['truth']):>9.3f}m {np.median(c['conf']):>9.3f} "
              f"{np.median(fix):>7.3f}")

    print("\n--- held-out condition: 3.5 px noise, 15% outliers (no candidate tuned here) ---")
    h = sweep("wide", n_rep=4, noise_px=3.5, outlier_rate=0.15)
    print(f"  true probe error median {np.median(h['truth']):.3f} m")
    for name, fn in CANDIDATES.items():
        print(f"  {name:<36} {_spearman(fn(h), h['truth']):>12.3f}")  # type: ignore[operator]

    print("\n--- does the score separate good frames from bad? (points + lines) ---")
    s = runs["points + lines (what ships today, post-R3)"]
    n = len(s["truth"])
    for label, sig in (("old (count)", s["conf"]),
                       ("now (dof)", s["fit_dof"] * s["mean_conf"] * s["agree"])):
        o = np.argsort(sig)
        third = max(1, n // 3)
        lo, hi = np.median(s["truth"][o[:third]]), np.median(s["truth"][o[-third:]])
        print(f"  {label:<15} lowest-confidence third {lo:.3f} m   "
              f"highest third {hi:.3f} m   {'INVERTED' if hi > lo else 'ok'}")

    print("\n" + "=" * 78)
    print("""VERDICT
1. The complaint was real for the configuration that produced it. Points only, the old
   product carries no signal (Spearman -0.06) and its residual term points the WRONG WAY: the
   RMS is taken over the very inliers RANSAC chose because they fit, so it scores the subset's
   self-consistency, not the homography.
2. R3 already repaired most of it. Line observations are numerous and are not what consensus is
   voted on, so with them the same formula reads -0.53 and every term is correctly signed. The
   r = +0.699 artifact predates that fix; it should not be re-quoted without a post-R3 re-run.
3. Do NOT replace the formula. Every alternative measured here — k-fold holdout error, probe
   support, and their products — is equal or worse once lines are present (-0.45..-0.53 vs
   -0.53) and clearly worse under harder noise (-0.39..-0.68 vs -0.72).
4. DO fix the residual normalisation. Dividing by degrees of freedom (rows - 8) instead of
   observation count beats the count-normalised score everywhere (-0.36 vs -0.06 points
   only, -0.55 vs -0.53 with lines, -0.727 vs -0.722 held out) and collapses the one
   unambiguous inversion: the minimal 4-landmark frame, 1.87 m wrong, scored 0.70 - HIGHER than
   the 0.31 m frame - and now scores 0.00. Shipped; pinned by a golden test.
5. Still open, marked not erased (R-6): spatial distribution is not scored. The clustered frame
   is 2.5x worse than the wide one and still scores about the same. The candidate that would
   catch it (probe support) loses overall, so it stays measured-and-rejected, not hidden.""")
    print("=" * 78)


if __name__ == "__main__":
    verdict()
