"""Turn named line segments into a camlab anchor, with camlab's own paint as the judge.

Step 2 of `camlab/docs/findings/automating-the-anchor-2026-08-13.md`. camlab needs one camera on
one frame and today a human drags it there; this produces the same file from labels.

**camlab is not modified.** Everything it supplies comes over its HTTP API, and everything this
writes goes into `runs/<clip>/camera_manual.json` — the store `solve/hand.py` already reads, and
which already refuses to rank by source:

    A store cannot have priority. … nothing here ranks by source: both stores offer candidates and
    the caller picks the one that fits the paint, which is the only thing that can settle it.

So a wrong anchor written here loses to the seed on the paint and `solve_carry.py` says it did.
That is the whole safety argument and none of it is new code.

**Nothing about the pitch is duplicated here.** The world markings come from `GET /api/pitch` and
the detected segments from `GET /api/run/{clip}/lines/{n}`, both in camlab's own frame, so no
convention can drift between the two repos. The one thing that is copied is the line-to-line DLT
(see `homography_from_lines`), because it is four lines of algebra AVATAR does not otherwise have.

**Ranking is camlab's, not ours.** Every surviving candidate is written to the manual store and
scored by `GET /api/run/{clip}/residual/{n}`, which re-reads the overlay, so the number that picks
the anchor is the same number that judges the finished solve. AVATAR contributes no scoring rule.

**The principal point is the trap.** A camera is only valid under the K it was solved with, and on
a cropped clip the optical axis is not the image centre — `fan`'s is at ``cy = -334``, not 304.
`cx, cy` therefore come from camlab's camera file over the API and are never assumed here.
`plane_camera._measure_focal` and `_k_inv` already take them; only
`camera_from_calibration` hardcodes the image centre, and this script does not go through it.

Run::

    python scripts/write_camlab_anchor.py --clip broadcast --frame 0 \
        --labels out/labeller/broadcast_f0/labels.json --which camera_smooth.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from itertools import combinations, permutations
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pitch3d.core.scene.plane_camera import (  # noqa: E402
    _decompose,
    _k_inv,
    _measure_focal,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_line_labeller import (  # noqa: E402
    EXPLAIN_PX,
    MARKING_TYPE,
    TYPE_FAMILY,
    point_to_polyline_px,
)

#: Physical plausibility, deliberately looser than camlab's `bootstrap_clip.plausible`
#: (900–12000 px, 5–45 m high, 35–140 m out). That gate is broadcast-shaped and camlab's own
#: `pitch-level-clips-2026-08-13.md` records that it *"would reject the true camera outright"* on a
#: phone held at 1.5 m. Every bound here is overridable on the command line for the same reason.
FOCAL_BOUNDS = (300.0, 20000.0)
HEIGHT_BOUNDS = (1.0, 80.0)
DISTANCE_BOUNDS = (10.0, 200.0)
#: Share of the pitch model that must land inside the frame. camlab's `MIN_IN_FRAME`, and it is
#: what stops the search settling on a camera that frames almost no pitch and so misses almost
#: nothing — the failure its random-search bootstrap converged on three times at different budgets.
MIN_IN_FRAME = 0.12
#: How far the decomposed camera may land from its own homography. Looser than
#: `plane_camera.REALIZABLE_PX = 1.0` because that gate is applied to a *fitted* camera over a
#: whole clip, and this one is applied to a single raw hypothesis before any refit.
REALIZABLE_PX = 3.0

#: How many survivors get sent to camlab's paint. The local pre-rank only measures agreement with
#: the very correspondences that built each hypothesis, so it cannot be the verdict — it is a
#: shortlist, and this is how long it is.
SHORTLIST = 12


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310 - localhost only
        return r.read()


def get_json(url: str) -> dict:
    return json.loads(fetch(url))


def line_through(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Homogeneous line through two 2-D points, **normalised so ``|n| = 1``**.

    Copied together with `homography_from_lines` from camlab `solve/bootstrap.py:_line`, and the
    normalisation is not cosmetic. Image lines are built from coordinates of order 10³ and world
    lines from coordinates of order 10¹, so unnormalised the two sides enter the DLT with
    determinants differing by six orders of magnitude and the SVD solves whichever rows are
    loudest. Measured, with this line left out: every one of 384 label-consistent hypotheses on
    `broadcast` f0 came back **771–3221 px** from reproducing its own homography, i.e. not one of
    them was a camera at all — while the same correspondences with it in place solve the frame.

    The lesson generalises past this function: a copied routine has to bring its helpers, because
    a missing one changes the numerics silently rather than raising.
    """
    line = np.cross(np.array([a[0], a[1], 1.0]), np.array([b[0], b[1], 1.0]))
    n = float(np.linalg.norm(line[:2]))
    return line / n if n > 1e-12 else line


def homography_from_lines(img_lines: list[np.ndarray],
                          world_lines: list[np.ndarray]) -> np.ndarray | None:
    """World→image homography from ≥4 line correspondences, or None if degenerate.

    Copied from camlab `src/camlab/solve/bootstrap.py:_homography_from_lines` (ADR-0013 §5 — copy
    with an origin header, never import a sibling lab). Lines are contravariant: ``l_world ∝
    Hᵀ l_image``, so writing ``m = Hᵀ l_image`` as a linear map on the nine entries of H gives
    ``l_world × m = 0``, two independent rows per correspondence.

    Kept rather than reimplemented from points because AVATAR's own DLT
    (`adapters/models/calibration.solve_homography`) takes point correspondences and point-on-line
    rows — neither is a line-to-line correspondence, and intersecting near-parallel named lines to
    manufacture points amplifies their localisation error exactly where it is largest.
    """
    rows = []
    for li, lw in zip(img_lines, world_lines, strict=True):
        m = np.zeros((3, 9))
        for i in range(3):
            for j in range(3):
                m[j, 3 * i + j] = li[i]
        cross = np.array([[0.0, -lw[2], lw[1]], [lw[2], 0.0, -lw[0]], [-lw[1], lw[0], 0.0]])
        rows.append(cross @ m)
    a = np.vstack(rows)
    if a.shape[0] < 8:
        return None
    _u, s, vt = np.linalg.svd(a)
    if s[-2] < 1e-12:
        return None
    h = vt[-1].reshape(3, 3)
    return None if abs(np.linalg.det(h)) < 1e-12 else h


def in_frame_share(h_w2i: np.ndarray, samples: np.ndarray, w: int, h: int) -> float:
    q = samples @ h_w2i.T
    z = np.where(np.abs(q[:, 2]) > 1e-9, q[:, 2], 1e-9)
    uv = q[:, :2] / z[:, None]
    ok = (q[:, 2] > 0) & (uv[:, 0] > 0) & (uv[:, 0] < w) & (uv[:, 1] > 0) & (uv[:, 1] < h)
    return float(ok.mean())


def project(h_w2i: np.ndarray, xy: np.ndarray) -> np.ndarray:
    q = np.column_stack([xy, np.ones(len(xy))]) @ h_w2i.T
    z = np.where(np.abs(q[:, 2]) > 1e-9, q[:, 2], 1e-9)
    return q[:, :2] / z[:, None]


def realizable_px(h_w2i: np.ndarray, rot: np.ndarray, t: np.ndarray,
                  focal: float, cx: float, cy: float, xy: np.ndarray) -> float:
    """How far the decomposed camera lands from its own homography, worst point, in pixels.

    **This check is not optional and its absence produced a wrong camera that looked right.**
    A homography fitted from LINE correspondences is not sign-determined — a line is the same line
    with its coefficients negated, so ``l_world × Hᵀ l_image = 0`` is satisfied by reflected
    solutions too. `_decompose` then orthogonalises ``[r1, r2, r1×r2]`` through an SVD, which turns
    a reflection into a *proper* rotation without complaint, and the recovered position stays
    physically plausible. Measured on `broadcast` f0: a candidate came back at focal 4169 px and
    (−5.11, −71.96, 17.19) m — within a few metres of the camera camlab believes — and scored
    327 px against the paint, the worst of its whole pool.

    The camera the pitch3d golden test pins has ``det(H) < 0``; the sign of a 3×3 determinant is
    not scale-invariant (``H → λH`` scales it by ``λ³``), so the determinant alone cannot be the
    test. Reprojection can: a reflected solve does not reproduce the homography it came from.
    Same quantity `core/scene/plane_camera.camera_from_calibration` gates on with ``REALIZABLE_PX``.
    """
    k = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    through_cam = (np.column_stack([xy, np.zeros(len(xy))]) @ rot.T + t) @ k.T
    with np.errstate(all="ignore"):
        got = through_cam[:, :2] / np.where(np.abs(through_cam[:, 2:3]) > 1e-9,
                                            through_cam[:, 2:3], 1e-9)
        want = project(h_w2i, xy)
        d = np.abs(got - want)
    ahead = through_cam[:, 2] > 0
    return float(d[ahead].max()) if ahead.any() else float("inf")


def candidates(segments: np.ndarray, labels: dict[str, str], world: dict[int, np.ndarray],
               width: int, height: int, args: argparse.Namespace) -> list[dict]:
    """Every label-consistent 2+2 assignment that yields a physically plausible camera.

    This is `solve/bootstrap.hypotheses` with the labels doing the work its enumeration does
    blindly: the family split is read off the labels instead of being found by vanishing-point
    consensus, and each segment may only be assigned the world markings its own label allows.
    """
    by_type: dict[str, list[int]] = {}
    for k, v in MARKING_TYPE.items():
        by_type.setdefault(v, []).append(k)

    fam: dict[int, list[int]] = {0: [], 1: []}
    for i in range(len(segments)):
        t = labels.get(str(i + 1))
        f = TYPE_FAMILY.get(str(t))
        if f is not None:
            fam[f].append(i)
    if len(fam[0]) < 2 or len(fam[1]) < 2:
        return []

    img_lines = [line_through(s[:2], s[2:]) for s in segments]
    world_lines = {k: line_through(v[0], v[-1]) for k, v in world.items()}
    samples = np.column_stack([
        np.concatenate([v[:, 0] for v in world.values()]),
        np.concatenate([v[:, 1] for v in world.values()]),
        np.ones(sum(len(v) for v in world.values())),
    ])

    out, seen = [], set()
    for pair0 in combinations(fam[0], 2):
        for pair1 in combinations(fam[1], 2):
            picks = pair0 + pair1
            allowed = [by_type.get(str(labels.get(str(i + 1))), []) for i in picks]
            for m0 in permutations(allowed[0] or [], 1):
                for m1 in permutations(allowed[1] or [], 1):
                    if m1[0] == m0[0]:
                        continue
                    for m2 in permutations(allowed[2] or [], 1):
                        for m3 in permutations(allowed[3] or [], 1):
                            if m3[0] == m2[0]:
                                continue
                            ms = (m0[0], m1[0], m2[0], m3[0])
                            h = homography_from_lines([img_lines[i] for i in picks],
                                                      [world_lines[m] for m in ms])
                            if h is None:
                                continue
                            if in_frame_share(h, samples, width, height) < MIN_IN_FRAME:
                                continue
                            focal = _measure_focal([h], args.cx, args.cy)
                            # A bound that is being HIT is a finding, not a setting
                            # (camlab `inherited-claims.md`). A focal pinned at the search
                            # floor is not a lens: on `broadcast` f0 two such candidates came
                            # back at 300 px seeing 1087 samples — four times the support of
                            # the real camera — because a camera that wide frames the whole
                            # pitch and so misses nothing. They outranked the true answer.
                            edge = 0.01 * (args.focal_max - args.focal_min)
                            if not (args.focal_min + edge < focal < args.focal_max - edge):
                                continue
                            rot, t = _decompose(h, _k_inv(focal, args.cx, args.cy))
                            real = realizable_px(h, rot, t, focal, args.cx, args.cy, samples[:, :2])
                            if real > args.realizable_px:
                                continue
                            centre = -rot.T @ t
                            if not (args.height_min < centre[2] < args.height_max):
                                continue
                            dist = float(np.linalg.norm(centre[:2]))
                            if not (args.distance_min < dist < args.distance_max):
                                continue
                            # Score only against the correspondences that built it: this cannot be
                            # the verdict, and camlab's paint is asked for that below.
                            err = []
                            for i, m in zip(picks, ms, strict=True):
                                p0, p1 = segments[i][:2], segments[i][2:]
                                pts = p0 + np.linspace(0, 1, 15)[:, None] * (p1 - p0)
                                poly = project(h, world[m])
                                err.append(np.median([point_to_polyline_px(p, poly) for p in pts]))
                            key = (round(focal), *np.round(centre, 1))
                            if key in seen:
                                continue
                            seen.add(key)
                            out.append({
                                "realizable_px": float(real),
                                "focal_px": float(focal),
                                "rotation": cv2.Rodrigues(rot)[0].ravel().tolist(),
                                "position": [float(v) for v in centre],
                                "self_px": float(np.max(err)),
                                "markings": list(ms),
                                "segments": [int(i) + 1 for i in picks],
                            })
    out.sort(key=lambda c: c["self_px"])
    return out


def refine(base: str, which: str, frame: int) -> dict:
    """Ask camlab to finish the aim — `POST /api/run/{clip}/refine/{n}`, its own auto-fit.

    **The stage this script did not have, and without it the ranking was meaningless.** A
    hypothesis built from four lines through *detected* segments is an aim, not a solve: measured on
    `broadcast` f0 with correct labels, the raw anchor scored 327.59 px worst / 6.07 px median, and
    the very same anchor after this call scored **4.04 px worst / 0.84 px median on 268 markings** —
    against 3.42 / 0.94 for the camera camlab itself believes for that frame.

    That is exactly the two-click flow camlab documents for a human — *"aim it roughly, the solver
    finishes it. A rough aim at 445 px comes back at 4.7 px on `broadcast` frame 0"* — so ranking
    raw aims by paint asks the wrong question. `refit._accept` takes the fit only if the worst
    offset fell and no correspondence was lost, so a refit can refuse but not damage.
    """
    req = urllib.request.Request(  # noqa: S310 - localhost only
        f"{base}/refine/{frame}", method="POST",
        data=json.dumps({"which": which}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310 - localhost only
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        # Under `MIN_MATCHED` matches camlab says "aim closer" rather than damaging the frame.
        return {"moved": False, "error": e.read().decode()[:200]}


def write_manual(run_dir: Path, which: str, frame: int, entry: dict) -> None:
    """Write one anchor into camlab's manual store, atomically.

    camlab's own writer takes a lock — two files were found corrupted by interleaved writers — so
    replace-by-rename rather than read-modify-write in place. There is no human dragging while this
    runs, which is the point of the exercise, but a half-written anchor is the frame the whole
    chain hangs off.
    """
    path = run_dir / "camera_manual.json"
    blob = json.loads(path.read_text()) if path.exists() else {}
    blob.setdefault(which, {})[str(frame)] = {
        "focal_px": entry["focal_px"],
        "rotation": entry["rotation"],
        "position": entry["position"],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob, indent=1))
    tmp.replace(path)


def solve_anchor(server: str, clip: str, frame: int, which: str, method: str,
                 labels: dict, run_dir: Path, opts: argparse.Namespace) -> dict:
    """Rank label-consistent anchors for one frame and leave camlab's store exactly as it was.

    Factored out of `main` so the review UI (`scripts/labeller_ui.py`) drives the identical path
    rather than a second implementation of it — the failure mode ADR-0013 §4 is about, and the one
    `#141` keeps producing.

    Ranking writes each candidate into the manual store to score it, because camlab's residual
    route reads the overlay from disk; the store is restored before returning, so **nothing is
    committed here**. The caller commits with `write_manual`.
    """
    base = f"{server}/api/run/{clip}"
    lines = get_json(f"{base}/lines/{frame}?method={method}&which={which}")
    cam = get_json(f"{base}/camera?which={which}")
    pitch = get_json(f"{server}/api/pitch")

    segments = np.asarray(lines["segments"], float).reshape(-1, 4)
    opts.cx, opts.cy = float(cam["cx"]), float(cam["cy"])
    world = {k: np.asarray(pitch["markings"][k], float) for k in MARKING_TYPE}
    out: dict = {"n_segments": len(segments), "cx": opts.cx, "cy": opts.cy,
                 "width": int(lines["width"]), "height": int(lines["height"])}

    pool = candidates(segments, labels, world, out["width"], out["height"], opts)
    out["pool"] = len(pool)
    out["baseline"] = baseline = get_json(f"{base}/residual/{frame}?which={which}")
    if not pool:
        out["scored"], out["twins"], out["floor"] = [], False, 0.0
        return out

    store = Path(run_dir) / clip / "camera_manual.json"
    backup = store.read_text() if store.exists() else None
    scored = []
    try:
        for c in pool[:opts.shortlist]:
            write_manual(store.parent, which, frame, c)
            c["raw_median_px"] = get_json(f"{base}/residual/{frame}?which={which}")["median_px"]
            c["refine"] = refine(base, which, frame)
            r = get_json(f"{base}/residual/{frame}?which={which}")
            c["median_px"] = r["median_px"]
            c["worst_line_px"] = r["worst_line_px"]
            c["n_scored"] = r["n_scored"]
            c["camera"] = json.loads(store.read_text())[which][str(frame)]
            scored.append(c)
    finally:
        if backup is None:
            store.unlink(missing_ok=True)
        else:
            store.write_text(backup)

    floor = 0.5 * (baseline["n_scored"] or 0)
    scored.sort(key=lambda c: (c["median_px"] if (c["n_scored"] or 0) >= floor and c["median_px"]
                               else 1e9))
    twins = False
    if len(scored) > 1:
        a = np.asarray(scored[0]["camera"]["position"], float)
        b = np.asarray(scored[1]["camera"]["position"], float)
        twins = float(np.linalg.norm(a[:2] + b[:2])) < 0.05 * float(np.linalg.norm(a[:2]))
    out["scored"], out["twins"], out["floor"] = scored, twins, floor
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--labels", required=True, help='JSON: {"1": "touchline", ...} or {"labels":…}')
    ap.add_argument("--which", default="camera_smooth.json", help="the solve the anchor overlays")
    ap.add_argument("--method", default="hough", choices=("hough", "lsd"))
    ap.add_argument("--server", default="http://127.0.0.1:8899")
    ap.add_argument("--run-dir", default="/home/chubuchnyi/camlab/runs",
                    help="where camlab keeps its runs; the anchor is written here")
    ap.add_argument("--focal-min", type=float, default=FOCAL_BOUNDS[0])
    ap.add_argument("--focal-max", type=float, default=FOCAL_BOUNDS[1])
    ap.add_argument("--height-min", type=float, default=HEIGHT_BOUNDS[0])
    ap.add_argument("--height-max", type=float, default=HEIGHT_BOUNDS[1])
    ap.add_argument("--distance-min", type=float, default=DISTANCE_BOUNDS[0])
    ap.add_argument("--distance-max", type=float, default=DISTANCE_BOUNDS[1])
    ap.add_argument("--realizable-px", type=float, default=float("inf"),
                    help="reject a hypothesis this far from reproducing its own "
                         "homography; report-only by default, because camlab's refit "
                         "rescues aims this gate would throw away (see realizable_px)")
    ap.add_argument("--shortlist", type=int, default=SHORTLIST)
    ap.add_argument("--dry-run", action="store_true", help="rank only; leave the store untouched")
    args = ap.parse_args()

    raw_labels = json.loads(Path(args.labels).read_text())
    labels = {str(k): str(v) for k, v in raw_labels.get("labels", raw_labels).items()}

    # `n_scored` is not decoration: a camera that has run away projects almost everything somewhere
    # unscoreable and posts a flattering median on the survivors, so a median alone would reward
    # exactly the failure to avoid. The reference is the support the SEED camera gets on this
    # frame — how much of the pitch is scoreable is a property of the frame, not of the method,
    # and taking half the best candidate instead let a 300 px focal set the bar for everyone.
    res = solve_anchor(args.server, args.clip, args.frame, args.which, args.method,
                       labels, Path(args.run_dir), args)
    print(f"{args.clip} f{args.frame}: {res['n_segments']} segments, {len(labels)} labelled, "
          f"principal point ({res['cx']:.1f}, {res['cy']:.1f}) from camlab")
    if not res["scored"]:
        print("  no label-consistent, physically plausible camera — nothing to write")
        return 1

    scored, baseline, floor = res["scored"], res["baseline"], res["floor"]
    n_try = min(args.shortlist, res["pool"])
    print(f"  {res['pool']} label-consistent cameras; refitting the best {n_try} with camlab's own "
          f"auto-fit and scoring them on its paint")
    print(f"  seed camera on this frame: median {baseline['median_px']} px on "
          f"{baseline['n_scored']} samples — the bar to beat, and the support reference")
    print(f"\n  median  worst  n_scored   raw→   focal   position              segments→markings"
          f"   (support floor {floor:.0f})")
    for c in scored[:n_try]:
        pos = " ".join(f"{v:7.2f}" for v in c["position"])
        med = "  n/a " if c["median_px"] is None else f"{c['median_px']:6.2f}"
        wl = "  n/a" if c["worst_line_px"] is None else f"{c['worst_line_px']:5.1f}"
        raw = "  n/a" if c["raw_median_px"] is None else f"{c['raw_median_px']:5.1f}"
        pairs = ",".join(f"{s}→{m}" for s, m in zip(c["segments"], c["markings"], strict=True))
        print(f"  {med} {wl}  {str(c['n_scored']):>8}  {raw}  {c['camera']['focal_px']:6.0f}  "
              f"{pos}  {pairs}")

    # The pitch is exactly symmetric under a half-turn and camlab measures the twin as scoring
    # bit for bit the same, so when the top two are each other's negation the paint has not
    # chosen and cannot. Say so rather than let the sort order look like a decision.
    if res["twins"]:
        print("\n  ! the top two are HALF-TURN TWINS — the paint scores both the same and\n"
              "    never will. Which end this is has to come from off the pitch: the\n"
              "    labeller's left/right call, or camlab's `flip 180` button.")

    best = scored[0]
    ok = (best["n_scored"] or 0) >= floor and best["median_px"] is not None
    if args.dry_run or not ok:
        why = "dry run" if args.dry_run else f"best is unsupported ({best['n_scored']} scored)"
        print(f"\n  store left as it was ({why})")
        return 0 if args.dry_run else 2

    store = Path(args.run_dir) / args.clip / "camera_manual.json"
    write_manual(store.parent, args.which, args.frame, best["camera"])
    print(f"\n  wrote anchor f{args.frame} → {store}")
    print(f"  median {best['median_px']:.2f} px, worst line {best['worst_line_px']:.2f} px on "
          f"{best['n_scored']} scored samples; focal {best['camera']['focal_px']:.0f}, "
          f"position {np.round(best['camera']['position'], 2).tolist()}")
    print(f"  (raw aim was {best['raw_median_px']:.2f} px median before camlab's auto-fit; "
          f"a segment counts as explained within {EXPLAIN_PX:.0f} px)")
    print(f"  (gate for reference: a detected segment counts as explained within "
          f"{EXPLAIN_PX:.0f} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
