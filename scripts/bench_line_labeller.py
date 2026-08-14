"""Can a vision model name camlab's detected pitch segments well enough to seed a camera?

This is the §7 measurement of `camlab/docs/findings/automating-the-anchor-2026-08-13.md`, and it
is deliberately the *first* thing built, because a negative result costs an afternoon and cancels
everything downstream of it.

**The question.** camlab's bootstrap fails on five of six anchors by *abstaining* — the generator
is right (4680 physically plausible cameras in 12 s) and the chooser lands 113 m out, because
anonymous paint cannot say which line is which. Names collapse that: `_homography_from_lines`
needs **four** correspondences, two per parallel family. So: given one frame with camlab's own
detected segments drawn and numbered, can a model name enough of them?

**Why this can be scored with no hand labelling at all.** camlab already has clips whose camera it
believes. Project the pitch model through that camera and every detected segment gets its true
class for free — `is-a-model-worth-training.md`'s self-labelling idea run backwards: instead of
making training data, it makes an answer key. The labeller never sees it.

**Why the vocabulary is coarse and half-turn symmetric.** It names types, not instances:
`touchline`, not "Side line top". Two reasons, both load-bearing:

* `solve/bootstrap.hypotheses` already resolves *instances* by order within a family, so reducing
  the type is all that is needed — and "is this the touchline or the goal line" is a far more
  reliable question than "is this Big rect. left top or Big rect. right top".
* the pitch is exactly symmetric under a half-turn (camlab measures it as bit-identical, 2.1 px on
  307 samples either way), so a vocabulary that *did* distinguish left from right would be asking
  the model to answer something the geometry cannot check. That stays a separate binary question;
  `bootstrap_clip.py` already emits both twins.

Two modes::

    # 1. build the labelling image + the answer key (the key is NOT shown to the labeller)
    python scripts/bench_line_labeller.py prepare --clip broadcast --frame 0 \
        --which camera_smooth.json --out out/labeller

    # 2. score a labeller's answer against the key
    python scripts/bench_line_labeller.py score --dir out/labeller/broadcast_f0 \
        --labels out/labeller/broadcast_f0/labels.json

camlab is driven over its HTTP API and is not modified: `GET /api/run/{clip}/lines/{n}` returns
both halves — its **own** detected segments, and the pitch model projected through the camera —
so a label lands one-to-one on the line the solver will actually use.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

#: A detected segment counts as explained by a model marking within this perpendicular distance.
#: Same constant as ``scripts/bench_markings_vs_camera.py``, where it was set against a camera
#: known to be right; on `broadcast` frame 0 the worst matched offset is 0.59 px, so the gate is
#: two orders of margin and the segments it rejects are rejected clearly.
EXPLAIN_PX = 12.0

#: camlab's straight-marking indices (``measure/line_error.straight_markings``) → the coarse type a
#: labeller is asked for. Derived from the world endpoints, not typed by hand:
#: ``family 0`` runs along the pitch's long axis, ``family 1`` across it.
MARKING_TYPE = {
    0: "touchline", 2: "touchline",
    1: "goal_line", 3: "goal_line",
    4: "halfway_line",
    7: "penalty_area_side", 9: "penalty_area_side",
    15: "penalty_area_side", 17: "penalty_area_side",
    8: "penalty_area_front", 16: "penalty_area_front",
    10: "goal_area_side", 12: "goal_area_side",
    18: "goal_area_side", 20: "goal_area_side",
    11: "goal_area_front", 19: "goal_area_front",
}

#: What the labeller may answer. `not_a_marking` is the important one — it is what the advertising
#: hoarding join must be called, and #14's open precision half is exactly this class.
VOCABULARY = [
    "touchline", "goal_line", "halfway_line",
    "penalty_area_front", "penalty_area_side",
    "goal_area_front", "goal_area_side",
    "not_a_marking",
]

#: Which parallel family each type belongs to. A hypothesis needs two correspondences per family,
#: so a type error *within* a family is far cheaper than one across it — scored separately.
TYPE_FAMILY = {
    "touchline": 0, "penalty_area_side": 0, "goal_area_side": 0,
    "goal_line": 1, "halfway_line": 1, "penalty_area_front": 1, "goal_area_front": 1,
    "not_a_marking": None,
}


#: Vanishing-point consensus tolerance for `split_families`. camlab's `VP_TOL_PX`.
VP_TOL_PX = 60.0


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310 - localhost only
        return r.read()


def _line(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Homogeneous line through two points, normalised so ``|n| = 1``. camlab `bootstrap._line`."""
    line = np.cross(np.array([a[0], a[1], 1.0]), np.array([b[0], b[1], 1.0]))
    n = float(np.linalg.norm(line[:2]))
    return line / n if n > 1e-12 else line


def split_families(segments: np.ndarray, tol_px: float = VP_TOL_PX) -> tuple[list[int], list[int]]:
    """Split detected segments into the two world-parallel families, from the image alone.

    Copied from camlab `src/camlab/solve/bootstrap.py:split_families` (ADR-0013 §5). Consensus on a
    shared vanishing point, **not** clustering on angle — camlab measured one real family's image
    directions spread across 22.1° to 32.5° against an 8° tolerance, while their vanishing point is
    shared exactly.

    It is here because the first blind labelling measured *this* as the thing a vision model gets
    wrong. On `fan` f8 the labeller grouped the segments almost correctly and then assigned the
    groups the wrong meaning — it called the across-pitch family "the long axis" — which flipped
    six labels of nine. Geometry does this reliably and for free, so the model should never be
    asked. What is left for it is the question no geometry here has answered: paint or not paint.
    """
    n = len(segments)
    if n < 4:
        return list(range(n)), []
    lines = [_line(np.asarray(s[:2], float), np.asarray(s[2:], float)) for s in segments]

    best: list[int] = []
    for i, j in combinations(range(n), 2):
        v = np.cross(lines[i], lines[j])
        v = np.array([v[0], v[1], 0.0]) if abs(v[2]) < 1e-9 else v / v[2]
        members = [k for k, line in enumerate(lines) if abs(float(line @ v)) < tol_px]
        if len(members) > len(best):
            best = members
    a = best
    b = [k for k in range(n) if k not in a]
    return (a, b) if len(a) >= len(b) else (b, a)


def point_to_polyline_px(pt: np.ndarray, poly: np.ndarray) -> float:
    """Shortest distance from a point to a polyline, in pixels."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    t = np.where(denom > 1e-9, np.einsum("ij,ij->i", pt - a, ab) / np.maximum(denom, 1e-9), 0.0)
    proj = a + np.clip(t, 0.0, 1.0)[:, None] * ab
    return float(np.min(np.linalg.norm(proj - pt, axis=1)))


def segment_to_marking(seg: np.ndarray, models: dict[int, np.ndarray]) -> tuple[int | None, float]:
    """Nearest projected marking to a detected segment, by the median over samples along it.

    The median rather than the mean, and over samples rather than endpoints: a detected fragment
    covering a quarter of a projected touchline is still that touchline, and a fragment lying
    across two markings should not be awarded to whichever one its endpoint happens to touch.
    """
    p0, p1 = seg[:2], seg[2:]
    samples = p0 + np.linspace(0.0, 1.0, 25)[:, None] * (p1 - p0)
    best_k, best_d = None, float("inf")
    for k, poly in models.items():
        d = float(np.median([point_to_polyline_px(s, poly) for s in samples]))
        if d < best_d:
            best_k, best_d = k, d
    return (best_k, best_d) if best_d <= EXPLAIN_PX else (None, best_d)


def draw(frame: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """The frame with every detected segment drawn and numbered, legibly at full resolution.

    Nothing else is drawn — no model, no camera, no residual. The labeller must see exactly what
    camlab detected and nothing that would leak the answer.
    """
    out = frame.copy()
    placed: list[tuple[int, int]] = []
    for i, s in enumerate(segments, start=1):
        p0 = (int(round(s[0])), int(round(s[1])))
        p1 = (int(round(s[2])), int(round(s[3])))
        cv2.line(out, p0, p1, (0, 0, 0), 9, cv2.LINE_AA)
        cv2.line(out, p0, p1, (0, 245, 255), 4, cv2.LINE_AA)
        mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        # Push the label off the line along its normal, or it covers the very pixels being judged.
        d = np.array([s[2] - s[0], s[3] - s[1]], float)
        n = np.array([-d[1], d[0]])
        n = n / max(float(np.linalg.norm(n)), 1e-6) * 44.0
        anchor = (int(mid[0] + n[0]), int(mid[1] + n[1]))
        # And keep it clear of labels already drawn. Measured: on `broadcast` f0 boxes 4 and 5
        # landed 23 px apart at the left edge, box 5 covered box 4, and the labeller reported
        # "there is no number 4 anywhere in the frame" — one segment silently dropped from the run.
        for _ in range(12):
            clash = next((q for q in placed if abs(q[0] - anchor[0]) < 46
                          and abs(q[1] - anchor[1]) < 40), None)
            if clash is None:
                break
            anchor = (anchor[0], anchor[1] + 44 if anchor[1] >= clash[1] else anchor[1] - 44)
        anchor = (int(np.clip(anchor[0], 30, out.shape[1] - 30)),
                  int(np.clip(anchor[1], 30, out.shape[0] - 30)))
        placed.append(anchor)
        text = str(i)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
        tl = (anchor[0] - tw // 2 - 10, anchor[1] - th // 2 - 10)
        br = (anchor[0] + tw // 2 + 10, anchor[1] + th // 2 + 10)
        cv2.rectangle(out, tl, br, (0, 0, 0), -1)
        cv2.rectangle(out, tl, br, (0, 245, 255), 3)
        cv2.putText(out, text, (anchor[0] - tw // 2, anchor[1] + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 245, 255), 4, cv2.LINE_AA)
    return out


def crop_tile(frame: np.ndarray, seg: np.ndarray, index: int, tile: int = 384) -> np.ndarray:
    """One segment's neighbourhood, upscaled, with the segment drawn thin and **dashed**.

    Dashed on purpose: the question is whether there is white paint under the line, and a solid
    overlay would be the only white the reader could see. Served one at a time by the review UI and
    tiled into a contact sheet by `crops` for a model that reads whole images.
    """
    cx_, cy_ = (seg[0] + seg[2]) / 2.0, (seg[1] + seg[3]) / 2.0
    half = max(float(np.hypot(seg[2] - seg[0], seg[3] - seg[1])) * 0.75, 60.0)
    x0, y0 = int(max(cx_ - half, 0)), int(max(cy_ - half, 0))
    x1, y1 = int(min(cx_ + half, frame.shape[1])), int(min(cy_ + half, frame.shape[0]))
    patch = frame[y0:y1, x0:x1].copy()
    if patch.size == 0:
        patch = np.zeros((10, 10, 3), np.uint8)
    scale = tile / max(patch.shape[0], patch.shape[1])
    patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    p0 = (int((seg[0] - x0) * scale), int((seg[1] - y0) * scale))
    p1 = (int((seg[2] - x0) * scale), int((seg[3] - y0) * scale))
    n_dash = 24
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    for d in range(0, n_dash, 2):
        a = (int(p0[0] + dx * d / n_dash), int(p0[1] + dy * d / n_dash))
        b = (int(p0[0] + dx * (d + 1) / n_dash), int(p0[1] + dy * (d + 1) / n_dash))
        cv2.line(patch, a, b, (0, 245, 255), 2, cv2.LINE_AA)
    canvas = np.zeros((tile + 40, tile, 3), np.uint8)
    canvas[40:40 + patch.shape[0], :patch.shape[1]] = patch
    cv2.putText(canvas, f"segment {index}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 245, 255), 2, cv2.LINE_AA)
    return canvas


def crops(frame: np.ndarray, segments: np.ndarray, tile: int = 384, cols: int = 3) -> np.ndarray:
    """A contact sheet of one zoomed tile per segment — the instrument the labeller asked for.

    **Built because two independent signals said the full frame is not enough.** The first two
    labeller runs both stalled trying to crop the image themselves, and the one that finished said
    in its own notes that its answer *"rests on a nesting argument rather than on directly readable
    paint"* — i.e. it reasoned about geometry because it could not see whether there was white
    paint under the line. On `fan` f8 (1080×608) that produced 1 correct label of 9.

    Each tile is the segment's neighbourhood upscaled, with the segment drawn **thin and dashed**
    so the paint underneath stays visible. A crop cannot answer "which marking is this" — that
    needs the whole pitch — so this is supplementary to the overview, never a replacement.
    """
    tiles = [crop_tile(frame, s, i, tile) for i, s in enumerate(segments, start=1)]
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def prepare(args: argparse.Namespace) -> int:
    base = f"{args.server}/api/run/{args.clip}"
    data = json.loads(fetch(f"{base}/lines/{args.frame}?method={args.method}&which={args.which}"))
    segments = np.asarray(data["segments"], float).reshape(-1, 4)
    if len(segments) == 0:
        print(f"{args.clip} f{args.frame}: no segments detected — nothing to label")
        return 1

    raw = np.frombuffer(fetch(f"{base}/frame/{args.frame}"), np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"could not decode {base}/frame/{args.frame}")
        return 1

    models = {int(e["marking"]): np.asarray(e["model"], float) for e in data["lines"]}
    key = {}
    for i, seg in enumerate(segments, start=1):
        k, dist = segment_to_marking(seg, models)
        key[str(i)] = {
            "marking": k,
            "type": MARKING_TYPE.get(k, "not_a_marking") if k is not None else "not_a_marking",
            "distance_px": round(dist, 2),
        }

    out_dir = Path(args.out) / f"{args.clip}_f{args.frame}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "numbered.png"), draw(frame, segments))
    cv2.imwrite(str(out_dir / "crops.png"), crops(frame, segments))
    (out_dir / "key.json").write_text(json.dumps({
        "clip": args.clip, "frame": args.frame, "which": args.which, "method": args.method,
        "explain_px": EXPLAIN_PX, "summary": data["summary"], "key": key,
    }, indent=2))
    (out_dir / "segments.json").write_text(json.dumps(segments.tolist(), indent=2))

    named = sum(1 for v in key.values() if v["marking"] is not None)
    print(f"{args.clip} f{args.frame}: {len(segments)} segments, {named} explained by the camera "
          f"within {EXPLAIN_PX:.0f} px, {len(segments) - named} not")
    print(f"  labelling image: {out_dir / 'numbered.png'}")
    print(f"  close-ups      : {out_dir / 'crops.png'}")
    print(f"  answer key     : {out_dir / 'key.json'}  (do not show this to the labeller)")
    return 0


def score(args: argparse.Namespace) -> int:
    d = Path(args.dir)
    key = json.loads((d / "key.json").read_text())["key"]
    labels = json.loads(Path(args.labels).read_text())
    labels = labels.get("labels", labels)

    rows, per_family = [], {0: 0, 1: 0}
    for i in sorted(key, key=int):
        truth = key[i]["type"]
        said = str(labels.get(i, "<missing>"))
        exact = said == truth
        fam_ok = (TYPE_FAMILY.get(said) is not None
                  and TYPE_FAMILY.get(said) == TYPE_FAMILY.get(truth))
        if fam_ok:
            per_family[TYPE_FAMILY[truth]] += 1
        rows.append((i, truth, said, exact, fam_ok, key[i]["distance_px"]))

    print("\n  #  truth                 said                  type  family   dist")
    print("  -- --------------------- --------------------- ----- ------- ------")
    for i, truth, said, exact, fam_ok, dist in rows:
        print(f"  {i:>2} {truth:<21} {said:<21} {'ok' if exact else '  ':<5} "
              f"{'ok' if fam_ok else '  ':<7} {dist:>6.1f}")

    n = len(rows)
    exact_n = sum(r[3] for r in rows)
    fam_n = sum(r[4] for r in rows)
    # A hypothesis needs four correspondences, two per family, and a wrong family is unrecoverable
    # while a wrong instance within a family is what `hypotheses()` already enumerates over.
    passes = per_family[0] >= 2 and per_family[1] >= 2
    print(f"\n  exact type      {exact_n}/{n}")
    print(f"  correct family  {fam_n}/{n}   (family 0: {per_family[0]}, family 1: {per_family[1]})")
    verdict = "PASS" if passes else "FAIL"
    print(f"\n  BAR: two correct-family correspondences per family — {verdict}")
    if not passes:
        print("  Below the bar, `_homography_from_lines` cannot be seeded from these labels.")
    return 0 if passes else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", default="http://127.0.0.1:8899", help="camlab server")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="build the labelling image and the answer key")
    p.add_argument("--clip", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--which", default="camera_smooth.json", help="the camera to believe")
    p.add_argument("--method", default="hough", choices=("hough", "lsd"))
    p.add_argument("--out", default="out/labeller")
    p.set_defaults(func=prepare)

    p = sub.add_parser("score", help="score a labeller's answer against the key")
    p.add_argument("--dir", required=True, help="the prepare/ output directory")
    p.add_argument("--labels", required=True, help='JSON: {"1": "touchline", ...}')
    p.set_defaults(func=score)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
