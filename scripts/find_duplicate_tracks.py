"""Find humans smeared across several track ids — without any labels.

The signature comes from the user's eye-verdict of 2026-08-13 and was then measured on the
crossing he named (`findings/quality-criterion-2026-08-13.md` §3.1): **two ids that are never
measured on the same frame and stay within half a metre are one human**, not two. At f33 of the
broadcast clip, t3 and t184 alternate — on every frame exactly one carries the detection and the
other coasts — while sitting 0.05-0.13 m apart. Meanwhile t3 and t5, a genuine pair, are measured
*together* and stand 1.8-2.6 m apart.

That asymmetry is mechanical, so it needs no labels:

* **co-measured frames** — a frame where BOTH are ``measured``. Two humans produce these; one
  human split across two ids cannot, because there is only ever one detection to give.
* **separation** — how far apart the two rows sit. The gate is **0.15 m** and it is a physical
  bound, not a tuned one: two standing humans' pelvises cannot be 15 cm apart. Swept on both arms —
  at 0.20 m the pair the eye called "t16 and t18 jostling" leaks in, at 0.50 m half the scene does.

A **run of consecutive frames** with zero co-measured frames and a small separation is one human
wearing two ids over that interval. The unit is the (pair, interval): t3 and t62 are one human for
14 frames and then genuinely two for the next hundred, so a verdict on the pair as a whole averages
the defect away. This is #135's П3 primitive ("simultaneously measured > 4 frames = two humans")
turned around and run over *every* pair and *every* interval instead of only over stitch
candidates — which is why nothing caught t184.

    PYTHONPATH=src .venv/bin/python scripts/find_duplicate_tracks.py \\
        --scene out/pod_ab/scene_b.json

Reports the pairs, then groups them transitively: a human split three ways shows up as one group
of three, and the headline number is **how many humans the scene smeared, and over how many ids**.

What it deliberately does NOT do: decide which id is the "real" one. That is the merge policy's
job (the user's rule: keep the parent's trajectory, take the fragment's pose) and it needs the eye.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _arr(x):
    """scene.json stores numpy arrays as ``{"__ndarray__": {...}}``."""
    if isinstance(x, dict) and "__ndarray__" in x:
        return np.asarray(x["__ndarray__"]["data"])
    return np.asarray(x)


def load_tracks(path: Path) -> dict[int, dict]:
    scene = json.load(open(path))["fields"]
    out: dict[int, dict] = {}
    for sub in scene["subjects"]:
        f = sub["fields"]
        pf = f["proposal"]["fields"]["pose"]["fields"]
        prov = [str(p) for p in _arr(pf["provenance"])]
        frames = _arr(pf["frames"]).astype(int)
        measured = np.array([p == "measured" for p in prov], dtype=bool)
        out[int(f["track_id"])] = {
            "frames": frames,
            "xy": _arr(pf["transl"]).astype(float)[:, :2],
            "measured": measured,
            "team": f.get("team_id"),
        }
    return out


def measured_span(t: dict) -> tuple[int, int] | None:
    """First..last frame this id was actually measured on. Invented prefix/suffix excluded."""
    idx = np.flatnonzero(t["measured"])
    if not idx.size:
        return None
    return int(t["frames"][idx[0]]), int(t["frames"][idx[-1]])


def duplicate_runs(a: dict, b: dict, *, max_sep: float, min_len: int) -> list[dict]:
    """Longest runs of consecutive frames on which the two ids behave as ONE human.

    The unit is the (pair, interval) and not the pair. Measured on the broadcast clip: t3 and t62
    alternate 0.1-0.7 m apart for 14 frames (f33-46) and are then measured *together* and walk to
    5.6 m for the next hundred. Judging the whole overlap averages those together and throws the
    duplicate away — which is exactly what the first version of this script did. The eye said the
    same thing in its own words: "correct until frame 39, then it became t62".

    A frame belongs to a run when the two rows are within ``max_sep`` AND not both ``measured``.
    """
    ia = {int(f): i for i, f in enumerate(a["frames"])}
    ib = {int(f): i for i, f in enumerate(b["frames"])}
    common = sorted(set(ia) & set(ib))
    runs, cur = [], []
    for f in common:
        i, j = ia[f], ib[f]
        both = bool(a["measured"][i]) and bool(b["measured"][j])
        sep = float(np.linalg.norm(a["xy"][i] - b["xy"][j]))
        if not both and sep <= max_sep:
            cur.append((f, sep, bool(a["measured"][i]) != bool(b["measured"][j])))
        else:
            if len(cur) >= min_len:
                runs.append(cur)
            cur = []
    if len(cur) >= min_len:
        runs.append(cur)

    out = []
    for r in runs:
        seps = np.array([s for _, s, _ in r])
        # A run of two coasting mannequins parked near each other proves nothing: demand that the
        # detection actually ALTERNATES, i.e. one of them is measured on most frames of the run.
        alternating = sum(1 for _, _, one in r if one)
        out.append({
            "window": (r[0][0], r[-1][0]),
            "length": len(r),
            "alternating": alternating,
            "median_sep": float(np.median(seps)),
            "max_sep": float(seps.max()),
        })
    return out


def group(pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Transitive closure — a human split three ways is one group, not three pairs."""
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    out: dict[int, list[int]] = {}
    for x in parent:
        out.setdefault(find(x), []).append(x)
    return [sorted(v) for v in out.values() if len(v) > 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--max-sep", type=float, default=0.15,
                    help="separation under which two ids cannot be two humans (m). NOT tuned: two "
                         "standing people's pelvises do not sit 15 cm apart. Swept 0.15/0.20/0.50 "
                         "on both arms — at 0.20 the jostling pair t16/t18 leaks in, and the eye "
                         "says they are two players in contact; at 0.50 half the scene leaks in.")
    ap.add_argument("--min-overlap", type=int, default=4,
                    help="consecutive frames a duplicate interval must last to be reported")
    ap.add_argument("--min-alternating", type=float, default=0.6,
                    help="fraction of the run on which exactly one of the two must be measured")
    ap.add_argument("--show-healthy", action="store_true",
                    help="also list the closest pairs that were REJECTED, to see the margin")
    args = ap.parse_args()

    tracks = load_tracks(Path(args.scene))
    ids = sorted(tracks)
    hits, healthy = [], []
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            a, b = ids[x], ids[y]
            runs = duplicate_runs(tracks[a], tracks[b], max_sep=args.max_sep,
                                  min_len=args.min_overlap)
            for r in runs:
                # Both coasting side by side proves nothing — demand the detection alternate.
                if r["alternating"] >= args.min_alternating * r["length"]:
                    hits.append((a, b, r))
                else:
                    healthy.append((a, b, r))

    print(f"scene: {args.scene}   {len(ids)} ids")
    print(f"rule: a run of >= {args.min_overlap} consecutive frames, never both measured, "
          f"separation <= {args.max_sep} m,\n      with the detection alternating on "
          f">= {args.min_alternating:.0%} of the run\n")

    if not hits:
        print("  no duplicate intervals")
    for a, b, r in sorted(hits, key=lambda t: -t[2]["length"]):
        print(f"  t{a:<4d} + t{b:<4d}  f{r['window'][0]}-{r['window'][1]} ({r['length']:3d} f)"
              f"  alternating {r['alternating']:3d}/{r['length']:<3d}"
              f"  sep median {r['median_sep']:.2f} m max {r['max_sep']:.2f} m")

    groups = group([(a, b) for a, b, _ in hits])
    smeared = sum(len(g) - 1 for g in groups)
    print(f"\n  {len(groups)} human(s) smeared over multiple ids; "
          f"{smeared} id(s) duplicate another over some interval")
    for g in sorted(groups, key=len, reverse=True):
        print("    one human as: " + " + ".join(f"t{t}" for t in g))
    frames_lost = sum(r["length"] for _, _, r in hits)
    print(f"\n  headline: {len(ids)} ids -> {len(ids) - smeared} distinct humans; "
          f"{frames_lost} subject-frame(s) carried by a duplicate id")

    if args.show_healthy:
        print("\n  runs rejected for not alternating (two mannequins parked together):")
        for a, b, r in sorted(healthy, key=lambda t: t[2]["median_sep"])[:10]:
            print(f"    t{a:<4d} + t{b:<4d}  f{r['window'][0]}-{r['window'][1]}  alternating "
                  f"{r['alternating']}/{r['length']}  sep {r['median_sep']:.2f} m -> both coasting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
