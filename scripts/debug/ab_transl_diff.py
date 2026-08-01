"""A vs B golden test — do the two pose backends agree on WHERE each player is?

Variants A (SMPLest-X) and B (SAM 3D Body) run on the *same* detect / track /
calibrate upstream; only the pose network differs.  Therefore the world root
translation of a given ``track_id`` — the player's position on the pitch — must
be (nearly) identical between A and B.  It's a units/origin/frame invariant, not
a modelling choice.  Where they diverge, one backend's world transform is wrong,
and the pitch-plausibility of each side says which one.

Reports, per scene: transl ranges + how many subjects sit on the pitch; then,
for every shared ``track_id``, the per-frame translation delta.

Usage:
    python scripts/debug/ab_transl_diff.py --a out/bakeoff/scene_A.json \
        --b out/bakeoff/scene_B.json --frame 0
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from poseannot.config import load as load_cfg
from poseannot.scene_state import build_scene_state

DEFAULT_VIDEO = "samples/video/Colombia-1-0-Congo-DR1080p.mp4"


def _load(scene: str, video: str):
    base = load_cfg()
    cfg = replace(
        base,
        scene_json=Path(scene).resolve(),
        source_video=Path(video).resolve(),
        corrections_out=Path("/tmp/pose_probe_noedits.json").resolve(),
    )
    return build_scene_state(cfg)


def _pitch_ok(t) -> bool:
    return abs(t[0]) <= 60 and abs(t[1]) <= 40 and 0.0 <= t[2] <= 3.0


def _summary(tag: str, st) -> None:
    tr = []
    for sub in st.subjects.values():
        tr.append(sub.transl[0])
    tr = np.array(tr, dtype=float)
    ok = sum(_pitch_ok(t) for t in tr)
    print(
        f"  {tag}: {len(st.subjects)} subjects  pitch_ok={ok}/{len(tr)}  "
        f"transl x[{tr[:,0].min():.1f},{tr[:,0].max():.1f}] "
        f"y[{tr[:,1].min():.1f},{tr[:,1].max():.1f}] "
        f"z[{tr[:,2].min():.2f},{tr[:,2].max():.2f}]"
    )


def diff(a: str, b: str, video: str, frame: int) -> None:
    sa, sb = _load(a, video), _load(b, video)
    print(f"\n=== A/B golden transl test  frame {frame} ===")
    _summary(f"A {Path(a).name}", sa)
    _summary(f"B {Path(b).name}", sb)

    common = sorted(set(sa.subjects) & set(sb.subjects))
    print(f"  shared track_ids: {len(common)}  ({common[:20]})")
    if not common:
        print("  (no shared track_ids at same frame — cannot pair directly; compare ranges above)")
        return
    print("  tid    A transl(x,y,z)        B transl(x,y,z)        |delta| m")
    for tid in common:
        ca, cb = sa.subjects[tid], sb.subjects[tid]
        ka = np.where(ca.frames == frame)[0]
        kb = np.where(cb.frames == frame)[0]
        if ka.size == 0 or kb.size == 0:
            continue
        ta = ca.transl[int(ka[0])]
        tb = cb.transl[int(kb[0])]
        d = float(np.linalg.norm(np.asarray(ta) - np.asarray(tb)))
        print(
            f"  t{tid:<3} ({ta[0]:7.1f},{ta[1]:6.1f},{ta[2]:5.2f})   "
            f"({tb[0]:8.1f},{tb[1]:7.1f},{tb[2]:6.2f})   {d:9.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()
    diff(args.a, args.b, args.video, args.frame)


if __name__ == "__main__":
    main()
