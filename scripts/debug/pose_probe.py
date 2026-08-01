"""Per-layer pose debug harness for the A/B bakeoff scenes.

For a chosen frame this dumps every subject along the full transform chain —
raw world ``transl`` → world joints → camera-space → pixels — and computes
objective, ground-truth-free health metrics so we can localise a bug to a
specific layer instead of eyeballing an overlay:

  * transl_ok : raw world transl sits on/near the pitch (|x|<=60, |y|<=40,
                0<=z<=3 m). A z-up world with origin at the centre mark means a
                standing player's root is ~(±52, ±34, ~0.9).  Fails ⇒ the pose
                net's world translation is wrong (wrong units / origin / frame).
  * in_front  : pelvis camera-space z > 0 (in front of the camera, not behind).
  * in_frame  : pelvis pixel lands inside the image (0<=u<W, 0<=v<H).
  * upright   : foot pixel-v > head pixel-v.  Gravity is ground truth: a standing
                player's feet sit LOWER in the image than the head, regardless of
                world frame.  Systematic upright=False ⇒ a vertical inversion.
  * scale_px  : skeleton pixel height.  A real broadcast player is ~40-140 px;
                a few px ⇒ collapsed/degenerate scale.

Auto-detects the subjects present on the frame; ``--tid`` restricts to one.

Usage:
    python scripts/debug/pose_probe.py --scene out/bakeoff/scene_A.json --frames 0 16
    python scripts/debug/pose_probe.py --scene out/bakeoff/scene_B.json --tid 3 --frames 0
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from poseannot.camera import frame_projector, project_points
from poseannot.config import load as load_cfg
from poseannot.scene_state import build_scene_state
from poseannot.video import frame_size

DEFAULT_VIDEO = "samples/video/Colombia-1-0-Congo-DR1080p.mp4"

# SMPL-X body joint indices (see poseannot.scene_state.BODY_JOINT_NAMES).
PELVIS, HEAD = 0, 15
L_ANKLE, R_ANKLE, L_FOOT, R_FOOT = 7, 8, 10, 11
FEET = (L_ANKLE, R_ANKLE, L_FOOT, R_FOOT)


def _load(scene: str, video: str):
    base = load_cfg()
    cfg = replace(
        base,
        scene_json=Path(scene).resolve(),
        source_video=Path(video).resolve(),
        corrections_out=Path("/tmp/pose_probe_noedits.json").resolve(),
    )
    st = build_scene_state(cfg)
    W, H = frame_size(str(cfg.source_video))
    return st, W, H


def _nanmax(vals):
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    return float(arr.max()) if arr.size else np.nan


def probe(scene: str, video: str, frame: int, only_tid: int | None) -> None:
    st, W, H = _load(scene, video)
    proj = frame_projector(st.scene.camera, frame, video_size=(W, H))
    print(
        f"\n=== {Path(scene).name}  frame {frame}  video {W}x{H}  "
        f"flip={proj.frame_flipped}  subjects_total={len(st.subjects)} ==="
    )
    rows = []
    for tid, sub in sorted(st.subjects.items()):
        if only_tid is not None and tid != only_tid:
            continue
        hit = np.where(sub.frames == frame)[0]
        if hit.size == 0:
            continue
        k = int(hit[0])
        j_world = sub.joints[k]                       # (22, 3) z-up world
        tx, ty, tz = (float(v) for v in sub.transl[k])
        transl_ok = abs(tx) <= 60 and abs(ty) <= 40 and 0.0 <= tz <= 3.0

        cam = j_world @ proj.R.T + proj.t             # (22, 3) camera space
        z_pelvis = float(cam[PELVIS, 2])
        in_front = z_pelvis > 1e-6

        pix = project_points(j_world, proj)           # (22, 2) pixels
        pu, pv = (float(x) for x in pix[PELVIS])
        in_frame = np.isfinite(pu) and np.isfinite(pv) and 0 <= pu < W and 0 <= pv < H

        head_v = float(pix[HEAD, 1])
        foot_v = _nanmax([pix[i, 1] for i in FEET])
        upright = np.isfinite(head_v) and np.isfinite(foot_v) and foot_v > head_v

        valid = np.isfinite(pix).all(axis=1)
        scale_px = float(pix[valid, 1].max() - pix[valid, 1].min()) if valid.any() else np.nan
        # World stature (m): head - lowest foot along Z-up.  ~1.7-1.9 for a real body.
        stature = float(j_world[HEAD, 2] - min(j_world[i, 2] for i in FEET))

        rows.append(dict(
            tid=tid, tx=tx, ty=ty, tz=tz, transl_ok=transl_ok,
            z_pelvis=z_pelvis, in_front=in_front, pu=pu, pv=pv, in_frame=in_frame,
            upright=upright, scale_px=scale_px, stature=stature,
        ))

    if not rows:
        print("  (no subjects on this frame)")
        return

    def cnt(key):
        return sum(1 for r in rows if r[key])

    n = len(rows)
    sc = np.array([r["scale_px"] for r in rows if np.isfinite(r["scale_px"])])
    txr = np.array([r["tx"] for r in rows])
    tyr = np.array([r["ty"] for r in rows])
    tzr = np.array([r["tz"] for r in rows])
    print(
        f"  present={n}  transl_ok={cnt('transl_ok')}/{n}  in_front={cnt('in_front')}/{n}  "
        f"in_frame={cnt('in_frame')}/{n}  upright={cnt('upright')}/{n}"
    )
    if sc.size:
        print(f"  scale_px  min={sc.min():.0f} med={np.median(sc):.0f} max={sc.max():.0f}  (want ~40-140)")
    print(
        f"  transl x[{txr.min():.1f},{txr.max():.1f}] y[{tyr.min():.1f},{tyr.max():.1f}] "
        f"z[{tzr.min():.2f},{tzr.max():.2f}]  (pitch: |x|<=52.5 |y|<=34 z~0.9)"
    )
    print("  tid    transl(x,y,z)            zc     pelvis(u,v)      scalepx statm up inF inFrm tOK")
    for r in sorted(rows, key=lambda r: r["tid"])[:24]:
        print(
            f"  t{r['tid']:<3} ({r['tx']:7.1f},{r['ty']:6.1f},{r['tz']:5.2f})  {r['z_pelvis']:7.1f}  "
            f"({r['pu']:7.0f},{r['pv']:6.0f})  {r['scale_px']:6.0f}  {r['stature']:5.2f}  "
            f"{int(r['upright'])}  {int(r['in_front'])}   {int(r['in_frame'])}    {int(r['transl_ok'])}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--frames", type=int, nargs="+", default=[0])
    ap.add_argument("--tid", type=int, default=None, help="restrict to one track id")
    args = ap.parse_args()
    for fr in args.frames:
        probe(args.scene, args.video, fr, args.tid)


if __name__ == "__main__":
    main()
