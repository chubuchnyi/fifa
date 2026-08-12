"""A/B the association step alone: ByteTrack vs BoT-SORT, on the SAME cached detections.

Why this is worth a script. #133 measured 78 mid-pitch identity births/deaths on shot 1, and
96 % of them had an unclaimed detection a median 6-23 px away — the boxes were there and the
matching lost them. W5 sharpened it on 2026-08-10: those births are *contested*, not orphaned
(a birth's best assignment cost is 0.176-0.221 against 0.250 for every column), which puts the
defect in **allocation**. Five cheap fixes have now hit the same plateau (#138).

BoT-SORT differs from ByteTrack in one term that matters here: it estimates a frame-to-frame
affine warp (GMC) and applies it to every track's Kalman prediction before matching. Ultralytics
documents ByteTrack as having "no camera-motion compensation", and the portrait fan clip's
dominant defect is one whip-pan at f38 breaking six identities at once (#135 §8). So the
question this script answers is narrow and falsifiable: **with the detections held fixed, does
a camera-motion term reduce mid-pitch identity events?**

Three arms, because "BoT-SORT is better" would not say *why*:

* ``bytetrack``      — the shipped :class:`ByteTrackBackend` (supervision). Control.
* ``botsort``        — BoT-SORT with ``sparseOptFlow`` GMC.
* ``botsort-nogmc``  — BoT-SORT with the camera model switched OFF. Isolates GMC from
  BoT-SORT's other differences (fuse_score, proximity gate, its own Kalman tuning).

All three run on identical detections and identical thresholds (match 0.8, activation 0.25,
buffer 30) — :class:`BotSortBackend` inherits them from :class:`ByteTrackBackend` for exactly
this reason.

    PYTHONPATH=src .venv/bin/python scripts/bench_association.py

An event is counted the way `identity_failure_kind.py` counts one: a track's first frame (birth)
or last frame (death), excluding the clip boundary (±2 frames) and excluding boxes at the frame
edge, where an entry or exit is the honest explanation.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pitch3d.core.ports.io import ClipRef  # noqa: E402
from pitch3d.core.ports.perception import Detection, Detections, FrameDetections  # noqa: E402

#: Fraction of frame width treated as "the edge" — a track that starts or ends here entered or
#: left the shot, which is not an identity failure. Same 6 % `identity_failure_kind.py` uses.
EDGE_FRAC = 0.06


def load_detections(path: Path) -> tuple[Detections, int]:
    """``dets_*.npz`` (ragged object arrays) → the port's :class:`Detections`."""
    z = np.load(path, allow_pickle=True)
    frames = []
    for f, bb, kk, ss in zip(z["frame"], z["boxes"], z["classes"], z["scores"], strict=True):
        items = [
            Detection(bbox_xyxy=np.asarray(b, dtype=float), cls=str(k), score=float(s))
            for b, k, s in zip(bb, kk, ss, strict=True)
        ]
        frames.append(FrameDetections(frame=int(f), items=items))
    frames.sort(key=lambda fd: fd.frame)
    return Detections(frames=frames), len(frames)


def mid_pitch_events(tracklets, width: int, height: int, last_frame: int) -> dict[str, int]:
    """Births + deaths that neither the clip boundary nor the frame edge explains."""
    edge_x, edge_y = EDGE_FRAC * width, EDGE_FRAC * height

    def at_edge(b) -> bool:
        return bool(
            b[0] < edge_x or b[2] > width - edge_x or b[1] < edge_y or b[3] > height - edge_y
        )

    births = deaths = 0
    for t in tracklets:
        order = np.argsort(t.frames)
        f_first, f_last = int(t.frames[order[0]]), int(t.frames[order[-1]])
        if f_first > 2 and not at_edge(t.bboxes_xyxy[order[0]]):
            births += 1
        if f_last < last_frame - 2 and not at_edge(t.bboxes_xyxy[order[-1]]):
            deaths += 1
    return {"births": births, "deaths": deaths, "events": births + deaths}


def summarise(name: str, tracklets, width, height, last_frame, seconds: float) -> dict:
    lengths = [int(t.frames.shape[0]) for t in tracklets] or [0]
    row = {
        "arm": name,
        "tracks": len(tracklets),
        "boxes": int(sum(lengths)),
        "median_len": int(statistics.median(lengths)),
        "longest": max(lengths),
        "seconds": round(seconds, 1),
    }
    row.update(mid_pitch_events(tracklets, width, height, last_frame))
    return row


def build_arm(name: str):
    from pitch3d.adapters.models.botsort_backend import BotSortBackend
    from pitch3d.adapters.models.tracking import ByteTrackBackend

    if name == "bytetrack":
        return ByteTrackBackend(device="cpu")
    if name == "botsort":
        return BotSortBackend(device="cpu", gmc_method="sparseOptFlow")
    if name == "botsort-nogmc":
        return BotSortBackend(device="cpu", gmc_method="none")
    raise SystemExit(f"unknown arm {name!r}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dets", default="out/phmr_ab/dets_coco_0_236.npz")
    p.add_argument("--clip", default="samples/video/Colombia-1-0-Congo-DR1080p.mp4")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=float, default=29.97)
    p.add_argument("--arms", default="bytetrack,botsort,botsort-nogmc")
    args = p.parse_args()

    dets, n_frames = load_detections(REPO / args.dets)
    frame_ids = np.array([fd.frame for fd in dets.frames], dtype=int)
    n_boxes = sum(len(fd.items) for fd in dets.frames)
    clip = ClipRef(
        source_id="bench-association",
        uri=str(REPO / args.clip),
        frames=frame_ids,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    last_frame = int(frame_ids.max())
    print(
        f"detections: {n_boxes} boxes over {n_frames} frames "
        f"({frame_ids.min()}..{last_frame}) @ {args.width}x{args.height}\n"
        f"clip: {clip.uri}\n"
    )

    rows = []
    for name in args.arms.split(","):
        backend = build_arm(name.strip())
        t0 = time.time()
        tracklets = backend.associate(clip, dets)
        elapsed = time.time() - t0
        rows.append(
            summarise(name.strip(), tracklets, args.width, args.height, last_frame, elapsed)
        )
        print(f"  {rows[-1]}")

    cols = [
        "arm", "tracks", "boxes", "median_len", "longest",
        "births", "deaths", "events", "seconds",
    ]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    base = next((r for r in rows if r["arm"] == "bytetrack"), None)
    if base and len(rows) > 1:
        print(f"\nbaseline bytetrack: {base['events']} mid-pitch events")
        for r in rows:
            if r["arm"] != "bytetrack":
                d = r["events"] - base["events"]
                print(f"  {r['arm']:>14}: {r['events']:>3}  ({d:+d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
