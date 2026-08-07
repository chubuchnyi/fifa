"""What does RF-DETR's input resolution cost us in found players? (#138)

RF-DETR resizes the whole frame to a ``resolution x resolution`` **square** — aspect ratio is not
preserved — and its default is **560**, which `RFDETRBackend` never overrode. Measured against the
box sizes in `findings/occlusion-stack-review-2026-08-07.md` §2:

    fan clip  1080x1920 -> 560x560 = 0.52x across, 0.29x DOWN   player 28 x 72 px -> 14 x 21 px
    broadcast 1920x1080 -> 560x560                              player 41 x 86 px -> 12 x 45 px

Every association cue downstream is capped by what survives that resize, so the question is not
"is a bigger input nicer" but "how many players does 560 simply fail to find". This measures it on
real frames, at the real thresholds, with wall time — because the answer has to be worth its cost.

Deliberately *not* the score-threshold sweep, which came back null (`human-physics-requirement`
§"Detector recall"): that test kept boxes the net had already found at 560. This changes what it
can find at all.

    python scripts/bench_detector_resolution.py --clip <mp4> --frames 60 \
        --resolutions 560 728 896 1064

Needs a GPU and the `cv` extra, so it runs on `demorig` (see `docs/local-gpu-box.md`), not here.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

#: COCO class id for `person` in RF-DETR's pretrained head.
PERSON = 1
#: The adapter's authoritative floor (`RFDETRDetector.score_threshold`) and the low floor whose
#: recall we measured as a 96 % association ceiling.
SCORES = (0.3, 0.1)


def frames_of(uri: str, n: int, start: int = 0):
    import cv2

    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {uri}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, img = cap.read()
        if not ok:
            break
        out.append(img)
    cap.release()
    return out, w, h


def run(images, resolution: int) -> dict:
    """Detect over ``images`` at one input resolution; return the counts that decide the knob."""
    from rfdetr import RFDETRBase

    model = RFDETRBase() if resolution is None else RFDETRBase(resolution=resolution)
    # One warm-up so the timing is inference, not cudnn autotune + weight upload.
    model.predict(images[0], threshold=0.5)

    t0 = time.perf_counter()
    boxes, scores = [], []
    for img in images:
        det = model.predict(img, threshold=min(SCORES))
        cid = np.asarray(det.class_id, dtype=int).reshape(-1)
        keep = cid == PERSON
        boxes.append(np.asarray(det.xyxy, dtype=float).reshape(-1, 4)[keep])
        scores.append(np.asarray(det.confidence, dtype=float).reshape(-1)[keep])
    elapsed = time.perf_counter() - t0

    all_b = np.concatenate(boxes) if boxes else np.zeros((0, 4))
    all_s = np.concatenate(scores) if scores else np.zeros(0)
    row = {"resolution": resolution, "s_per_frame": elapsed / max(1, len(images))}
    for thr in SCORES:
        m = all_s >= thr
        row[f"n@{thr}"] = int(m.sum())
        row[f"per_frame@{thr}"] = float(m.sum()) / max(1, len(images))
    if all_b.size:
        h = all_b[:, 3] - all_b[:, 1]
        row["median_h"] = float(np.median(h[all_s >= 0.3])) if (all_s >= 0.3).any() else 0.0
        row["p05_h"] = float(np.percentile(h[all_s >= 0.3], 5)) if (all_s >= 0.3).any() else 0.0
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--resolutions", type=int, nargs="*", default=[560, 728, 896, 1064])
    ap.add_argument("--json-out")
    args = ap.parse_args()

    images, w, h = frames_of(args.clip, args.frames, args.start)
    print(f"{args.clip}: {len(images)} frames at {w}x{h}\n")
    print(f"{'res':>6} {'sq scale x/y':>14} {'players/frame @0.3':>19} {'@0.1':>8} "
          f"{'median box h':>13} {'p05 h':>7} {'s/frame':>8}")

    rows = []
    for r in args.resolutions:
        row = run(images, r)
        rows.append(row)
        sx, sy = r / w, r / h
        print(f"{r:>6} {sx:>6.2f}x/{sy:.2f}y {row['per_frame@0.3']:>19.1f} "
              f"{row['per_frame@0.1']:>8.1f} {row.get('median_h', 0):>13.1f} "
              f"{row.get('p05_h', 0):>7.1f} {row['s_per_frame']:>8.3f}", flush=True)

    base = next((r for r in rows if r["resolution"] == 560), rows[0])
    print(f"\nagainst {base['resolution']}:")
    for r in rows:
        d = r["per_frame@0.3"] - base["per_frame@0.3"]
        cost = r["s_per_frame"] / max(1e-9, base["s_per_frame"])
        print(f"  {r['resolution']:>5}: {d:+.1f} players/frame at 0.3, {cost:.2f}x the time")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"clip": args.clip, "width": w, "height": h,
                       "frames": len(images), "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
