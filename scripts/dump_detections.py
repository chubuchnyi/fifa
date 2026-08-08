"""Dump RF-DETR detections at a chosen input resolution, in the npz the CPU probes already read.

W1 measured what the detector's input square costs in *players found per frame* and concluded the
default should stay 560. That measurement had a hole the user pointed at: **players/frame is not
the thing we care about.** Identity churn, handovers, stitch quality and the assignment margin all
sit downstream of the detections, and none of them were re-measured at a higher resolution.

This closes that. It writes exactly the format `out/phmr_ab/dets_coco_0_236.npz` has — `frame`,
`boxes`, `classes`, `scores` — so every CPU probe in `scripts/` can be re-run against a different
resolution with no other change and no GPU:

    scripts/bench_handover_stitch.py   --dets <npz>     # W3: П3 pre-pose, kit clashes
    scripts/bench_assignment_margin.py --dets <npz>     # W5: margins + mid-pitch events
    scripts/bench_kit_split_timing.py  --dets <npz>     # W13: where the kit split cuts
    scripts/mask_cue_ab.py             --dets <npz>     # #133: the cue A/B

Run on the GPU box (`docs/local-gpu-box.md`); the consumers all run on CPU.

    python scripts/dump_detections.py --clip <mp4> --frames 236 \
        --resolutions 560 1064 --out-dir out/dets
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

#: RF-DETR's *base* checkpoint is COCO-pretrained (91-class, 1-indexed): person=1, sports ball=37.
#: Same map and same floor the pipeline's own `RFDETRDetector` applies, so the dumped npz is what
#: the adapter would have produced — see `adapters/models/detection.py`.
COCO_BASE_CLASSES = {1: "player", 37: "ball"}


def frames_of(clip: str, n: int, start: int = 0):
    import cv2

    cap = cv2.VideoCapture(clip)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    out = []
    for _ in range(n):
        ok, img = cap.read()
        if not ok:
            break
        out.append(img)
    cap.release()
    return out


def detect(images, resolution: int | None, threshold: float):
    from PIL import Image
    from rfdetr import RFDETRBase

    model = RFDETRBase() if resolution is None else RFDETRBase(resolution=resolution)
    frames, boxes, classes, scores = [], [], [], []
    t0 = time.time()
    for i, bgr in enumerate(images):
        pil = Image.fromarray(bgr[:, :, ::-1])
        det = model.predict(pil, threshold=threshold)
        keep = [k for k, cid in enumerate(np.asarray(det.class_id).tolist())
                if int(cid) in COCO_BASE_CLASSES]
        xyxy = np.asarray(det.xyxy, dtype=float).reshape(-1, 4)
        frames.append(i)
        boxes.append(xyxy[keep] if keep else np.zeros((0, 4)))
        classes.append([COCO_BASE_CLASSES[int(det.class_id[k])] for k in keep])
        scores.append([float(det.confidence[k]) for k in keep])
    return frames, boxes, classes, scores, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", type=int, default=236)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="the pipeline's own score floor, not the backend's")
    ap.add_argument("--resolutions", type=int, nargs="*", default=[560, 1064])
    ap.add_argument("--out-dir", default="out/dets")
    args = ap.parse_args()

    for r in args.resolutions:
        if r is not None and r % 56:
            raise SystemExit(f"resolution must be divisible by 56 (RF-DETR's patch stride): {r}")

    images = frames_of(args.clip, args.frames, args.start)
    print(f"decoded {len(images)} frames from {args.clip}", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for res in args.resolutions:
        frames, boxes, classes, scores, secs = detect(images, res, args.threshold)
        total = sum(len(s) for s in scores)
        players = sum(sum(1 for c in cl if c == "player") for cl in classes)
        dst = out_dir / f"dets_r{res}_{args.start}_{len(images)}.npz"
        np.savez(
            dst,
            frame=np.asarray(frames, dtype=np.int64),
            boxes=np.array(boxes, dtype=object),
            classes=np.array(classes, dtype=object),
            scores=np.array(scores, dtype=object),
            allow_pickle=True,
        )
        print(f"res {res:>5}: {total:5d} dets ({players} players) "
              f"= {players / max(len(images), 1):5.2f} players/frame · "
              f"{secs / max(len(images), 1):.3f} s/frame -> {dst}", flush=True)


if __name__ == "__main__":
    main()
