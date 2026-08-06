"""Propagate one mask per track across a clip — the evidence McByte's cue consumes (#133).

Two passes, on purpose. MOT benchmarks force McByte to be online; our pipeline is offline, and
being offline buys a much simpler design that exploits exactly what a propagator gives:

1. **Track once, normally.** The tracks come out broken — that is the defect.
2. **Seed a mask per track** at its first frame (SAM, prompted by the box) and let Cutie carry it
   forward *by appearance*. When the first pass loses a player mid-crossing, his mask keeps going
   anyway, because propagation never consulted the tracker.
3. **Track again with the cue on**, and the surviving mask is there to claim the detection the
   first pass dropped.

    .venv/bin/python scripts/build_track_masks.py --frames 236 --out out/phmr_ab/track_masks.npz

Writes one ``(H, W)`` int label image per frame, values = track ids, 0 = background. Cutie holds
every seeded object simultaneously, so ids never collide.

Cost warning, measured before you spend it: Cutie is a video model at full 1920x1080 and this is
the slow half of the whole idea. Time a short ``--frames`` before committing to the clip.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
parser.add_argument('--tracks', default='out/phmr_ab/tracks_split.npz',
                    help='pass-1 tracks; their boxes seed the masks')
parser.add_argument('--start', type=int, default=0,
                    help='first frame; tracks already alive here are seeded from their box AT it')
parser.add_argument('--frames', type=int, default=236)
parser.add_argument('--out', default='out/phmr_ab/track_masks.npz')
parser.add_argument('--sam', default='facebook/sam-vit-base')
parser.add_argument('--device', default='cpu')
parser.add_argument('--threads', type=int, default=6)
parser.add_argument('--max-objects', type=int, default=0,
                    help='cap simultaneous objects (0 = no cap); Cutie memory grows with them')
args = parser.parse_args()

import torch  # noqa: E402

torch.set_num_threads(args.threads)

z = np.load(REPO / args.tracks, allow_pickle=True)
H, W = int(z['height']), int(z['width'])


def plausible(b):
    return 25 < b[3] - b[1] < 0.45 * H and b[2] - b[0] < 0.30 * W


# track id -> its first frame and the box there; that is where its mask gets seeded.
END = args.start + args.frames
first: dict[int, tuple[int, np.ndarray]] = {}
for tid, frames, boxes in zip(z['track_ids'], z['frames'], z['boxes'], strict=True):
    for f, b in zip(np.asarray(frames).tolist(), np.asarray(boxes), strict=True):
        b = np.asarray(b, float)[:4]
        # A track already running at --start is seeded from its box THERE, not from a box in the
        # past we are not going to decode: the window has to stand on its own.
        f = max(int(f), args.start)
        if args.start <= f < END and plausible(b) and (
                int(tid) not in first or f < first[int(tid)][0]):
            first[int(tid)] = (f, b)
seeds: dict[int, list[tuple[int, np.ndarray]]] = {}
for tid, (f, b) in first.items():
    seeds.setdefault(f, []).append((tid, b))
print(f'{len(first)} tracks to seed, spread over {len(seeds)} frames')

from PIL import Image  # noqa: E402
from transformers import SamModel, SamProcessor  # noqa: E402

from pitch3d.adapters.models.mask_propagation import CutiePropagator  # noqa: E402

proc = SamProcessor.from_pretrained(args.sam)
sam = SamModel.from_pretrained(args.sam).eval()
prop = CutiePropagator(repo_dir=str(REPO / 'backends/McByte/mask_propagation/Cutie'),
                       device=args.device)


def sam_mask(rgb, box):
    inp = proc(Image.fromarray(rgb), input_boxes=[[box.tolist()]], return_tensors='pt')
    with torch.no_grad():
        o = sam(**inp, multimask_output=False)
    return proc.image_processor.post_process_masks(
        o.pred_masks.cpu(), inp['original_sizes'].cpu(),
        inp['reshaped_input_sizes'].cpu())[0][0, 0].numpy().astype(bool)


# Absolute paths win over the repo-relative default: on the pod the clip lives OUTSIDE the
# checkout (/workspace/…), and silently reading zero frames used to surface only at save time as
# "need at least one array to stack".
clip_path = args.clip if os.path.isabs(args.clip) else str(REPO / args.clip)
cap = cv2.VideoCapture(clip_path)
if not cap.isOpened():
    raise SystemExit(f'cannot open {clip_path!r} — pass --clip with the path on THIS machine')
if args.start:
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
labels: dict[int, np.ndarray] = {}
live: list[int] = []
t0 = time.time()
for n in range(args.start, END):
    ok, frame = cap.read()
    if not ok:
        print(f'  clip ended at frame {n} (asked for {END})', flush=True)
        break
    new = seeds.get(n, [])
    if args.max_objects:
        new = new[: max(0, args.max_objects - len(live))]
    if new:
        # A seed frame carries evidence: hand Cutie a label image with the new masks painted in
        # alongside whatever it already holds, and the objects list it should track from here.
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        lab = labels.get(n - 1, np.zeros((H, W), dtype=np.int64)).astype(np.int64).copy()
        for tid, box in new:
            m = sam_mask(rgb, box)
            lab[m] = tid                  # SAM's silhouette, not the box — Cutie refines it anyway
            live.append(tid)
        # Cutie numbers objects in the order they are FIRST added and asserts those internal
        # ids come in ascending order, so the list must stay in insertion order. Passing it
        # sorted by track id worked on the first window by luck and asserted on the second.
        out = prop.seed(frame, lab, list(dict.fromkeys(live)))
    else:
        out = prop.step(frame)
    labels[n] = out.astype(np.int32)
    if n % 20 == 0 or n == END - 1:
        el = time.time() - t0
        ids = int(len(np.unique(out)) - 1)
        print(f'  frame {n:4d}/{END}  {ids:3d} live masks  '
              f'{el:6.1f}s  {el / max(1, n - args.start + 1):.2f}s/frame', flush=True)
cap.release()

out_path = REPO / args.out
out_path.parent.mkdir(parents=True, exist_ok=True)
if not labels:
    raise SystemExit(f'no frames decoded from {clip_path!r} — nothing to write')
np.savez_compressed(
    out_path,
    frames=np.array(sorted(labels)),
    labels=np.stack([labels[k] for k in sorted(labels)]).astype(np.int32),
    width=W, height=H,
)
print(f'\nwrote {out_path}  ({len(labels)} frames, {time.time() - t0:.0f}s total)')
