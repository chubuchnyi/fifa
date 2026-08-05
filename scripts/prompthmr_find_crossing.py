"""Find the frames where two tracked players actually overlap — the case #132 is about.

Runs the pipeline's own RF-DETR + ByteTrack on CPU so the boxes are exactly the ones the real
pipeline produces, then ranks frames by the largest pairwise box IoU. Writes the tracks and the
ranking to an npz for the mask-vs-box A/B that follows.

    .venv/bin/python scripts/prompthmr_find_crossing.py --frames 60 --out out/phmr_ab/tracks.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pitch3d.adapters.models.detection import DETECTOR_CLASS_MAPS, RFDETRDetector  # noqa: E402
from pitch3d.adapters.models.tracking import ByteTrackTracker  # noqa: E402
from pitch3d.core.ports.io import ClipRef  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--frames', type=int, default=60)
parser.add_argument('--out', default='out/phmr_ab/tracks.npz')
parser.add_argument('--classes', default='coco', choices=sorted(DETECTOR_CLASS_MAPS),
                    help='"coco" matches the free base weights; "sports" needs the gated ckpt')
args = parser.parse_args()

import cv2  # noqa: E402

cap = cv2.VideoCapture(args.clip)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
print(f'clip {width}x{height} @ {fps:.2f} fps, {total} frames')

clip = ClipRef(
    source_id='colombia',
    uri=args.clip,
    frames=np.arange(args.start, args.start + args.frames),
    width=width,
    height=height,
    fps=fps,
)

print('detecting (RF-DETR, cpu) ...', flush=True)
detections = RFDETRDetector(device='cpu', class_map=DETECTOR_CLASS_MAPS[args.classes]).detect(clip)
n_det = sum(len(f.items) for f in detections.frames)
print(f'  {n_det} detections over {len(detections.frames)} frames', flush=True)

print('tracking (ByteTrack) ...', flush=True)
tracks = ByteTrackTracker(device='cpu', min_track_frames=4).track(clip, detections)
players = [t for t in tracks.tracklets if t.cls == 'player']
print(f'  {len(tracks.tracklets)} tracklets, {len(players)} players')


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


per_frame = {}
for t in players:
    for f, box in zip(t.frames.tolist(), t.bboxes_xyxy, strict=True):
        per_frame.setdefault(int(f), []).append((int(t.track_id), np.asarray(box, float)))

ranking = []
for f, entries in sorted(per_frame.items()):
    best, pair = 0.0, (-1, -1)
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            v = iou(entries[i][1], entries[j][1])
            if v > best:
                best, pair = v, (entries[i][0], entries[j][0])
    ranking.append((f, best, pair[0], pair[1], len(entries)))

ranking.sort(key=lambda r: -r[1])
print('\nframes ranked by strongest player-player box overlap:')
print('  frame   IoU   tracks      n_players')
for f, v, a, b, n in ranking[:10]:
    print(f'  {f:5d}  {v:.3f}  {a:3d} vs {b:3d}   {n}')

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(
    args.out,
    frames=np.array([t.frames for t in players], dtype=object),
    boxes=np.array([t.bboxes_xyxy for t in players], dtype=object),
    track_ids=np.array([t.track_id for t in players]),
    ranking=np.array(ranking, dtype=float),
    width=width, height=height, fps=fps, start=args.start, n_frames=args.frames,
    allow_pickle=True,
)
print(f'\nwrote {args.out}')
