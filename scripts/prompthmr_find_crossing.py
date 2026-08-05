"""Find the frames where one player is actually *hidden* behind another — the case #132 is about.

Runs the pipeline's own RF-DETR + ByteTrack on CPU so the boxes are exactly the ones the real
pipeline produces, then ranks frames two ways and writes both to an npz.

    .venv/bin/python scripts/prompthmr_find_crossing.py --frames 334        # the whole clip
    .venv/bin/python scripts/prompthmr_find_crossing.py --frames 334 --verify-masks 8

**`ranking` — largest pairwise box IoU.** The original, kept so the numbers already quoted in
`docs/findings/occlusion-pose-research-2026-08-04.md` stay reproducible. It is a poor proxy:
its 2026-08-05 winner (frame 29, IoU 0.511) turned out to be two *adjacent* players, both fully
visible, which is not the failure #132 describes.

**`occlusion` — how much of the BACK player's box the front one covers.** Players stand on a
plane and the camera is raised, so the one whose box bottom sits lower in the frame is nearer;
`cover = inter / area(back)` then reads as "fraction of the far player the near one sits over".
That is the quantity that breaks a per-crop pose, and it is what to rank on.

Box cover still over-reports: two boxes can overlap while the pixels do not. `--verify-masks K`
settles it on the top K candidates by running SAM and measuring the back player's **visible
fill** = mask_area / box_area. An unoccluded standing player fills roughly 0.3-0.45 of his box,
so a markedly lower fill is real hiding rather than box arithmetic.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pitch3d.adapters.models.detection import DETECTOR_CLASS_MAPS, RFDETRDetector  # noqa: E402
from pitch3d.adapters.models.tracking import ByteTrackTracker  # noqa: E402
from pitch3d.core.ports.io import ClipRef  # noqa: E402
from pitch3d.core.ports.perception import Detection, Detections, FrameDetections  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--frames', type=int, default=60)
parser.add_argument('--out', default='out/phmr_ab/tracks.npz')
parser.add_argument('--classes', default='coco', choices=sorted(DETECTOR_CLASS_MAPS),
                    help='"coco" matches the free base weights; "sports" needs the gated ckpt')
parser.add_argument('--verify-masks', type=int, default=0, metavar='K',
                    help='run SAM on the top K occlusion candidates and report visible fill')
parser.add_argument('--sam', default='facebook/sam-vit-base')
parser.add_argument('--no-kit-split', action='store_true',
                    help='disable the #132 team-change split (pre-fix behaviour)')
parser.add_argument('--also-nosplit', metavar='OUT.npz',
                    help='additionally track the SAME detections with the split off, as a control')
parser.add_argument('--det-cache', metavar='NPZ',
                    help='reuse detections from here (written on first run); tracker A/Bs are free')
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

cache = Path(args.det_cache or
             f'out/phmr_ab/dets_{args.classes}_{args.start}_{args.frames}.npz')
if cache.exists():
    # Detection is minutes of CPU and does not depend on anything downstream, so an A/B over
    # tracker settings should never pay for it twice.
    c = np.load(cache, allow_pickle=True)
    detections = Detections(frames=[
        FrameDetections(frame=int(f), items=[
            Detection(bbox_xyxy=b, cls=str(k), score=float(s))
            for b, k, s in zip(bb, kk, ss, strict=True)])
        for f, bb, kk, ss in zip(c['frame'], c['boxes'], c['classes'], c['scores'], strict=True)])
    print(f'detections from cache {cache}', flush=True)
else:
    print('detecting (RF-DETR, cpu) ...', flush=True)
    detections = RFDETRDetector(device='cpu',
                                class_map=DETECTOR_CLASS_MAPS[args.classes]).detect(clip)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache,
             frame=np.array([fd.frame for fd in detections.frames]),
             boxes=np.array([np.stack([d.bbox_xyxy for d in fd.items]) if fd.items
                             else np.zeros((0, 4)) for fd in detections.frames], dtype=object),
             classes=np.array([[d.cls for d in fd.items] for fd in detections.frames],
                              dtype=object),
             scores=np.array([[d.score for d in fd.items] for fd in detections.frames],
                             dtype=object),
             allow_pickle=True)
    print(f'  cached -> {cache}', flush=True)
n_det = sum(len(f.items) for f in detections.frames)
print(f'  {n_det} detections over {len(detections.frames)} frames', flush=True)

print('tracking (ByteTrack) ...', flush=True)
tracks = ByteTrackTracker(device='cpu', min_track_frames=4,
                          kit_split=not args.no_kit_split).track(clip, detections)
players = [t for t in tracks.tracklets if t.cls == 'player']
print(f'  {len(tracks.tracklets)} tracklets, {len(players)} players'
      f'  (kit_split={"off" if args.no_kit_split else "on"})')

if args.also_nosplit:
    # Same clip, same detections, same association -- only the #132 kit split differs, so any
    # difference between the two npz files is that fix and nothing else.
    ctrl = ByteTrackTracker(device='cpu', min_track_frames=4, kit_split=False).track(
        clip, detections)
    cp = [t for t in ctrl.tracklets if t.cls == 'player']
    Path(args.also_nosplit).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.also_nosplit,
             frames=np.array([t.frames for t in cp], dtype=object),
             boxes=np.array([t.bboxes_xyxy for t in cp], dtype=object),
             track_ids=np.array([t.track_id for t in cp]),
             ranking=np.zeros((0, 5)), occlusion=np.zeros((0, 5)), visible_fill=np.zeros((0, 5)),
             width=width, height=height, fps=fps, start=args.start, n_frames=args.frames,
             allow_pickle=True)
    print(f'  control arm (kit_split off) -> {args.also_nosplit}: {len(cp)} players', flush=True)


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

def intersection(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


ranking, occlusion = [], []
for f, entries in sorted(per_frame.items()):
    best, pair = 0.0, (-1, -1)
    best_cov, cov_pair = 0.0, (-1, -1)
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            (tid_i, box_i), (tid_j, box_j) = entries[i], entries[j]
            v = iou(box_i, box_j)
            if v > best:
                best, pair = v, (tid_i, tid_j)
            inter = intersection(box_i, box_j)
            if inter <= 0:
                continue
            # Feet lower in the frame = nearer the raised camera, so that player is the occluder.
            back, front = ((tid_i, box_i), (tid_j, box_j)) if box_i[3] < box_j[3] \
                else ((tid_j, box_j), (tid_i, box_i))
            area_back = (back[1][2] - back[1][0]) * (back[1][3] - back[1][1])
            cov = float(inter / area_back) if area_back > 0 else 0.0
            if cov > best_cov:
                best_cov, cov_pair = cov, (back[0], front[0])
    ranking.append((f, best, pair[0], pair[1], len(entries)))
    occlusion.append((f, best_cov, cov_pair[0], cov_pair[1], len(entries)))

ranking.sort(key=lambda r: -r[1])
occlusion.sort(key=lambda r: -r[1])

print('\nframes ranked by strongest player-player box overlap:')
print('  frame   IoU   tracks      n_players')
for f, v, a, b, n in ranking[:10]:
    print(f'  {f:5d}  {v:.3f}  {a:3d} vs {b:3d}   {n}')

print('\nframes ranked by how much of the BACK player the front one covers:')
print('  frame  cover   back <- front   n_players')
for f, v, a, b, n in occlusion[:10]:
    print(f'  {f:5d}  {v:.3f}  {a:3d} <- {b:3d}      {n}')

fill = []
if args.verify_masks:
    # Box cover over-reports: two boxes can overlap while the pixels do not. Cut the back
    # player out with SAM and see how much of his box he still fills.
    print(f'\nverifying the top {args.verify_masks} with {args.sam} ...', flush=True)
    import torch
    from PIL import Image
    from transformers import SamModel, SamProcessor

    sam_proc = SamProcessor.from_pretrained(args.sam)
    sam = SamModel.from_pretrained(args.sam).eval()
    cap = cv2.VideoCapture(args.clip)
    print('  frame  cover   back <- front   visible fill   verdict')
    for f, cov, back_id, front_id, _n in occlusion[:args.verify_masks]:
        box = next(b for t, b in per_frame[int(f)] if t == int(back_id))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = sam_proc(Image.fromarray(rgb), input_boxes=[[box[:4].tolist()]],
                          return_tensors='pt')
        with torch.no_grad():
            out = sam(**inputs, multimask_output=False)
        mask = sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs['original_sizes'].cpu(),
            inputs['reshaped_input_sizes'].cpu())[0][0, 0].numpy().astype(bool)
        area = (box[2] - box[0]) * (box[3] - box[1])
        vf = float(mask.sum() / area) if area > 0 else 0.0
        fill.append((f, cov, back_id, front_id, vf))
        verdict = 'HIDDEN' if vf < 0.28 else ('partial' if vf < 0.34 else 'clear')
        print(f'  {int(f):5d}  {cov:.3f}  {int(back_id):3d} <- {int(front_id):3d}'
              f'         {vf:.3f}   {verdict}')
    cap.release()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(
    args.out,
    frames=np.array([t.frames for t in players], dtype=object),
    boxes=np.array([t.bboxes_xyxy for t in players], dtype=object),
    track_ids=np.array([t.track_id for t in players]),
    ranking=np.array(ranking, dtype=float),
    occlusion=np.array(occlusion, dtype=float),
    visible_fill=np.array(fill, dtype=float).reshape(-1, 5),
    width=width, height=height, fps=fps, start=args.start, n_frames=args.frames,
    allow_pickle=True,
)
print(f'\nwrote {args.out}')
