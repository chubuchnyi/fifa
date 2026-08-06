"""Is ByteTrack's IoU match threshold why players keep losing their id? — swept, both ways.

`scripts/identity_failure_kind.py` found that **55 of 78** mid-pitch identity births/deaths in
shot 1 had an unclaimed detection sitting a median 6-23 px from where the dying track was heading,
and that 72 % of those orphans scored under 0.4. We pass supervision's `sv.ByteTrack` only
`frame_rate`, so `minimum_matching_threshold` sits at its default **0.8**.

**Read that parameter carefully — I did not, at first.** It is a *distance* threshold on
``1 - IoU``, not an IoU floor: 0.8 already means "match anything sharing more than 0.2 IoU", which
is permissive. Lowering it TIGHTENS matching, and the first sweep duly showed churn rising from 66
events at 0.80 to 162 at 0.50. The interesting direction is upward.

    .venv/bin/python scripts/match_threshold_sweep.py

Loosening a match threshold trades one error for another, so this reports both and neither alone
is the verdict:

* **mid-pitch identity events** — births and deaths away from the frame border, i.e. players
  appearing from nowhere or vanishing. Down is the point.
* **kit changes** — a track whose team colour flips is one id covering two humans. Up means the
  looser threshold is now matching the *wrong* player, which is worse than the churn it removed.
  The #132 kit split is OFF by default here **on purpose**: it cuts those tracks apart, so with it
  on this column reads 0 whatever the threshold does, and the sweep is blind to its own downside.

Detections come from the cache, so every arm sees identical boxes and only the association differs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from pitch3d.adapters.models.tracking import ByteTrackBackend, ByteTrackTracker  # noqa: E402
from pitch3d.core.ports.io import ClipRef  # noqa: E402
from pitch3d.core.ports.perception import (  # noqa: E402
    Detection,
    Detections,
    FrameDetections,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
parser.add_argument('--dets', default='out/phmr_ab/dets_coco_0_236.npz')
parser.add_argument('--frames', type=int, default=236)
parser.add_argument('--thresholds', type=float, nargs='*',
                    default=[0.7, 0.8, 0.85, 0.9, 0.95])
parser.add_argument('--kit-split', action='store_true',
                    help='leave the #132 kit split ON; default OFF so swaps stay VISIBLE')
parser.add_argument('--min-track-frames', type=int, default=4)
args = parser.parse_args()

c = np.load(REPO / args.dets, allow_pickle=True)
dets = Detections(frames=[
    FrameDetections(frame=int(f), items=[
        Detection(bbox_xyxy=b, cls=str(k), score=float(s))
        for b, k, s in zip(bb, kk, ss, strict=True)])
    for f, bb, kk, ss in zip(c['frame'], c['boxes'], c['classes'], c['scores'], strict=True)])
clip = ClipRef(source_id='colombia', uri=str(REPO / args.clip),
               frames=np.arange(args.frames), width=1920, height=1080, fps=29.97)
W, H = clip.width, clip.height
EDGE = 0.06 * W


def plausible(b):
    return 25 < b[3] - b[1] < 0.45 * H and b[2] - b[0] < 0.30 * W


def at_edge(b):
    return b[0] < EDGE or b[2] > W - EDGE or b[1] < EDGE * H / W or b[3] > H - EDGE * H / W


import cv2  # noqa: E402


def kit_of(b, img):
    h, w = b[3] - b[1], b[2] - b[0]
    p = img[int(b[1] + 0.20 * h):int(b[1] + 0.50 * h), int(b[0] + 0.25 * w):int(b[0] + 0.75 * w)]
    if p.size == 0:
        return '?'
    bgr = p.reshape(-1, 3).mean(0)
    if bgr[0] > 110 and bgr[0] > bgr[2]:
        return 'BLU'
    if bgr[2] > 120 and bgr[1] > 110:
        return 'YEL'
    return '?'


print('reading frames for the kit check ...', flush=True)
cap = cv2.VideoCapture(str(REPO / args.clip))
frames_img = {}
for n in range(args.frames):
    ok, img = cap.read()
    if not ok:
        break
    frames_img[n] = img
cap.release()

print(f'\n{"match":>6}  {"tracks":>6}  {"mid-pitch births/deaths":>23}  {"kit changes":>11}')
for thr in args.thresholds:
    tracker = ByteTrackTracker(
        device='cpu', min_track_frames=args.min_track_frames, kit_split=args.kit_split,
        backend=ByteTrackBackend(device='cpu', match_threshold=thr),
    )
    tracks = tracker.track(clip, dets)
    players = [t for t in tracks.tracklets if t.cls == 'player']

    events, swaps = 0, 0
    for t in players:
        fr = np.asarray(t.frames, dtype=int)
        bx = np.asarray(t.bboxes_xyxy, dtype=float)
        m = np.array([plausible(b) for b in bx])
        if m.sum() < 2:
            continue
        fr, bx = fr[m], bx[m]
        if fr.min() > 2 and not at_edge(bx[0]):
            events += 1
        if fr.max() < args.frames - 3 and not at_edge(bx[-1]):
            events += 1
        # Kit run-length over this track: two colours held >=3 boxes each = it changed player.
        seq = [k for k in (kit_of(b, frames_img[f]) for f, b in zip(fr.tolist(), bx, strict=True))
               if k != '?']
        runs = []
        for k in seq:
            if runs and runs[-1][0] == k:
                runs[-1][1] += 1
            else:
                runs.append([k, 1])
        held = {k for k, n in runs if n >= 3}
        if len(held) > 1:
            swaps += 1
    print(f'{thr:6.2f}  {len(players):6d}  {events:23d}  {swaps:11d}', flush=True)

print('\nDown on the middle column is the point; UP on the right one means the looser threshold')
print('started matching the wrong player, which costs more than the churn it removed.')
