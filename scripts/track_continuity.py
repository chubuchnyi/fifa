"""Does the tracker survive a crossing? — the first half of #132, measured.

`prompthmr_find_crossing.py` says *where* players overlap. This says what the tracker does to
their identities there, reading the same `tracks.npz` so no model has to run again.

    .venv/bin/python scripts/track_continuity.py --window 115 135
    .venv/bin/python scripts/track_continuity.py --window 115 135 --pair 97 110

Three things are reported, and each is a distinct way #132 shows up:

* **gaps** — frames inside a track's own span where it has no box. The subject did not leave;
  the tracker lost him. Every gap is a stretch the pose path must interpolate or drop.
* **births / deaths** — a track starting or ending inside the window. A player who neither
  entered nor left the frame but gained a new id is an identity switch, and downstream that is
  a *different person*: new avatar, new kit assignment, motion history reset.
* **near-coincident boxes** — pairs whose boxes overlap so heavily that a per-crop estimator is
  handed effectively the same pixels twice, and cannot tell which of the two it should fit.
  That is the mechanism behind the "fuse per-crop poses" half of the ticket.

Boxes are bounded to plausible players, because RF-DETR/COCO also emits crowd, touchline
officials and (in this clip's second shot) a 738x806 blob over half the frame.
"""
import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tracks', default='out/phmr_ab/tracks.npz')
parser.add_argument('--window', type=int, nargs=2, default=(115, 135), metavar=('FIRST', 'LAST'))
parser.add_argument('--pair', type=int, nargs=2, default=None, metavar=('A', 'B'),
                    help='follow these two ids frame by frame')
parser.add_argument('--coincident-iou', type=float, default=0.5,
                    help='box IoU above which a per-crop estimator sees one scene, not two')
args = parser.parse_args()

REPO = Path(__file__).resolve().parent.parent
z = np.load(REPO / args.tracks, allow_pickle=True)
W, H = int(z['width']), int(z['height'])
first, last = args.window

box = {}
span = {}
for tid, frames, boxes in zip(z['track_ids'], z['frames'], z['boxes'], strict=True):
    f = np.asarray(frames)
    span[int(tid)] = (int(f.min()), int(f.max()), len(f))
    for fr, b in zip(f.tolist(), np.asarray(boxes), strict=True):
        box[(int(tid), int(fr))] = np.asarray(b, float)[:4]


def plausible(b):
    """A broadcast player, not a crowd blob or a touchline official."""
    return 25 < b[3] - b[1] < 0.45 * H and b[2] - b[0] < 0.30 * W


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    both = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (both - inter))


live = [t for t, (a, b, _n) in span.items() if a <= last and b >= first]
print(f'window {first}-{last}: {len(live)} tracks alive')

print('\ngaps, births and deaths inside the window')
print('  tid   span        n   gaps in window            event')
trouble = 0
for tid in sorted(live):
    a, b, n = span[tid]
    have = {f for (t, f) in box if t == tid}
    gaps = [f for f in range(max(a, first), min(b, last) + 1) if f not in have]
    events = []
    if first <= a <= last:
        events.append(f'BORN @{a}')
    if first <= b <= last:
        events.append(f'DIED @{b}')
    if not gaps and not events:
        continue
    trouble += 1
    g = ','.join(map(str, gaps[:10])) + ('...' if len(gaps) > 10 else '')
    print(f'  {tid:4d}  {a:4d}-{b:<4d} {n:4d}   {g or "-":24s}  {" ".join(events) or "-"}')
print(f'  {trouble} of {len(live)} tracks are broken somewhere in the window')

print(f'\nbox pairs above IoU {args.coincident_iou} '
      '(one crop, two players -- a per-crop estimator cannot tell them apart)')
print('  frame   a    b    IoU   centres')
worst = []
for f in range(first, last + 1):
    here = [(t, box[(t, f)]) for t in live if (t, f) in box and plausible(box[(t, f)])]
    for i in range(len(here)):
        for j in range(i + 1, len(here)):
            v = iou(here[i][1], here[j][1])
            if v >= args.coincident_iou:
                ca = ((here[i][1][0] + here[i][1][2]) / 2, (here[i][1][1] + here[i][1][3]) / 2)
                cb = ((here[j][1][0] + here[j][1][2]) / 2, (here[j][1][1] + here[j][1][3]) / 2)
                worst.append((v, f, here[i][0], here[j][0], ca, cb))
for v, f, ta, tb, ca, cb in sorted(worst, reverse=True):
    print(f'  {f:5d}  {ta:3d}  {tb:3d}  {v:.3f}   '
          f'({ca[0]:.0f},{ca[1]:.0f}) vs ({cb[0]:.0f},{cb[1]:.0f})')
if not worst:
    print('  none')

if args.pair:
    ta, tb = args.pair
    print(f'\ntracks {ta} and {tb}, frame by frame')
    print(f'  frame       {ta:<4d}           {tb:<4d}      centre gap')
    for f in range(first, last + 1):
        ba, bb = box.get((ta, f)), box.get((tb, f))
        def show(b):
            if b is None:
                return '    MISSING   '
            return f'({(b[0] + b[2]) / 2:4.0f},{(b[1] + b[3]) / 2:4.0f})  '
        gap = ''
        if ba is not None and bb is not None:
            dx = (ba[0] + ba[2]) / 2 - (bb[0] + bb[2]) / 2
            dy = (ba[1] + ba[3]) / 2 - (bb[1] + bb[3]) / 2
            gap = f'{np.hypot(dx, dy):6.1f} px   IoU {iou(ba, bb):.3f}'
        print(f'  {f:5d}   {show(ba)} {show(bb)} {gap}')
