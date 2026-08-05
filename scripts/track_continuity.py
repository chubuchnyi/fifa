"""Does the tracker survive a crossing? — the first half of #132, measured.

`prompthmr_find_crossing.py` says *where* players overlap. This says what the tracker does to
their identities there, reading the same `tracks.npz` so no model has to run again.

    .venv/bin/python scripts/track_continuity.py --window 115 135
    .venv/bin/python scripts/track_continuity.py --window 115 135 --pair 97 110
    .venv/bin/python scripts/track_continuity.py --window 115 135 --render out/ids.png
    .venv/bin/python scripts/track_continuity.py --window 0 235 --kit-scan

Four things are reported, and each is a distinct way #132 shows up:

* **gaps** — frames inside a track's own span where it has no box. The subject did not leave;
  the tracker lost him. Every gap is a stretch the pose path must interpolate or drop.
* **births / deaths** — a track starting or ending inside the window. A player who neither
  entered nor left the frame but gained a new id is an identity switch, and downstream that is
  a *different person*: new avatar, new kit assignment, motion history reset.
* **near-coincident boxes** — pairs whose boxes overlap so heavily that a per-crop estimator is
  handed effectively the same pixels twice, and cannot tell which of the two it should fit.
  That is the mechanism behind the "fuse per-crop poses" half of the ticket.
* **kit scan** (`--kit-scan`) — the two teams wear light blue and yellow, so the mean torso
  colour inside a box says *which player* an id is currently on. That turns two suspicions into
  measurements: a track whose kit **flips** is an id swap carrying an avatar onto a new human,
  and an overlapping same-kit pair of near-equal box size is one man **detected twice**, not an
  occlusion. Both look identical in IoU, which is how the frame-124 "crossing" was misread as
  two players in identical kit when it is a single blue player boxed twice.

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
parser.add_argument('--render', metavar='OUT.png',
                    help='contact sheet of the window, every box labelled with its id')
parser.add_argument('--render-frames', type=int, nargs=2, default=None, metavar=('FIRST', 'LAST'),
                    help='sub-range to draw; defaults to the whole window')
parser.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
parser.add_argument('--zoom', type=int, default=4)
parser.add_argument('--kit-scan', action='store_true',
                    help='read torso kit colour per box: finds id swaps and duplicate detections')
parser.add_argument('--min-run', type=int, default=3,
                    help='frames a kit colour must hold before it can call a swap')
parser.add_argument('--dup-iou', type=float, default=0.35,
                    help='same-kit pairs above this IoU at similar box height are one man twice')
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

if args.kit_scan:
    import cv2

    def kit_of(b, img):
        """Mean torso BGR -> team. Upper-middle of the box is shirt, clear of shorts and grass."""
        h, w = b[3] - b[1], b[2] - b[0]
        patch = img[int(b[1] + 0.20 * h):int(b[1] + 0.50 * h),
                    int(b[0] + 0.25 * w):int(b[0] + 0.75 * w)]
        if patch.size == 0:
            return '?', None
        bgr = patch.reshape(-1, 3).mean(0)
        if bgr[0] > 110 and bgr[0] > bgr[2]:
            return 'BLU', bgr
        if bgr[2] > 120 and bgr[1] > 110:
            return 'YEL', bgr
        return '?', bgr

    cap = cv2.VideoCapture(args.clip)
    kit = {}
    dups, real = {}, []
    for f in range(first, last + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            break
        here = [(t, box[(t, f)]) for t in sorted(live)
                if (t, f) in box and plausible(box[(t, f)])]
        for t, b in here:
            k, _ = kit_of(b, img)
            if k != '?':
                kit[(t, f)] = k
        for i in range(len(here)):
            for j in range(i + 1, len(here)):
                (ta, ba), (tb, bb) = here[i], here[j]
                v = iou(ba, bb)
                ha, hb = ba[3] - ba[1], bb[3] - bb[1]
                ka, kb = kit.get((ta, f)), kit.get((tb, f))
                if ka is None or kb is None:
                    continue
                if ka == kb:
                    if v >= args.dup_iou and abs(ha - hb) / max(ha, hb) < 0.25:
                        dups.setdefault(f, []).append((ta, tb, v, ka))
                    continue
                # Different kits, so genuinely two men. Lower box bottom = nearer the raised
                # camera, so that one is the occluder and the other is what gets hidden.
                back, front = ((ta, ba), (tb, bb)) if ba[3] < bb[3] else ((tb, bb), (ta, ba))
                x0, y0 = max(ba[0], bb[0]), max(ba[1], bb[1])
                x1, y1 = min(ba[2], bb[2]), min(ba[3], bb[3])
                inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                area = (back[1][2] - back[1][0]) * (back[1][3] - back[1][1])
                if inter > 0 and area > 0:
                    real.append((inter / area, f, back[0], front[0],
                                 kit[(back[0], f)], kit[(front[0], f)]))
    cap.release()

    print('\nid swaps: a track whose kit colour changes is carrying an avatar onto a NEW human')
    print('  tid   kit history                                    verdict')
    swaps = 0
    for tid in sorted(live):
        seq = [(f, kit[(tid, f)]) for f in range(first, last + 1) if (tid, f) in kit]
        if not seq:
            continue
        runs = []
        for f, k in seq:
            if not runs or runs[-1][0] != k:
                runs.append([k, f, f, 1])
            else:
                runs[-1][2], runs[-1][3] = f, runs[-1][3] + 1
        # A brief flip is the torso patch catching a neighbour, not an id changing hands. Drop
        # runs under --min-run frames, then merge what that leaves adjacent, so only a colour
        # the track actually *held* can call a swap.
        solid = [r for r in runs if r[3] >= args.min_run]
        merged = []
        for r in solid:
            if merged and merged[-1][0] == r[0]:
                merged[-1][2], merged[-1][3] = r[2], merged[-1][3] + r[3]
            else:
                merged.append(list(r))
        if len({r[0] for r in merged}) > 1:
            swaps += 1
            hist = ' '.join(f'{k}@{a}-{b}' for k, a, b, _n in merged)
            print(f'  {tid:4d}  {hist:44s}  SWAP')
    print(f'  {swaps} track(s) change team inside {first}-{last}')

    print(f'\nduplicate detections: same kit, IoU >= {args.dup_iou}, box heights within 25%')
    print('  frame   a    b    IoU  kit   -- one player boxed twice, NOT an occlusion')
    for f in sorted(dups):
        for ta, tb, v, k in dups[f]:
            print(f'  {f:5d}  {ta:3d}  {tb:3d}  {v:.3f}  {k}')
    if not dups:
        print('  none')

    print('\nREAL occlusions: opposing kits, ranked by how much of the back player is covered')
    print('  frame  cover   back <- front   kits')
    for cov, f, tb_, tf_, kb, kf in sorted(real, reverse=True)[:12]:
        print(f'  {f:5d}  {cov:.3f}  {tb_:3d} <- {tf_:3d}      {kb} behind {kf}')
    if not real:
        print('  none')

if args.render:
    # The numbers above describe boxes, and boxes are not players: at frame 124 the pair with the
    # highest IoU in this clip turned out to be one man detected twice, which no amount of IoU
    # could have said. Draw them.
    import cv2

    rf, rl = args.render_frames or (first, last)
    anchor = args.pair or [t for _, t in sorted(
        ((span[t][2], t) for t in live), reverse=True)][:2]
    seen = [box[(t, f)] for t in anchor for f in range(rf, rl + 1) if (t, f) in box]
    if not seen:
        raise SystemExit(f'no boxes for {anchor} in {rf}-{rl}')
    pad = 45
    x0 = max(0, int(min(b[0] for b in seen)) - pad)
    y0 = max(0, int(min(b[1] for b in seen)) - pad)
    x1 = min(W, int(max(b[2] for b in seen)) + pad)
    y1 = min(H, int(max(b[3] for b in seen)) + pad)

    cap = cv2.VideoCapture(args.clip)
    panels = []
    for f in range(rf, rl + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        tile = cv2.resize(frame[y0:y1, x0:x1].copy(), None, fx=args.zoom, fy=args.zoom,
                          interpolation=cv2.INTER_NEAREST)
        for t in sorted(live):
            b = box.get((t, f))
            if b is None or not plausible(b):
                continue
            colour = ((0, 255, 0), (0, 165, 255))[anchor.index(t)] if t in anchor \
                else (200, 200, 200)
            p0 = (int((b[0] - x0) * args.zoom), int((b[1] - y0) * args.zoom))
            p1 = (int((b[2] - x0) * args.zoom), int((b[3] - y0) * args.zoom))
            if p1[0] < 0 or p0[0] > tile.shape[1]:
                continue
            cv2.rectangle(tile, p0, p1, colour, 2)
            cv2.putText(tile, str(t), (p0[0] + 2, max(14, p0[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        cv2.putText(tile, f'f{f}', (6, tile.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(tile)
    cap.release()
    sheet = np.hstack(panels)
    Path(args.render).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.render, sheet)
    print(f'\nwrote {args.render}  ({len(panels)} frames, crop {x1 - x0}x{y1 - y0} @ x{args.zoom})')

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
