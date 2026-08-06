"""Why does a track die mid-pitch — did the detector miss him, or did association drop him?

Shot 1 carries 60 identity births/deaths that no entry or exit explains (~1 per track per 8 s,
`human-physics-requirement-2026-08-06.md`). Before adopting anything to fix that, the failure has
to be split, because the two causes have opposite cures:

* **detection miss** — no box exists at all in the following frames. A better association cue
  (masks, re-ID, McByte) has nothing to associate and cannot help; this wants detector recall.
* **association miss** — a box IS there, unclaimed by any track, right where the dying track was
  heading. That is the association step losing a player it was still being shown, and it is
  exactly what a mask/appearance cue is for.

    .venv/bin/python scripts/identity_failure_kind.py

Reads the cached detections and the tracker's own output, so it costs nothing and re-runs no model.
An "orphan" is a detection that no live track claims at that frame (IoU below --claim-iou against
every track box). A death is *recoverable* if an orphan sits within --search-radius box-widths of
where constant-velocity extrapolation puts the dying track, in the next --horizon frames.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tracks', default='out/phmr_ab/tracks_split.npz')
parser.add_argument('--dets', default='out/phmr_ab/dets_coco_0_236.npz')
parser.add_argument('--last-frame', type=int, default=235)
parser.add_argument('--claim-iou', type=float, default=0.5,
                    help='a detection above this IoU with a track box is claimed, not orphaned')
parser.add_argument('--search-radius', type=float, default=2.5,
                    help='orphan must sit within this many box-widths of the extrapolated position')
parser.add_argument('--horizon', type=int, default=4, help='frames to look ahead / back')
args = parser.parse_args()

z = np.load(REPO / args.tracks, allow_pickle=True)
W, H = int(z['width']), int(z['height'])


def plausible(b):
    return 25 < b[3] - b[1] < 0.45 * H and b[2] - b[0] < 0.30 * W


track_boxes: dict[int, dict[int, np.ndarray]] = {}
for tid, frames, boxes in zip(z['track_ids'], z['frames'], z['boxes'], strict=True):
    seq = {}
    for f, b in zip(np.asarray(frames).tolist(), np.asarray(boxes), strict=True):
        b = np.asarray(b, float)[:4]
        if f <= args.last_frame and plausible(b):
            seq[int(f)] = b
    if len(seq) >= 2:
        track_boxes[int(tid)] = seq

c = np.load(REPO / args.dets, allow_pickle=True)
dets: dict[int, list[np.ndarray]] = {}
for f, bb, kk in zip(c['frame'], c['boxes'], c['classes'], strict=True):
    keep = [np.asarray(b, float) for b, k in zip(bb, kk, strict=True)
            if str(k) != 'ball' and plausible(np.asarray(b, float))]
    dets[int(f)] = keep


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if i <= 0:
        return 0.0
    return float(i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i))


def orphans(f):
    """Detections at frame f that no track claims — the boxes association threw away."""
    live = [s[f] for s in track_boxes.values() if f in s]
    return [d for d in dets.get(f, []) if all(iou(d, t) < args.claim_iou for t in live)]


def centre(b):
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


EDGE = 0.06 * W


def at_edge(b):
    return b[0] < EDGE or b[2] > W - EDGE or b[1] < EDGE * H / W or b[3] > H - EDGE * H / W


def velocity(seq, f, back=3):
    """Constant-velocity estimate from the track's own tail, in px/frame."""
    fs = sorted(k for k in seq if k <= f)[-back - 1:]
    if len(fs) < 2:
        return np.zeros(2)
    return (centre(seq[fs[-1]]) - centre(seq[fs[0]])) / max(1, fs[-1] - fs[0])


print(f'tracks: {len(track_boxes)}   detections cached for {len(dets)} frames')
print(f'orphan = det with IoU < {args.claim_iou} against every live track box\n')

rows = {'death': [], 'birth': []}
for tid, seq in sorted(track_boxes.items()):
    fs = sorted(seq)
    for kind, f0, direction in (('death', fs[-1], +1), ('birth', fs[0], -1)):
        if kind == 'death' and f0 >= args.last_frame - 2:
            continue
        if kind == 'birth' and f0 <= 2:
            continue
        b0 = seq[f0]
        if at_edge(b0):
            continue                     # left or entered the frame: legitimate
        v = velocity(seq, f0) * direction
        w = max(8.0, b0[2] - b0[0])
        found, dist = None, None
        for k in range(1, args.horizon + 1):
            f = f0 + direction * k
            if f not in dets:
                continue
            pred = centre(b0) + v * k
            for d in orphans(f):
                dd = float(np.linalg.norm(centre(d) - pred))
                if dd <= args.search_radius * w and (dist is None or dd < dist):
                    found, dist = f, dd
            if found is not None:
                break
        rows[kind].append((tid, f0, found, dist))

for kind in ('death', 'birth'):
    rs = rows[kind]
    rec = [r for r in rs if r[2] is not None]
    print(f'{kind}s mid-pitch: {len(rs)}')
    print(f'  ASSOCIATION miss — an unclaimed detection was right there: {len(rec)} '
          f'({100 * len(rec) / max(1, len(rs)):.0f}%)')
    print(f'  DETECTION miss  — no box at all within {args.horizon} frames: '
          f'{len(rs) - len(rec)} ({100 * (len(rs) - len(rec)) / max(1, len(rs)):.0f}%)')
    if rec:
        d = np.array([r[3] for r in rec])
        print(f'  orphan distance from the extrapolated position: median {np.median(d):.0f} px, '
              f'max {d.max():.0f} px')
    print()

tot = len(rows['death']) + len(rows['birth'])
recov = sum(1 for k in rows for r in rows[k] if r[2] is not None)
print(f'=> {recov} of {tot} mid-pitch identity events had a detection available and unused.')
print('   That share is the ceiling on what a better ASSOCIATION cue (masks, re-ID) can fix;')
print('   the rest needs detector recall, and no tracker change will reach it.')
