"""Where does this clip cut between cameras? — and is the threshold that decides it defensible?

    .venv/bin/python scripts/shot_cuts.py --clip samples/video/Colombia-1-0-Congo-DR1080p.mp4

Reports every shot in the clip plus the *distribution* of consecutive-frame histogram distances,
because a cut detector is only as good as the gap between within-shot motion and a real cut. If
the top within-shot distance sits near the threshold, the threshold is a guess; if it sits far
below, the detector is measuring something real.

The pure half is :mod:`pitch3d.core.orchestration.shots`; this script only supplies the decode.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from pitch3d.core.orchestration.shots import (  # noqa: E402
    DEFAULT_CUT_THRESHOLD,
    find_shot_cuts,
    histogram_distances,
    shot_bounds,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
parser.add_argument('--frames', type=int, default=0, help='0 = the whole clip')
parser.add_argument('--threshold', type=float, default=DEFAULT_CUT_THRESHOLD)
parser.add_argument('--bins', type=int, default=8, help='per channel; 8 → 512 bins')
args = parser.parse_args()

import cv2  # noqa: E402

from pitch3d.adapters.models.shot_detect import clip_histograms  # noqa: E402

cap = cv2.VideoCapture(args.clip)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
n = args.frames or total
print(f'{args.clip}: {total} frames, scanning {n}')

hists = clip_histograms(args.clip, n_frames=n, bins=args.bins)
d = histogram_distances(hists)
cuts = find_shot_cuts(hists, threshold=args.threshold)

print(f'\nconsecutive-frame histogram distance over {d.size} pairs (L1, range 0-2)')
print(f'  median {np.median(d):.4f}   p95 {np.percentile(d, 95):.4f}   '
      f'p99 {np.percentile(d, 99):.4f}   max {d.max():.4f}')

order = np.argsort(-d)[:6]
print('\n  largest distances (frame pair -> distance)')
for i in order.tolist():
    mark = '  <-- CUT' if (i + 1) in cuts else ''
    print(f'    {i:4d}->{i + 1:<4d}  {d[i]:.4f}{mark}')

within = d[[i for i in range(d.size) if (i + 1) not in cuts]]
if cuts:
    print(f'\n  threshold {args.threshold}: '
          f'largest WITHIN-shot distance {within.max():.4f}, '
          f'smallest CUT distance {min(d[c - 1] for c in cuts):.4f}')
    print(f'  → the gap the threshold sits in is '
          f'{within.max():.4f} .. {min(d[c - 1] for c in cuts):.4f}')

print(f'\n{len(cuts) + 1} shot(s):')
for k, (a, b) in enumerate(shot_bounds(n, cuts)):
    print(f'  shot {k}: frames {a}-{b}  ({b - a + 1} frames)')
if not cuts:
    print('  (single shot — safe to reconstruct end to end)')
