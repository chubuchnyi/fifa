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

from pitch3d.adapters.models.shot_detect import VALIDATED_BINS, clip_histograms  # noqa: E402
from pitch3d.core.orchestration.shots import (  # noqa: E402
    DEFAULT_CUT_RATIO,
    cut_threshold,
    find_shot_cuts,
    histogram_distances,
    shot_bounds,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
parser.add_argument('--frames', type=int, default=0, help='0 = the whole clip')
parser.add_argument('--ratio', type=float, default=DEFAULT_CUT_RATIO,
                    help='multiple of the clip median a cut must exceed')
parser.add_argument('--threshold', type=float, default=None,
                    help='absolute distance, replacing the adaptive rule (measure first)')
parser.add_argument('--bins', type=int, default=VALIDATED_BINS,
                    help='per channel; raising it makes histograms sparse and DEGRADES separation')
args = parser.parse_args()

import cv2  # noqa: E402

cap = cv2.VideoCapture(args.clip)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
n = args.frames or total
print(f'{args.clip}: {total} frames, scanning {n}')

hists = clip_histograms(args.clip, n_frames=n, bins=args.bins)
d = histogram_distances(hists)
cuts = find_shot_cuts(hists, ratio=args.ratio, threshold=args.threshold)
thr = args.threshold if args.threshold is not None else cut_threshold(d, args.ratio)
med = float(np.median(d))

print(f'\nconsecutive-frame histogram distance over {d.size} pairs (L1, range 0-2), '
      f'bins={args.bins}')
print(f'  median {med:.4f}   p95 {np.percentile(d, 95):.4f}   '
      f'p99 {np.percentile(d, 99):.4f}   max {d.max():.4f}')
print(f'  threshold {thr:.4f} = max({args.ratio} x median, floor)')

order = np.argsort(-d)[:6]
print('\n  largest distances (frame pair -> distance, and as a multiple of the median)')
for i in order.tolist():
    mark = '  <-- CUT' if (i + 1) in cuts else ''
    print(f'    {i:4d}->{i + 1:<4d}  {d[i]:.4f}  ({d[i] / med:5.2f}x median){mark}')

# Separation is the number that decides whether this detector is measuring anything. Print it
# whether or not a cut was found, so a bad binning is visible instead of silently degrading.
if cuts:
    within = d[[i for i in range(d.size) if (i + 1) not in cuts]]
    lo, hi = float(within.max()), float(min(d[c - 1] for c in cuts))
    print(f'\n  SEPARATION  within-shot max {lo:.4f} ({lo / med:.2f}x) .. '
          f'smallest cut {hi:.4f} ({hi / med:.2f}x)')
    if hi <= lo:
        print('  !! within-shot motion reaches the cuts — this binning cannot separate them')
    else:
        print(f'  the threshold sits in a {hi / lo:.1f}x gap')
else:
    print(f'\n  SEPARATION  no cut found; the largest pair is {d.max() / med:.2f}x the median '
          f'against a {args.ratio}x bar')

print(f'\n{len(cuts) + 1} shot(s):')
for k, (a, b) in enumerate(shot_bounds(n, cuts)):
    print(f'  shot {k}: frames {a}-{b}  ({b - a + 1} frames)')
if not cuts:
    print('  (single shot — safe to reconstruct end to end)')
