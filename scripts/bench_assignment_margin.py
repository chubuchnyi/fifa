"""Is the Hungarian assignment margin a usable dispatch signal? (W5, selective mask propagation)

Selective Mask Propagation (arXiv 2606.13033) is training-free: it keeps the base tracker and the
VOS as black boxes and fires the VOS **only where the tracker is unsure**, using the *assignment
margin in the Hungarian cost matrix* as the trigger. That is attractive here for one measured
reason: our McByte mask cue costs **686 s on GPU / 4.2 h on CPU** per pass and bought mid-pitch
identity events **28 → 24**. Paying that everywhere for a 14 % gain is the thing SMP claims to fix.

But the whole idea rests on a premise nobody has checked on our data: **that the margin is low
where the identity errors are.** If the tracker is equally unsure everywhere, or unsure in places
that never break, then firing selectively saves compute and loses the gain, and SMP is not for us.

So this measures the premise before anything is built, and it needs no GPU and no Cutie:

1. replay the pipeline's own tracker over cached detections, recording every cost matrix that
   `supervision`'s `matching.iou_distance` produces — the exact seam the mask cue already patches;
2. reduce each to a per-track **margin** = (second-best cost − best cost), the SMP trigger;
3. find the **mid-pitch identity events** with the same definition `scripts/mask_cue_ab.py` uses
   (a track born or dying away from the frame edge), each with its frame;
4. ask whether the margins at those frames are lower than everywhere else, and what fraction of
   the work a threshold would actually skip.

    PYTHONPATH=src .venv/bin/python scripts/bench_assignment_margin.py
    PYTHONPATH=src .venv/bin/python scripts/bench_assignment_margin.py --frames 236
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from pitch3d.adapters.models.tracking import ByteTrackTracker  # noqa: E402
from pitch3d.core.ports.io import ClipRef  # noqa: E402
from pitch3d.core.ports.perception import (  # noqa: E402
    Detection,
    Detections,
    FrameDetections,
)

ap = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--dets', default='out/phmr_ab/dets_coco_0_236.npz')
ap.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
ap.add_argument('--frames', type=int, default=236)
ap.add_argument('--max-cost', type=float, default=0.9,
                help='a row whose BEST cost is above this has no real candidate — not ambiguity')
ap.add_argument('--window', type=int, default=2,
                help='frames either side of an event that count as "at" it')
args = ap.parse_args()

W, H = 1920, 1080
c = np.load(REPO / args.dets, allow_pickle=True)
dets = Detections(frames=[
    FrameDetections(frame=int(f), items=[
        Detection(bbox_xyxy=b, cls=str(k), score=float(s))
        for b, k, s in zip(bb, kk, ss, strict=True)])
    for f, bb, kk, ss in zip(c['frame'], c['boxes'], c['classes'], c['scores'], strict=True)
    if int(f) < args.frames])
clip = ClipRef(source_id='colombia', uri=str(REPO / args.clip),
               frames=np.arange(args.frames), width=W, height=H, fps=29.97)

# ---------------------------------------------------------------- record every cost matrix
#
# No monkey-patching of our own: the backend already patches `matching.iou_distance` whenever a
# `mask_cue` is set, and stamps `cue.frame` before each frame's association. So a recorder that
# quacks like a MaskCue gets both the matrix and its frame number through the *real* seam — the
# same one selective propagation would dispatch from.
#: frame -> per-track margins from every cost matrix with >= 2 candidate detections
per_frame: dict[int, list[float]] = {}
#: frame -> rows considered = the units of VOS work a dispatch decision would be made over
rows_seen: dict[int, int] = {}
#: (frame, track_id) -> that track's OWN margin — the per-track test, not the per-frame proxy
by_track: dict[tuple[int, int], float] = {}


class MarginRecorder:
    """A no-op MaskCue that measures instead of discounting."""

    def __init__(self, max_cost: float) -> None:
        self.frame = -1
        self.max_cost = max_cost

    def apply(self, cost, track_ids, boxes):
        a = np.asarray(cost, dtype=float)
        f = int(self.frame)
        if a.ndim == 2 and a.shape[0] and a.shape[1]:
            rows_seen[f] = rows_seen.get(f, 0) + a.shape[0]
            if a.shape[1] >= 2:
                part = np.partition(a, 1, axis=1)
                best, second = part[:, 0], part[:, 1]
                # A row with no plausible candidate is not "ambiguous", it is empty. Only rows
                # that actually have something to confuse are evidence about confusion.
                live = best <= self.max_cost
                per_frame.setdefault(f, []).extend((second - best)[live].tolist())
                for i, tid in enumerate(track_ids):
                    if tid is not None and live[i]:
                        key = (f, int(tid))
                        m = float(second[i] - best[i])
                        by_track[key] = min(by_track.get(key, 1e9), m)
        return cost                      # unchanged: this must not alter the tracking it measures


from pitch3d.adapters.models.tracking import ByteTrackBackend  # noqa: E402

recorder = MarginRecorder(args.max_cost)
tracker = ByteTrackTracker(
    device='cpu', min_track_frames=2, kit_split=True,
    backend=ByteTrackBackend(device='cpu', mask_cue=recorder),
)
tracks = tracker.track(clip, dets)

tls = list(tracks.tracklets)
allm = np.array([m for v in per_frame.values() for m in v], dtype=float)
total_rows = sum(rows_seen.values())
print(f'{len(tls)} tracklets · {len(per_frame)} frames produced a cost matrix · '
      f'{allm.size} ambiguous-capable rows of {total_rows} total rows')
if allm.size == 0:
    raise SystemExit('no margins recorded — the frame stamping did not take effect')

# ---------------------------------------------------------------- mid-pitch events, same rule
EDGE = 0.06 * W


def plausible(b):
    return 25 < b[3] - b[1] < 0.45 * H and b[2] - b[0] < 0.30 * W


def at_edge(b):
    return (b[0] < EDGE or b[2] > W - EDGE
            or b[1] < EDGE * H / W or b[3] > H - EDGE * H / W)


events: list[tuple[int, int, str]] = []          # (frame, track_id, 'born'|'died')
for t in tls:
    fr = np.asarray(t.frames, dtype=int).reshape(-1)
    bx = np.asarray(t.bboxes_xyxy, dtype=float).reshape(-1, 4)
    keep = np.array([plausible(b) for b in bx])
    if keep.sum() < 2:
        continue
    fr, bx = fr[keep], bx[keep]
    if fr.min() > 2 and not at_edge(bx[0]):
        events.append((int(fr.min()), int(t.track_id), 'born'))
    if fr.max() < args.frames - 3 and not at_edge(bx[-1]):
        events.append((int(fr.max()), int(t.track_id), 'died'))

event_frames = {f for f, _t, _k in events}
near = {f + d for f in event_frames for d in range(-args.window, args.window + 1)}
print(f'{len(events)} mid-pitch identity events on {len(event_frames)} distinct frames')

at = np.array([m for f, v in per_frame.items() if f in near for m in v], dtype=float)
away = np.array([m for f, v in per_frame.items() if f not in near for m in v], dtype=float)


def line(name: str, a: np.ndarray) -> None:
    if a.size == 0:
        print(f'  {name:<28} (none)')
        return
    print(f'  {name:<28} n={a.size:6d}  p10 {np.percentile(a, 10):.3f}  '
          f'p25 {np.percentile(a, 25):.3f}  median {np.median(a):.3f}  mean {a.mean():.3f}')


print(f'\n== is the margin lower where identities break? (±{args.window} frames) ==')
line('at an event', at)
line('away from any event', away)
if at.size and away.size:
    print(f'  median at/away ratio {np.median(at) / max(np.median(away), 1e-9):.2f}  '
          f'(1.00 = the signal says nothing)')

# ---------------------------------------------------------------- what would a threshold cost?
print('\n== per-FRAME dispatch, against a random trigger of the SAME size ==')
print(f'  The null matters: {100.0 * at.size / max(allm.size, 1):.0f} % of rows already sit within '
      f'±{args.window} frames of an event, so a trigger firing at')
print('  random would hit event frames anyway. "random" draws the same number of rows, 200x.')
print(f'\n{"margin <":>10} {"rows":>7} {"of all":>8} {"event frames hit":>17} '
      f'{"random hits":>14} {"lift":>6}')
rng = np.random.default_rng(0)
flat = [(f, m) for f, v in per_frame.items() for m in v]
for thr in (0.02, 0.05, 0.10, 0.20, 0.40):
    fired = [f for f, m in flat if m < thr]
    hit = len(set(fired) & event_frames)
    n = len(fired)
    if n == 0:
        print(f'{thr:10.2f} {0:7d} {0.0:7.1f}% {0:9d}/{len(event_frames):<7d}')
        continue
    draws = []
    for _ in range(200):
        idx = rng.choice(len(flat), size=n, replace=False)
        draws.append(len({flat[i][0] for i in idx} & event_frames))
    exp = float(np.mean(draws))
    print(f'{thr:10.2f} {n:7d} {100.0 * n / max(total_rows, 1):7.1f}% '
          f'{hit:9d}/{len(event_frames):<7d} {exp:14.1f} {hit / max(exp, 1e-9):5.2f}x')

print('\nLift is the verdict. 1.0x means the margin is not selecting anything at this level — and')
print('that is what it says: per FRAME the signal is worthless, because on a crowded frame')
print('something is always ambiguous. The per-TRACK table below is the fair test.')


# ---------------------------------------------------------------- the per-track test
# The per-frame table above is a proxy: it asks whether SOMETHING was ambiguous on a frame where
# an identity broke. The sharper question is whether the *breaking track itself* was ambiguous.
print('\n== per-TRACK: was the breaking track\'s OWN row ambiguous at its own frame? ==')
print(f'  {len(by_track)} (frame, track) margins recorded')
own, missing = [], 0
for f, tid, _kind in events:
    hit = [by_track[(g, tid)] for g in range(f - args.window, f + args.window + 1)
           if (g, tid) in by_track]
    if hit:
        own.append(min(hit))
    else:
        missing += 1
own = np.asarray(own, dtype=float)
allv = np.asarray(list(by_track.values()), dtype=float)
if own.size:
    print(f'  breaking tracks   n={own.size:5d}  p10 {np.percentile(own, 10):.3f}  '
          f'p25 {np.percentile(own, 25):.3f}  median {np.median(own):.3f}')
    print(f'  every (frame,trk)  n={allv.size:5d}  p10 {np.percentile(allv, 10):.3f}  '
          f'p25 {np.percentile(allv, 25):.3f}  median {np.median(allv):.3f}')
    print(f'  {missing} of {len(events)} events had no recorded row for that track at all '
          f'(the track was not in the matrix — a birth has no prior row)')
    for thr in (0.05, 0.10, 0.20, 0.40):
        rec = 100.0 * float((own < thr).mean())
        base = 100.0 * float((allv < thr).mean())
        print(f'  margin < {thr:.2f}: catches {rec:5.1f} % of breaking tracks · '
              f'{base:5.1f} % of all rows fire  ->  lift {rec / max(base, 1e-9):.2f}x')
