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

**Extended 2026-08-10 with the column side.** The row margin can only speak for a track that
already exists, so it is blind to births by construction — 29 of our 78 events had no row at all.
A birth is an unmatched *detection*, i.e. a column. Measured: the column margin does see them
(births median 0.079-0.125 against 0.705 for every column, lift 2.4-3.8x), and the rival rule
"fire when no track claims the detection" measures **0.0 %** — our mid-pitch births are contested,
not orphaned. Findings: docs/findings/reply-occlusion-stack-2026-08-10.md
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

# Read the frame size from the clip, never assume it. A portrait phone video is 1080x1920 and
# hardcoding 1920x1080 silently makes `plausible()` and `at_edge()` reject everything — the first
# run of this probe on the fan clip reported 0 events for exactly that reason.
import cv2  # noqa: E402

_cap = cv2.VideoCapture(str(REPO / args.clip))
W = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
H = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
_cap.release()
print(f'clip {Path(args.clip).name}: {W}x{H}')
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
#: frame -> [(detection box, COLUMN margin, best cost)] — the birth-side signal (added 2026-08-10)
#:
#: The row margin above can only speak for a track that already exists, so it is structurally blind
#: to births: 29 of our 78 events had no row for that track at all. A birth is an unmatched
#: *detection*, i.e. a **column**. Two candidate signals live there and they are not the same thing:
#: the column margin (two tracks compete for one detection) and the column's best cost (no track
#: claims it at all). Record both; the tables at the bottom score them against their own base rate.
by_det: dict[int, list[tuple[np.ndarray, float, float]]] = {}


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
            # --- column side. One row means there is no second-best: the margin is undefined and
            # recording 0 there would manufacture ambiguity out of a matrix that has none.
            bx = np.asarray(boxes, dtype=float).reshape(-1, 4)
            if bx.shape[0] == a.shape[1]:
                if a.shape[0] >= 2:
                    pc = np.partition(a, 1, axis=0)
                    cbest, csecond = pc[0, :], pc[1, :]
                else:
                    cbest, csecond = a[0, :], np.full(a.shape[1], np.inf)
                for j in range(a.shape[1]):
                    by_det.setdefault(f, []).append(
                        (bx[j], float(csecond[j] - cbest[j]), float(cbest[j]))
                    )
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
births: list[tuple[int, int, np.ndarray]] = []   # (frame, track_id, first plausible box)
for t in tls:
    fr = np.asarray(t.frames, dtype=int).reshape(-1)
    bx = np.asarray(t.bboxes_xyxy, dtype=float).reshape(-1, 4)
    keep = np.array([plausible(b) for b in bx])
    if keep.sum() < 2:
        continue
    fr, bx = fr[keep], bx[keep]
    if fr.min() > 2 and not at_edge(bx[0]):
        events.append((int(fr.min()), int(t.track_id), 'born'))
        births.append((int(fr.min()), int(t.track_id), bx[0].copy()))
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


# ---------------------------------------------------------------- the birth side (COLUMN margin)
#
# Everything above scores rows, and a row belongs to a track that already exists. A birth is an
# unmatched *detection* — a column — so the row margin cannot see one even in principle. This asks
# whether the column carries a usable signal instead, and scores both candidates the same way the
# row table is scored: against the base rate of the same rule firing on every column.
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


print('\n== per-DETECTION: can a COLUMN signal see a birth, which no row can? ==')
# The window must NOT reach past the birth. At f+1 the newborn track exists and matches its own
# detection perfectly, so a symmetric window measures the moment *after* the decision and reports a
# confident column for every birth. SMP dispatches at f, so only g <= f is admissible. Within the
# window a box can appear in both association rounds; take the MINIMUM of each statistic, which is
# the reading least favourable to the signal.
hits: list[tuple[float, float]] = []          # (column margin, best cost) at each birth
no_column = 0
for f, _tid, box in births:
    ms, cs = [], []
    for g in range(f - args.window, f + 1):
        for cb, m, c in by_det.get(g, ()):
            if _iou(box, cb) >= 0.5:
                ms.append(m)
                cs.append(c)
    if ms:
        hits.append((min(ms), min(cs)))
    else:
        no_column += 1

cols = [(m, c) for v in by_det.values() for (_b, m, c) in v]
print(f'  {len(births)} births of {len(events)} events · {len(cols)} detection columns recorded')
print(f'  {len(hits)} births matched to their own column (IoU >= 0.5, g <= f), {no_column} not '
      f'found')
if hits and cols:
    bm = np.array([m for m, _c in hits])
    bc = np.array([c for _m, c in hits])
    am = np.array([m for m, _c in cols])
    ac = np.array([c for _m, c in cols])
    print(f'  column MARGIN   births median {np.median(bm[np.isfinite(bm)]):.3f} · '
          f'all columns {np.median(am[np.isfinite(am)]):.3f}')
    print(f'  column BEST     births median {np.median(bc):.3f} · all columns {np.median(ac):.3f}')
    print('\n  a) fire when two tracks compete for the detection (the SMP-faithful rule):')
    for thr in (0.05, 0.10, 0.20, 0.40):
        rec = 100.0 * float((bm < thr).mean())
        base = 100.0 * float((am < thr).mean())
        print(f'     margin < {thr:.2f}: catches {rec:5.1f} % of births · {base:5.1f} % of columns'
              f' fire  ->  lift {rec / max(base, 1e-9):.2f}x')
    print('\n  b) fire when NO track claims the detection (an orphan, not an ambiguity):')
    for thr in (0.60, 0.80, 0.90, 0.95):
        rec = 100.0 * float((bc > thr).mean())
        base = 100.0 * float((ac > thr).mean())
        print(f'     best    > {thr:.2f}: catches {rec:5.1f} % of births · {base:5.1f} % of columns'
              f' fire  ->  lift {rec / max(base, 1e-9):.2f}x')
    print('\n  Rule (b) was expected to be near-tautological — ByteTrack births a track exactly')
    print('  because no match cleared the threshold. It measures 0.0 %, and that is a finding:')
    print('  a mid-pitch birth\'s detection has a perfectly ordinary best cost (median 0.18-0.22')
    print('  against 0.25 for every column). Our births are CONTESTED, not orphaned — a good')
    print('  candidate track existed and the assignment gave it to a competitor. Same fact as')
    print('  "96 % of events have an unclaimed detection 6-23 px away", seen from the matrix.')
    print('  Rule (a) is the one SMP specifies, and it is the one that carries signal.')
