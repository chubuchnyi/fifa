"""How many identities does one real player end up with? — the #132 scoreboard.

Splitting a track that changed player (`kit_split`) and stitching fragments back together
(`core/orchestration/continuity.py`) pull in opposite directions on the *count* while pulling the
same way on *correctness*. This measures both ends at once, over the same cached detections, so
the four numbers are comparable:

    .venv/bin/python scripts/identity_budget.py --frames 236

    kit_split  stitch   identities
    off        off      38          <- one id may cover two humans
    off        on       ...
    on         off      56          <- every id covers one human, but a human may hold several
    on         on       ...

The target is not "as few ids as possible" — that is what the broken tracker already scores well
on by fusing two players into one id. It is **one id per human for the whole shot**, and the shot
has roughly 22 players visible. So read the two columns together: `kit_split` fixes who an id is,
stitching fixes how many ids a human needs, and only the pair gets near the target.

Detections come from the cache `prompthmr_find_crossing.py` writes, so this never re-runs RF-DETR.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from pitch3d.adapters.models.tracking import ByteTrackTracker  # noqa: E402
from pitch3d.core.orchestration.continuity import (  # noqa: E402
    StitchConfig,
    stitch_tracks_with_report,
)
from pitch3d.core.ports.io import ClipRef  # noqa: E402
from pitch3d.core.ports.perception import Detection, Detections, FrameDetections  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--frames', type=int, default=236)
parser.add_argument('--classes', default='coco')
parser.add_argument('--det-cache', metavar='NPZ')
parser.add_argument('--min-run', type=int, default=3)
parser.add_argument('--gap-sweep', type=int, nargs='*', metavar='MAX_GAP',
                    help='sweep StitchConfig.max_gap and score each by identity count AND by the '
                         'implied speed across every merged gap (a wrong merge teleports a body)')
args = parser.parse_args()

import cv2  # noqa: E402

cap = cv2.VideoCapture(args.clip)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
cap.release()

cache = Path(args.det_cache or
             f'out/phmr_ab/dets_{args.classes}_{args.start}_{args.frames}.npz')
if not cache.exists():
    raise SystemExit(f'no detection cache at {cache} -- run prompthmr_find_crossing.py --frames '
                     f'{args.frames} first, which writes it')
c = np.load(cache, allow_pickle=True)
detections = Detections(frames=[
    FrameDetections(frame=int(f), items=[
        Detection(bbox_xyxy=b, cls=str(k), score=float(s))
        for b, k, s in zip(bb, kk, ss, strict=True)])
    for f, bb, kk, ss in zip(c['frame'], c['boxes'], c['classes'], c['scores'], strict=True)])
print(f'detections from {cache}: '
      f'{sum(len(f.items) for f in detections.frames)} over {len(detections.frames)} frames')

clip = ClipRef(source_id='colombia', uri=args.clip,
               frames=np.arange(args.start, args.start + args.frames),
               width=width, height=height, fps=fps)

def _seam_speeds(before, after, report):
    """Centre speed (px/frame) implied by every gap a merge bridged, vs normal motion.

    The stitcher's own risk is stated in its docstring: "a missed merge leaves two fragments, but
    a *wrong* merge teleports a body". So widening `max_gap` cannot be judged on the identity
    count alone -- a merge that needs a player to cross the pitch in half a second is a wrong one,
    and it shows up here as a seam speed far outside the clip's own distribution.
    """
    box = {}
    for t in before.tracklets:
        for f, b in zip(t.frames.tolist(), t.bboxes_xyxy, strict=True):
            box[(t.track_id, int(f))] = np.asarray(b, float)

    normal = []
    for t in before.tracklets:
        fr, bb = t.frames.tolist(), t.bboxes_xyxy
        for (f0, b0), (f1, b1) in zip(zip(fr, bb, strict=True), zip(fr[1:], bb[1:], strict=True),
                                      strict=False):
            if f1 - f0 == 1:
                c0 = ((b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2)
                c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
                normal.append(float(np.hypot(c1[0] - c0[0], c1[1] - c0[1])))

    seams = []
    for ids in report.merges:
        spans = []
        for tid in ids:
            fs = sorted(f for (t, f) in box if t == tid)
            if fs:
                spans.append((fs[0], fs[-1], tid))
        spans.sort()
        for (_a0, a1, ta), (b0, _b1, tb) in zip(spans, spans[1:], strict=False):
            ca, cb = box[(ta, a1)], box[(tb, b0)]
            d = np.hypot((cb[0] + cb[2]) / 2 - (ca[0] + ca[2]) / 2,
                         (cb[1] + cb[3]) / 2 - (ca[1] + ca[3]) / 2)
            seams.append(float(d / max(1, b0 - a1)))
    return np.array(normal), np.array(seams)


if args.gap_sweep is not None:
    grid = ([('max_gap', g) for g in (args.gap_sweep or [12, 24, 48, 72])]
            + [('max_center_dist', d) for d in (1.5, 2.5, 4.0, 6.0, 10.0)]
            + [('max_size_ratio', r) for r in (1.6, 2.2, 3.0)])
    tracks = ByteTrackTracker(device='cpu', min_track_frames=4, kit_split=True,
                              kit_split_min_run=args.min_run).track(clip, detections)
    n_in = len([t for t in tracks.tracklets if t.cls == 'player'])
    print(f'\nkit_split on, {n_in} player ids before stitching. '
          f'Median players visible per frame is the real target.')
    print('\n  gate                    ids  merges   worst seam speed   normal p95   verdict')
    for name, val in grid:
        out, rep = stitch_tracks_with_report(tracks, StitchConfig(**{name: val}))
        n = len([t for t in out.tracklets if t.cls == 'player'])
        normal, seams = _seam_speeds(tracks, out, rep)
        p95 = float(np.percentile(normal, 95)) if normal.size else 0.0
        worst = float(seams.max()) if seams.size else 0.0
        bad = int((seams > p95).sum()) if seams.size else 0
        verdict = 'clean' if bad == 0 else f'{bad} merge(s) above normal p95'
        print(f'  {name}={val:<8} {n:6d}  {len(rep.merges):6d}   {worst:14.1f} px/f  '
              f'{p95:9.1f} px/f   {verdict}', flush=True)
    raise SystemExit(0)

print('\n  kit_split  stitch   player ids   merges  dropped')
for split in (False, True):
    tracks = ByteTrackTracker(device='cpu', min_track_frames=4, kit_split=split,
                              kit_split_min_run=args.min_run).track(clip, detections)
    for stitch in (False, True):
        if stitch:
            out, rep = stitch_tracks_with_report(tracks, StitchConfig())
            n = len([t for t in out.tracklets if t.cls == 'player'])
            extra = f'{len(rep.merges):6d}  {len(rep.dropped):7d}'
        else:
            n = len([t for t in tracks.tracklets if t.cls == 'player'])
            extra = f'{"-":>6s}  {"-":>7s}'
        print(f'  {"on " if split else "off"}        {"on " if stitch else "off"}      '
              f'{n:6d}       {extra}', flush=True)
