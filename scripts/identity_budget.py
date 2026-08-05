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
