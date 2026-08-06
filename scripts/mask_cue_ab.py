"""Does McByte's mask cue actually fix the identity churn? — the measurement it was built for.

The chain that leads here, all measured (#133): 78 mid-pitch identity births/deaths in shot 1;
96 % of them have an unclaimed detection a median 6-23 px away, so the boxes exist; and three
cheap fixes came back null — the match threshold in both directions and the detector threshold.
What is missing is an association *cue*, and a mask propagated from earlier frames is one, because
it does not depend on the detection being judged.

    .venv/bin/python scripts/mask_cue_ab.py --masks out/phmr_ab/track_masks.npz

Same detections, same association parameters, same stitch — only the cue differs. Reported both
ways, because a cue that grabs the wrong player trades one error for a worse one:

* **mid-pitch births/deaths** — players appearing from nowhere or vanishing. Down is the point.
* **kit changes** — a track whose team colour flips is one id covering two humans. Up means the
  cue is confidently wrong. Measured with the #132 kit split **off**, since that split cuts such
  tracks apart and would hide the very thing this column exists to catch.
* **subjects after stitching** — what the render actually receives.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from pitch3d.adapters.models.tracking import (  # noqa: E402
    ByteTrackBackend,
    ByteTrackTracker,
    MaskCue,
)
from pitch3d.core.orchestration.continuity import (  # noqa: E402
    StitchConfig,
    stitch_tracks_with_report,
)
from pitch3d.core.ports.io import ClipRef  # noqa: E402
from pitch3d.core.ports.perception import (  # noqa: E402
    Detection,
    Detections,
    FrameDetections,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
parser.add_argument('--dets', default='out/phmr_ab/dets_coco_0_236.npz')
parser.add_argument('--masks', default='out/phmr_ab/track_masks.npz')
parser.add_argument('--frames', type=int, default=236)
parser.add_argument('--kit-split', action='store_true',
                    help='leave the #132 split ON; default OFF so wrong matches stay visible')
parser.add_argument('--out-prefix', default='out/phmr_ab/cue')
args = parser.parse_args()

clip_path = args.clip if Path(args.clip).is_absolute() else str(REPO / args.clip)

c = np.load(REPO / args.dets, allow_pickle=True)
dets = Detections(frames=[
    FrameDetections(frame=int(f), items=[
        Detection(bbox_xyxy=b, cls=str(k), score=float(s))
        for b, k, s in zip(bb, kk, ss, strict=True)])
    for f, bb, kk, ss in zip(c['frame'], c['boxes'], c['classes'], c['scores'], strict=True)])

m = np.load(REPO / args.masks)
labels = {int(f): lab for f, lab in zip(m['frames'], m['labels'], strict=True)}
ids = sorted({int(v) for lab in labels.values() for v in np.unique(lab)} - {0})
print(f'masks: {len(labels)} frames, {len(ids)} distinct track ids carried')

W, H = 1920, 1080
clip = ClipRef(source_id='colombia', uri=clip_path,
               frames=np.arange(args.frames), width=W, height=H, fps=29.97)
EDGE = 0.06 * W


def plausible(b):
    return 25 < b[3] - b[1] < 0.45 * H and b[2] - b[0] < 0.30 * W


def at_edge(b):
    return b[0] < EDGE or b[2] > W - EDGE or b[1] < EDGE * H / W or b[3] > H - EDGE * H / W


cap = cv2.VideoCapture(clip_path)
if not cap.isOpened():
    raise SystemExit(f'cannot open {clip_path!r}')
imgs = {}
for n in range(args.frames):
    ok, img = cap.read()
    if not ok:
        break
    imgs[n] = img
cap.release()


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


def score(tracks):
    """(mid-pitch events, kit changes) over the player tracks."""
    events = swaps = 0
    for t in tracks:
        fr = np.asarray(t.frames, dtype=int)
        bx = np.asarray(t.bboxes_xyxy, dtype=float)
        keep = np.array([plausible(b) for b in bx])
        if keep.sum() < 2:
            continue
        fr, bx = fr[keep], bx[keep]
        if fr.min() > 2 and not at_edge(bx[0]):
            events += 1
        if fr.max() < args.frames - 3 and not at_edge(bx[-1]):
            events += 1
        seq = [k for k in (kit_of(b, imgs[f]) for f, b in zip(fr.tolist(), bx, strict=True)
                           if f in imgs) if k != '?']
        runs = []
        for k in seq:
            if runs and runs[-1][0] == k:
                runs[-1][1] += 1
            else:
                runs.append([k, 1])
        if len({k for k, n in runs if n >= 3}) > 1:
            swaps += 1
    return events, swaps


print(f'\n{"cue":>5} {"tracks":>6} {"after stitch":>12} {"mid-pitch ev":>12} {"kit chg":>7}')
for on in (False, True):
    cue = MaskCue(labels=labels) if on else None
    tracker = ByteTrackTracker(
        device='cpu', min_track_frames=4, kit_split=args.kit_split,
        backend=ByteTrackBackend(device='cpu', mask_cue=cue),
    )
    tracks = tracker.track(clip, dets)
    players = [t for t in tracks.tracklets if t.cls == 'player']
    stitched, _ = stitch_tracks_with_report(tracks, StitchConfig())
    after = [t for t in stitched.tracklets if t.cls == 'player']
    ev, sw = score(after)
    print(f'{"ON" if on else "OFF":>5} {len(players):6d} {len(after):12d} {ev:12d} {sw:7d}',
          flush=True)
    out = f'{args.out_prefix}_{"on" if on else "off"}.npz'
    (REPO / out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(REPO / out,
             frames=np.array([t.frames for t in after], dtype=object),
             boxes=np.array([t.bboxes_xyxy for t in after], dtype=object),
             track_ids=np.array([t.track_id for t in after]),
             ranking=np.zeros((0, 5)), occlusion=np.zeros((0, 5)), visible_fill=np.zeros((0, 5)),
             width=W, height=H, fps=29.97, start=0, n_frames=args.frames, allow_pickle=True)
    print(f'      -> {out}')

print('\nDown on mid-pitch events is the point; UP on kit changes means the cue is confidently')
print('wrong, which costs more than the churn it removed.')
