"""The #132 kit split fires — but does it fire *where the kit flips*? (W13)

Three separate measurements on 2026-08-07 said no, and none of them was looking for it:

* **W3.** t17 reads blue for f0-22 and **yellow from f26 to f34** — its id has walked onto a
  yellow player — yet the split cut it at **f35**, leaving 9 contaminated frames inside a track
  labelled B. Those 9 frames are what made П3-pre-pose propose merging t17 into a blue track.
* **W10.** t11 reads `?YYYYYY???B?BBB…` and t13 reads `BBBB??YYY???BBB…`, both labelled B, both
  never cut at all.
* **W3 again.** t3 reads 19 yellow against 10 blue and is labelled **B**.

A late cut is worse than no cut: the piece keeps its avatar, its team label and its motion history
across a human change, and everything downstream — stitch, identity gate, pose — treats it as one
person.

This replays the tracker's own appearance sampler and centroids, then puts three rows side by
side per track: what the **centroids** say each frame is, what the **video pixels** say, and where
the split actually cut.

    PYTHONPATH=src .venv/bin/python scripts/bench_kit_split_timing.py
    PYTHONPATH=src .venv/bin/python scripts/bench_kit_split_timing.py --only 17,11,13,3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from pitch3d.adapters.models.tracking import (  # noqa: E402
    TEAM_CLASSES,
    ByteTrackTracker,
    _hsv_to_feature,
    _kmeans,
    _majority_class,
    split_on_kit_change,
)
from pitch3d.core.ports.io import ClipRef  # noqa: E402
from pitch3d.core.ports.perception import (  # noqa: E402
    Detection,
    Detections,
    FrameDetections,
)

ap = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
ap.add_argument('--dets', default='out/phmr_ab/dets_coco_0_236.npz')
ap.add_argument('--frames', type=int, default=60)
ap.add_argument('--only', default='', help='comma-separated track ids; default = every track')
ap.add_argument('--min-run', type=int, default=None, help='override kit_split_min_run')
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

trk = ByteTrackTracker(device='cpu', min_track_frames=2, kit_split=False)
if args.min_run is not None:
    trk.kit_split_min_run = int(args.min_run)
backend = trk.backend or trk._default_backend()
raw = [r for r in backend.associate(clip, dets)
       if r.frames.shape[0] >= trk.min_track_frames]
print(f'{len(raw)} raw tracklets, kit_split_min_run={trk.kit_split_min_run}')

# The tracker's own centroid fit, verbatim: over every frame of every team-bearing track at once.
team_bearing = [r for r in raw
                if r.appearance_series is not None and _majority_class(r.classes) in TEAM_CLASSES]
feats = _hsv_to_feature(np.concatenate([r.appearance_series for r in team_bearing]))
labels = _kmeans(feats, 2)
centroids = np.stack([feats[labels == k].mean(axis=0) for k in range(2) if np.any(labels == k)])

# Name each centroid by its own hue, so the printed letter means something.
names = []
for cen in centroids:
    ang = np.degrees(np.arctan2(cen[1], cen[0])) % 360.0
    names.append('Y' if 20.0 <= ang / 2.0 <= 55.0 else ('B' if 160.0 <= ang <= 280.0 else '?'))
hues = [round(float(np.degrees(np.arctan2(c[1], c[0])) % 360), 1) for c in centroids]
print(f'centroid hues: {hues} -> named {names}')

# --------------------------------------------------------------- the pixels, independently
cap = cv2.VideoCapture(str(REPO / args.clip))
cache: dict[int, np.ndarray] = {}


def img(i: int):
    if i not in cache:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        cache[i] = im if ok else None
    return cache[i]


def pixel_kit(f: int, box) -> str:
    x0, y0, x1, y1 = box
    h, w = y1 - y0, x1 - x0
    a, b = int(y0 + 0.20 * h), int(y0 + 0.45 * h)
    c0, c1 = int(x0 + 0.25 * w), int(x0 + 0.75 * w)
    im = img(f)
    if im is None or b <= a or c1 <= c0 or min(a, c0) < 0:
        return '-'
    patch = im[a:b, c0:c1]
    if patch.size == 0:
        return '-'
    hsv = cv2.cvtColor(np.uint8([[np.median(patch.reshape(-1, 3), axis=0)]]),
                       cv2.COLOR_BGR2HSV)[0][0]
    if 18 <= hsv[0] <= 48 and hsv[1] > 90:
        return 'Y'
    if 85 <= hsv[0] <= 135 and hsv[1] > 55:
        return 'B'
    return '?'


wanted = {int(x) for x in args.only.split(',') if x.strip()} if args.only else None
next_id = max(r.track_id for r in raw) + 1
n_late = n_missed = n_split = 0
print(f'\n{"":8}centroid = what the split sees · pixels = what the shirt is · ^ = where it cut\n')
for r in sorted(raw, key=lambda x: x.track_id):
    if wanted is not None and r.track_id not in wanted:
        continue
    if r.appearance_series is None:
        continue
    f = np.asarray(r.frames, dtype=int).reshape(-1)
    lab = np.argmin(np.sum((_hsv_to_feature(r.appearance_series)[:, None, :]
                            - centroids[None, :, :]) ** 2, axis=2), axis=1)
    cen_s = ''.join(names[k] for k in lab)
    pix_s = ''.join(pixel_kit(int(f[i]), r.bboxes_xyxy[i]) for i in range(f.shape[0]))

    pieces = split_on_kit_change(r, centroids, trk.kit_split_min_run, next_id)
    cut_at = np.cumsum([p.frames.shape[0] for p in pieces])[:-1].tolist()
    marker = ''.join('^' if i in cut_at else ' ' for i in range(f.shape[0]))

    # Where do the PIXELS flip, ignoring '?' and '-'? That is the truth the cut should match.
    solid = [(i, ch) for i, ch in enumerate(pix_s) if ch in 'YB']
    pix_cuts = [solid[j + 1][0] for j in range(len(solid) - 1)
                if solid[j][1] != solid[j + 1][1]]
    # collapse flips closer than min_run apart — those are crossings, not handovers
    pruned: list[int] = []
    for x in pix_cuts:
        if not pruned or x - pruned[-1] >= trk.kit_split_min_run:
            pruned.append(x)

    if len(pieces) > 1:
        n_split += 1
    verdict = ''
    if pruned and not cut_at:
        verdict = f'  <== NEVER CUT, pixels flip at index {pruned}'
        n_missed += 1
    elif pruned and cut_at:
        off = [min(abs(x - y) for y in cut_at) for x in pruned]
        if max(off) >= trk.kit_split_min_run:
            verdict = (f'  <== CUT LATE/EARLY: cut at {cut_at}, pixels flip at {pruned} '
                       f'(off by {off})')
            n_late += 1
    if not pruned and not cut_at and wanted is None:
        continue                                   # clean track, nothing to say

    print(f't{r.track_id:<4d} f{f[0]}-{f[-1]}{verdict}')
    print(f'  centroid  {cen_s}')
    print(f'  pixels    {pix_s}')
    print(f'  cut       {marker}')

cap.release()
print(f'\n{n_split} track(s) split · {n_late} cut in the wrong place · '
      f'{n_missed} flipped without ever being cut')
