"""Did the OpenCV 5 YUV→RGB change move the kit colours? (#103 / W10, П7)

R9 migrated the repo to OpenCV 5 on 2026-07-29 with "zero code changes", and recorded one real
behaviour change: the video decode switched its YUV→RGB matrix **BT.601 → BT.709**, shifting
pixels on 92 % of the frame. Kit colour and jersey OCR were never re-measured against it, and the
question is now load-bearing — the handover merge's one flagged decision (t10 + t77) turns on
whether the shirt reader is telling the truth, and the user has said outright that they had been
taking the reconstruction's team colours as truth and suspect they are wrong.

`ffprobe` settles half of it before any pixels move: the target clip declares
**`color_space=bt709`, `color_range=tv`**. So BT.709 is what the file asks for, OpenCV 5 is
right, and it is the *pre-migration* numbers that were decoded wrong.

What is left to measure is whether it matters where we look — on a shirt, at our subject size —
and whether the HSV thresholds the kit reader uses still sit in the right place.

    PYTHONPATH=src .venv/bin/python scripts/bench_kit_colour_matrix.py

Decodes the raw YUV planes once with ffmpeg (no matrix applied), converts them **both** ways in
numpy, and reads every tracked shirt under each. Also checks what `cv2.VideoCapture` actually
returns, because a library being on the right matrix and a library handling `tv` range correctly
are two different claims.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
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
ap.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
ap.add_argument('--dets', default='out/phmr_ab/dets_coco_0_236.npz')
ap.add_argument('--frames', type=int, default=60)
args = ap.parse_args()

W, H = 1920, 1080
CLIP = REPO / args.clip

# --------------------------------------------------------------- raw YUV, no matrix applied
raw = subprocess.run(
    ['ffmpeg', '-v', 'error', '-i', str(CLIP), '-vframes', str(args.frames),
     '-pix_fmt', 'yuv420p', '-f', 'rawvideo', '-'],
    capture_output=True, check=True).stdout
per = W * H * 3 // 2
n = len(raw) // per
print(f'{n} frames of raw yuv420p decoded ({len(raw) / 1e6:.0f} MB)')
buf = np.frombuffer(raw, dtype=np.uint8)


def planes(i: int):
    f = buf[i * per:(i + 1) * per]
    y = f[:W * H].reshape(H, W).astype(np.float64)
    u = f[W * H:W * H + W * H // 4].reshape(H // 2, W // 2).astype(np.float64)
    v = f[W * H + W * H // 4:].reshape(H // 2, W // 2).astype(np.float64)
    return y, np.repeat(np.repeat(u, 2, 0), 2, 1), np.repeat(np.repeat(v, 2, 0), 2, 1)


#: Inverse conversion coefficients. Both are the *limited-range* ("tv") forms, which is what the
#: file declares — using the full-range form on a tv-range file is a second, independent way to
#: get the colour wrong, and it is not the one R9 changed.
KR_KB = {'bt601': (0.299, 0.114), 'bt709': (0.2126, 0.0722)}


def to_bgr(i: int, matrix: str) -> np.ndarray:
    y, u, v = planes(i)
    kr, kb = KR_KB[matrix]
    kg = 1.0 - kr - kb
    yl = (y - 16.0) * (255.0 / 219.0)          # tv range 16..235 -> 0..255
    cb = (u - 128.0) * (255.0 / 224.0)
    cr = (v - 128.0) * (255.0 / 224.0)
    r = yl + 2.0 * (1.0 - kr) * cr
    b = yl + 2.0 * (1.0 - kb) * cb
    g = (yl - kr * r - kb * b) / kg
    return np.clip(np.stack([b, g, r], axis=2), 0, 255).astype(np.uint8)


# --------------------------------------------------------------- what does cv2 actually return?
cap = cv2.VideoCapture(str(CLIP))
cv_frames = {}
for i in range(n):
    ok, im = cap.read()
    if not ok:
        break
    cv_frames[i] = im
cap.release()
print(f'cv2 {cv2.__version__} returned {len(cv_frames)} frames')

probe = min(5, n - 1)
a601, a709, acv = to_bgr(probe, 'bt601'), to_bgr(probe, 'bt709'), cv_frames[probe]
d = np.abs(a601.astype(int) - a709.astype(int))
print(f'\n== whole frame f{probe}: BT.601 vs BT.709 ==')
print(f'   mean |delta| {d.mean():5.2f} / 255   max {d.max()}   '
      f'pixels changed at all: {100.0 * (d.max(axis=2) > 0).mean():.1f} %')
print(f'   pixels changed by >4/255: {100.0 * (d.max(axis=2) > 4).mean():.1f} %')
for name, arr in (('BT.601', a601), ('BT.709', a709)):
    e = np.abs(arr.astype(int) - acv.astype(int))
    print(f'   cv2 vs {name}: mean |delta| {e.mean():5.2f}   max {e.max()}'
          + ('   <== this is what cv2 gives us' if e.mean() < 2.0 else ''))

# --------------------------------------------------------------- the shirts
c = np.load(REPO / args.dets, allow_pickle=True)
dets = Detections(frames=[
    FrameDetections(frame=int(f), items=[
        Detection(bbox_xyxy=b, cls=str(k), score=float(s))
        for b, k, s in zip(bb, kk, ss, strict=True)])
    for f, bb, kk, ss in zip(c['frame'], c['boxes'], c['classes'], c['scores'], strict=True)
    if int(f) < n])
clip = ClipRef(source_id='colombia', uri=str(CLIP), frames=np.arange(n),
               width=W, height=H, fps=29.97)
tracks = ByteTrackTracker(device='cpu', min_track_frames=2, kit_split=True).track(clip, dets)
print(f'\n{len(tracks.tracklets)} tracklets')

cache: dict[tuple[int, str], np.ndarray] = {}


def frame_as(i: int, matrix: str) -> np.ndarray:
    key = (i, matrix)
    if key not in cache:
        cache[key] = to_bgr(i, matrix)
    return cache[key]


def shirt_hsv(img: np.ndarray, box) -> np.ndarray | None:
    x0, y0, x1, y1 = box
    h, w = y1 - y0, x1 - x0
    a, b = int(y0 + 0.20 * h), int(y0 + 0.45 * h)
    c0, c1 = int(x0 + 0.25 * w), int(x0 + 0.75 * w)
    if b <= a or c1 <= c0 or min(a, c0) < 0 or b > H or c1 > W:
        return None
    patch = img[a:b, c0:c1]
    if patch.size == 0:
        return None
    med = np.uint8([[np.median(patch.reshape(-1, 3), axis=0)]])
    return cv2.cvtColor(med, cv2.COLOR_BGR2HSV)[0][0]


def classify(hsv) -> str:
    """The exact thresholds `scripts/track_quality.py --kit` uses."""
    if hsv is None:
        return '-'
    if 18 <= hsv[0] <= 48 and hsv[1] > 90:
        return 'Y'
    if 85 <= hsv[0] <= 135 and hsv[1] > 55:
        return 'B'
    return '?'


rows, flips, seen = [], 0, 0
dh, ds, dv = [], [], []
for t in sorted(tracks.tracklets, key=lambda x: int(x.track_id)):
    fr = np.asarray(t.frames, dtype=int).reshape(-1)
    bx = np.asarray(t.bboxes_xyxy, dtype=float).reshape(-1, 4)
    s601 = s709 = ''
    for k, f in enumerate(fr):
        h6 = shirt_hsv(frame_as(int(f), 'bt601'), bx[k])
        h7 = shirt_hsv(frame_as(int(f), 'bt709'), bx[k])
        c6, c7 = classify(h6), classify(h7)
        s601 += c6
        s709 += c7
        if h6 is not None and h7 is not None:
            dh.append(abs(int(h6[0]) - int(h7[0])))
            ds.append(abs(int(h6[1]) - int(h7[1])))
            dv.append(abs(int(h6[2]) - int(h7[2])))
            seen += 1
            flips += c6 != c7

    def modal(s: str) -> str:
        y, b = s.count('Y'), s.count('B')
        return 'Y' if y > b else ('B' if b > y else '?')

    rows.append((int(t.track_id), str(t.team_id), modal(s601), modal(s709), s601, s709))

print(f'\n== shirt patch under each matrix, {seen} measured boxes ==')
print(f'   hue   |delta| mean {np.mean(dh):5.2f}  p95 {np.percentile(dh, 95):5.1f}  '
      f'max {max(dh)} (of 180)')
print(f'   sat   |delta| mean {np.mean(ds):5.2f}  p95 {np.percentile(ds, 95):5.1f}  max {max(ds)}')
print(f'   value |delta| mean {np.mean(dv):5.2f}  p95 {np.percentile(dv, 95):5.1f}  max {max(dv)}')
print(f'   per-box classification flips: {flips} of {seen} = {100.0 * flips / max(seen, 1):.2f} %')

print(f'\n{"track":>6} {"team":>5} {"601":>4} {"709":>4}   per-frame (601 over 709)')
changed = 0
for tid, team, m6, m7, s6, s7 in rows:
    mark = '   <== the matrix changes this track' if m6 != m7 else ''
    changed += m6 != m7
    print(f'{tid:6d} {team:>5} {m6:>4} {m7:>4}   {s6}')
    if s6 != s7:
        print(f'{"":6} {"":>5} {"":>4} {"":>4}   {s7}{mark}')
print(f'\n{changed} of {len(rows)} tracks get a different modal kit under the other matrix.')
