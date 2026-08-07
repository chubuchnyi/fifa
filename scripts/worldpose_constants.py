"""What are the real numbers? — our guessed kinematic constants against WorldPose ground truth.

Several thresholds in this repo are **guesses that were never measured against a real football
player**: the 10.5 m/s speed ceiling and 8 m/s² acceleration ceiling in the physics gates, the
0.5 m radius that `track_quality.py` calls a "twin", and the ~0.4 m pelvis rise we cite as what a
jump looks like. WorldPose is ground truth for exactly our problem — multi-person SMPL in world
coordinates, from broadcast football — so it can replace every one of them with a distribution.

**89 clips, 1080p 50 Hz broadcast, up to 22 players each, NaN where a player is not visible.**
Poses and cameras are on disk (`AVATAR/WorldPose/{poses,cameras}`); the *video* is FIFA-gated and
is not needed here — these are world-space translations already.

    .venv/bin/python scripts/worldpose_constants.py
    .venv/bin/python scripts/worldpose_constants.py --clips 12    # quick pass

Two things this deliberately does NOT do. It does not smooth: a ceiling fitted to smoothed GT
would be a ceiling on our smoother, not on football. And it reports **p99.9 and max separately**,
because a gate set at the max passes everything and a gate set at p99 clips real football.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WP = REPO / 'WorldPose'
FPS = 50.0          # "raw 1080p 50Hz TV" — WorldPose paper (arXiv 2501.02771v2), §3

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--clips', type=int, default=0, help='0 = all')
ap.add_argument('--twin-radius', type=float, default=0.5,
                help='the radius track_quality.py calls a twin')
args = ap.parse_args()

files = sorted((WP / 'poses').glob('*.npz'))
if args.clips:
    files = files[:args.clips]
print(f'WorldPose: {len(files)} clips at {FPS:.0f} Hz\n')

# ---------------------------------------------------------------- which axis is up
head = np.load(files[0])['transl']                       # (N, T, 3)
spans = np.nanmax(head.reshape(-1, 3), axis=0) - np.nanmin(head.reshape(-1, 3), axis=0)
UP = int(np.argmin(spans))
GROUND = [i for i in range(3) if i != UP]
print(f'axis spans on {files[0].stem}: {spans.round(1)} m  ->  up axis is {"xyz"[UP]}, '
      f'ground plane {"".join("xyz"[i] for i in GROUND)}')


def _smooth(a: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average along axis 0, edges held. `a` must be gap-free."""
    if a.shape[0] < w:
        return a
    k = np.ones(w) / w
    return np.stack([np.convolve(a[:, j], k, mode='same') for j in range(a.shape[1])], axis=1)


def _runs(ok: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, stop) spans where `ok` is True — a gap must never be differentiated."""
    out, i = [], 0
    while i < ok.size:
        if ok[i]:
            j = i
            while j < ok.size and ok[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


speeds: list[np.ndarray] = []
accels: list[np.ndarray] = []
speeds_s: list[np.ndarray] = []      # same, on a 100 ms moving average
accels_s: list[np.ndarray] = []
vert_range: list[float] = []          # per player-clip, total vertical excursion
vert_burst: list[float] = []          # per player-clip, largest rise inside 0.5 s (a jump)
near: list[np.ndarray] = []           # per frame, nearest-neighbour distance between players
n_pairs_close = 0
n_pairs_seen = 0
frames_total = 0

for f in files:
    tr = np.load(f)['transl'].astype(float)              # (N, T, 3)
    n, t, _ = tr.shape
    frames_total += t

    for i in range(n):
        p = tr[i]
        ok = np.isfinite(p).all(axis=1)
        if ok.sum() < 5:
            continue
        # Differentiate only across frames that are BOTH visible and adjacent, so a
        # NaN gap never becomes a teleport.
        step = ok[:-1] & ok[1:]
        if step.sum() >= 1:
            d = np.linalg.norm(np.diff(p[:, GROUND], axis=0)[step], axis=1) * FPS
            speeds.append(d)
        step2 = ok[:-2] & ok[1:-1] & ok[2:]
        if step2.sum() >= 1:
            a = np.linalg.norm(np.diff(p[:, GROUND], n=2, axis=0)[step2], axis=1) * FPS * FPS
            accels.append(a)
        # The same two quantities on a 100 ms moving average, per gap-free run. A ceiling fitted
        # to raw 50 Hz GT is a ceiling on GT jitter; our own poses reach the gates smoothed.
        for lo, hi in _runs(ok):
            if hi - lo < 7:
                continue
            q = _smooth(p[lo:hi][:, GROUND], 5)[2:-2]     # drop the edge-held samples
            if q.shape[0] >= 3:
                speeds_s.append(np.linalg.norm(np.diff(q, axis=0), axis=1) * FPS)
                accels_s.append(np.linalg.norm(np.diff(q, n=2, axis=0), axis=1) * FPS * FPS)
        z = p[ok, UP]
        vert_range.append(float(z.max() - z.min()))
        w = int(0.5 * FPS)
        if z.size > w:
            # largest rise inside any half-second window: a jump, not a slope
            roll_max = np.array([z[k:k + w].max() - z[k:k + w].min()
                                 for k in range(0, z.size - w, 5)])
            vert_burst.append(float(roll_max.max()))

    # nearest neighbour per frame: is 0.5 m ever legitimate between two REAL players?
    for k in range(0, t, 5):                              # every 5th frame is plenty
        pts = tr[:, k, :][:, GROUND]
        vis = np.isfinite(pts).all(axis=1)
        pts = pts[vis]
        if pts.shape[0] < 2:
            continue
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        iu = np.triu_indices(pts.shape[0], k=1)
        dd = d[iu]
        n_pairs_seen += dd.size
        n_pairs_close += int((dd < args.twin_radius).sum())
        near.append(dd.min(axis=None, keepdims=True))

sp = np.concatenate(speeds)
ac = np.concatenate(accels)
nn = np.concatenate(near)
vr = np.asarray(vert_range)
vb = np.asarray(vert_burst)


def row(name: str, a: np.ndarray, unit: str, ours: str) -> None:
    print(f'  {name:<26} p50 {np.percentile(a, 50):7.2f}  p99 {np.percentile(a, 99):7.2f}  '
          f'p99.9 {np.percentile(a, 99.9):7.2f}  max {a.max():7.2f} {unit}   ours: {ours}')


print(f'\n== root kinematics, {len(sp):,} frame-to-frame samples over {frames_total:,} frames ==')
row('horizontal speed', sp, 'm/s', '10.5 m/s ceiling')
row('horizontal accel', ac, 'm/s²', '8 m/s² ceiling')
print('  -- the same, on a 100 ms moving average (raw 50 Hz double differences are mostly '
      'GT jitter) --')
row('horizontal speed, 100 ms', np.concatenate(speeds_s), 'm/s', '10.5 m/s ceiling')
row('horizontal accel, 100 ms', np.concatenate(accels_s), 'm/s²', '8 m/s² ceiling')

print(f'\n== vertical, {vr.size} player-clips (up axis = {"xyz"[UP]}) ==')
row('root range, whole clip', vr, 'm', 'best scene ever: 0.234 m')
row('root rise, 0.5 s window', vb, 'm', 'we cite ~0.4 m for a jump')

print(f'\n== how close do two REAL players get? {n_pairs_seen:,} player-pairs sampled ==')
row('nearest neighbour', nn, 'm', f'twin radius {args.twin_radius} m')
frac = 100.0 * n_pairs_close / max(n_pairs_seen, 1)
print(f'  pairs closer than {args.twin_radius} m: {n_pairs_close:,} of {n_pairs_seen:,} '
      f'= {frac:.3f} %')
for r in (0.3, 0.4, 0.5, 0.75, 1.0):
    print(f'    < {r:.2f} m: {100.0 * (nn < r).mean():6.2f} % of frames have SOME pair that close')
