"""Do the skeletons sit on the players? — the three residuals, measured (step 1, #140).

The eye said the overlay does not line up, worse toward the frame edges. This turns that into
numbers, and separates the two causes it can have:

1. **pitch paint** — is the drawn pitch line on real paint? Answers the camera on the ground plane.
2. **subject foot vs detector box bottom** — the goal is players, not paint. Both halves already
   exist: the scene's projected root and the cached detections.
3. **common-mode vs per-player scatter** — all subjects displaced the same way means the camera;
   scattered means grounding or association. This is the only test that separates them, and it
   costs nothing once (2) is computed.

    PYTHONPATH=src .venv/bin/python scripts/bench_overlay_residual.py \\
        --scene out/res_ab/res896.json out/res_ab/res896_rigid.json \\
        --dets out/dets/dets_r896_0_236.npz

Residuals are reported **by radius from the principal point**, because a focal or distortion error
grows with radius while an extrinsics error does not.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

_spec = importlib.util.spec_from_file_location('tq', REPO / 'scripts' / 'track_quality.py')
_tq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tq)

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--scene', nargs='+', required=True)
ap.add_argument('--dets', default='out/dets/dets_r896_0_236.npz')
ap.add_argument('--max-px', type=float, default=250.0,
                help='a match further than this is a different player, not a residual')
args = ap.parse_args()

d = np.load(REPO / args.dets, allow_pickle=True)
boxes_by_frame: dict[int, np.ndarray] = {}
for f, bb, cc in zip(d['frame'], d['boxes'], d['classes'], strict=True):
    keep = [i for i, c in enumerate(cc) if c == 'player']
    if keep:
        boxes_by_frame[int(f)] = np.asarray(bb, dtype=float).reshape(-1, 4)[keep]

print(f'{len(boxes_by_frame)} frames of cached detections\n')
print(f'{"scene":<34} {"fx":>7} {"source":>16} {"n":>5} {"median":>7} {"p90":>7} '
      f'{"common":>7} {"scatter":>8}')

for path in args.scene:
    scene = _tq.load_scene(REPO / path)
    cam = scene['camera']
    k = cam['intrinsics']
    project = _tq.projector_from_scene(cam)
    src = cam.get('source', '(unmarked)')

    per_frame: dict[int, list[np.ndarray]] = {}
    radii, errs = [], []
    for tr in scene['tracks'].values():
        prov = np.asarray(tr['prov'])
        for i, f in enumerate(tr['frames']):
            if prov[i] != 'measured' or int(f) not in boxes_by_frame:
                continue
            root = tr['transl'][i]
            uv = project(int(f), np.array([root[0], root[1], root[2] - 0.92]))  # the feet
            if uv is None:
                continue
            bx = boxes_by_frame[int(f)]
            foot = np.column_stack(((bx[:, 0] + bx[:, 2]) / 2.0, bx[:, 3]))
            dist = np.linalg.norm(foot - np.asarray(uv), axis=1)
            j = int(np.argmin(dist))
            if dist[j] > args.max_px:
                continue
            vec = foot[j] - np.asarray(uv)
            per_frame.setdefault(int(f), []).append(vec)
            errs.append(float(dist[j]))
            radii.append(float(np.hypot(uv[0] - k['cx'], uv[1] - k['cy'])))

    if not errs:
        print(f'{Path(path).name:<34} {k["fx"]:7.0f} {str(src):>16}     0   — no matched subjects')
        continue
    e = np.asarray(errs)
    r = np.asarray(radii)
    # common mode = the length of the mean displacement per frame; scatter = spread about it
    groups = [v for v in per_frame.values() if len(v) >= 3]
    common = np.mean([np.linalg.norm(np.mean(v, axis=0)) for v in groups])
    scatter = np.mean([np.mean(np.linalg.norm(np.asarray(v) - np.mean(v, axis=0), axis=1))
                       for v in groups])
    print(f'{Path(path).name:<34} {k["fx"]:7.0f} {str(src):>16} {e.size:5d} '
          f'{np.median(e):7.1f} {np.percentile(e, 90):7.1f} {common:7.1f} {scatter:8.1f}')

    # by radius — a focal or distortion error grows with it, extrinsics do not
    edges = np.percentile(r, [0, 25, 50, 75, 100])
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (r >= lo) & (r <= hi)
        if m.sum():
            bins.append(f'{lo:4.0f}-{hi:4.0f}px: {np.median(e[m]):5.1f}')
    print(f'{"":<34} by radius  ' + '  ·  '.join(bins))

print('\ncommon = mean per-frame displacement, all subjects together -> the camera')
print('scatter = mean spread about it                              -> grounding or association')
