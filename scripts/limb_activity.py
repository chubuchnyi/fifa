"""Does a player move his limbs while he moves? — the phantom test the user's eye found.

From the 2026-08-06 eye review (`eye-review-2026-08-06.md`):

> до пересечения скелет 3 фактически бежал, а 66 перемещался не двигая конечностями … после
> пересечения наоборот … Позиция на поле верная у того, который двигает конечностями. И вообще
> если нет движения конечностями совсем, то это признак того, что человек — фантом.

Two claims worth measuring, and neither needs a new model — the joint angles are already in the
scene:

1. **Translating without articulating is a phantom signature.** Standing still is not: a player
   waiting for the ball has low limb activity *and* low speed. The discriminating quantity is the
   pair, so this reports **speed against activity**, never activity alone.
2. **At a crossing the limb-active track holds the true position.** If so, a track going still at
   the same frame another comes alive is a handover between two ids of one human — a stitching cue
   from a quantity we already have.

    .venv/bin/python scripts/limb_activity.py --scene out/cue/scene_off.json
    .venv/bin/python scripts/limb_activity.py --scene out/cue/scene_off.json --pair 3 66

Activity is the per-frame sum of absolute body-joint angle change (radians/frame over 21 joints),
which is what `config/physics.yaml` half-names as `pose_motion_sync.joint_activity_threshold`.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scene', default='out/cue/scene_off.json')
parser.add_argument('--fps', type=float, default=29.97)
parser.add_argument('--pair', type=int, nargs=2, default=None, metavar=('A', 'B'),
                    help='trace two tracks frame by frame — the handover the eye described')
parser.add_argument('--moving-mps', type=float, default=1.0,
                    help='speed above which a subject counts as translating')
parser.add_argument('--still-rad', type=float, default=0.15,
                    help='per-frame limb activity below which a subject counts as unarticulated')
args = parser.parse_args()


def _unwrap(o):
    if isinstance(o, dict):
        if '__type__' in o and 'fields' in o:
            return _unwrap(o['fields'])
        return {k: _unwrap(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_unwrap(x) for x in o]
    return o


def _arr(v):
    nd = v['__ndarray__'] if isinstance(v, dict) and '__ndarray__' in v else v
    if isinstance(nd, dict) and 'data' in nd:
        return np.asarray(nd['data'], float).reshape(nd.get('shape') or -1)
    return np.asarray(nd, float)


scene = _unwrap(json.loads((REPO / args.scene).read_text()))
subjects = scene['subjects']
print(f'{args.scene}: {len(subjects)} subjects')

tracks = {}
for s in subjects:
    pose = s['proposal']['pose']
    fr = _arr(pose['frames']).astype(int)
    body = _arr(pose['body_pose']).reshape(len(fr), -1, 3)
    tr = _arr(pose['transl']).reshape(len(fr), 3)
    if len(fr) < 3:
        continue
    dt = np.diff(fr) / args.fps
    dt[dt <= 0] = 1.0 / args.fps
    # Limb activity: how much the ARTICULATION changed, root orientation excluded on purpose —
    # a body carried along by a moving box still has a changing global orient.
    act = np.abs(np.diff(body, axis=0)).sum(axis=(1, 2)) / dt
    spd = np.linalg.norm(np.diff(tr[:, :2], axis=0), axis=1) / dt
    tracks[int(s['track_id'])] = {'frames': fr[1:], 'act': act, 'spd': spd,
                                  'team': s.get('team_id')}

print(f'\n{"tid":>4} {"n":>4} {"team":>5} {"speed p50":>10} {"speed p90":>10} '
      f'{"limb p50":>9} {"limb p90":>9} {"moving-but-still":>17}')
phantoms = []
for tid, t in sorted(tracks.items()):
    moving = t['spd'] > args.moving_mps
    frozen_while_moving = float((moving & (t['act'] < args.still_rad)).sum()) / max(1, moving.sum())
    flag = ''
    if moving.sum() >= 5 and frozen_while_moving > 0.5:
        flag = '  <-- PHANTOM?'
        phantoms.append(tid)
    print(f'{tid:4d} {len(t["frames"]):4d} {str(t["team"]):>5} '
          f'{np.percentile(t["spd"], 50):10.2f} {np.percentile(t["spd"], 90):10.2f} '
          f'{np.percentile(t["act"], 50):9.2f} {np.percentile(t["act"], 90):9.2f} '
          f'{100 * frozen_while_moving:16.0f}%{flag}')

print(f'\n{len(phantoms)} track(s) translate above {args.moving_mps} m/s while their limbs stay '
      f'under {args.still_rad} rad/frame for most of it: {phantoms}')
print('A standing player is NOT flagged: he fails the speed test, which is why the pair is used.')

if args.pair:
    a, b = args.pair
    if a not in tracks or b not in tracks:
        raise SystemExit(f'need both tracks in this scene; have {sorted(tracks)}')
    ta, tb = tracks[a], tracks[b]
    common = sorted(set(ta['frames'].tolist()) & set(tb['frames'].tolist()))
    print(f'\ntracks {a} and {b}: {len(common)} shared frame(s)')
    print(f'{"frame":>6} {f"{a} spd":>8} {f"{a} limb":>9} {f"{b} spd":>8} {f"{b} limb":>9}  '
          f'who is articulating')
    ia = {f: i for i, f in enumerate(ta['frames'].tolist())}
    ib = {f: i for i, f in enumerate(tb['frames'].tolist())}
    for f in common:
        sa, la = ta['spd'][ia[f]], ta['act'][ia[f]]
        sb, lb = tb['spd'][ib[f]], tb['act'][ib[f]]
        who = a if la > lb * 1.5 else (b if lb > la * 1.5 else '~')
        print(f'{f:6d} {sa:8.2f} {la:9.2f} {sb:8.2f} {lb:9.2f}  {who}')
