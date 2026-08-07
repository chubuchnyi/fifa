"""Is this reconstructed player correct? — the criteria, measured, and scored against the eye.

The user judged every track of the reference reconstruction by eye on 2026-08-07 (verbatim in
`docs/findings/track-correctness-criteria-2026-08-07.md`) and then asked for the *признаки* — the
features by which we decide a player is correct — to be written down. This script is the measured
half of that document: it computes the features and, given the eye labels, reports where they
agree and where they do not.

Everything here rests on one mechanical fact about the pipeline, verified below by
``--explain-imputed``: a pose frame marked ``imputed`` has **exactly zero** limb articulation
change while the root keeps coasting. So an imputed run is a sliding mannequin — which is what the
eye reads as a phantom ("перемещался не двигая конечностями"). ``interpolated`` is a different
animal: anchored on both sides, it carries real limb motion and is not a defect.

The features, in the order the report prints them:

* **shape** — where the measured frames sit. ``FULL`` (whole clip), ``HEAD`` (dies mid-clip),
  ``TAIL`` (born mid-clip), ``CORE`` (imputed at both ends).
* **in-frame** — does the root project inside the image? An imputed run **off** frame is correct
  behaviour (nobody can measure a player who left the picture, and R-6 says mark, never hide); the
  same run **in** frame is a phantom.
* **handover** — a HEAD that dies where a TAIL is born, same team, small frame gap: one human,
  two ids. That is a stitch, not two players.
* **twin** — two tracks closer than a body width for many frames: at most one of them is real.
* **vertical** — root Z range. A scene where nobody leaves the ground cannot reconstruct a jump.

    .venv/bin/python scripts/track_quality.py --scene out/cue/scene_off.json
    .venv/bin/python scripts/track_quality.py --scene out/cue/scene_off.json \
        --camera calib/Colombia-1-0-Congo-DR1080p.npz --labels docs/findings/track-labels.json
    .venv/bin/python scripts/track_quality.py --scene out/cue/scene_off.json --explain-imputed

**Use --camera.** A scene reconstructed without the #129 rigid fit stores an *invented* fallback
camera (772 px @ 1280x720, principal point dead centre) whose field of view is so wide that every
subject lands inside the image — which silently turns the in-frame test into a constant. The
script says so when it detects one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

SYM = {'measured': 'M', 'imputed': '.', 'interpolated': '~'}
FALLBACK_FOCAL = 772.0          # controller.py's invented camera; see the docstring


def _unwrap(o):
    if isinstance(o, dict):
        if '__type__' in o and 'fields' in o:
            return _unwrap(o['fields'])
        if '__enum__' in o:
            return o['__enum__']['value']
        if '__tuple__' in o:
            return tuple(_unwrap(x) for x in o['__tuple__'])
        return {k: _unwrap(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_unwrap(x) for x in o]
    return o


def _arr(v):
    nd = v['__ndarray__'] if isinstance(v, dict) and '__ndarray__' in v else v
    if isinstance(nd, dict) and 'data' in nd:
        return np.asarray(nd['data']).reshape(nd.get('shape') or -1)
    return np.asarray(nd)


def load_scene(path: Path) -> dict:
    scene = _unwrap(json.loads(path.read_text()))
    tracks = {}
    for s in scene['subjects']:
        pose = s['proposal']['pose']
        frames = _arr(pose['frames']).astype(int)
        tracks[int(s['track_id'])] = {
            'frames': frames,
            'prov': [str(x) for x in _arr(pose['provenance'])],
            'transl': _arr(pose['transl']).astype(float).reshape(len(frames), 3),
            'body': _arr(pose['body_pose']).astype(float).reshape(len(frames), -1, 3),
            'team': s.get('team_id'),
            'role': s.get('role'),
        }
    return {'tracks': tracks, 'camera': scene['camera']}


def projector_from_scene(camera: dict):
    """(frame, xyz) -> (u, v, width, height) using the camera stored in the scene."""
    from pitch3d.core.scene.projection import quat_to_rotation_matrix
    k = camera['intrinsics']
    frames = _arr(camera['frames']).astype(int)
    quat = _arr(camera['rotation_quat']).astype(float)
    transl = _arr(camera['translation']).astype(float)
    poses = {int(f): (quat_to_rotation_matrix(quat[i]), transl[i]) for i, f in enumerate(frames)}
    invented = abs(k['fx'] - FALLBACK_FOCAL) < 1.0
    return _make_projector(poses, k['fx'], k['fy'], k['cx'], k['cy'], k['width'], k['height'],
                           invented)


def projector_from_npz(path: Path, width: int, height: int):
    """(frame, xyz) -> (u, v, w, h) using the measured one-camera fit (#119 / #129)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('arc', REPO / 'scripts' / 'apply_rigid_camera.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    blob = np.load(path, allow_pickle=True)
    focal = float(blob['focal'])
    centre = np.asarray(blob['centre'], dtype=float)
    frames = np.asarray(blob['frames']).astype(int)
    poses = {}
    for i, f in enumerate(frames):
        rot = mod.rot_from_rvec(blob['rvecs'][i])
        poses[int(f)] = (rot, -rot @ centre)
    return _make_projector(poses, focal, focal, width / 2.0, height / 2.0, width, height, False)


def _make_projector(poses, fx, fy, cx, cy, width, height, invented):
    def project(frame: int, xyz: np.ndarray):
        pose = poses.get(int(frame))
        if pose is None:
            return None
        rot, t = pose
        cam = rot @ np.asarray(xyz, dtype=float) + t
        if cam[2] <= 1e-6:
            return None
        return fx * cam[0] / cam[2] + cx, fy * cam[1] / cam[2] + cy

    project.width, project.height, project.invented = width, height, invented
    project.focal = fx
    return project


def runs_of(prov: list[str]) -> list[tuple[str, int, int]]:
    """Contiguous provenance runs as ``(kind, first_index, last_index)``."""
    out: list[list] = []
    for i, p in enumerate(prov):
        if out and out[-1][0] == p:
            out[-1][2] = i
        else:
            out.append([p, i, i])
    return [(k, a, b) for k, a, b in out]


def classify(tid: int, track: dict, project, min_run: int, off_frac: float) -> dict:
    prov, frames, transl = track['prov'], track['frames'], track['transl']
    measured = [i for i, p in enumerate(prov) if p == 'measured']
    n = len(prov)
    inside = []
    for i, f in enumerate(frames):
        uv = project(int(f), transl[i])
        inside.append(uv is not None and 0 <= uv[0] < project.width and 0 <= uv[1] < project.height)

    if not measured:
        shape = 'GHOST'
    elif measured[0] <= 2 and measured[-1] >= n - 3:
        shape = 'FULL'
    elif measured[0] <= 2:
        shape = 'HEAD'
    elif measured[-1] >= n - 3:
        shape = 'TAIL'
    else:
        shape = 'CORE'

    bad, off = [], []
    for kind, a, b in runs_of(prov):
        if kind != 'imputed' or (b - a + 1) < min_run:
            continue
        out_frac = 1.0 - (sum(inside[a:b + 1]) / (b - a + 1))
        (off if out_frac >= off_frac else bad).append((a, b, out_frac))

    if not bad and not off:
        verdict = 'OK'
    elif not bad:
        verdict = 'OK_OFF_FRAME'
    else:
        verdict = 'PHANTOM'
    return {
        'tid': tid, 'shape': shape, 'verdict': verdict, 'n': n,
        'n_measured': len(measured), 'first_m': measured[0] if measured else None,
        'last_m': measured[-1] if measured else None,
        'timeline': ''.join(SYM.get(p, '?') if inside[i] else '_' for i, p in enumerate(prov)),
        'phantom_runs': bad, 'off_frame_runs': off,
        'in_frame_frac': sum(inside) / n, 'team': track['team'],
    }


def stitch_candidates(tracks: dict, verdicts: dict, max_gap: int, max_dist: float) -> list:
    heads = [t for t, v in verdicts.items() if v['shape'] == 'HEAD']
    tails = [t for t, v in verdicts.items() if v['shape'] == 'TAIL']
    out = []
    for h in heads:
        for t in tails:
            if tracks[h]['team'] != tracks[t]['team']:
                continue
            die, born = verdicts[h]['last_m'], verdicts[t]['first_m']
            gap = born - die
            if abs(gap) > max_gap:
                continue
            here, there = tracks[h]['transl'][die][:2], tracks[t]['transl'][born][:2]
            dist = float(np.linalg.norm(here - there))
            if dist > max_dist:
                continue
            out.append({'head': h, 'tail': t, 'gap': gap, 'dist': dist})
    return sorted(out, key=lambda c: c['dist'])


def twins(tracks: dict, radius: float, min_frames: int) -> list:
    ids = sorted(tracks)
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            fa = {int(f): k for k, f in enumerate(tracks[a]['frames'])}
            fb = {int(f): k for k, f in enumerate(tracks[b]['frames'])}
            common = sorted(set(fa) & set(fb))
            if not common:
                continue
            d = np.array([np.linalg.norm(tracks[a]['transl'][fa[f]][:2]
                                         - tracks[b]['transl'][fb[f]][:2]) for f in common])
            close = [f for f, dist in zip(common, d, strict=True) if dist < radius]
            if len(close) < min_frames:
                continue
            # Which of the two was *invented* while it stood inside the other? A body that
            # interpenetrates another only on its imputed frames is a phantom, not a collision.
            imp = {}
            for tid, idx in ((a, fa), (b, fb)):
                imp[tid] = sum(1 for f in close if tracks[tid]['prov'][idx[f]] == 'imputed')
            out.append({'a': a, 'b': b, 'frames': len(close), 'min': float(d.min()),
                        'imputed': imp})
    return sorted(out, key=lambda t: -t['frames'])


def explain_imputed(tracks: dict) -> None:
    print('\n== is an imputed run a frozen mannequin? per run: root travel vs limb travel ==')
    print(f'{"track":>6}  {"run":>12} {"root move":>10} {"limb move":>11}')
    for tid in sorted(tracks):
        tr = tracks[tid]
        for kind, a, b in runs_of(tr['prov']):
            if b == a:
                continue
            move = float(np.linalg.norm(np.diff(tr['transl'][a:b + 1, :2], axis=0), axis=1).sum())
            joints = float(np.abs(np.diff(tr['body'][a:b + 1], axis=0)).sum())
            print(f'{tid:6d}  {SYM[kind]}[{a:2d}-{b:2d}]{"":>4} {move:8.2f} m {joints:9.2f} rad')
    print('\nimputed runs carry 0.00 rad of limb motion by construction while the root keeps')
    print('moving; interpolated runs are anchored on both sides and DO articulate.')
    print('Only imputed is a phantom.')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', default='out/cue/scene_off.json')
    ap.add_argument('--camera', default=None,
                    help='measured camera npz (auto npz -> this flag -> the scene\'s own camera)')
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--labels', default=None, help='eye verdicts to score the criteria against')
    ap.add_argument('--min-run', type=int, default=6,
                    help='an imputed run shorter than this is not held against a track')
    ap.add_argument('--off-frac', type=float, default=0.5,
                    help='fraction of a run projecting outside the image to call it off-frame')
    ap.add_argument('--max-gap', type=int, default=14, help='handover frame gap for a stitch pair')
    ap.add_argument('--max-dist', type=float, default=6.0, help='handover metres for a stitch pair')
    ap.add_argument('--explain-imputed', action='store_true',
                    help='print the per-run root-vs-limb travel the criteria rest on')
    args = ap.parse_args()

    scene = load_scene(REPO / args.scene)
    tracks = scene['tracks']
    if args.camera:
        project = projector_from_npz(REPO / args.camera, args.width, args.height)
        print(f'{args.scene}: {len(tracks)} tracks · camera {args.camera} '
              f'(focal {project.focal:.0f} px @ {args.width}x{args.height})')
    else:
        project = projector_from_scene(scene['camera'])
        print(f'{args.scene}: {len(tracks)} tracks · camera from the scene '
              f'(focal {project.focal:.0f} px @ {project.width}x{project.height})')
        if project.invented:
            print('  !! that is the INVENTED fallback camera. Its field of view is so wide that '
                  'every\n     subject lands inside the image, so the in-frame test below is a '
                  'constant.\n     Pass --camera calib/<clip>.npz for a real answer.')

    verdicts = {tid: classify(tid, tr, project, args.min_run, args.off_frac)
                for tid, tr in sorted(tracks.items())}

    print('\n== provenance timeline (M measured · ~ interpolated · . imputed · '
          '_ root outside image)')
    print(f'{"track":>6} {"tm":>3} {"shape":>5} {"in%":>4}  timeline')
    for tid, v in verdicts.items():
        print(f'{tid:6d} {str(v["team"]):>3} {v["shape"]:>5} {100 * v["in_frame_frac"]:3.0f}%  '
              f'{v["timeline"]}')

    print('\n== verdict per track ==')
    for tid, v in verdicts.items():
        note = ''
        if v['phantom_runs']:
            note = 'in-frame imputed ' + ', '.join(f'f{a}-{b}' for a, b, _ in v['phantom_runs'])
        elif v['off_frame_runs']:
            note = 'left the picture ' + ', '.join(f'f{a}-{b}' for a, b, _ in v['off_frame_runs'])
        print(f'  t{tid:<3d} {v["verdict"]:<13} {v["shape"]:<5} '
              f'measured {v["n_measured"]:2d}/{v["n"]:2d}  {note}')

    cands = stitch_candidates(tracks, verdicts, args.max_gap, args.max_dist)
    print(f'\n== stitch candidates (HEAD dies where TAIL is born, same team, gap <= {args.max_gap} '
          f'frames, <= {args.max_dist} m) ==')
    for c in cands:
        print(f'  t{c["head"]} -> t{c["tail"]}   handover {c["dist"]:5.2f} m   frame gap '
              f'{c["gap"]:+3d}')
    if not cands:
        print('  (none)')

    tw = twins(tracks, radius=0.5, min_frames=3)
    print('\n== twins: two tracks inside 0.5 m of each other for 3+ frames '
          '(at most one is real) ==')
    for t in tw:
        imp = t['imputed']
        who = ', '.join(f't{k} imputed on {v}' for k, v in imp.items() if v)
        print(f'  t{t["a"]} / t{t["b"]}   {t["frames"]} frames closer than 0.5 m, '
              f'min {t["min"]:.2f} m' + (f'  ({who} of them)' if who else '  (both measured)'))
    if not tw:
        print('  (none)')

    zr = {tid: float(tr['transl'][:, 2].max() - tr['transl'][:, 2].min())
          for tid, tr in tracks.items()}
    top = max(zr.values())
    print(f'\n== vertical: largest root-Z excursion in the whole scene is {top:.3f} m '
          f'(t{max(zr, key=zr.get)})')
    print('   A jump moves a pelvis ~0.4 m. Under ~0.2 m here means the scene has no vertical DOF,')
    print('   so "player X did not jump" is a scene-wide property, not that track\'s defect.')

    if args.explain_imputed:
        explain_imputed(tracks)

    if args.labels:
        score(verdicts, cands, json.loads((REPO / args.labels).read_text()))


def score(verdicts: dict, cands: list, labels: dict) -> None:
    print('\n== criteria vs the eye ==')
    eye = {int(k): v for k, v in labels['tracks'].items()}
    agree, disagree, unjudged = [], [], []
    for tid, v in verdicts.items():
        want = eye.get(tid)
        if want is None:
            unjudged.append(tid)
            continue
        got = v['verdict']
        ok = ((want in ('correct', 'off_frame_ok') and got in ('OK', 'OK_OFF_FRAME'))
              or (want in ('stitch', 'misplaced') and got == 'PHANTOM'))
        (agree if ok else disagree).append((tid, want, got))
    print(f'  agree {len(agree)}/{len(agree) + len(disagree)} judged tracks; '
          f'{len(unjudged)} not judged by eye: {unjudged}')
    for tid, want, got in disagree:
        print(f'  DISAGREE t{tid}: eye says "{want}", criteria say {got}')
    pairs = {tuple(sorted(p)) for p in labels.get('stitch_pairs', [])}
    found = {tuple(sorted((c['head'], c['tail']))) for c in cands}
    print(f'  stitch pairs the eye named: {sorted(pairs)}')
    print(f'  found by the handover test:  {sorted(found)}')
    print(f'  matched {sorted(pairs & found)} · missed {sorted(pairs - found)} · '
          f'extra {sorted(found - pairs)}')


if __name__ == '__main__':
    main()
