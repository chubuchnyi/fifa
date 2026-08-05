"""Did the #132 kit split make the render better or just busier? — the eye's arithmetic.

Splitting a track that changed player is correct by construction, but correctness is not the bar
here: the bar is the picture. More identities means more avatars, and a *short* identity is one
that pops into the render and vanishes — visible churn that could cost more than the wrong-kit
frames the split removes.

    .venv/bin/python scripts/ab_subject_stability.py out/ab_nosplit out/ab_split

So this compares two exported scenes on the things that actually show:

* **subject count** — how many avatars the render has to carry.
* **frames per subject** — the churn measure. A subject alive for 4 of 48 frames pops.
* **coverage** — total subject-frames. If the split only *relabels*, this barely moves; if it
  is dropping or duplicating work, it moves a lot.

It reads the exported `scene.json`, i.e. what the renderer is actually handed, not an
intermediate the pipeline might still change.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('arms', nargs=2, metavar=('BEFORE', 'AFTER'))
parser.add_argument('--short', type=int, default=8,
                    help='subjects alive fewer frames than this are counted as churn')
args = parser.parse_args()


def _unwrap(o):
    """The export wraps every dataclass as ``{"__type__": ..., "fields": {...}}``. Strip that."""
    if isinstance(o, dict):
        if '__type__' in o and 'fields' in o:
            return _unwrap(o['fields'])
        return {k: _unwrap(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_unwrap(x) for x in o]
    return o


def _array(v):
    """Canonical-JSON arrays land as ``{"__ndarray__"/"data": [...]}``; length is all we need."""
    if isinstance(v, dict):
        for key in ('data', '__ndarray__', 'values'):
            if key in v:
                return v[key]
        return next(iter(v.values()), [])
    return v or []


def load(run_dir):
    path = Path(run_dir) / 'export' / 'scene.json'
    if not path.exists():
        raise SystemExit(f'no export at {path}')
    scene = _unwrap(json.loads(path.read_text()))
    out = {}
    for s in scene.get('subjects', []):
        frames = _array(s.get('proposal', {}).get('pose', {}).get('frames'))
        out[str(s.get('track_id'))] = {
            'n': len(frames),
            'team': s.get('team_id'),
            'role': s.get('role'),
        }
    if not out:
        raise SystemExit(f'{path} parsed but held no subjects — the export schema moved')
    return out


rows = []
for arm in args.arms:
    subs = load(arm)
    lens = sorted(v['n'] for v in subs.values())
    # A fake pose adapter emits a fixed handful of keyframes per subject, so every subject comes
    # back the same tiny length and this script would report "identical" for *any* input. Refuse
    # to give a verdict on that rather than give a confident wrong one.
    if len(set(lens)) == 1 and lens and lens[0] <= 5:
        raise SystemExit(
            f'{arm}: every subject has exactly {lens[0]} pose frames — that is a fake pose '
            'adapter, not a measurement. Re-run with a real --pose backend, or compare at the '
            'tracklet level instead.'
        )
    rows.append((arm, subs, lens))

print(f'{"run":<22} {"subjects":>9} {"frames/subj":>22} {"short":>7} {"coverage":>9}')
for arm, subs, lens in rows:
    short = sum(1 for n in lens if n < args.short)
    lo, med, hi = (lens[0], lens[len(lens) // 2], lens[-1]) if lens else (0, 0, 0)
    span = f'min {lo} med {med} max {hi}'
    print(f'{arm:<22} {len(subs):9d} {span:>22} {short:7d} {sum(lens):9d}')

before, after = rows[0][1], rows[1][1]
print(f'\nlengths BEFORE: {rows[0][2]}')
print(f'lengths AFTER : {rows[1][2]}')

for name, subs in ((args.arms[0], before), (args.arms[1], after)):
    teams = Counter(v['team'] for v in subs.values())
    print(f'{name:<22} teams {dict(teams)}')

d = len(after) - len(before)
cov = sum(rows[1][2]) - sum(rows[0][2])
print(f'\nverdict inputs: {d:+d} subjects, {cov:+d} subject-frames, '
      f'short subjects {sum(1 for n in rows[0][2] if n < args.short)} -> '
      f'{sum(1 for n in rows[1][2] if n < args.short)}')
print('A split that only relabels moves coverage ~0. A split that shreds adds short subjects.')
