"""Why does the 2D stitcher not join the pairs the eye calls one player? (W3 / П3)

The user judged every track of `out/cue/scene_off.json` by eye on 2026-08-07 and named three
pairs that are one human each. `scripts/track_quality.py` finds those handovers *after* pose, in
metres, off the `provenance` timeline, and scores **20/21** against the eye. The pipeline's own
stitcher runs **before** pose, in pixels — and merged none of them.

This replays the tracker over cached detections and asks the question in one place:

1. **Which gate rejects each pair the eye named** (`--pairs`), and by how much. A gate that
   rejects by 3 % is a threshold; a gate that rejects structurally is a design decision.
2. **What does П3 find if you run it pre-pose?** The measured camera un-projects each tracklet's
   endpoint box to the pitch, so the same endpoint-distance-in-metres rule the post-pose criterion
   uses can run on 2D tracks — no pose, no GPU.
3. **Is each candidate merge right?** Scored against the *video pixels*, not a label file: a merge
   that joins two humans joins two shirts. At our subject size the shirt is the only appearance
   signal that survives (28 x 72 px player = ~573 px of shirt shared by eleven teammates), so it
   cannot give identity — but it is decisive for "these two are not the same man".

    PYTHONPATH=src .venv/bin/python scripts/bench_handover_stitch.py
    PYTHONPATH=src .venv/bin/python scripts/bench_handover_stitch.py --propose

Reads `out/phmr_ab/dets_coco_0_236.npz` (cached RF-DETR output) and
`calib/Colombia-1-0-Congo-DR1080p.npz` (the measured 60-frame camera), so it needs no GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from pitch3d.adapters.models.tracking import ByteTrackTracker  # noqa: E402
from pitch3d.core.orchestration.continuity import (  # noqa: E402
    StitchConfig,
    _summarize,
    stitch_tracks_with_report,
)
from pitch3d.core.ports.io import ClipRef  # noqa: E402
from pitch3d.core.ports.perception import (  # noqa: E402
    Detection,
    Detections,
    FrameDetections,
)

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--dets', default='out/phmr_ab/dets_coco_0_236.npz')
ap.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
ap.add_argument('--camera', default='calib/Colombia-1-0-Congo-DR1080p.npz')
ap.add_argument('--frames', type=int, default=60)
ap.add_argument('--pairs', default='3-66,10-77,15-71,15-25',
                help="the eye's stitch list, as head-tail ids")
ap.add_argument('--no-kit-split', dest='kit_split', action='store_false',
                help='the wiring default is ON (#132); turn it off to see the uncut tracks')
ap.add_argument('--max-gap', type=int, default=14, help='П3 handover frame gap')
ap.add_argument('--max-dist', type=float, default=6.0, help='П3 handover metres')
ap.add_argument('--max-both', type=int, default=4,
                help='simultaneous measured frames above which two tracks are two humans')
ap.add_argument('--propose', action='store_true',
                help='diff a relaxed StitchConfig against the shipped one')
args = ap.parse_args()

c = np.load(REPO / args.dets, allow_pickle=True)
dets = Detections(frames=[
    FrameDetections(frame=int(f), items=[
        Detection(bbox_xyxy=b, cls=str(k), score=float(s))
        for b, k, s in zip(bb, kk, ss, strict=True)])
    for f, bb, kk, ss in zip(c['frame'], c['boxes'], c['classes'], c['scores'], strict=True)
    if int(f) < args.frames])

W, H = 1920, 1080
clip = ClipRef(source_id='colombia', uri=str(REPO / args.clip),
               frames=np.arange(args.frames), width=W, height=H, fps=29.97)

# The wiring's own parameters (`app/wiring.py`), so this is the tracker the pipeline builds.
tracker = ByteTrackTracker(device='cpu', min_track_frames=2, kit_split=args.kit_split)
tracks = tracker.track(clip, dets)
tls = {int(t.track_id): t for t in tracks.tracklets}
frames_of = {tid: np.asarray(t.frames, dtype=int).reshape(-1) for tid, t in tls.items()}
boxes_of = {tid: np.asarray(t.bboxes_xyxy, dtype=float).reshape(-1, 4) for tid, t in tls.items()}
print(f'{len(tls)} tracklets over {args.frames} frames '
      f'(kit_split {"ON" if args.kit_split else "OFF"})')


# ------------------------------------------------------------------ pitch metres, no pose
def _ground_projector(npz: Path):
    """(frame, u, v) -> (X, Y) on the pitch plane, from the measured one-camera fit (#119/#129).

    The stitcher's own gate is a *predicted* pixel position after constant-velocity
    extrapolation. Over a 16-28 frame gap that diverges — which is most of why it misses long
    handovers — so this measures the thing the post-pose criterion measures instead: how far
    apart the two endpoints are **on the grass**.
    """
    spec = importlib.util.spec_from_file_location('arc', REPO / 'scripts' / 'apply_rigid_camera.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    blob = np.load(npz, allow_pickle=True)
    focal = float(blob['focal'])
    centre = np.asarray(blob['centre'], dtype=float)
    rots = {int(f): mod.rot_from_rvec(blob['rvecs'][i])
            for i, f in enumerate(np.asarray(blob['frames']).astype(int))}
    # The solved camera is 180-degree rolled on some clips; auto-detect exactly as the repo does.
    any_rot = next(iter(rots.values()))
    rolled = -any_rot[1][2] < 0

    def to_pitch(frame: int, u: float, v: float):
        rot = rots.get(int(frame))
        if rot is None:
            return None
        if rolled:
            u, v = W - u, H - v
        d_cam = np.array([(u - W / 2.0) / focal, (v - H / 2.0) / focal, 1.0])
        d = rot.T @ d_cam
        if abs(d[2]) < 1e-9:
            return None
        s = -centre[2] / d[2]
        if s <= 0:
            return None
        return (centre + s * d)[:2]

    to_pitch.rolled = rolled
    return to_pitch


to_pitch = _ground_projector(REPO / args.camera)
print(f'camera {args.camera}: 180-degree roll {"DETECTED" if to_pitch.rolled else "absent"}')


def foot_metres(tid: int, k: int):
    """Where on the pitch is this tracklet's k-th box standing? Bottom-centre = the feet."""
    b = boxes_of[tid][k]
    return to_pitch(int(frames_of[tid][k]), (b[0] + b[2]) / 2.0, b[3])


# ------------------------------------------------------------------ shirt colour, from pixels
def _kit_reader():
    import cv2
    cap = cv2.VideoCapture(str(REPO / args.clip))
    cache: dict[int, object] = {}

    def img(i: int):
        if i not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, im = cap.read()
            cache[i] = im if ok else None
        return cache[i]

    def kit(tid: int, k: int) -> str:
        f = int(frames_of[tid][k])
        x0, y0, x1, y1 = boxes_of[tid][k]
        h, w = y1 - y0, x1 - x0
        a, b = int(y0 + 0.20 * h), int(y0 + 0.45 * h)   # below the head, above the shorts
        c0, c1 = int(x0 + 0.25 * w), int(x0 + 0.75 * w)
        im = img(f)
        if im is None or b <= a or c1 <= c0 or min(a, c0) < 0:
            return '?'
        patch = im[a:b, c0:c1]
        if patch.size == 0:
            return '?'
        med = np.uint8([[np.median(patch.reshape(-1, 3), axis=0)]])
        hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)[0][0]
        if 18 <= hsv[0] <= 48 and hsv[1] > 90:
            return 'Y'
        if 85 <= hsv[0] <= 135 and hsv[1] > 55:
            return 'B'
        return '?'

    def kit_run(tid: int, at_end: bool, n: int = 6) -> str:
        """Modal shirt over the last / first `n` boxes — one occluded frame must not decide."""
        idx = range(max(0, len(frames_of[tid]) - n), len(frames_of[tid])) if at_end else range(
            min(n, len(frames_of[tid])))
        s = [kit(tid, k) for k in idx]
        y, b = s.count('Y'), s.count('B')
        return 'Y' if y > b else ('B' if b > y else '?')

    kit_run.close = cap.release
    return kit_run


kit_run = _kit_reader()


# ------------------------------------------------------------- 1. the eye's pairs, gate by gate
cfg = StitchConfig()
summaries = {tid: _summarize(t, cfg.velocity_window) for tid, t in tls.items()}


def diagnose(a_id: int, b_id: int) -> None:
    if a_id not in summaries or b_id not in summaries:
        print(f'  t{a_id} -> t{b_id}: not both present in this run')
        return
    a, b = summaries[a_id], summaries[b_id]
    gap = b.start_frame - a.end_frame - 1
    shared = sorted(set(frames_of[a_id].tolist()) & set(frames_of[b_id].tolist()))
    print(f'\n  t{a_id} (f{a.start_frame}-{a.end_frame}, team {a.team_id}) '
          f'-> t{b_id} (f{b.start_frame}-{b.end_frame}, team {b.team_id})   '
          f'gap {gap:+d}, shared {len(shared)}')

    why = []
    if gap < 0:
        why.append(f'GAP<0 ({len(shared)} measured frames overlap)')
    elif gap > cfg.max_gap:
        why.append(f'GAP {gap} > {cfg.max_gap}')
    for name, sa, sb in (('w', a.mean_w, b.mean_w), ('h', a.mean_h, b.mean_h)):
        r = max(sa, sb) / max(min(sa, sb), 1e-6)
        if r > cfg.max_size_ratio:
            why.append(f'SIZE {name} {r:.2f} > {cfg.max_size_ratio}')
    dt = max(b.start_frame - a.end_frame, 1)
    dist = float(np.linalg.norm(a.end_center + a.velocity * dt - b.start_center))
    scale = cfg.max_center_dist * (a.mean_w + b.mean_w) / 2.0
    if dist > scale:
        why.append(f'CENTRE {dist:.0f} px > {scale:.0f} px')
    here, there = foot_metres(a_id, len(frames_of[a_id]) - 1), foot_metres(b_id, 0)
    both_seen = here is not None and there is not None
    m = float(np.linalg.norm(here - there)) if both_seen else float('nan')
    print(f'    extrapolated {dist:6.1f} px (budget {scale:.0f})   '
          f'endpoint-to-endpoint on the pitch {m:5.2f} m   '
          f'kit {kit_run(a_id, True)} -> {kit_run(b_id, False)}')
    print('    ' + ('rejected by: ' + ' | '.join(why) if why else 'WOULD LINK'))


print("\n== the eye's pairs, against the shipped StitchConfig ==")
for spec in args.pairs.split(','):
    x, y = spec.split('-')
    diagnose(int(x), int(y))

# ------------------------------------------------------------------ 2. П3, run pre-pose
print(f'\n== П3 handovers found pre-pose (|gap| <= {args.max_gap}, <= {args.max_dist} m, '
      f'<= {args.max_both} simultaneous measured frames) ==')
cands = []
for h in tls:
    for t in tls:
        if h == t:
            continue
        die, born = int(frames_of[h][-1]), int(frames_of[t][0])
        gap = born - die
        if abs(gap) > args.max_gap:
            continue
        both = len(set(frames_of[h].tolist()) & set(frames_of[t].tolist()))
        if both > args.max_both:
            continue
        here, there = foot_metres(h, len(frames_of[h]) - 1), foot_metres(t, 0)
        if here is None or there is None:
            continue
        d = float(np.linalg.norm(here - there))
        if d > args.max_dist:
            continue
        cands.append({'head': h, 'tail': t, 'gap': gap, 'dist': d, 'both': both})

taken, pairs = set(), []
for cd in sorted(cands, key=lambda x: x['dist']):
    if cd['head'] in taken or cd['tail'] in taken:
        continue
    taken.add(cd['head'])
    taken.add(cd['tail'])
    pairs.append(cd)

accepted = {(p['head'], p['tail']) for p in pairs}
n_ok = n_clash = 0
for cd in sorted(cands, key=lambda x: x['dist']):
    ka, kb = kit_run(cd['head'], True), kit_run(cd['tail'], False)
    verdict = 'same kit' if ka == kb != '?' else ('KIT CLASH' if '?' not in (ka, kb) else 'unclear')
    mark = 'x' if (cd['head'], cd['tail']) in accepted else ' '
    if mark == 'x':
        n_ok += verdict == 'same kit'
        n_clash += verdict == 'KIT CLASH'
    print(f'  {mark} t{cd["head"]:<3d} -> t{cd["tail"]:<3d}  {cd["dist"]:5.2f} m  '
          f'gap {cd["gap"]:+3d}  both {cd["both"]}  kit {ka}->{kb}  {verdict}')
if not cands:
    print('  (none)')
print(f'  assignment accepts {len(pairs)}: {n_ok} same-kit, {n_clash} kit clash, '
      f'{len(pairs) - n_ok - n_clash} unclear')

# ------------------------------------------------------------------ 3. what the shipped stitch does
_out, rep = stitch_tracks_with_report(tracks, cfg)
print(f'\n== shipped stitch: {rep.n_in} -> {rep.n_out}, {len(rep.merges)} merges {rep.merges}, '
      f'dropped {rep.dropped}')
shipped = {tuple(sorted(m)) for m in rep.merges}
found = {tuple(sorted((p['head'], p['tail']))) for p in pairs}
print(f'  П3 pairs the shipped stitch already covers: '
      f'{sorted(p for p in found if any(set(p) <= set(m) for m in shipped))}')
print(f'  П3 pairs it misses:                        '
      f'{sorted(p for p in found if not any(set(p) <= set(m) for m in shipped))}')
kit_run.close()
