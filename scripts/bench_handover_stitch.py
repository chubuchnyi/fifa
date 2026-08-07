"""Why does the 2D stitcher not join the pairs the eye calls one player? (W3 / П3)

The user judged every track of `out/cue/scene_off.json` by eye on 2026-08-07 and named three
pairs that are one human each: **3+66, 10+77, 15+71** (the last disputed, geometry prefers
15+25). `scripts/track_quality.py` finds those handovers *after* pose, in metres, off the
`provenance` timeline. The pipeline's own stitcher runs **before** pose, in pixels — and it
merged none of them.

This replays the tracker over the cached detections, then reports, for every pair the eye named,
which gate in `core/orchestration/continuity.py` rejects the link and by how much. A gate that
rejects by 3 % is a threshold; a gate that rejects structurally is a design decision.

    PYTHONPATH=src .venv/bin/python scripts/bench_handover_stitch.py
    PYTHONPATH=src .venv/bin/python scripts/bench_handover_stitch.py --frames 236

Reads `out/phmr_ab/dets_coco_0_236.npz` (cached RF-DETR output), so it needs no GPU and no video.
"""
from __future__ import annotations

import argparse
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
ap.add_argument('--frames', type=int, default=60)
ap.add_argument('--pairs', default='3-66,10-77,15-71,15-25',
                help='the eye\'s stitch list, as head-tail ids')
ap.add_argument('--no-kit-split', dest='kit_split', action='store_false',
                help='the wiring default is ON (#132); turn it off to see the uncut tracks')
args = ap.parse_args()

c = np.load(REPO / args.dets, allow_pickle=True)
dets = Detections(frames=[
    FrameDetections(frame=int(f), items=[
        Detection(bbox_xyxy=b, cls=str(k), score=float(s))
        for b, k, s in zip(bb, kk, ss, strict=True)])
    for f, bb, kk, ss in zip(c['frame'], c['boxes'], c['classes'], c['scores'], strict=True)
    if int(f) < args.frames])

W, H = 1920, 1080
clip = ClipRef(source_id='colombia', uri=str(REPO / 'samples/video/Colombia-1-0-Congo-DR1080p.mp4'),
               frames=np.arange(args.frames), width=W, height=H, fps=29.97)

# The wiring's own parameters (`app/wiring.py`), so this is the tracker the pipeline builds.
tracker = ByteTrackTracker(device='cpu', min_track_frames=2, kit_split=args.kit_split)
tracks = tracker.track(clip, dets)
tls = {int(t.track_id): t for t in tracks.tracklets}
print(f'{len(tls)} tracklets over {args.frames} frames (kit_split '
      f'{"ON" if args.kit_split else "OFF"}): {sorted(tls)}')
print('  spans: ' + '  '.join(
    f't{tid}:f{int(np.min(t.frames))}-{int(np.max(t.frames))}({len(np.asarray(t.frames).reshape(-1))})'
    for tid, t in sorted(tls.items())))

cfg = StitchConfig()
summaries = {tid: _summarize(t, cfg.velocity_window) for tid, t in tls.items()}


def report(a_id: int, b_id: int) -> None:
    if a_id not in summaries or b_id not in summaries:
        print(f'  t{a_id} -> t{b_id}: not both present in this run')
        return
    a, b = summaries[a_id], summaries[b_id]
    gap = b.start_frame - a.end_frame - 1
    fa = set(np.asarray(tls[a_id].frames, dtype=int).reshape(-1).tolist())
    fb = set(np.asarray(tls[b_id].frames, dtype=int).reshape(-1).tolist())
    shared = sorted(fa & fb)
    print(f'\n  t{a_id} (f{a.start_frame}-{a.end_frame}, {a.n_frames} frames, team {a.team_id}) '
          f'-> t{b_id} (f{b.start_frame}-{b.end_frame}, {b.n_frames} frames, team {b.team_id})')
    print(f'    gap {gap:+d}   shared frames {len(shared)}'
          + (f' {shared[:8]}' if shared else ''))

    verdicts = []
    if gap < 0:
        verdicts.append(f'GAP<0 rejects: the two overlap on {len(shared)} measured frame(s)')
    elif gap > cfg.max_gap:
        verdicts.append(f'GAP rejects: {gap} > max_gap {cfg.max_gap}')
    if cfg.require_same_cls and a.cls != b.cls:
        verdicts.append(f'CLS rejects: {a.cls} != {b.cls}')
    if (cfg.require_same_team and a.team_id is not None and b.team_id is not None
            and a.team_id != b.team_id):
        verdicts.append(f'TEAM rejects: {a.team_id} != {b.team_id}')
    for name, sa, sb in (('w', a.mean_w, b.mean_w), ('h', a.mean_h, b.mean_h)):
        lo, hi = min(sa, sb), max(sa, sb)
        r = hi / max(lo, 1e-6)
        flag = ' <== rejects' if r > cfg.max_size_ratio else ''
        print(f'    size {name}: {sa:6.1f} vs {sb:6.1f} px  ratio {r:4.2f} '
              f'(max {cfg.max_size_ratio}){flag}')
        if r > cfg.max_size_ratio:
            verdicts.append(f'SIZE {name} rejects: ratio {r:.2f}')

    dt = max(b.start_frame - a.end_frame, 1)
    predicted = a.end_center + a.velocity * dt
    dist = float(np.linalg.norm(predicted - b.start_center))
    scale = cfg.max_center_dist * (a.mean_w + b.mean_w) / 2.0
    print(f'    centre: predicted {predicted.round(1)} vs actual {b.start_center.round(1)}  '
          f'-> {dist:6.1f} px  (budget {scale:.1f} px)'
          + ('  <== rejects' if dist > scale else ''))
    if dist > scale:
        verdicts.append(f'CENTRE rejects: {dist:.1f} px > {scale:.1f} px')

    # How coincident are the two boxes on the frames they share? Two ids on ONE human overlap
    # almost exactly; two humans standing close do not. This is the guard an overlap-tolerant
    # link would need, so measure it whether or not the gap gate fires.
    if shared:
        ia = {int(f): k for k, f in enumerate(np.asarray(tls[a_id].frames, dtype=int).reshape(-1))}
        ib = {int(f): k for k, f in enumerate(np.asarray(tls[b_id].frames, dtype=int).reshape(-1))}
        ba = np.asarray(tls[a_id].bboxes_xyxy, dtype=float).reshape(-1, 4)
        bb_ = np.asarray(tls[b_id].bboxes_xyxy, dtype=float).reshape(-1, 4)
        ious = []
        for f in shared:
            p, q = ba[ia[f]], bb_[ib[f]]
            ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
            iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
            inter = ix * iy
            union = ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter)
            ious.append(inter / union if union > 0 else 0.0)
        print(f'    shared-frame IoU: min {min(ious):.2f} mean {np.mean(ious):.2f} '
              f'max {max(ious):.2f}   -> {"ONE human" if np.mean(ious) > 0.5 else "two bodies"}')

    print('    ' + ('VERDICT: ' + ' | '.join(verdicts) if verdicts else 'VERDICT: would link'))


print('\n== the eye\'s pairs, against the shipped StitchConfig ==')
for spec in args.pairs.split(','):
    x, y = spec.split('-')
    report(int(x), int(y))

out, rep = stitch_tracks_with_report(tracks, cfg)
print(f'\n== shipped stitch: {rep.n_in} -> {rep.n_out} tracklets, '
      f'{len(rep.merges)} merges {rep.merges}, dropped {rep.dropped}')
