"""Does Deep-EIoU's expansion trick fix OUR association failure? (#132/#133 follow-up)

Deep-EIoU (arXiv 2306.13074, WACV'24 RWS) drops the Kalman filter entirely and associates on
**ExpansionIoU**: both boxes are inflated by a scale factor before IoU, so two boxes that no longer
overlap after a fast nonlinear move still score above the match threshold. It plus OSNet won both
2025 soccer tracking challenges (SoccerTrack-2025 GTATrack, SoccerNet-2025 GSR KIST-GSR).

That is a direct answer to what we measured in `human-physics-requirement-2026-08-06.md`:

    96 % of mid-pitch identity births/deaths have an unclaimed detection a median 6-23 px away,
    and 72 % of those orphans score under 0.4 IoU -- under the 0.8 match threshold, so the match
    is refused and the detection is orphaned.

Inflating the boxes raises exactly that IoU. The full method also swaps in a sports-fine-tuned
ReID embedder, which we cannot evaluate without importing it -- but the *geometric* half costs one
patched function and needs no new weights, so it is measurable today and separates the two claims:
is the win in the expansion, or in the appearance features?

    .venv/bin/python scripts/bench_expansion_iou.py --frames 236

Scored the way #132 scores everything: identities before and after the stitcher, plus the seam
speed of every merge the expansion enables, because a wrong merge teleports a body.

Two properties of this path a reader would otherwise have to assume, both measured 2026-08-10
after the table below was read as confounded:

  * **Kit labels are present and the cue binds.** 0 of 56 player tracks are ``team_id=None`` here,
    and ``require_same_team`` costs 3 identities (36 vs 33). ``preflight()`` prints both, because
    ``StitchConfig.require_same_team`` treats ``None`` as a wildcard — a run whose tracker failed
    to label teams measures the geometry with the only 28-px-proof appearance cue switched off.
  * **There is no run noise.** Three independent runs at ``scale=1.0`` give 56 / 36 / 14 every
    time. So spread across the scale column is a deterministic non-monotonic response, not
    scatter, and must not be excused as noise.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from pitch3d.adapters.models.tracking import ByteTrackTracker  # noqa: E402
from pitch3d.core.orchestration.continuity import (  # noqa: E402
    StitchConfig,
    stitch_tracks_with_report,
)
from pitch3d.core.ports.io import ClipRef  # noqa: E402
from pitch3d.core.ports.perception import (  # noqa: E402
    Detection,
    Detections,
    FrameDetections,
    Tracks,
)

DEFAULT_SCALES = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def expand(boxes: np.ndarray, scale: float) -> np.ndarray:
    """Inflate each xyxy box about its own centre. ``scale=1.0`` is a no-op."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    cx, cy = (b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0
    hw, hh = (b[:, 2] - b[:, 0]) / 2.0 * scale, (b[:, 3] - b[:, 1]) / 2.0 * scale
    return np.stack([cx - hw, cy - hh, cx + hw, cy + hh], axis=1)


def patch_expansion(scale: float):
    """Wrap supervision's ``matching.iou_distance`` so the cost is computed on inflated boxes.

    Same seam the McByte mask cue uses (`ByteTrackTracker._patch_matching`): it is the one place
    where evidence can change a pairing without forking the validated tracker.
    """
    from supervision.tracker.byte_tracker import matching  # noqa: PLC0415

    original = matching.iou_distance

    def patched(atracks, btracks):
        if scale == 1.0 or not atracks or not btracks:
            return original(atracks, btracks)
        saved = []
        for t in list(atracks) + list(btracks):
            saved.append((t, np.array(t.tlbr, dtype=float)))
        try:
            for t, box in saved:
                grown = expand(box[None, :], scale)[0]
                # supervision's STrack stores `mean`/`tlwh`; write through the tlwh the property
                # derives from so `tlbr` reports the inflated box for this call only.
                t.mean = None if t.mean is None else t.mean
                t._tlwh = np.array(  # noqa: SLF001  (deliberate: the only writable seam)
                    [grown[0], grown[1], grown[2] - grown[0], grown[3] - grown[1]], dtype=float
                )
            return original(atracks, btracks)
        finally:
            for t, box in saved:
                t._tlwh = np.array(  # noqa: SLF001
                    [box[0], box[1], box[2] - box[0], box[3] - box[1]], dtype=float
                )

    matching.iou_distance = patched
    return lambda: setattr(matching, 'iou_distance', original)


def seam_speeds(tracks: Tracks, merges: list[list[int]]) -> np.ndarray:
    """px/frame implied by each bridged gap — the wrong-merge detector from `identity_budget`."""
    box = {(t.track_id, int(f)): np.asarray(b, float)
           for t in tracks.tracklets
           for f, b in zip(t.frames.tolist(), t.bboxes_xyxy, strict=True)}
    out = []
    for ids in merges:
        spans = sorted((min(f for (t, f) in box if t == i),
                        max(f for (t, f) in box if t == i), i) for i in ids
                       if any(t == i for (t, _f) in box))
        for (_a0, a1, ta), (b0, _b1, tb) in zip(spans, spans[1:], strict=False):
            ca, cb = box[(ta, a1)], box[(tb, b0)]
            d = np.hypot((cb[0] + cb[2] - ca[0] - ca[2]) / 2, (cb[1] + cb[3] - ca[1] - ca[3]) / 2)
            out.append(float(d / max(1, b0 - a1)))
    return np.array(out)


def preflight(clip: ClipRef, dets: Detections) -> None:
    """Show that the kit cue is present and binding, so the sweep cannot be read as confounded."""
    tracks = ByteTrackTracker(device='cpu', min_track_frames=4, kit_split=True).track(clip, dets)
    players = [t for t in tracks.tracklets if t.cls == 'player']
    blank = sum(1 for t in players if t.team_id is None)
    print(f'  kit labels {dict(Counter(str(t.team_id) for t in players))} '
          f'-- unlabelled {blank}/{len(players)}')
    for req in (True, False):
        merged, report = stitch_tracks_with_report(tracks, StitchConfig(require_same_team=req))
        n = len([t for t in merged.tracklets if t.cls == 'player'])
        print(f'  require_same_team={str(req):<5} -> {n:2d} ids, {len(report.merges)} merges')
    print('  Equal rows would mean the kit cue is idle and the sweep below measures geometry '
          'alone.\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
    ap.add_argument('--det-cache', default='out/phmr_ab/dets_coco_0_236.npz')
    ap.add_argument('--frames', type=int, default=236)
    ap.add_argument('--scales', type=float, nargs='*', default=DEFAULT_SCALES)
    args = ap.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.clip)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    c = np.load(args.det_cache, allow_pickle=True)
    dets = Detections(frames=[
        FrameDetections(frame=int(f), items=[
            Detection(bbox_xyxy=b, cls=str(k), score=float(s))
            for b, k, s in zip(bb, kk, ss, strict=True)])
        for f, bb, kk, ss in zip(c['frame'], c['boxes'], c['classes'], c['scores'], strict=True)])
    clip = ClipRef(source_id='c', uri=args.clip, frames=np.arange(args.frames),
                   width=w, height=h, fps=fps)

    print(f'{sum(len(f.items) for f in dets.frames)} detections over {len(dets.frames)} frames')
    print('kit_split on, the #132 scoreboard configuration.\n')
    preflight(clip, dets)
    print('  expand   player ids   after stitch   merges   seam speed px/f (median / max)')

    for scale in args.scales:
        unpatch = patch_expansion(scale)
        try:
            tracks: Tracks = ByteTrackTracker(
                device='cpu', min_track_frames=4, kit_split=True
            ).track(clip, dets)
        finally:
            unpatch()
        pre = [t for t in tracks.tracklets if t.cls == 'player']
        merged, report = stitch_tracks_with_report(tracks, StitchConfig())
        post = [t for t in merged.tracklets if t.cls == 'player']
        sp = seam_speeds(tracks, report.merges)
        med = f'{np.median(sp):6.1f}' if sp.size else '     -'
        mx = f'{sp.max():6.1f}' if sp.size else '     -'
        print(f'  {scale:5.2f}   {len(pre):10d}   {len(post):12d}   {len(report.merges):6d}   '
              f'{med} / {mx}', flush=True)


if __name__ == '__main__':
    main()
