"""Draw the pitch markings the *pixels* contain, on every frame, without a camera.

Everything else in this repo that draws markings projects the pitch model through a solved
camera, so it can only ever agree with itself: if the calibration is wrong the overlay is
wrong in exactly the same way and still looks confident. This script never sees the camera,
the calibration or the pitch model. It reads pixels and reports what is painted there.

That makes it the independent side of every overlay check, and it is also the only thing here
that runs on a clip with no calibration at all.

Three stages, each measured on two clips (broadcast 1920x1080, fan-phone portrait 1080x1920):

  1. paint      poseannot.pitch_evidence._masks -- the repo's canonical ridge filter with the
                turf-on-both-sides test. Not re-implemented here; there are already four
                copies of this idea in the tree and a fifth would be debt.
  2. extent     LSD (present in OpenCV 5.0.0) over the ridge band, then a length floor.
                Extent is the discriminator, which bench_camera_swim.py:364 already measured:
                brightness alone marks 2.5 % of the pitch and a bare ridge filter is worse at
                3-8 %, because floodlit grass texture is full of short ridges. A marking is
                long. The goal net is the clearest case -- the frame inside the net yields
                1080 LSD segments and *not one* is 80 px long, so the length floor deletes it
                whole while keeping every real marking. What length does not catch is the goal
                *frame* -- posts and crossbar are genuinely long and straight -- so those go by
                mesh density instead: a net puts a few hundred ridge pixels into a 31x31
                window where a painted line puts about 28, and dilating that region reaches
                the frame that bounds it. Worth 1.8 % of segments on both clips.
  3. carry      frames with no evidence keep the last measured markings, warped by pixel
                motion, and are drawn dimmed and labelled. R-6: marked, never hidden.

**There is no arc or circle detection here, and that is a deletion, not an omission.** A first
version fitted ellipses to the ridge blobs the straight segments left over. Measured: **67
ellipses on the fan clip and 0 on the broadcast clip** -- exactly backwards, because the fan
clip is a goalmouth view with no centre circle in shot while the broadcast clip shows both the
circle and a penalty D. Every one of the 67 was net mesh or a goalkeeper: median major axis
**85 px, max 127**, where a real centre circle at this camera is many hundreds of pixels
across. Recall 0, precision 0.

It is not a threshold that was wrong, it is the shape of the algorithm. It looked for one big
curved connected component, and a real arc never is one: players break it and the ridge spine
is dotted, so it arrives as short, locally *straight* fragments -- which the straightness test
then rejects on purpose. Detecting arcs properly means grouping fragments across gaps and
fitting a conic to the group. That is a different algorithm and it is not written.

**Coverage is not 100 % and the honest answer is that it should not be.** Measured over every
frame of both clips:

  | clip            | frames | >=1 long segment | >=2 direction families | blind |
  |-----------------|--------|------------------|------------------------|-------|
  | fan portrait    |   355  |      99.4 %      |         92.7 %         |   2   |
  | broadcast       |   334  |      73.1 %      |         71.0 %         |  90   |

The broadcast clip's 90 blind frames are not hard frames, they are a *different shot*: one
contiguous 2 s block (f272-330) plus three short ones, all of them the goalmouth close-up
after the goal, where the playing surface drops from 51 % of the frame to 31 % and no marking
is long enough to see. A detector that reported markings there would be inventing them.

## Where this does not work

Run against all nine sample clips, seven of which it was never tuned on. **It handles 5 of
9**, and the two it was tuned on are both floodlit night matches:

| clip | med. markings | frames with >=4 | verdict |
|---|---|---|---|
| 14604731 portrait *(tuned)* | 10 | 100 % | works |
| 14604680 portrait | 9 | 100 % | works |
| 14604660 portrait | 7 | 100 % | works |
| 13386302 4K overhead | 26 | 75 % | works |
| Colombia broadcast *(tuned)* | 10 | 67 % | works |
| 11710897 daylight amateur | 0 | 0 % | **fails** |
| 15449383 daylight stadium | 0 | 0 % | **fails** |
| 15449387 daylight stadium | 1 | 0 % | **fails** |
| 15750079 daylight amateur | 0 | 0 % | **fails** |

The four failures are all daylight, and the failure is **upstream of anything in this
file** — in `pitch_evidence._masks`. Two different shapes:

- **fragmentation** (11710897): the paint mask is a healthy 8941 ridge px, the same as the
  working fan clip's 8955, but the longest LSD segment is **50 px** against a 60 px floor.
  The ridge arrives dotted, so no long line ever forms.
- **noise** (15750079): **460 587** ridge px, 50x normal — 55 530 per megapixel against
  3 384-8 744 on every clip that works. The mask is measuring turf texture, not paint, and
  LSD cannot find a line in it.

So `ridge px per megapixel` is a usable health signal: above ~10 000 the paint stage has
failed and the output will be empty or nonsense, whatever this file does. Fixing those
clips means the ridge filter's constants (`_RIDGE_CONTRAST`, `_RIDGE_MIN_V`, the hue band),
which are tuned for floodlit night and are not touched here.

Run:
    .venv/bin/python scripts/detect_markings.py --clip samples/video/<clip>.mp4 \
        --out out/markings.mp4 --json out/markings.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poseannot.pitch_evidence import _masks  # noqa: E402

#: A marking is long. Below this a segment is grass texture, a kit fold or the goal net.
#: Swept, with the inside-the-net frame as the false-positive control (it contains no real
#: marking at all, so every segment it yields is noise). Numbers are merged markings and
#: their total drawn length:
#:
#:   MIN_LEN |  fan f0     |  fan f50    |  bcast f67  |  NET f333 (all false)
#:      30   | 16 (3.1k)   | 14 (1.6k)   | 16 (3.8k)   | 136 (5.8k)
#:      40   | 10 (2.7k)   | 11 (1.4k)   | 10 (3.5k)   |  52 (2.3k)
#:      60   | 11 (2.2k)   | 11 (1.3k)   |  8 (3.2k)   |   3 (0.2k)   <-- here
#:      80   |  9 (1.8k)   |  6 (0.9k)   |  7 (2.8k)   |   0 (0.0k)
#:
#: 80 is where the net finally reaches zero, and it is not worth it: it also deletes 25-31 %
#: of the real markings on every ordinary frame to clean up three false segments on one frame
#: that has no markings to get wrong. 60 keeps the recall and leaves the residue to be
#: labelled honestly by the "no evidence" verdict.
MIN_LEN_PX = 60.0

#: Two LSD segments are the same marking if they are near-parallel, near-collinear AND their
#: spans along the line actually meet. LSD returns both edges of the 3 px ridge band as
#: separate segments, so without merging every marking is drawn twice.
#: The gap limit is the part that is easy to leave out and it is not optional: collinearity
#: alone merged the goal-area line with an unrelated fragment 500 px away and drew a diagonal
#: straight across the penalty area that no marking follows. Two collinear stretches with
#: nothing painted between them are two markings.
MERGE_ANGLE_DEG = 4.0
MERGE_OFFSET_PX = 6.0
MERGE_MAX_GAP_PX = 40.0

#: The net is a mesh and the markings are not. A net puts a few hundred ridge pixels into a
#: 31x31 window (measured: p99 = 250) where an ordinary painted line puts about 28. The 41 px
#: growth then reaches a little past the weave, which is usually enough to take the post or
#: crossbar beside it.
#:
#: **This keys on the net, not on the goal.** Rendered on a fan frame, the >90 core is two
#: blobs covering the densest part of the weave -- 0.45 % of the frame, 1.38 % after growth --
#: not the goal outline. A goal with no net, or a net too fine or too motion-blurred to
#: resolve, leaves its posts detected. The growth is also a fixed pixel distance, so it does
#: not scale with how large the goal appears.
#:
#: The physically correct test -- "the goal frame is above the ground plane, the paint is on
#: it" -- is unavailable by construction: this script never forms a camera, so it cannot ask
#: what is on the plane. Mesh density is a proxy for it and should be replaced if this ever
#: gets a camera.
NET_WINDOW = 31
NET_DENSITY = 90
NET_GROW = 41

#: A goal is small. If the "mesh" covers more of the frame than this, the ridge mask is noisy
#: and the density test is measuring that noise, not a net -- so the filter refuses to act
#: rather than delete the frame's markings. Measured over the nine sample clips: every clip
#: the detector handles has a core of 0.00-0.75 % of frame, while the two it fails on read
#: **12.3 % and 39.3 %**. Without this ceiling those two lost their only surviving segment to
#: a "goal" covering 26 % and 43 % of the frame. 3 % is 4x the worst honest case.
NET_MAX_FRAME = 0.03


def _angle(seg: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0])) % 180.0


def merge_collinear(segs: np.ndarray) -> np.ndarray:
    """Collapse LSD's duplicate band edges and broken runs into one segment per marking."""
    if not len(segs):
        return segs
    ang = _angle(segs)
    order = np.argsort(-np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))
    taken = np.zeros(len(segs), bool)
    out = []
    for i in order:
        if taken[i]:
            continue
        p0 = segs[i, :2]
        u = segs[i, 2:] - p0
        u /= np.linalg.norm(u) + 1e-9
        n = np.array([-u[1], u[0]])
        da = np.abs(ang - ang[i])
        near_angle = np.minimum(da, 180.0 - da) <= MERGE_ANGLE_DEG
        mids = 0.5 * (segs[:, :2] + segs[:, 2:])
        near_line = np.abs((mids - p0) @ n) <= MERGE_OFFSET_PX
        # spans along the seed direction, so "do they meet?" is a 1-D question
        ta = (segs[:, :2] - p0) @ u
        tb = (segs[:, 2:] - p0) @ u
        lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
        cand = np.flatnonzero(near_angle & near_line & ~taken)
        grown = [i]
        span_lo, span_hi = lo[i], hi[i]
        changed = True
        while changed:  # a chain of fragments joins end to end, one hop at a time
            changed = False
            for j in cand:
                if j in grown:
                    continue
                if lo[j] - span_hi <= MERGE_MAX_GAP_PX and span_lo - hi[j] <= MERGE_MAX_GAP_PX:
                    grown.append(j)
                    span_lo, span_hi = min(span_lo, lo[j]), max(span_hi, hi[j])
                    changed = True
        taken[grown] = True
        out.append(np.concatenate([p0 + span_lo * u, p0 + span_hi * u]))
    return np.array(out)


def goal_mask(dist: np.ndarray) -> np.ndarray:
    """The goal structure: mesh density, grown to reach the posts and crossbar that bound it."""
    ridge = (dist == 0).astype(np.uint8)
    dens = cv2.boxFilter(ridge, -1, (NET_WINDOW, NET_WINDOW), normalize=False)
    core = (dens > NET_DENSITY).astype(np.uint8)
    if core.mean() > NET_MAX_FRAME:
        return np.zeros_like(core)  # not a goal -- a noisy ridge mask. Refuse, do not delete.
    grow = np.ones((NET_GROW, NET_GROW), np.uint8)
    return cv2.dilate(core, grow)


def _on_mask(seg: np.ndarray, mask: np.ndarray) -> bool:
    """Does most of this segment lie inside ``mask``?"""
    t = np.linspace(0.0, 1.0, 24)[:, None]
    along = seg[:2] * (1 - t) + seg[2:] * t
    uu = np.clip(np.rint(along[:, 0]).astype(int), 0, mask.shape[1] - 1)
    vv = np.clip(np.rint(along[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return bool(mask[vv, uu].mean() >= 0.5)


def detect(bgr: np.ndarray, lsd: cv2.LineSegmentDetector) -> dict:
    """Every painted marking this frame can prove, as vectors. No camera, no pitch model."""
    dist, surface = _masks(bgr)
    band = (dist <= 1.0).astype(np.uint8) * 255
    raw = lsd.detect(band)[0]
    raw = np.zeros((0, 4)) if raw is None else raw.reshape(-1, 4)
    long_enough = raw[np.hypot(raw[:, 2] - raw[:, 0], raw[:, 3] - raw[:, 1]) >= MIN_LEN_PX]
    segs = merge_collinear(long_enough)
    if len(segs):
        goal = goal_mask(dist)
        keep = np.array([not _on_mask(s, goal) for s in segs])
        segs, dropped = segs[keep], int((~keep).sum())
    else:
        dropped = 0
    ang = _angle(segs) if len(segs) else np.zeros(0)
    return {
        "segments": segs,
        "families": int(len(np.unique((ang // 15).astype(int)))),
        "surface": float((surface > 0).mean()),
        "raw_segments": int(len(raw)),
        "on_goal": dropped,
    }


def warp_segments(segs: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Carry markings across a blind frame on measured pixel motion (R-6: marked, not hidden)."""
    if not len(segs):
        return segs
    pts = np.vstack([segs[:, :2], segs[:, 2:]]).astype(np.float32).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(pts, h).reshape(-1, 2)
    return np.hstack([moved[: len(segs)], moved[len(segs) :]])


def motion(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray | None:
    """Image-to-image homography from the pixels alone -- nothing here has seen the pitch."""
    p0 = cv2.goodFeaturesToTrack(prev_gray, 600, 0.01, 8)
    if p0 is None or len(p0) < 12:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
    if p1 is None or int(st.sum()) < 12:
        return None
    h, _ = cv2.findHomography(p0[st == 1], p1[st == 1], cv2.RANSAC, 3.0)
    return h


def draw(bgr: np.ndarray, det: dict, carried: bool) -> np.ndarray:
    """Measured markings solid, carried ones dashed and dimmed, verdict burnt into the frame."""
    out = bgr.copy()
    colour = (0, 200, 255) if carried else (0, 0, 255)
    thick = 2 if carried else 3
    for x1, y1, x2, y2 in det["segments"]:
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), colour, thick, cv2.LINE_AA)
    n = len(det["segments"])
    if carried:
        txt = f"carried {n} markings - no evidence this frame"
    elif n:
        txt = f"{n} markings measured - {det['families']} directions"
    else:
        txt = "no evidence - markings not visible"
    cv2.rectangle(out, (0, 0), (out.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(out, txt, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 200, 255) if carried else (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="out/markings.mp4")
    ap.add_argument("--json", default="")
    ap.add_argument("--frames", type=int, default=0, help="0 = the whole clip")
    ap.add_argument("--no-carry", action="store_true", help="leave blind frames empty")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        print(f"cannot open {args.clip}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hgt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (wid, hgt))

    lsd = cv2.createLineSegmentDetector()
    rows: list[dict] = []
    last: dict | None = None
    prev_gray: np.ndarray | None = None
    measured = carried_n = blind = 0

    fi = 0
    while args.frames <= 0 or fi < args.frames:
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        det = detect(bgr, lsd)
        carried = False

        if len(det["segments"]):
            last = det
            measured += 1
        elif last is not None and not args.no_carry and prev_gray is not None:
            h = motion(prev_gray, gray)
            if h is not None:
                last = {**last, "segments": warp_segments(last["segments"], h)}
                det, carried = last, True
                carried_n += 1
            else:
                blind += 1
        else:
            blind += 1

        vw.write(draw(bgr, det, carried))
        rows.append({
            "frame": fi,
            "carried": carried,
            "n": int(len(det["segments"])),
            "families": int(det["families"]),
            "surface": round(det["surface"], 4),
            "segments": np.asarray(det["segments"], float).round(1).tolist(),
        })
        if fi % 50 == 0:
            print(f"  f{fi:4d}  {len(det['segments']):2d} markings"
                  f"{'  (carried)' if carried else ''}", flush=True)
        prev_gray, fi = gray, fi + 1

    cap.release()
    vw.release()
    print(f"\n{fi} frames -> {args.out}")
    print(f"  measured {measured} ({measured / max(fi, 1):.1%}) - "
          f"carried {carried_n} ({carried_n / max(fi, 1):.1%}) - "
          f"blind {blind} ({blind / max(fi, 1):.1%})")
    if args.json:
        Path(args.json).write_text(json.dumps(rows))
        print(f"  per-frame detections -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
