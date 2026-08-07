#!/usr/bin/env python3
"""Crop a phone/fan clip down to the broadcast framing the calibrator was built for.

The calibrator wants a frame that is mostly pitch. A vertical fan clip is mostly stand, sky and
scoreboard, with the pitch as a band across part of it — PnLCalib then finds too few keypoints and
the #125 gate refuses the run. Measured on ``14604731_1080_1920_30fps.mp4`` (1080x1920, 355 f):
grass covers 37.2% of the frame and only starts at y=1088. Raw, the pipeline solved 0/8 calibration
frames; through this script's own crop (``1080x608+0+1200``, 84.2% grass) it solved 8/8 at
confidence 0.473–0.558. That crop frames the penalty box, both goalposts and the goal line — the
keypoints the calibrator looks for. On a clip that is *already* broadcast this is a no-op by
construction, and measured as one: the target clip returns ``1920x1080+0+0`` with grass unchanged
at 53.2%.

**One rect for a whole phone clip is a measurement of nothing** (fixed 2026-08-07). Re-measured in
30-frame windows, this clip's grass band walks from ``y 1262..1920`` to ``968..1920`` and back to
``1128..1920`` as the fan zooms: the band centre moves 177 px and its height changes by 354 px
(59 %). The single ``1080x608+0+1200`` above is a compromise wrong at both ends — stand at the
start, pitch cut off in the middle. The script now measures per window and emits **one rect per
framing**: four segments here at 82.4 / 91.3 / 91.7 / 90.3 % grass. A clip on a tripod does not
move and collapses to exactly one segment, so the broadcast path is unchanged (the target clip
still returns ``1920x1080+0+0``); ``--single`` forces the old behaviour.

Each segment is a **separate reconstruction**: its calibration belongs to its own pixels. Feeding a
later segment through an earlier segment's crop is the same class of error as grounding a foot
through a carried homography (``core/scene/field.py`` ``MIN_SOLVED_CONFIDENCE``).

A crop still cannot rescue a clip whose camera outruns the pitch, though: past frame ~155 this one
zooms until only the goal mouth is left, PnLCalib has nothing to solve with, and the plane is
undetermined. That is not a cropping problem — it is a refusal, and since 2026-08-07 the pipeline
makes it one instead of carrying a stale homography onto zoomed pixels. See
``docs/findings/open-items-2026-08-01.md`` (#131).

The band is *measured*, not assumed — a fan can hold the phone any way — and the measurement is
overridable, because a clip whose pitch is half-occluded by a crowd will mis-measure and the
operator's eye should win:

    python scripts/broadcast_crop.py --clip in.mp4                     # measure and print only
    python scripts/broadcast_crop.py --clip in.mp4 --out crop.mp4      # one file per segment
    python scripts/broadcast_crop.py --clip in.mp4 --out crop.mp4 --rect 1080:608:0:1312
    python scripts/broadcast_crop.py --clip in.mp4 --single            # one rect, pre-2026-08-07

This is a *pre*-processing step: it changes which pixels the pipeline sees, so the calibration it
produces belongs to the cropped clip. Feed the same cropped file to every downstream stage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import cv2
import numpy as np

#: Broadcast-green in HSV. Wide on hue because floodlights push the grass cool and the mown
#: stripes split it in two; the saturation floor is what keeps grey stand out of the mask.
GRASS_HSV_LO = (30, 40, 40)
GRASS_HSV_HI = (95, 255, 255)


def grass_row_cover(
    clip: str, n_samples: int = 11, first: int = 0, last: int | None = None
) -> tuple[np.ndarray, int, int, float]:
    """Per-row grass fraction averaged over ``n_samples`` frames of ``[first, last]``.

    Returns ``(per-row cover, width, height, overall fraction)``. The frame range exists because
    one measurement over a whole phone clip is a measurement of nothing: see :func:`segments`.
    """
    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end = total - 1 if last is None else min(int(last), total - 1)
    rows, taken = np.zeros(h, dtype=float), 0
    for i in np.linspace(max(0, int(first)), max(int(first), end), n_samples, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(GRASS_HSV_LO), np.array(GRASS_HSV_HI))
        rows += mask.mean(axis=1) / 255.0
        taken += 1
    cap.release()
    if not taken:
        raise SystemExit(f"read no frames from {clip}")
    rows /= taken
    return rows, w, h, float(rows.mean())


def longest_band(cover: np.ndarray, min_cover: float) -> tuple[int, int]:
    """The longest run of rows whose grass fraction clears ``min_cover`` — [y0, y1)."""
    hot = cover >= min_cover
    if not hot.any():
        raise SystemExit(
            f"no row reaches {min_cover:.0%} grass (max {cover.max():.1%}) — "
            "lower --min-cover or pass --rect"
        )
    best = cur = (0, 0)
    for y, on in enumerate(hot):
        cur = (cur[0], y + 1) if on else (y + 1, y + 1)
        if cur[1] - cur[0] > best[1] - best[0]:
            best = cur
    return best


def crop_rect(w: int, h: int, band: tuple[int, int], aspect: float) -> tuple[int, int, int, int]:
    """The widest ``aspect`` window that fits the frame, centred on the grass band."""
    y0, y1 = band
    cw = w
    ch = min(int(round(cw / aspect)) // 2 * 2, h)
    cw = min(int(round(ch * aspect)) // 2 * 2, w)
    y = int(round((y0 + y1) / 2 - ch / 2))
    return cw, ch, (w - cw) // 2, max(0, min(y, h - ch))


def frame_count(clip: str) -> int:
    cap = cv2.VideoCapture(clip)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 1)


def segments(
    clip: str, window: int, samples: int, min_cover: float, aspect: float, tolerance: int
) -> list[tuple[int, int, tuple[int, int, int, int], float]]:
    """Split the clip into runs whose crop is the same rect, and return one entry per run.

    One rect for a whole phone clip is a measurement of nothing. Measured on
    ``14604731_1080_1920_30fps.mp4`` in 30-frame windows, the grass band walks from
    ``y 1262..1920`` to ``968..1920`` and back to ``1128..1920`` — the band **centre** moves 177 px
    and its height changes by 354 px (59 %) as the fan zooms. The single clip-wide crop that
    produced (``1080x608+0+1200``) is a compromise wrong at both ends: it keeps stand at the start
    and cuts pitch in the middle.

    A broadcast clip on a tripod does not move, so it collapses to exactly one segment and the
    output is byte-identical to the pre-segmentation behaviour. That is the point of ``tolerance``:
    it decides "did the framing actually change", not "how finely should I chop".

    Returns ``[(first_frame, last_frame, (w, h, x, y), grass_fraction_kept), ...]``.
    """
    total = frame_count(clip)
    out: list[tuple[int, int, tuple[int, int, int, int], float]] = []
    for a in range(0, total, window):
        b = min(a + window - 1, total - 1)
        cover, w, h, _overall = grass_row_cover(clip, samples, a, b)
        try:
            band = longest_band(cover, min_cover)
        except SystemExit:
            continue  # no pitch in this window at all — reported by the caller as a gap
        rect = crop_rect(w, h, band, aspect)
        out.append((a, b, rect, float(cover[rect[3]:rect[3] + rect[1]].mean())))
    return merge_windows(out, tolerance)


def merge_windows(
    windows: list[tuple[int, int, tuple[int, int, int, int], float]], tolerance: int
) -> list[tuple[int, int, tuple[int, int, int, int], float]]:
    """Collapse consecutive windows whose crop is the same framing into one segment.

    Two windows are the same framing when their crop has the same size and their vertical offset
    differs by at most ``tolerance`` px. Size is compared exactly on purpose: :func:`crop_rect`
    locks the aspect, so on a given clip the size is constant and every real change shows up in
    ``y`` — a size difference means the source itself changed and must not be glued together.
    """
    out: list[tuple[int, int, tuple[int, int, int, int], float]] = []
    for a, b, rect, kept in windows:
        if out and out[-1][2][:3] == rect[:3] and abs(out[-1][2][3] - rect[3]) <= tolerance:
            first, _last, prev_rect, prev_kept = out[-1]
            out[-1] = (first, b, prev_rect, (prev_kept + kept) / 2)
        else:
            out.append((a, b, rect, kept))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="", help="write the cropped mp4 here (default: measure only)")
    ap.add_argument("--rect", default="", help="override the measurement: w:h:x:y in source pixels")
    ap.add_argument("--aspect", type=float, default=16 / 9)
    ap.add_argument("--scale", default="1920:1080", help="output size, or 'none' to keep the crop")
    ap.add_argument("--min-cover", type=float, default=0.25, help="grass fraction that marks pitch")
    ap.add_argument("--samples", type=int, default=11)
    ap.add_argument("--window", type=int, default=30,
                    help="frames per measurement window; the framing is re-measured this often")
    ap.add_argument("--tolerance", type=int, default=48, metavar="PX",
                    help="how far the crop may drift before it counts as a new framing. A tripod "
                         "never moves, so a broadcast clip collapses to one segment either way")
    ap.add_argument("--single", action="store_true",
                    help="force ONE rect for the whole clip (the pre-2026-08-07 behaviour)")
    args = ap.parse_args()

    cover, w, h, overall = grass_row_cover(args.clip, args.samples)
    y0, y1 = longest_band(cover, args.min_cover)
    print(f"clip      {args.clip}  {w}x{h}  {frame_count(args.clip)} frames")
    print(f"grass     {overall:.1%} of frame · band y={y0}..{y1} ({y1 - y0} px, "
          f"{y0 / h:.0%}..{y1 / h:.0%} down)   [whole clip]")

    if args.rect:
        cw, ch, cx, cy = (int(v) for v in args.rect.split(":"))
        segs = [(0, frame_count(args.clip) - 1, (cw, ch, cx, cy),
                 float(cover[cy:cy + ch].mean()))]
        print(f"crop      {cw}x{ch}+{cx}+{cy}  (manual --rect, measurement overridden)")
    elif args.single:
        cw, ch, cx, cy = crop_rect(w, h, (y0, y1), args.aspect)
        segs = [(0, frame_count(args.clip) - 1, (cw, ch, cx, cy),
                 float(cover[cy:cy + ch].mean()))]
        print(f"crop      {cw}x{ch}+{cx}+{cy}  (measured, --single)")
    else:
        segs = segments(args.clip, args.window, max(3, args.samples // 2),
                        args.min_cover, args.aspect, args.tolerance)
        if not segs:
            raise SystemExit("no window of this clip contains enough pitch to crop to")

    print(f"\nsegments  {len(segs)} (a fixed camera gives 1; each extra one is the framing "
          f"actually changing)")
    for first, last, (cw, ch, cx, cy), kept in segs:
        print(f"  f{first:<5d}-{last:<5d} {cw}x{ch}+{cx}+{cy}   {kept:.1%} grass inside the crop")
    if len(segs) > 1:
        print("  NB every segment is a separate reconstruction: its calibration belongs to its own "
              "pixels.\n     Feeding a later segment through an earlier segment's crop is the same "
              "class of error\n     as grounding a foot through a carried homography.")

    cmds = []
    for first, last, (cw, ch, cx, cy), _kept in segs:
        vf = f"crop={cw}:{ch}:{cx}:{cy}"
        if args.scale != "none":
            vf += f",scale={args.scale}:flags=lanczos"
        out = args.out or "out.mp4"
        if len(segs) > 1:
            stem = out.rsplit(".", 1)
            out = f"{stem[0]}_{first:05d}-{last:05d}.{stem[1] if len(stem) > 1 else 'mp4'}"
        trim = ["-vf", f"select='between(n\\,{first}\\,{last})',{vf}", "-vsync", "0"] \
            if len(segs) > 1 else ["-vf", vf]
        cmds.append(["ffmpeg", "-y", "-i", args.clip, *trim,
                     "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-an",
                     out])
    if not args.out:
        print()
        for c in cmds:
            print("$ " + " ".join(c))
        return 0
    rc = 0
    for c in cmds:
        print("\n$ " + " ".join(c))
        rc = subprocess.run(c, check=False).returncode or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
