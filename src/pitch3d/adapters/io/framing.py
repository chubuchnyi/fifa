"""Measure the framing a clip needs — a pipeline capability, not a step someone runs by hand.

The calibrator wants a frame that is mostly pitch. A phone clip from the stand is mostly stand,
sky and scoreboard, and PnLCalib then finds too few landmarks: measured on
``14604731_1080_1920_30fps.mp4``, grass covers **36.6 %** of the raw frame and the pipeline solved
**0 of 8** calibration frames; inside the measured crop it is 82–92 % and it solves.

`scripts/broadcast_crop.py` has measured this since #136 and prints an ffmpeg command. That made
it a **manual pre-step**, and a manual pre-step is a per-clip artefact — the same class of thing as
a hand-fitted `calib/<clip>.npz`: it makes one clip work and generalises to nothing. The goal is
any clip, so the measurement belongs in the pipeline, applied through the one decoder every stage
already reads (`io.frames.iter_clip_frames`), with the chosen rect recorded on the `ClipRef`.

**What this does not do.** A clip whose framing changes mid-way needs **one reconstruction per
segment** — its calibration belongs to its own pixels, exactly as a carried homography does not
belong to a later frame. :func:`measure_segments` reports the segments; running each as its own
reconstruction is a pipeline change that is not built. Until it is, a caller handed a multi-segment
clip must either restrict its frame range to one segment or accept that the crop fits the first.

**And a crop cannot rescue a clip whose camera outruns the pitch.** Past frame ~155 of that clip
the zoom leaves only the goal mouth; there are no landmarks at any crop. Measured: the segment with
the *least* grass (82.4 %) solves 98 % of its frames and the one with the *most* (91.7 %) solves
9 %. Grass fraction is not the predictor — landmark count is. That case is a refusal, not a crop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: HSV window for pitch grass, OpenCV ranges. Wide on purpose: floodlit night grass is
#: desaturated compared with daylight, and the band only has to separate turf from stand.
GRASS_HSV_LO = (35, 40, 40)
GRASS_HSV_HI = (85, 255, 255)

#: Fraction of a row that must be grass for it to count as pitch.
DEFAULT_MIN_COVER = 0.25
#: A clip already framed like a broadcast returns its own full frame and nothing changes.
BROADCAST_ASPECT = 16 / 9


@dataclass(frozen=True)
class Framing:
    """The crop a clip's frames should be read through, and the evidence for it.

    ``rect`` is ``(w, h, x, y)`` in source pixels. ``is_identity`` when the clip is already
    framed like a broadcast — the common case, and a no-op by construction.
    """

    rect: tuple[int, int, int, int]
    source_size: tuple[int, int]
    grass_before: float
    grass_after: float
    n_segments: int = 1

    @property
    def is_identity(self) -> bool:
        w, h, x, y = self.rect
        return (x, y) == (0, 0) and (w, h) == self.source_size


def longest_band(cover: np.ndarray, min_cover: float) -> tuple[int, int] | None:
    """Longest run of rows whose grass fraction clears ``min_cover``, as ``[y0, y1)``.

    ``None`` when no row reaches it — a clip with no visible pitch, which is a refusal upstream,
    not a crop to guess at.
    """
    hot = np.asarray(cover) >= min_cover
    if not hot.any():
        return None
    best = cur = (0, 0)
    for y, on in enumerate(hot):
        cur = (cur[0], y + 1) if on else (y + 1, y + 1)
        if cur[1] - cur[0] > best[1] - best[0]:
            best = cur
    return best


def crop_rect(
    w: int, h: int, band: tuple[int, int], aspect: float = BROADCAST_ASPECT
) -> tuple[int, int, int, int]:
    """The widest ``aspect`` window that fits the frame, centred on the grass band."""
    y0, y1 = band
    cw = w
    ch = min(int(round(cw / aspect)) // 2 * 2, h)
    cw = min(int(round(ch * aspect)) // 2 * 2, w)
    y = int(round((y0 + y1) / 2 - ch / 2))
    return cw, ch, (w - cw) // 2, max(0, min(y, h - ch))


def grass_row_cover(
    uri: str, n_samples: int = 11, first: int = 0, last: int | None = None
) -> tuple[np.ndarray, int, int, float]:
    """Per-row grass fraction over ``n_samples`` frames of ``[first, last]``.

    Returns ``(per-row cover, width, height, overall fraction)``. The frame range is not optional
    detail: one measurement over a whole phone clip is a measurement of nothing, because the
    framing moves.
    """
    import cv2

    from .frames import resolve_source_path

    cap = cv2.VideoCapture(resolve_source_path(uri))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open clip for framing measurement: {uri}")
    try:
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
    finally:
        cap.release()
    if not taken:
        raise ValueError(f"read no frames from {uri}")
    rows /= taken
    return rows, w, h, float(rows.mean())


def measure_framing(
    uri: str,
    *,
    first: int = 0,
    last: int | None = None,
    n_samples: int = 11,
    min_cover: float = DEFAULT_MIN_COVER,
    aspect: float = BROADCAST_ASPECT,
) -> Framing | None:
    """The crop this clip's frames should be read through, or ``None`` if it cannot be measured.

    ``None`` means no row of the sampled frames reaches ``min_cover`` grass — there is no pitch to
    centre on, and inventing a rect would hand the calibrator a confidently wrong frame. The caller
    keeps the full frame and the calibration refuses on its own terms.
    """
    try:
        cover, w, h, before = grass_row_cover(uri, n_samples=n_samples, first=first, last=last)
    except (FileNotFoundError, ValueError, ImportError):
        return None
    band = longest_band(cover, min_cover)
    if band is None:
        return None
    rect = crop_rect(w, h, band, aspect)
    cw, ch, x, y = rect
    after = float(cover[y:y + ch].mean()) if ch else before
    return Framing(rect=rect, source_size=(w, h), grass_before=before, grass_after=after)


def measure_segments(
    uri: str,
    *,
    window: int = 30,
    n_samples: int = 5,
    min_cover: float = DEFAULT_MIN_COVER,
    aspect: float = BROADCAST_ASPECT,
    tolerance: int = 40,
    n_frames: int | None = None,
) -> list[tuple[int, int, Framing]]:
    """Runs of frames that share one crop, as ``(first, last, Framing)``.

    A clip on a tripod collapses to exactly one segment, so the broadcast path is unchanged. More
    than one means the framing genuinely moved, and **each segment is a separate reconstruction**.
    """
    import cv2

    from .frames import resolve_source_path

    if n_frames is None:
        cap = cv2.VideoCapture(resolve_source_path(uri))
        n_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        cap.release()

    out: list[tuple[int, int, Framing]] = []
    for start in range(0, n_frames, window):
        stop = min(start + window - 1, n_frames - 1)
        f = measure_framing(uri, first=start, last=stop, n_samples=n_samples,
                            min_cover=min_cover, aspect=aspect)
        if f is None:
            continue
        if out and abs(out[-1][2].rect[3] - f.rect[3]) <= tolerance \
                and out[-1][2].rect[:3] == f.rect[:3]:
            first, _, prev = out[-1]
            out[-1] = (first, stop, prev)
        else:
            out.append((start, stop, f))
    return out
