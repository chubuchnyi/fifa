"""Frame decoding — turn a clip URI + frame indices into decoded pixels (adapter job).

The core never holds pixels (``core.ports.io``); decoding is an adapter concern. This is the one
cv2-backed decoder shared by every adapter that needs real frames: the detector reads the whole
clip for tracking, the measured-avatar texturer (M2-8b) reads a handful of reference frames to
sample appearance. ``cv2`` is imported lazily so importing this module never requires it.

``clip.uri`` may be a video file (seek per frame index) or a directory of frames (index into the
sorted image list). Decoded images are returned in OpenCV's native **BGR** order — the caller
converts to RGB if it needs to (the avatar texturer does, since the PLY stores RGB).
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence

import numpy as np


def resolve_source_path(uri: str) -> str:
    """Strip a ``file://`` scheme so the path can be opened by cv2 / :func:`os.path.exists`."""
    return uri[len("file://"):] if uri.startswith("file://") else uri


def apply_crop(img: np.ndarray, crop: tuple[int, int, int, int] | None,
               out_size: tuple[int, int] | None = None) -> np.ndarray:
    """Cut ``(w, h, x, y)`` out of a decoded frame, optionally rescaling to ``out_size``.

    The crop is clamped to the frame rather than raising: a rect measured on one segment of a
    clip whose framing moved can overhang a later frame, and a black bar is a better failure than
    a dead run. `None` is the identity.
    """
    if crop is None:
        return img
    import cv2

    h_img, w_img = img.shape[:2]
    w, h, x, y = (int(v) for v in crop)
    x = max(0, min(x, max(w_img - 1, 0)))
    y = max(0, min(y, max(h_img - 1, 0)))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    out = img[y:y + h, x:x + w]
    if out_size is not None and (out.shape[1], out.shape[0]) != tuple(out_size):
        out = cv2.resize(out, tuple(out_size), interpolation=cv2.INTER_LANCZOS4)
    return out


def iter_clip_frames(
    uri: str, frames: Sequence[int], crop: tuple[int, int, int, int] | None = None,
    out_size: tuple[int, int] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:  # pragma: no cover - heavy decode path (needs cv2 + media)
    """Yield ``(frame_index, BGR uint8 image)`` for each requested frame of ``uri``.

    ``uri`` may be a video file (seek per index) or a directory of frames (index into the sorted
    image list). Lazy ``cv2``. Raises if a requested frame cannot be decoded — an unreadable source
    is surfaced, never silently skipped.

    ``crop`` is applied here, at the **one** point every adapter decodes through, so a framing
    decision reaches the detector, the calibrator and the texturer identically and without anyone
    writing a new mp4. Pass `ClipRef.crop`.
    """
    import cv2

    wanted = [int(f) for f in frames]
    path = resolve_source_path(uri)
    if os.path.isdir(path):
        files = sorted(
            f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        for idx in wanted:
            img = cv2.imread(os.path.join(path, files[idx]))
            if img is None:
                raise FileNotFoundError(f"frame {idx} unreadable in {path}")
            yield idx, apply_crop(img, crop, out_size)
        return

    cap = cv2.VideoCapture(path)
    try:
        for idx in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {idx} of {uri}")
            yield idx, apply_crop(img, crop, out_size)
    finally:
        cap.release()


__all__ = ["iter_clip_frames", "resolve_source_path"]
