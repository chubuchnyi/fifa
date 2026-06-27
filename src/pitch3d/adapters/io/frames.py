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


def iter_clip_frames(
    uri: str, frames: Sequence[int]
) -> Iterator[tuple[int, np.ndarray]]:  # pragma: no cover - heavy decode path (needs cv2 + media)
    """Yield ``(frame_index, BGR uint8 image)`` for each requested frame of ``uri``.

    ``uri`` may be a video file (seek per index) or a directory of frames (index into the sorted
    image list). Lazy ``cv2``. Raises if a requested frame cannot be decoded — an unreadable source
    is surfaced, never silently skipped.
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
            yield idx, img
        return

    cap = cv2.VideoCapture(path)
    try:
        for idx in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {idx} of {uri}")
            yield idx, img
    finally:
        cap.release()


__all__ = ["iter_clip_frames", "resolve_source_path"]
