"""Lightweight references to pixel data that cross port boundaries.

The core never holds decoded frames — decoding video/images is an adapter job
(FFmpeg/GStreamer). Ports therefore receive *references* (`ClipRef`, `CropRef`):
a URI plus the frame indices and geometry needed to locate the pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClipRef:
    """A reference to a contiguous set of frames of one source.

    Attributes:
        source_id: The owning :class:`~pitch3d.core.scene.scene.Source`.
        uri: Where the pixels live (video file / frame directory).
        frames: Frame indices to process, shape ``(T,)``.
        width, height: Frame geometry (px).
        fps: Frame rate (for any time-based math in adapters).
    """

    source_id: str
    uri: str
    frames: np.ndarray
    width: int
    height: int
    fps: float
    #: Optional ``(w, h, x, y)`` in source pixels that every decode of this clip is read through.
    #: The pipeline measures it (``adapters.io.framing.measure_framing``) so a phone clip reaches
    #: the calibrator framed like a broadcast, **without anyone cutting a new file by hand** — a
    #: hand-cut mp4 is a per-clip artefact and the goal is any clip. ``None`` means the full frame,
    #: which is what an already-broadcast clip measures to.
    crop: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=int).reshape(-1)

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])


@dataclass
class CropRef:
    """A reference crop of one subject (input to avatar building / inpainting)."""

    subject_track_id: int
    uri: str
    frame: int
    bbox_xyxy: np.ndarray  # (4,) image px

    def __post_init__(self) -> None:
        self.bbox_xyxy = np.asarray(self.bbox_xyxy, dtype=float).reshape(4)
