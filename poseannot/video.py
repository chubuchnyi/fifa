"""Video frame extraction for poseannot — cached JPEGs served by FastAPI."""

from __future__ import annotations

import subprocess
import threading
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

_VIDEO_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _open_capture(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    return cap


def frame_size(video_path: str) -> tuple[int, int]:
    cap = _open_capture(video_path)
    return (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )


def read_frame(video_path: str, frame_index: int) -> np.ndarray:
    """Return a BGR frame at ``frame_index`` (0-based).

    OpenCV's seek is not thread-safe on a shared capture, so we serialize.
    """
    cap = _open_capture(video_path)
    with _VIDEO_LOCK:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    if not ok:
        raise IndexError(f"could not read frame {frame_index} from {video_path}")
    return frame


def encode_jpeg(bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return bytes(buf)
