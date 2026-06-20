"""FFmpeg/ffprobe ingest — media file → canonical ``Source`` + ``ClipRef`` (M1 step 1, FR-2).

Split like the model adapters so the *logic* is testable with **no ffmpeg**:

* :func:`parse_probe` / :func:`_parse_rate` — **pure** mapping of ``ffprobe``'s JSON to a
  :class:`ProbeResult` (fps/res/frame-count/timecode). Unit-tested on captured JSON.
* :func:`probe` — the **subprocess** half: runs ``ffprobe`` and feeds :func:`parse_probe`.
  ``ffprobe`` is a plain CLI dependency (no Python extra, no GPU); it raises an actionable
  error if the binary is missing.

The core stays pixel-free: the ingestor returns *references*, never decoded frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from ...core.ports.io import ClipRef
from ...core.scene.scene import Source, SourceKind
from ...core.scene.units import TimeBase


@dataclass
class ProbeResult:
    """The subset of ``ffprobe`` metadata the scene model needs.

    Attributes:
        width, height: Frame geometry (px).
        fps: Average frame rate (``avg_frame_rate``, falling back to ``r_frame_rate``).
        frame_count: ``nb_frames`` when present, else ``round(fps * duration)``.
        duration_s: Stream/container duration in seconds (0 if unknown).
        codec: Video codec name, if reported.
        start_timecode: SMPTE start timecode from stream/container tags, if present.
    """

    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float = 0.0
    codec: str | None = None
    start_timecode: str | None = None


def _parse_rate(rate: str | None) -> float:
    """Parse an ffprobe rational rate string (``"30000/1001"``) to fps.

    Returns 0.0 for the ``"0/0"`` / ``"N/A"`` / unparsable cases ffprobe emits when a
    stream has no meaningful rate.
    """
    if not rate or rate in ("N/A", "0/0"):
        return 0.0
    try:
        return float(Fraction(rate))
    except (ZeroDivisionError, ValueError):
        return 0.0


def parse_probe(probe_json: dict) -> ProbeResult:
    """Map ``ffprobe -print_format json -show_format -show_streams`` output to a ProbeResult.

    Pure: no subprocess, no I/O. Raises :class:`ValueError` if there is no video stream.
    """
    streams = probe_json.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("ffprobe output has no video stream")

    fmt = probe_json.get("format", {})
    width = int(video["width"])
    height = int(video["height"])
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    duration = float(video.get("duration") or fmt.get("duration") or 0.0)

    nb_frames = video.get("nb_frames")
    if nb_frames not in (None, "N/A"):
        frame_count = int(nb_frames)
    else:
        frame_count = int(round(fps * duration))

    stream_tags = video.get("tags") or {}
    fmt_tags = fmt.get("tags") or {}
    timecode = stream_tags.get("timecode") or fmt_tags.get("timecode")

    return ProbeResult(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=duration,
        codec=video.get("codec_name"),
        start_timecode=timecode,
    )


def probe(uri: str) -> ProbeResult:
    """Run ``ffprobe`` on ``uri`` and return a :class:`ProbeResult` (subprocess half)."""
    import shutil
    import subprocess

    exe = shutil.which("ffprobe")
    if exe is None:
        raise RuntimeError(
            "ffprobe not found on PATH. Install ffmpeg to ingest real clips "
            "(e.g. `apt install ffmpeg`), or build the ClipRef yourself."
        )
    cmd = [
        exe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", uri,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return parse_probe(json.loads(completed.stdout))


def _default_source_id(uri: str) -> str:
    return Path(uri).stem or "source"


@dataclass
class FFmpegIngestor:
    """Probe a media file and emit the canonical :class:`Source` / :class:`ClipRef` (FR-2).

    Holds no state and decodes nothing — every call shells out to ``ffprobe`` for metadata
    only. Frame *pixels* are pulled later, lazily, by whichever model backend needs them.
    """

    def ingest(self, uri: str, *, source_id: str | None = None) -> Source:
        """Return the :class:`Source` for ``uri`` (fps/res/frame-count/timecode)."""
        pr = probe(uri)
        return Source(
            id=source_id or _default_source_id(uri),
            uri=uri,
            kind=SourceKind.VIDEO,
            time_base=TimeBase(fps=pr.fps, start_timecode=pr.start_timecode),
            width=pr.width,
            height=pr.height,
            frame_count=pr.frame_count,
        )

    def clip(
        self,
        uri: str,
        *,
        start: int = 0,
        end: int | None = None,
        step: int = 1,
        max_frames: int | None = None,
        source_id: str | None = None,
    ) -> ClipRef:
        """Build a :class:`ClipRef` for a frame range of ``uri``.

        ``end`` defaults to the last frame; ``step`` strides; ``max_frames`` caps the count
        (contiguous from ``start``) so a long clip can be sampled into a manageable episode.
        """
        pr = probe(uri)
        if pr.frame_count <= 0:
            raise ValueError(f"{uri}: ffprobe reported no frames (frame_count={pr.frame_count})")
        last = pr.frame_count - 1
        end = last if end is None else min(int(end), last)
        frames = np.arange(int(start), end + 1, max(1, int(step)))
        if max_frames is not None and frames.shape[0] > max_frames:
            frames = frames[:max_frames]
        return ClipRef(
            source_id=source_id or _default_source_id(uri),
            uri=uri,
            frames=frames,
            width=pr.width,
            height=pr.height,
            fps=pr.fps,
        )
