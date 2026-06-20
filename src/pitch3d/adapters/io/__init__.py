"""I/O adapters: turn media files into the core's references (no ML, no GPU).

Decoding/probing video is an adapter job (the core never holds pixels — see
``core.ports.io``). The FFmpeg ingestor probes a file with ``ffprobe`` and produces the
canonical :class:`~pitch3d.core.scene.scene.Source` plus a
:class:`~pitch3d.core.ports.io.ClipRef` for a chosen frame range (M1 step 1, FR-2).
"""

from __future__ import annotations

from .ffmpeg import FFmpegIngestor, ProbeResult, parse_probe, probe

__all__ = ["FFmpegIngestor", "ProbeResult", "parse_probe", "probe"]
