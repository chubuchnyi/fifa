"""FFmpeg ingest (M1 step 1, FR-2): pure ffprobe-JSON mapping + a hermetic integration test.

The mapping (rate parsing, frame-count fallback, timecode) is verified on captured JSON with
**no ffmpeg**. One integration test synthesizes a tiny clip with ``ffmpeg`` (skipped when the
binary is absent), so the subprocess path is exercised without depending on the large
``samples/`` media.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from pitch3d.adapters.io import FFmpegIngestor, ProbeResult, parse_probe
from pitch3d.adapters.io.ffmpeg import _parse_rate
from pitch3d.core.ports.io import ClipRef
from pitch3d.core.scene.scene import Source, SourceKind

_HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _probe_json(**video):
    base = {"codec_type": "video", "width": 1920, "height": 1080,
            "avg_frame_rate": "25/1", "r_frame_rate": "25/1", "nb_frames": "250",
            "duration": "10.0", "codec_name": "h264"}
    base.update(video)
    return {"streams": [{"codec_type": "audio"}, base], "format": {"duration": "10.0"}}


@pytest.mark.parametrize(
    "rate,expected",
    [("25/1", 25.0), ("30000/1001", 29.97003), ("60/1", 60.0),
     ("0/0", 0.0), ("N/A", 0.0), (None, 0.0), ("", 0.0)],
)
def test_parse_rate(rate, expected):
    assert _parse_rate(rate) == pytest.approx(expected, abs=1e-4)


def test_parse_probe_maps_core_fields():
    pr = parse_probe(_probe_json())
    assert isinstance(pr, ProbeResult)
    assert (pr.width, pr.height) == (1920, 1080)
    assert pr.fps == pytest.approx(25.0)
    assert pr.frame_count == 250
    assert pr.codec == "h264"


def test_parse_probe_prefers_avg_over_r_frame_rate():
    pr = parse_probe(_probe_json(avg_frame_rate="30000/1001", r_frame_rate="30/1"))
    assert pr.fps == pytest.approx(29.97003, abs=1e-4)


def test_parse_probe_frame_count_falls_back_to_fps_times_duration():
    # No nb_frames → round(fps * duration) = round(25 * 10) = 250.
    pr = parse_probe(_probe_json(nb_frames="N/A", duration="10.0"))
    assert pr.frame_count == 250


def test_parse_probe_reads_timecode_from_tags():
    pr = parse_probe(_probe_json(tags={"timecode": "01:00:00:00"}))
    assert pr.start_timecode == "01:00:00:00"


def test_parse_probe_without_video_stream_is_actionable():
    with pytest.raises(ValueError, match="no video stream"):
        parse_probe({"streams": [{"codec_type": "audio"}], "format": {}})


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_ingest_real_synthesized_clip(tmp_path):
    """End-to-end: synthesize a 2s/30fps/320x240 clip, then ingest → Source + ClipRef."""
    media = tmp_path / "synthetic.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=duration=2:size=320x240:rate=30", str(media)],
        check=True,
    )
    ing = FFmpegIngestor()

    source = ing.ingest(str(media))
    assert isinstance(source, Source)
    assert source.kind is SourceKind.VIDEO
    assert (source.width, source.height) == (320, 240)
    assert source.time_base.fps == pytest.approx(30.0, abs=0.5)
    assert source.frame_count >= 55  # ~60 frames; allow encoder rounding

    clip = ing.clip(str(media), max_frames=12)
    assert isinstance(clip, ClipRef)
    assert clip.n_frames == 12
    assert (clip.width, clip.height) == (320, 240)
    assert clip.frames[0] == 0 and np.all(np.diff(clip.frames) == 1)  # contiguous from start
