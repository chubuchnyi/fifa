"""The detector input square is per-clip data, not a constant in code.

Measured 2026-08-08 on two clips: 896 beat 560 by 31 % and 36 % on mid-pitch identity events, and
1288/1512 were worse than 896 on both. Two clips is not a law, so the value lives in
`config/detector_resolution.yaml` keyed by clip file name, and the lookup has to degrade safely —
a missing or broken config must not stop a run.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.adapters.models.detection import (
    RESOLUTION_CONFIG,
    RFDETRBackend,
    RFDETRDetector,
    resolution_for_clip,
)
from pitch3d.core.ports.io import ClipRef


def _clip(uri: str) -> ClipRef:
    return ClipRef(source_id="s", uri=uri, frames=np.arange(4), width=1920, height=1080, fps=30.0)


def test_the_shipped_config_is_readable_and_covers_both_measured_clips():
    assert RESOLUTION_CONFIG.exists(), f"missing {RESOLUTION_CONFIG}"
    assert resolution_for_clip(_clip("samples/video/Colombia-1-0-Congo-DR1080p.mp4")) == 896
    assert resolution_for_clip(_clip("/anywhere/14604731_1080_1920_30fps.mp4")) == 896


def test_the_lookup_is_by_file_name_not_by_path():
    """The same clip is at different paths on the laptop, the GPU box and the pod."""
    a = resolution_for_clip(_clip("samples/video/Colombia-1-0-Congo-DR1080p.mp4"))
    b = resolution_for_clip(_clip("/workspace/fifa/samples/video/Colombia-1-0-Congo-DR1080p.mp4"))
    assert a == b == 896


def test_an_unlisted_clip_falls_back_to_the_config_default():
    assert resolution_for_clip(_clip("/x/some-clip-we-never-measured.mp4")) == 896


def test_a_missing_config_returns_none_rather_than_raising(tmp_path):
    """A run must not die because a config file is absent; the backend then uses its own default."""
    assert resolution_for_clip(_clip("/x/a.mp4"), path=tmp_path / "nope.yaml") is None


def test_a_corrupt_config_returns_none_rather_than_raising(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("clips: [this is not a mapping\n")
    assert resolution_for_clip(_clip("/x/a.mp4"), path=bad) is None


def test_a_clip_entry_beats_the_default(tmp_path):
    cfg = tmp_path / "r.yaml"
    cfg.write_text("default: 896\nclips:\n  odd.mp4: 1064\n")
    assert resolution_for_clip(_clip("/x/odd.mp4"), path=cfg) == 1064
    assert resolution_for_clip(_clip("/x/other.mp4"), path=cfg) == 896


def test_an_explicit_resolution_skips_the_lookup_entirely():
    """`--detector-resolution` must win over the config, per the auto -> flag -> default rule."""
    det = RFDETRDetector(resolution=1064)
    backend = det._default_backend(_clip("samples/video/Colombia-1-0-Congo-DR1080p.mp4"))
    assert isinstance(backend, RFDETRBackend)
    assert backend.resolution == 1064


def test_no_explicit_resolution_takes_the_clip_s_own_value():
    det = RFDETRDetector()
    backend = det._default_backend(_clip("samples/video/Colombia-1-0-Congo-DR1080p.mp4"))
    assert backend.resolution == 896


def test_the_backend_itself_holds_no_hardcoded_optimum():
    """The measured number lives in the config. Constructing the backend bare gives RF-DETR's 560.

    This is the regression guard for the thing the user objected to: a value measured on two
    clips had been written into the code as a default.
    """
    assert RFDETRBackend().resolution is None


@pytest.mark.parametrize("bad", [500, 900, 1000])
def test_a_resolution_off_the_patch_stride_is_rejected_with_the_nearest_valid_one(bad: int):
    with pytest.raises(ValueError, match="divisible by 56"):
        RFDETRBackend(resolution=bad)
