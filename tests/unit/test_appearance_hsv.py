"""HSV appearance provider — starter Re-ID adapter for identity_gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pitch3d.adapters.models.appearance_hsv import (
    HsvAppearanceConfig,
    _hsv_feature,
    make_hsv_appearance_provider,
)
from pitch3d.core.ports.io import ClipRef
from pitch3d.core.ports.perception import Tracklet


def _write_bgr_frames(tmp_path: Path, rgb_frames: list[tuple[int, int, int]]) -> Path:
    """Write each RGB tuple as one 32x32 BGR PNG frame; return the directory."""
    d = tmp_path / "frames"
    d.mkdir()
    for i, rgb in enumerate(rgb_frames):
        img = np.zeros((32, 32, 3), np.uint8)
        img[..., 0] = rgb[2]  # B
        img[..., 1] = rgb[1]  # G
        img[..., 2] = rgb[0]  # R
        cv2.imwrite(str(d / f"{i:04d}.png"), img)
    return d


def _clip(uri: str, n_frames: int = 8) -> ClipRef:
    return ClipRef(
        source_id="test", uri=uri, frames=np.arange(n_frames, dtype=int),
        width=32, height=32, fps=30.0,
    )


def test_hsv_feature_shape():
    crop = np.full((16, 16, 3), 128, dtype=np.uint8)
    cfg = HsvAppearanceConfig()
    feat = _hsv_feature(crop, cfg)
    # hue_bins + [mean_sat, mean_val, chroma_frac]
    assert feat.shape == (cfg.hue_bins + 3,)


def test_hsv_feature_yellow_vs_azure_disjoint_hue_bins():
    """A pure-yellow crop and a pure-azure crop populate different hue bins."""
    yellow = np.zeros((16, 16, 3), np.uint8); yellow[..., 1] = 220; yellow[..., 2] = 220
    azure = np.zeros((16, 16, 3), np.uint8); azure[..., 0] = 220; azure[..., 1] = 180
    cfg = HsvAppearanceConfig(hue_bins=16, sat_min=0.10, val_min=0.10)
    fy = _hsv_feature(yellow, cfg)
    fa = _hsv_feature(azure, cfg)
    dot = float(np.dot(fy[:16], fa[:16]))
    # normalised hue histograms of yellow vs azure barely overlap
    assert dot < 0.1, dot


def test_hsv_feature_zero_on_low_chroma():
    """A mid-grey crop below sat_min → chroma_frac ~ 0, hue histogram zeros."""
    grey = np.full((16, 16, 3), 100, dtype=np.uint8)
    cfg = HsvAppearanceConfig(sat_min=0.30)
    feat = _hsv_feature(grey, cfg)
    assert feat[-1] < 0.05   # chroma_frac
    assert feat[:16].sum() == 0.0


def test_provider_returns_dense_feature_per_frame(tmp_path: Path):
    """Provider decodes the tracklet's frame set and returns (T, D)."""
    # 4 yellow frames + 4 azure frames
    yellow_rgb = (220, 220, 0)
    azure_rgb = (0, 100, 220)
    d = _write_bgr_frames(tmp_path, [yellow_rgb] * 4 + [azure_rgb] * 4)
    clip = _clip(str(d))
    tracklet = Tracklet(
        track_id=1,
        frames=np.arange(8, dtype=int),
        bboxes_xyxy=np.tile([2.0, 2.0, 30.0, 30.0], (8, 1)),
        cls="player",
    )
    provider = make_hsv_appearance_provider(clip, HsvAppearanceConfig())
    feats = provider(tracklet)
    assert feats is not None
    assert feats.shape == (8, HsvAppearanceConfig().hue_bins + 3)
    # first 4 rows vs last 4 rows in different hue bins
    hist_y = feats[:4, :16].mean(axis=0)
    hist_a = feats[4:, :16].mean(axis=0)
    assert float(np.dot(hist_y, hist_a)) < 0.05


def test_provider_produces_features_dbscan_can_split(tmp_path: Path):
    """End-to-end wiring: HSV provider + identity_gate splits a yellow→azure track."""
    from pitch3d.core.config.gates import IdentityConfig
    from pitch3d.core.orchestration.identity import identity_gate
    from pitch3d.core.ports.perception import Tracks

    yellow_rgb = (220, 220, 0)
    azure_rgb = (0, 100, 220)
    d = _write_bgr_frames(tmp_path, [yellow_rgb] * 5 + [azure_rgb] * 5)
    clip = _clip(str(d), n_frames=10)
    tracklet = Tracklet(
        track_id=1,
        frames=np.arange(10, dtype=int),
        bboxes_xyxy=np.tile([2.0, 2.0, 30.0, 30.0], (10, 1)),
        cls="player",
    )
    provider = make_hsv_appearance_provider(clip, HsvAppearanceConfig())
    cfg = IdentityConfig(
        enabled=True, dbscan_eps=0.25, dbscan_min_samples=3,
        min_split_gap_frames=3, merge_enabled=False,
    )
    tracks, report = identity_gate(
        Tracks(tracklets=[tracklet], teams=[]), cfg,
        appearance_provider=provider,
    )
    assert report.tracks_split == 1
    assert len(tracks.tracklets) == 2


def test_provider_returns_none_on_unreadable_clip():
    """When decode raises, provider returns None so the gate skips the track."""
    clip = _clip("/nonexistent/path", n_frames=3)
    tracklet = Tracklet(
        track_id=1, frames=np.arange(3, dtype=int),
        bboxes_xyxy=np.tile([2.0, 2.0, 30.0, 30.0], (3, 1)), cls="player",
    )
    provider = make_hsv_appearance_provider(clip, HsvAppearanceConfig())
    feats = provider(tracklet)
    assert feats is None
