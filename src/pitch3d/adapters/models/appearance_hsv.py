"""Cheap HSV appearance provider for :mod:`pitch3d.core.orchestration.identity`.

The identity gate needs per-frame appearance features to detect ID swaps and
merge disjoint tracklets of the same physical player. A production system
plugs OSNet or CLIP-ReIdent here; this adapter is the numpy-only starter that
ships without torch — samples the tracklet's bbox crops in HSV space and
returns a small histogram feature per frame.

For a broadcast football clip the kit is the dominant colour → HSV mean +
hue histogram is enough to separate team A from team B and catch mid-track
identity swaps. Not enough to distinguish two players ON THE SAME TEAM — that
needs Re-ID embeddings and is deferred to the next tier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.orchestration.identity import AppearanceProvider
from ...core.ports.io import ClipRef
from ...core.ports.perception import Tracklet
from ..io.frames import iter_clip_frames


@dataclass(frozen=True)
class HsvAppearanceConfig:
    """Knobs for :func:`make_hsv_appearance_provider`."""

    hue_bins: int = 16              # circular hue histogram bin count
    sat_min: float = 0.15           # per-pixel saturation floor (skip low-chroma pixels)
    val_min: float = 0.10           # per-pixel value floor (skip near-black)
    crop_shrink: float = 0.15       # fraction of each bbox edge to shrink inward
    max_crop_pixels: int = 4000     # cap on crop area for the histogram (subsample if larger)
    stride: int = 1                 # process every Nth frame; interpolate rest


def _shrink_bbox(bbox: np.ndarray, shrink: float, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    dx = (x1 - x0) * float(shrink) * 0.5
    dy = (y1 - y0) * float(shrink) * 0.5
    x0 = int(max(0, x0 + dx))
    y0 = int(max(0, y0 + dy))
    x1 = int(min(w, x1 - dx))
    y1 = int(min(h, y1 - dy))
    return x0, y0, x1, y1


def _hsv_feature(
    crop_bgr: np.ndarray, cfg: HsvAppearanceConfig,
) -> np.ndarray:
    """Return a fixed-length feature vector for one crop.

    Layout: ``[hue_hist (hue_bins), mean_sat, mean_val, chroma_frac]``.
    ``hue_hist`` is normalised so its L1 sum equals 1 (only over the chroma-
    keeping pixels).
    """
    import cv2
    if crop_bgr.size == 0:
        return np.zeros(cfg.hue_bins + 3, dtype=float)
    if crop_bgr.shape[0] * crop_bgr.shape[1] > cfg.max_crop_pixels:
        step = int(np.ceil(np.sqrt(
            (crop_bgr.shape[0] * crop_bgr.shape[1]) / cfg.max_crop_pixels
        )))
        crop_bgr = crop_bgr[::step, ::step]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV_FULL)
    H = hsv[..., 0].astype(np.float32) * (360.0 / 256.0)
    S = hsv[..., 1].astype(np.float32) / 255.0
    V = hsv[..., 2].astype(np.float32) / 255.0
    keep = (S >= cfg.sat_min) & (V >= cfg.val_min)
    chroma_frac = float(keep.mean())
    if not keep.any():
        hist = np.zeros(cfg.hue_bins, dtype=float)
    else:
        Hk = H[keep]
        bins = np.linspace(0.0, 360.0, cfg.hue_bins + 1)
        hist, _ = np.histogram(Hk, bins=bins)
        s = float(hist.sum())
        hist = (hist.astype(float) / s) if s > 0 else hist.astype(float)
    mean_sat = float(S[keep].mean()) if keep.any() else 0.0
    mean_val = float(V[keep].mean()) if keep.any() else 0.0
    return np.concatenate([hist, [mean_sat, mean_val, chroma_frac]])


def make_hsv_appearance_provider(
    clip: ClipRef, cfg: HsvAppearanceConfig | None = None,
) -> AppearanceProvider:
    """Return an :data:`AppearanceProvider` that samples HSV features from the clip.

    The provider is closure-bound to ``clip``; call it once per pipeline run.
    Frames are decoded lazily via :func:`iter_clip_frames`, which is a hot path,
    so we deduplicate frame indices across all tracklets in a first pass and
    slice the per-frame features back to each tracklet's frame set.
    """
    cfg = cfg or HsvAppearanceConfig()

    def provider(tracklet: Tracklet) -> np.ndarray | None:
        frames = np.asarray(tracklet.frames, dtype=int).reshape(-1)
        boxes = np.asarray(tracklet.bboxes_xyxy, dtype=float).reshape(-1, 4)
        if frames.size == 0:
            return None
        stride = max(1, int(cfg.stride))
        want = frames[::stride]
        # decode each requested frame in order
        idx_to_feature: dict[int, np.ndarray] = {}
        want_set = set(int(f) for f in want)
        try:
            for idx, bgr in iter_clip_frames(clip.uri, sorted(want_set)):
                # find box for this frame
                row_positions = np.where(frames == idx)[0]
                if row_positions.size == 0:
                    continue
                r = int(row_positions[0])
                h, w = bgr.shape[:2]
                x0, y0, x1, y1 = _shrink_bbox(boxes[r], cfg.crop_shrink, w, h)
                if x1 <= x0 or y1 <= y0:
                    idx_to_feature[idx] = np.zeros(cfg.hue_bins + 3, dtype=float)
                    continue
                crop = bgr[y0:y1, x0:x1]
                idx_to_feature[idx] = _hsv_feature(crop, cfg)
        except (FileNotFoundError, RuntimeError):
            return None
        # dense per-frame feature array (rows aligned with tracklet.frames)
        out = np.zeros((frames.shape[0], cfg.hue_bins + 3), dtype=float)
        last: np.ndarray | None = None
        for i, f in enumerate(frames):
            if int(f) in idx_to_feature:
                last = idx_to_feature[int(f)]
                out[i] = last
            elif last is not None:
                out[i] = last   # hold last sampled feature for skipped frames
            # else out[i] stays zeros (early frames before first sample)
        return out

    return provider


__all__ = [
    "HsvAppearanceConfig",
    "make_hsv_appearance_provider",
]
