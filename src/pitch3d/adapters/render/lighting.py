"""Estimate the clip's lighting — the *measured* light colour for the night-match render.

The target broadcast is a floodlit NIGHT match (measured: no daytime sky, near-neutral faintly cool
floodlights, soft even multi-directional shadows). So "light from the clip" is **not** a sun
azimuth/elevation recovered from one hard shadow; it is the floodlights' **illuminant colour**,
which we read off the brightest near-neutral surfaces (white kit, pitch lines, bright grass) — a
white-patch Retinex estimate. That colour drives the fixed night-world *shape* (a dark sky + a ring
of soft high suns) built by :func:`pitch3d.adapters.blender.scene_builders.build_stadium_lighting`;
the blender CLI can override any field (the manual half of auto+manual). Lazy decode; numpy core.
"""

from __future__ import annotations

import numpy as np

from ..io.frames import iter_clip_frames

# Measured floodlight-night model for the target clip (project note 2026-06-28): a neutral, faintly
# cool light; a dark world at low strength; a ring of soft, high suns → even, low-contrast, faint
# multi-directional shadows. These travel in lighting.npz so the render is self-describing, and each
# is overridable from the blender_animate CLI. They also back-stop a clip with no measurable light.
NIGHT_LIGHT_RGB = (0.96, 0.96, 1.0)
NIGHT_KEY_ENERGY = 1.6
NIGHT_SKY_STRENGTH = 0.03  # ≈ the clip's dark upper region (≈0.17 display sRGB) — a dark night sky
NIGHT_SUN_COUNT = 4
NIGHT_SUN_ELEVATION_DEG = 65.0
NIGHT_SUN_ANGLE_DEG = 9.0


def estimate_light_color(
    images, *, bright_pct: float = 99.0, sat_max: float = 0.25, dark_min: float = 0.2
) -> np.ndarray:
    """Illuminant (floodlight) colour from frames: white-patch on the bright, near-neutral pixels.

    A floodlit white/grey surface reflects the illuminant almost unchanged, so the colour of the
    brightest *low-saturation* pixels is the light's colour. Per frame we keep pixels whose
    saturation ``(max-min)/max`` is below ``sat_max`` (neutral surfaces — not the green grass or
    a red shirt) and that are not near-black (``max > dark_min``, dropping the dark night sky
    whose hue is just noise). Across all kept pixels we take the ``bright_pct`` percentile per
    channel — a high percentile, not the lone max, rejects single specular hot pixels. The
    result is normalised so its max channel is 1.0: a tint like ``[0.96, 0.96, 1.0]`` = neutral
    with a faint cool cast. Falls back to the measured night default when no bright neutral
    pixel exists (e.g. a blank clip).

    ``images`` is any iterable of ``(H, W, 3)`` RGB arrays in ``[0, 1]``.
    """
    neutral = []
    for img in images:
        a = np.asarray(img, dtype=np.float32).reshape(-1, 3)
        if a.size == 0:
            continue
        mx = a.max(axis=1)
        mn = a.min(axis=1)
        sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 1e-4)
        keep = (sat < sat_max) & (mx > dark_min)
        if keep.any():
            neutral.append(a[keep])
    if not neutral:
        return np.asarray(NIGHT_LIGHT_RGB, dtype=np.float32)
    px = np.concatenate(neutral, axis=0)
    illum = np.percentile(px, bright_pct, axis=0).astype(np.float32)
    peak = float(illum.max())
    if peak <= 1e-4:
        return np.asarray(NIGHT_LIGHT_RGB, dtype=np.float32)
    return (illum / peak).astype(np.float32)


def estimate_lighting_from_clip(video_uri: str, frames, *, max_frames: int = 12) -> dict:
    """Measure the floodlight colour from an even spread of clip frames → the lighting.npz model.

    Decodes up to ``max_frames`` frames evenly spread across ``frames`` (the colour is
    resolution- and rotation-invariant, so no resize / 180° roll is needed — a rotation only
    reorders pixels), estimates the illuminant, and returns it alongside the fixed night-world
    defaults. The returned dict is ready for ``np.savez(path, **d)``.
    """
    frames = np.asarray(frames, dtype=int)
    if frames.size > max_frames:
        sel = np.unique(np.linspace(0, frames.size - 1, max_frames).round().astype(int))
        frames = frames[sel]
    images = [
        bgr[:, :, ::-1].astype(np.float32) / 255.0
        for _idx, bgr in iter_clip_frames(video_uri, frames.tolist())
    ]
    light_rgb = (
        estimate_light_color(images) if images else np.asarray(NIGHT_LIGHT_RGB, dtype=np.float32)
    )
    return {
        "light_rgb": np.asarray(light_rgb, dtype=np.float32),
        "key_energy": np.float32(NIGHT_KEY_ENERGY),
        "sky_strength": np.float32(NIGHT_SKY_STRENGTH),
        "sun_count": np.int64(NIGHT_SUN_COUNT),
        "elevation_deg": np.float32(NIGHT_SUN_ELEVATION_DEG),
        "sun_angle_deg": np.float32(NIGHT_SUN_ANGLE_DEG),
    }
