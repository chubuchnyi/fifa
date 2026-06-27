"""'Needs attention' UI panel + confidence map — the visual half of UX-4/FR-17 (M3-5).

The *ranking* (which spots are worst) is pure core (:func:`core.scene.review.attention_list`); the
per-marker confidence highlight (:func:`confidence_to_color`) already drives the overlay/radar. What
was missing is the **picture the operator/LLM reads**: a dense per-subject×per-frame confidence
heatmap and a ranked 'needs attention' panel. Both are pure numpy + the stdlib PNG encoder (no
GPU/Blender, no font engine — colour encodes the category and bar length the severity), so they are
deterministic and pixel-testable, and the :class:`SceneObserver` returns them as the ``UI``
observation (ADR-0008: the agent literally sees what needs fixing).
"""

from __future__ import annotations

import numpy as np

from ...core.scene.review import attention_list
from ...core.scene.scene import Scene
from .overlay import (
    _BACKGROUND,
    _BALL_COLOR,
    _LOW_CONF_COLOR,
    _encode_png,
    _subject_color,
    confidence_to_color,
)

# Each attention reason gets a distinct hue so the panel reads without text.
_REASON_COLOR: dict[str, tuple[int, int, int]] = {
    "low_confidence": _LOW_CONF_COLOR,     # red
    "high_reprojection": (255, 140, 0),    # orange
    "low_ball_height": _BALL_COLOR,        # gold
}
_OK_COLOR = (60, 200, 90)       # green — shown when nothing needs attention
_DIVIDER_COLOR = (90, 90, 90)


def _bands(n: int, total: int) -> list[tuple[int, int]]:
    """Split ``total`` pixels into ``n`` contiguous (start, end) bands via integer edges."""
    edges = np.linspace(0, total, n + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]


def render_confidence_map(
    scene: Scene, *, width: int = 192, height: int = 96
) -> np.ndarray:
    """Dense per-subject × per-frame confidence heatmap (rows = subjects + ball, cols = frames).

    Each subject row is its team colour pulled toward the warning red as confidence drops (the same
    :func:`confidence_to_color` ramp the markers use); the last row is the ball's height confidence.
    A subject/scene with no confidence data stays at the flat background — honest: there is nothing
    measured to show (R-6), not a falsely-green 'all good'.
    """
    fb = np.empty((int(height), int(width), 3), dtype=np.uint8)
    fb[:, :] = _BACKGROUND

    conf = scene.confidence
    rows: list[tuple[tuple[int, int, int], np.ndarray | None]] = []
    for subj in scene.subjects:
        arr = conf.subject_frame_conf.get(subj.track_id) if conf is not None else None
        rows.append((
            _subject_color(subj, scene.teams),
            np.asarray(arr, dtype=float) if arr is not None else None,
        ))
    if scene.ball is not None and scene.ball.height_confidence is not None:
        rows.append((_BALL_COLOR, np.asarray(scene.ball.height_confidence, dtype=float)))

    if not rows:
        return fb
    for (base, arr), (y0, y1) in zip(rows, _bands(len(rows), int(height)), strict=True):
        if arr is None or arr.size == 0:
            continue  # unknown → leave background (never fabricate a confidence)
        for c, (x0, x1) in zip(arr, _bands(arr.shape[0], int(width)), strict=True):
            fb[y0:y1, x0:x1] = confidence_to_color(base, float(c))
    return fb


def render_attention_panel(
    scene: Scene,
    *,
    width: int = 192,
    height: int = 96,
    max_items: int = 8,
    conf_threshold: float = 0.5,
    reproj_threshold_px: float = 10.0,
) -> np.ndarray:
    """Ranked 'needs attention' bars: one row per flagged problem, most urgent on top.

    Bar length ∝ severity normalized to the worst item; hue ∝ reason (red = low confidence, orange =
    high reprojection, gold = ball height). No font engine — the rank order, bar length and hue
    carry the meaning, so the panel is deterministic and pixel-testable. When nothing is flagged a
    single short green bar signals 'all clear' (distinct from the empty background).
    """
    fb = np.empty((int(height), int(width), 3), dtype=np.uint8)
    fb[:, :] = _BACKGROUND
    items = attention_list(
        scene, conf_threshold=conf_threshold, reproj_threshold_px=reproj_threshold_px,
        max_items=max_items,
    )
    if not items:
        y0, y1 = _bands(1, int(height))[0]
        fb[y0:y1, : max(1, int(width * 0.15))] = _OK_COLOR
        return fb
    max_score = max(it.score for it in items)
    for it, (y0, y1) in zip(items, _bands(len(items), int(height)), strict=True):
        color = _REASON_COLOR.get(it.reason, _LOW_CONF_COLOR)
        frac = float(it.score) / max_score if max_score > 0 else 1.0
        bar_w = max(1, int(round(width * float(np.clip(frac, 0.0, 1.0)))))
        pad = 1 if (y1 - y0) > 2 else 0
        fb[y0 + pad:y1 - pad, :bar_w] = color
    return fb


def render_attention_ui(
    scene: Scene,
    *,
    width: int = 192,
    height: int = 160,
    max_items: int = 8,
    conf_threshold: float = 0.5,
    reproj_threshold_px: float = 10.0,
) -> bytes:
    """Composite the editor 'needs attention' UI as a PNG: the ranked panel, a 1px divider, then the
    confidence heatmap. This is the screenshot the :class:`SceneObserver` returns for the ``UI``
    observation, closing the visual half of UX-4 (the LLM/operator sees the prioritized problems).
    """
    fb = np.empty((int(height), int(width), 3), dtype=np.uint8)
    fb[:, :] = _BACKGROUND
    split = int(height) // 2
    fb[:split] = render_attention_panel(
        scene, width=int(width), height=split, max_items=max_items,
        conf_threshold=conf_threshold, reproj_threshold_px=reproj_threshold_px,
    )
    fb[split:split + 1] = _DIVIDER_COLOR
    fb[split + 1:] = render_confidence_map(scene, width=int(width), height=int(height) - split - 1)
    return _encode_png(fb)


__all__ = ["render_attention_panel", "render_confidence_map", "render_attention_ui"]
