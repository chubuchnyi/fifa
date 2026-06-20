"""Top-down tactical radar — a pure 2D minimap of the resolved scene (M1 step 10, FR-14, UX-3).

The radar is the read side of the "tactical / top" view: it drops the camera entirely and draws
the **resolved** world straight down onto the pitch plane — a schematic pitch rectangle with a
team-coloured dot per subject and a gold dot for the ball, both confidence-tinted exactly like the
reprojection overlay. That gives the operator (and the LLM observation loop, ADR-0008) an
angle-free "where is everyone" read that the broadcast camera can't, and it needs no GPU/Blender.

Interactive radar *placement* (drag a dot to move a player) is a GUI affordance that maps onto a
``ROOT_TRANSLATION`` correction; this module only renders the picture. It reuses the overlay's
resolved-marker assembly, confidence tint, stdlib PNG encoder and square painter so the two views
stay pixel-consistent.
"""

from __future__ import annotations

import numpy as np

from ...core.scene.scene import Scene
from .overlay import _draw_marker, _encode_png, _points_at_frame, _resolved_markers

_BACKGROUND = (12, 14, 12)     # near-black surround
_PITCH = (28, 84, 44)          # playing-surface green (lighter than the overlay background)
_LINE = (210, 220, 210)        # pitch markings


def world_to_radar(
    world_xy: np.ndarray, *, length: float, width: float, px_w: int, px_h: int, margin: int
) -> np.ndarray:
    """Map pitch-plane world XY (meters, origin at centre) to radar pixels ``(N, 2)``.

    World ``+X`` runs along the pitch length (→ image right) and ``+Y`` along the width (→ image
    *up*, so the picture reads like a tactics board). Points are placed inside the ``margin``-px
    border that frames the pitch; out-of-pitch points map outside it and are clipped when drawn.
    """
    xy = np.asarray(world_xy, dtype=float).reshape(-1, 2)
    inner_w = max(px_w - 2 * margin, 1)
    inner_h = max(px_h - 2 * margin, 1)
    u = margin + (xy[:, 0] + length / 2.0) / length * inner_w
    v = margin + (width / 2.0 - xy[:, 1]) / width * inner_h
    return np.column_stack([u, v])


def radar_to_world(
    radar_uv: np.ndarray, *, length: float, width: float, px_w: int, px_h: int, margin: int
) -> np.ndarray:
    """Inverse of :func:`world_to_radar`: radar pixels ``(N, 2)`` → pitch-plane world XY (meters).

    The exact algebraic inverse, so a dragged radar dot becomes the world position a
    ``ROOT_TRANSLATION`` correction targets (the interactive-placement seam, ADR-0010, UX-3).
    Pixels outside the pitch border map to off-pitch world coordinates (not clamped) — the caller
    decides whether an out-of-bounds drop is meaningful.
    """
    uv = np.asarray(radar_uv, dtype=float).reshape(-1, 2)
    inner_w = max(px_w - 2 * margin, 1)
    inner_h = max(px_h - 2 * margin, 1)
    x = (uv[:, 0] - margin) / inner_w * length - length / 2.0
    y = width / 2.0 - (uv[:, 1] - margin) / inner_h * width
    return np.column_stack([x, y])


def _draw_pitch(fb: np.ndarray, margin: int) -> None:
    """Fill the playing surface and paint the outline + halfway line (pure numpy slicing)."""
    height, width = fb.shape[:2]
    x0, y0, x1, y1 = margin, margin, width - margin, height - margin
    if x1 <= x0 or y1 <= y0:
        return
    fb[y0:y1, x0:x1] = _PITCH
    fb[y0, x0:x1] = fb[y1 - 1, x0:x1] = _LINE      # touchlines
    fb[y0:y1, x0] = fb[y0:y1, x1 - 1] = _LINE      # goal lines
    fb[y0:y1, (x0 + x1) // 2] = _LINE              # halfway line


def render_radar(
    scene: Scene,
    frame: int,
    *,
    width: int = 96,
    height: int = 64,
    margin: int = 4,
    dot_radius: int = 2,
) -> bytes:
    """Render a top-down radar PNG of the resolved scene at ``frame`` (bytes; stdlib only).

    Draws the pitch, then one confidence-tinted dot per subject/ball present at the frame. Reads
    only resolved state and never mutates the scene (same contract as the reprojection overlay).
    """
    dims = scene.field.dimensions
    fb = np.empty((int(height), int(width), 3), dtype=np.uint8)
    fb[:] = _BACKGROUND
    _draw_pitch(fb, margin)

    pts, colors = _points_at_frame(_resolved_markers(scene), int(frame))
    if pts.shape[0]:
        uv = world_to_radar(
            pts[:, :2], length=dims.length, width=dims.width,
            px_w=int(width), px_h=int(height), margin=margin,
        )
        for (u, v), color in zip(uv, colors, strict=True):
            _draw_marker(fb, u, v, color, dot_radius)
    return _encode_png(fb)


__all__ = ["radar_to_world", "render_radar", "world_to_radar"]
