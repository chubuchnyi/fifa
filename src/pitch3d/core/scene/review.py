"""Review analysis: the "needs attention" prioritization (UX-4, FR-17).

Pure functions over a :class:`Scene` and its :class:`ConfidenceMap`. No GUI, no
infrastructure — the actual list widget is a Blender-adapter concern, but the *ranking*
is core logic so it is deterministic and unit-testable. Higher ``score`` = more urgent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import Scene


@dataclass
class AttentionItem:
    """One flagged problem the operator should look at."""

    score: float                 # urgency; higher first
    reason: str                  # "low_confidence" | "high_reprojection" | "low_ball_height"
    track_id: int | None = None  # subject (None for ball/global)
    frame: int | None = None
    detail: str | None = None


def attention_list(
    scene: Scene,
    *,
    conf_threshold: float = 0.5,
    reproj_threshold_px: float = 10.0,
    max_items: int | None = None,
) -> list[AttentionItem]:
    """Rank problem spots by severity (UX-4).

    Combines three signals, each normalized so they are comparable:
        * per-frame subject confidence below ``conf_threshold``;
        * reprojection error above ``reproj_threshold_px`` (FR-16);
        * ball height confidence below ``conf_threshold`` (R-4).

    Returns items sorted by descending ``score``; ``max_items`` truncates the list.
    Returns an empty list when the scene has no confidence data.
    """
    items: list[AttentionItem] = []
    conf = scene.confidence

    if conf is not None:
        for track_id, arr in conf.subject_frame_conf.items():
            arr = np.asarray(arr, dtype=float)
            for frame, c in enumerate(arr):
                if c < conf_threshold:
                    items.append(
                        AttentionItem(
                            score=(conf_threshold - float(c)) / max(conf_threshold, 1e-9),
                            reason="low_confidence",
                            track_id=int(track_id),
                            frame=int(frame),
                            detail=f"confidence {c:.2f}",
                        )
                    )
        for track_id, arr in conf.reprojection_error_px.items():
            arr = np.asarray(arr, dtype=float)
            for frame, e in enumerate(arr):
                if e > reproj_threshold_px:
                    items.append(
                        AttentionItem(
                            score=float(e) / max(reproj_threshold_px, 1e-9),
                            reason="high_reprojection",
                            track_id=int(track_id),
                            frame=int(frame),
                            detail=f"reproj {e:.1f}px",
                        )
                    )

    if scene.ball is not None:
        hc = np.asarray(scene.ball.height_confidence, dtype=float)
        for i, c in enumerate(hc):
            if c < conf_threshold:
                items.append(
                    AttentionItem(
                        score=(conf_threshold - float(c)) / max(conf_threshold, 1e-9),
                        reason="low_ball_height",
                        track_id=None,
                        frame=int(scene.ball.frames[i]),
                        detail=f"ball height conf {c:.2f}",
                    )
                )

    items.sort(key=lambda it: it.score, reverse=True)
    return items if max_items is None else items[:max_items]
