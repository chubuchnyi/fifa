"""Concise textual scene state for the LLM agent (pairs with the visual snapshots).

Pure read-only digest of a :class:`Scene`: who's in it, ball-height/calibration confidence,
and the prioritized "needs attention" list (UX-4). The MCP adapter sends this alongside the
rendered viewpoints so the agent reasons over numbers *and* pixels.
"""

from __future__ import annotations

from ..scene.review import attention_list
from ..scene.scene import Scene


def scene_summary(scene: Scene, *, max_items: int = 8) -> str:
    """A short, LLM-friendly description of the scene's current (resolved-input) state."""
    lines: list[str] = [
        f"Scene {scene.id} (episode {scene.episode_id}): "
        f"{len(scene.subjects)} subject(s), {len(scene.corrections)} correction(s)."
    ]
    for s in scene.subjects:
        p = s.proposal.pose
        span = (int(p.frames[0]), int(p.frames[-1])) if p.frames.shape[0] else (0, 0)
        role = s.role.value if hasattr(s.role, "value") else s.role
        lines.append(
            f"  subject {s.track_id}: role={role} team={s.team_id} "
            f"frames {span[0]}-{span[1]} ({p.n_frames}f, {p.n_joints} joints)"
        )
    if scene.ball is not None and scene.ball.height_confidence.size:
        hc = scene.ball.height_confidence
        lines.append(
            f"  ball: {scene.ball.frames.shape[0]}f, "
            f"height_confidence mean={hc.mean():.2f} min={hc.min():.2f}"
        )
    if scene.field is not None and scene.field.calibration is not None:
        fc = scene.field.calibration.confidence
        if fc.size:
            lines.append(f"  field calibration confidence mean={fc.mean():.2f}")

    items = attention_list(scene, max_items=max_items)
    if items:
        lines.append(f"Needs attention ({len(items)}, highest first):")
        for it in items:
            where = []
            if it.track_id is not None:
                where.append(f"subject {it.track_id}")
            if it.frame is not None:
                where.append(f"frame {it.frame}")
            loc = ", ".join(where) or "scene"
            detail = f" — {it.detail}" if it.detail else ""
            lines.append(f"  - [{it.score:.2f}] {it.reason} ({loc}){detail}")
    else:
        lines.append("No frames flagged for attention.")
    return "\n".join(lines)
