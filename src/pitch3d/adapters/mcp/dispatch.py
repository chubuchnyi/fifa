"""Pure dispatch: MCP tool call → application use-case → content blocks (ADR-0008).

This is the import-free heart of the MCP driving adapter. :func:`dispatch_tool` maps each
:class:`~pitch3d.adapters.mcp.tools.McpTool` 1:1 onto an :class:`Application` method and turns
its result into a list of *content blocks* — :class:`TextBlock` (JSON / summaries) and
:class:`ImageBlock` (PNG bytes of each rendered viewpoint/overlay/UI). The blocks are plain data
with no MCP SDK dependency, so the whole control surface is unit-testable on the fake-wired app
without the optional ``mcp`` extra; :mod:`.server` only converts blocks to SDK content types.

The agent loop this powers: ``observe`` → reason over images+summary → mutate via a correction
tool → ``observe`` again (the visual-feedback loop, ADR-0008).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...core.correction.engine import make_keyframes, make_offset, make_refit, make_smoothing
from ...core.scene.layers import Correction, CorrectionTarget, TargetKind

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@dataclass(frozen=True)
class TextBlock:
    """A textual content block (a JSON result or an observation summary)."""

    text: str


@dataclass(frozen=True)
class ImageBlock:
    """An image content block — raw bytes the agent can look at, plus its MIME type."""

    data: bytes
    mime_type: str = "image/png"
    note: str | None = None


Block = TextBlock | ImageBlock


def dispatch_tool(app: Any, name: str, arguments: dict | None = None) -> list[Block]:
    """Run one MCP tool against ``app`` and return its content blocks.

    ``app`` is duck-typed to the :class:`Application` controller (one method per tool), so this
    function never imports the app layer. Unknown tool names raise ``ValueError``.
    """
    a = dict(arguments or {})
    if name == "list_episodes":
        eps = app.list_episodes()
        return [TextBlock(_json([_episode_payload(e) for e in eps]))]
    if name == "run_reconstruction":
        scene_id = app.run_reconstruction(a["episode_id"])
        return [TextBlock(_json({"scene_id": scene_id}))]
    if name == "observe":
        obs = app.observe(
            a["scene_id"],
            frame=a.get("frame"),
            viewpoints=a.get("viewpoints"),
            n_orbit=a.get("n_orbit", 0),
            include_ui=a.get("include_ui", False),
        )
        return observation_blocks(obs)
    if name == "get_attention":
        items = app.get_attention(a["scene_id"], max_items=a.get("max_items", 10))
        return [TextBlock(_json([_attention_payload(i) for i in items]))]
    if name == "apply_offset":
        corr = app.apply_offset(
            a["scene_id"], a["target"], a["frame_range"], a["delta"], note=a.get("note")
        )
        return [TextBlock(_json(_correction_payload(corr)))]
    if name == "apply_keyframes":
        corr = app.apply_keyframes(
            a["scene_id"], a["target"], a["frame_range"], a["key_frames"], a["key_values"],
            interp=a.get("interp", "linear"), note=a.get("note"),
        )
        return [TextBlock(_json(_correction_payload(corr)))]
    if name == "apply_smoothing":
        corr = app.apply_smoothing(
            a["scene_id"], a["target"], a["frame_range"],
            window=a.get("window", 5), method=a.get("method", "moving_average"),
            sigma=a.get("sigma", 1.0), note=a.get("note"),
        )
        return [TextBlock(_json(_correction_payload(corr)))]
    if name == "apply_refit":
        corr = app.apply_refit(
            a["scene_id"], a["target"], a["frame_range"], a.get("constraints"), note=a.get("note")
        )
        return [TextBlock(_json(_correction_payload(corr)))]
    if name == "set_correction_enabled":
        corr = app.set_correction_enabled(a["scene_id"], a["correction_id"], a["enabled"])
        return [TextBlock(_json(_correction_payload(corr)))]
    if name == "preview":
        result = app.preview(a["scene_id"], candidate_correction(a["candidate"]))
        return [TextBlock(_json(result))]
    if name == "render":
        res = app.render(a["scene_id"], quality=a.get("quality", "preview"))
        return [TextBlock(_json(_render_payload(res)))]
    if name == "export":
        res = app.export(a["scene_id"], a["format"], a["out_path"])
        return [TextBlock(_json(_export_payload(res)))]
    raise ValueError(f"unknown MCP tool {name!r}")


def candidate_correction(spec: dict) -> Correction:
    """Build a non-stored candidate :class:`Correction` from a JSON spec (for ``preview``).

    ``mode`` defaults to ``constant_offset``; the payload keys mirror the ``apply_*`` tools.
    """
    target = _target(spec["target"])
    frame_range = tuple(spec["frame_range"])
    mode = spec.get("mode", "constant_offset")
    note = spec.get("note")
    if mode == "constant_offset":
        delta = np.asarray(spec["delta"], float)
        return make_offset("candidate", target, frame_range, delta, note=note)
    if mode == "keyframe_interp":
        return make_keyframes(
            "candidate", target, frame_range, spec["key_frames"], spec["key_values"],
            interp=spec.get("interp", "linear"), note=note,
        )
    if mode == "temporal_smoothing":
        return make_smoothing(
            "candidate", target, frame_range, window=spec.get("window", 5),
            method=spec.get("method", "moving_average"), sigma=spec.get("sigma", 1.0), note=note,
        )
    if mode == "refit":
        return make_refit("candidate", target, frame_range, spec.get("constraints"), note=note)
    raise ValueError(f"unknown candidate mode {mode!r}")


def observation_blocks(obs: Any) -> list[Block]:
    """Turn an :class:`Observation` into a manifest TextBlock + one ImageBlock per snapshot."""
    lines = [f"scene={obs.scene_id} frame={obs.frame} images={len(obs.images)}"]
    for i, img in enumerate(obs.images):
        vp = img.viewpoint.value if img.viewpoint else "-"
        lines.append(f"  [{i}] {img.kind.value} vp={vp} frame={img.frame} {img.uri}")
    if obs.summary:
        lines.append("--- summary ---")
        lines.append(obs.summary)
    blocks: list[Block] = [TextBlock("\n".join(lines))]
    for img in obs.images:
        path = Path(img.uri)
        if path.is_file():
            mime = _MIME.get(path.suffix.lower(), "image/png")
            blocks.append(ImageBlock(path.read_bytes(), mime_type=mime, note=img.kind.value))
        else:
            blocks.append(TextBlock(f"[missing image artifact: {img.uri}]"))
    return blocks


# --- serializers (canonical types → JSON-friendly dicts) ----------------------------


def _target(spec: dict) -> CorrectionTarget:
    return CorrectionTarget(
        kind=TargetKind(spec["kind"]),
        subject_track_id=spec.get("subject_track_id"),
        joint_index=spec.get("joint_index"),
    )


def _episode_payload(ep: Any) -> dict:
    return {
        "id": ep.id,
        "name": ep.name,
        "source_id": ep.source_id,
        "start_frame": int(ep.start_frame),
        "end_frame": int(ep.end_frame),
        "n_frames": int(ep.n_frames),
        "origin": ep.origin.value,
    }


def _attention_payload(item: Any) -> dict:
    return {
        "score": float(item.score),
        "reason": item.reason,
        "track_id": item.track_id,
        "frame": item.frame,
        "detail": item.detail,
    }


def _correction_payload(corr: Any) -> dict:
    return {
        "id": corr.id,
        "mode": corr.mode.value,
        "target": {
            "kind": corr.target.kind.value,
            "subject_track_id": corr.target.subject_track_id,
            "joint_index": corr.target.joint_index,
        },
        "frame_range": [corr.frame_range.start, corr.frame_range.end],
        "enabled": bool(corr.enabled),
        "note": corr.note,
    }


def _render_payload(res: Any) -> dict:
    return {
        "uri": res.uri,
        "n_frames": int(res.n_frames),
        "quality": res.quality.value,
        "is_video": bool(res.is_video),
        "note": res.note,
    }


def _export_payload(res: Any) -> dict:
    return {"format": res.fmt.value, "paths": list(res.paths), "note": res.note}


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)!r}")


__all__ = [
    "Block",
    "ImageBlock",
    "TextBlock",
    "candidate_correction",
    "dispatch_tool",
    "observation_blocks",
]
