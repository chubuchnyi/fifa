"""The MCP tool catalog — the control surface an LLM agent drives (ADR-0008).

This is the *contract*, expressed as data, for the operations the agent can perform on an
episode. It is intentionally pure (no MCP SDK import) so the design is inspectable and
testable without the optional ``mcp`` extra, and so the same catalog can document both the
live server and the CLI. Each tool maps 1:1 to an application use-case (Task 7); the agent's
loop is: ``observe`` → reason over images+summary → mutate via a correction tool → ``observe``
again (closing the visual-feedback loop the user asked for).

Image feedback flows back as MCP image content blocks built from
:class:`~pitch3d.core.ports.observation.ObservationImage` URIs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class McpTool:
    """One agent-callable operation: a name, a description, and a JSON input schema."""

    name: str
    description: str
    input_schema: dict
    mutates: bool = False  # whether it creates/edits corrections (vs. read-only)


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


_FRAME_RANGE = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 2,
    "maxItems": 2,
    "description": "Inclusive [start, end] frame range.",
}
_TARGET = _obj(
    {
        "kind": {
            "type": "string",
            "enum": [
                "pose_body_joint",
                "root_orientation",
                "root_translation",
                "shape_beta",
                "ball_position",
            ],
        },
        "subject_track_id": {"type": ["integer", "null"]},
        "joint_index": {"type": ["integer", "null"]},
    },
    ["kind"],
)


def tool_catalog() -> list[McpTool]:
    """Return the full set of agent-callable tools (stable order)."""
    return [
        McpTool(
            "list_episodes",
            "List episodes in the open project (id, name, frame span, source).",
            _obj({}),
        ),
        McpTool(
            "run_reconstruction",
            "Run the proposal pipeline (DETECT→TRACK→CALIBRATE→POSE→BALL) for an episode and "
            "return the assembled scene id. Cached; cheap on a re-run with unchanged inputs.",
            _obj({"episode_id": {"type": "string"}}, ["episode_id"]),
            mutates=True,
        ),
        McpTool(
            "observe",
            "Capture visual + textual feedback for a scene: the resolved 3D from several "
            "viewpoints, an optional source-frame reprojection overlay, and an optional UI "
            "screenshot, plus the prioritized 'needs attention' summary. This is how the "
            "agent SEES the consequences of its edits.",
            _obj(
                {
                    "scene_id": {"type": "string"},
                    "frame": {"type": ["integer", "null"]},
                    "viewpoints": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["current", "front", "back", "left", "right", "top",
                                     "broadcast", "orbit"],
                        },
                    },
                    "n_orbit": {"type": "integer", "default": 0},
                    "include_ui": {"type": "boolean", "default": False},
                },
                ["scene_id"],
            ),
        ),
        McpTool(
            "get_attention",
            "Return the ranked 'needs attention' list (low confidence / high reprojection "
            "error) so the agent knows where to look first (UX-4).",
            _obj({"scene_id": {"type": "string"}, "max_items": {"type": "integer", "default": 10}}),
        ),
        McpTool(
            "apply_offset",
            "Add a CONSTANT_OFFSET correction (vector add, or axis-angle compose for rotations) "
            "over a frame range.",
            _obj(
                {
                    "scene_id": {"type": "string"},
                    "target": _TARGET,
                    "frame_range": _FRAME_RANGE,
                    "delta": {"type": "array", "items": {"type": "number"}},
                },
                ["scene_id", "target", "frame_range", "delta"],
            ),
            mutates=True,
        ),
        McpTool(
            "apply_keyframes",
            "Add a KEYFRAME_INTERP correction (linear for vectors, slerp for rotations) from "
            "operator/agent keyframes.",
            _obj(
                {
                    "scene_id": {"type": "string"},
                    "target": _TARGET,
                    "frame_range": _FRAME_RANGE,
                    "key_frames": {"type": "array", "items": {"type": "integer"}},
                    "key_values": {"type": "array", "items": {"type": "array",
                                   "items": {"type": "number"}}},
                },
                ["scene_id", "target", "frame_range", "key_frames", "key_values"],
            ),
            mutates=True,
        ),
        McpTool(
            "apply_smoothing",
            "Add a TEMPORAL_SMOOTHING correction (quaternion-aware on rotations) over a range.",
            _obj(
                {
                    "scene_id": {"type": "string"},
                    "target": _TARGET,
                    "frame_range": _FRAME_RANGE,
                    "window": {"type": "integer", "default": 5},
                    "method": {"type": "string", "enum": ["moving_average", "gaussian"],
                               "default": "moving_average"},
                    "sigma": {"type": "number", "default": 1.0},
                },
                ["scene_id", "target", "frame_range"],
            ),
            mutates=True,
        ),
        McpTool(
            "apply_refit",
            "Add a REFIT correction: re-run constraint-guided HMR on a range via the pose port.",
            _obj(
                {
                    "scene_id": {"type": "string"},
                    "target": _TARGET,
                    "frame_range": _FRAME_RANGE,
                    "constraints": {"type": "object"},
                },
                ["scene_id", "target", "frame_range"],
            ),
            mutates=True,
        ),
        McpTool(
            "set_correction_enabled",
            "Toggle a correction on/off without deleting it (compare / reset-to-model, UX-5).",
            _obj(
                {
                    "scene_id": {"type": "string"},
                    "correction_id": {"type": "string"},
                    "enabled": {"type": "boolean"},
                },
                ["scene_id", "correction_id", "enabled"],
            ),
            mutates=True,
        ),
        McpTool(
            "preview",
            "Resolve the scene AS IF a candidate correction were applied, without storing it "
            "(FR-23) — then observe, to let the agent check before committing.",
            _obj(
                {"scene_id": {"type": "string"}, "candidate": {"type": "object"}},
                ["scene_id", "candidate"],
            ),
        ),
        McpTool(
            "render",
            "Render the resolved scene along a camera path (PREVIEW by default; FINAL is "
            "expensive and cached).",
            _obj(
                {
                    "scene_id": {"type": "string"},
                    "quality": {"type": "string", "enum": ["preview", "final"],
                                "default": "preview"},
                },
                ["scene_id"],
            ),
            mutates=True,
        ),
        McpTool(
            "export",
            "Export the resolved scene to a format (gltf/glb/usd/fbx/alembic/json/...).",
            _obj(
                {"scene_id": {"type": "string"}, "format": {"type": "string"},
                 "out_path": {"type": "string"}},
                ["scene_id", "format", "out_path"],
            ),
            mutates=True,
        ),
    ]
