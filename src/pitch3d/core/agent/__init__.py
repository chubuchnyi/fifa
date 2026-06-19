"""Agent support: pure viewpoint camera math + textual scene digest for LLM feedback.

The control surface the agent drives is the application API (shared by the CLI and the MCP
adapter); this package holds only the core, port-free pieces the feedback loop needs.
"""

from __future__ import annotations

from .summary import scene_summary
from .viewpoints import (
    action_centroid,
    camera_at,
    default_intrinsics,
    look_at,
    standard_viewpoints,
)

__all__ = [
    "action_centroid",
    "camera_at",
    "default_intrinsics",
    "look_at",
    "scene_summary",
    "standard_viewpoints",
]
