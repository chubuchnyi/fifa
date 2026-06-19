"""MCP driving adapter — exposes the editor to an LLM agent (ADR-0008).

The tool catalog is import-free and usable now; the live server is gated behind the ``mcp``
extra and the application controller.
"""

from __future__ import annotations

from .server import build_catalog, serve
from .tools import McpTool, tool_catalog

__all__ = ["McpTool", "build_catalog", "serve", "tool_catalog"]
