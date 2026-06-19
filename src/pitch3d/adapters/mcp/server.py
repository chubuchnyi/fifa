"""MCP server — the live LLM control surface (driving adapter, ADR-0008).

Hexagonally this is a *driving* adapter, parallel to the CLI: it translates MCP tool calls
into application use-cases and streams visual feedback (rendered viewpoints, frame overlays,
UI screenshots) back to the model as image content blocks. It depends inward on the
application controller (Task 7) and on nothing in ``core`` directly.

The real server needs the optional ``mcp`` extra and the wired application; per the scaffold
convention every heavy adapter is an honest stub. The *catalog* in :mod:`.tools` is real and
import-free, so the agreed control surface is testable today.
"""

from __future__ import annotations

from .tools import McpTool, tool_catalog


def build_catalog() -> list[McpTool]:
    """Expose the agent tool catalog (pure; safe to call without the ``mcp`` extra)."""
    return tool_catalog()


def serve(app: object, *, transport: str = "stdio") -> None:
    """Start the MCP server bound to an application controller.

    Args:
        app: The application controller exposing the use-cases the tools map to
            (run_reconstruction, observe, apply_*, preview, resolve, render, export).
        transport: ``"stdio"`` (Claude Desktop / CLI agents) or ``"sse"`` (HTTP).

    The wiring is: for each :class:`McpTool` in :func:`build_catalog`, register an MCP tool
    whose handler validates against ``tool.input_schema`` and calls the matching ``app``
    method; ``observe``/``preview`` results are returned as text (the summary) plus one image
    content block per :class:`~pitch3d.core.ports.observation.ObservationImage`.
    """
    raise NotImplementedError(
        "Live MCP server requires the optional 'mcp' extra and the wired application "
        "controller (Task 7). The tool catalog is available now via build_catalog(). "
        "Install with `pip install pitch3d[mcp]` once the app layer lands."
    )


def main() -> int:
    """Console entry point (``pitch3d-mcp``). Prints the catalog; serving is not yet wired."""
    for tool in build_catalog():
        flag = "  (mutates)" if tool.mutates else ""
        print(f"- {tool.name}{flag}: {tool.description}")
    print(
        "\nThis is the MCP tool catalog (ADR-0008). The live server is not wired yet; "
        "see serve() and Task 7 (app controller)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
