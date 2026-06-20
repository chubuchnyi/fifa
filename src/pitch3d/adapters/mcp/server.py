"""MCP server — the live LLM control surface (driving adapter, ADR-0008).

Hexagonally this is a *driving* adapter, parallel to the CLI: it translates MCP tool calls
into application use-cases and streams visual feedback (rendered viewpoints, frame overlays,
UI screenshots) back to the model as image content blocks. It depends inward on the
application controller and on nothing in ``core`` directly.

The tool→use-case→content-block mapping lives in :mod:`.dispatch` and is import-free, so it is
fully unit-tested on the fake-wired app with **no** ``mcp`` extra. :func:`serve` is the only
part that needs the SDK: it registers the catalog and a call handler that delegates to
:func:`~pitch3d.adapters.mcp.dispatch.dispatch_tool`, then runs the stdio transport. The SDK is
lazy-imported and gated behind the optional ``mcp`` extra with an actionable install error.
"""

from __future__ import annotations

import base64
from typing import Any

from .dispatch import ImageBlock, TextBlock, dispatch_tool
from .tools import McpTool, tool_catalog

_SERVER_NAME = "pitch3d"


def build_catalog() -> list[McpTool]:
    """Expose the agent tool catalog (pure; safe to call without the ``mcp`` extra)."""
    return tool_catalog()


def _load_mcp():
    """Lazy-import the MCP SDK, raising an actionable error when the extra is absent."""
    try:
        import anyio
        from mcp import types
        from mcp.server.lowlevel import Server
        from mcp.server.stdio import stdio_server
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The live MCP server needs the optional 'mcp' extra: "
            "pip install 'pitch3d[mcp]'. The tool catalog is available without it "
            "via build_catalog()."
        ) from exc
    return anyio, types, Server, stdio_server


def serve(app: Any, *, transport: str = "stdio") -> None:
    """Start the MCP server bound to an application controller.

    Args:
        app: The :class:`~pitch3d.app.controller.Application` exposing the use-cases the tools
            map to (run_reconstruction, observe, apply_*, preview, render, export).
        transport: ``"stdio"`` (Claude Desktop / CLI agents). Only stdio is wired today.

    Each :class:`McpTool` in :func:`build_catalog` is advertised verbatim; a single call handler
    routes every tool through :func:`dispatch_tool`, returning the resulting text and image
    content blocks. ``observe``/``preview`` thus stream the rendered viewpoints straight back to
    the model. Blocks until the transport closes.
    """
    if transport != "stdio":
        raise ValueError(f"unsupported transport {transport!r}; only 'stdio' is wired")
    anyio, types, Server, stdio_server = _load_mcp()
    _run_stdio(app, anyio=anyio, types=types, Server=Server, stdio_server=stdio_server)


def _run_stdio(app, *, anyio, types, Server, stdio_server) -> None:  # pragma: no cover
    """Register the catalog + call handler on a low-level Server and run it over stdio.

    Only reachable with the 'mcp' extra installed (lazy-imported SDK + a live stdio loop).
    """
    server = Server(_SERVER_NAME)

    @server.list_tools()
    async def _list_tools() -> list[Any]:
        return [
            types.Tool(name=t.name, description=t.description, inputSchema=t.input_schema)
            for t in build_catalog()
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None) -> list[Any]:
        return _to_content(dispatch_tool(app, name, arguments), types)

    async def _main() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_main)


def _to_content(blocks, types) -> list[Any]:  # pragma: no cover
    """Convert pure dispatch blocks to MCP SDK content types (needs the 'mcp' extra)."""
    out: list[Any] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            out.append(types.TextContent(type="text", text=b.text))
        elif isinstance(b, ImageBlock):
            out.append(
                types.ImageContent(
                    type="image",
                    data=base64.b64encode(b.data).decode("ascii"),
                    mimeType=b.mime_type,
                )
            )
    return out


def main(argv: list[str] | None = None) -> int:
    """Console entry point (``pitch3d-mcp``): serve a fake-wired app over stdio.

    With the ``mcp`` extra installed this starts a real, runnable MCP server backed by the
    dependency-free fakes (so an agent can drive the whole golden path with no GPU). Without the
    extra it degrades gracefully: print the actionable install hint and the tool catalog, exit 0.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="pitch3d-mcp", description="Serve the pitch3d MCP tools.")
    parser.add_argument("--out-dir", default="out/mcp", help="where artifacts are written")
    parser.add_argument("--transport", default="stdio", choices=["stdio"])
    args = parser.parse_args(argv)

    from ...app.wiring import build_app

    app = build_app(out_dir=args.out_dir)
    try:
        serve(app, transport=args.transport)
    except RuntimeError as exc:
        print(exc)
        print("\nTool catalog (available without the extra):")
        for tool in build_catalog():
            flag = "  (mutates)" if tool.mutates else ""
            print(f"- {tool.name}{flag}: {tool.description}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
