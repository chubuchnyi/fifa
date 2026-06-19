"""Application layer — the composition root and the shared control surface (ADR-0008).

:class:`Application` is the single set of use-cases that both driving adapters call: the CLI
dry-run (:func:`main`) and the MCP server. :func:`build_app` is dependency injection (fakes by
default); :func:`default_ports` is the all-fakes wiring.
"""

from __future__ import annotations

from .cli import main, run_dry_run
from .controller import Application
from .wiring import AppPorts, build_app, default_ports

__all__ = [
    "AppPorts",
    "Application",
    "build_app",
    "default_ports",
    "main",
    "run_dry_run",
]
