"""Blender adapter — the editor's proxy surface + LLM SCENE_3D feedback (ADR-0008, M1).

Blender ships its own Python, so the heavy half runs **out of process** rather than as a lazy
import. The dependency-free :mod:`.proxy` half assembles a serializable build plan from a scene
(unit-tested with no Blender); :mod:`.runner` drives ``blender --background`` to realize it as the
editable ``.blend`` editing surface (:class:`BlenderProxyBuilder`, M1 step 10) and to render proxy
``SCENE_3D`` viewpoints (:class:`BlenderSceneObserver`, A-7). A missing Blender binary raises an
actionable error; :class:`~pitch3d.adapters.fakes.FakeSceneObserver` still backs the loop
without it.
"""

from __future__ import annotations

from .cycles_plan import CyclesPlan, build_cycles_plan, write_cycles_plan
from .live import apply_drag, launch_live_session, resolved_root_at, serve_edits
from .proxy import (
    ProxyObject,
    ProxyPlan,
    ProxyView,
    build_proxy_plan,
    parse_subject_name,
    subject_object_name,
)
from .runner import (
    BlenderProxyBuilder,
    BlenderSceneObserver,
    locate_blender,
    run_blender,
    run_cycles_render,
)

__all__ = [
    "BlenderProxyBuilder",
    "BlenderSceneObserver",
    "CyclesPlan",
    "ProxyObject",
    "ProxyPlan",
    "ProxyView",
    "apply_drag",
    "build_cycles_plan",
    "build_proxy_plan",
    "launch_live_session",
    "locate_blender",
    "parse_subject_name",
    "resolved_root_at",
    "run_blender",
    "run_cycles_render",
    "serve_edits",
    "subject_object_name",
    "write_cycles_plan",
]
