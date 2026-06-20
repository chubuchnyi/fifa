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

from .proxy import ProxyObject, ProxyPlan, ProxyView, build_proxy_plan
from .runner import BlenderProxyBuilder, BlenderSceneObserver, locate_blender, run_blender

__all__ = [
    "BlenderProxyBuilder",
    "BlenderSceneObserver",
    "ProxyObject",
    "ProxyPlan",
    "ProxyView",
    "build_proxy_plan",
    "locate_blender",
    "run_blender",
]
