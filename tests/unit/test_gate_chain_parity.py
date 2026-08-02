"""The Studio gate registry may not silently fall behind the real pipeline.

There are two places that run the physics gate chain, and they are not merged because they
legitimately differ. ``Application.run_reconstruction`` runs it on a live reconstruction, where
the pelvis-target and foot-position providers exist. ``poseannot/rerun.py`` re-runs it inside the
Studio UI against a loaded ``scene.json``, where those providers do not exist — four gates cannot
run there at all and are surfaced as ``available: false`` rather than faked (R-6: mark, never
hide).

What is *not* legitimate is drift. ``rerun.py``'s own docstring says its registry "mirrors
``app/controller.run_reconstruction``", and that mirror is maintained by hand: nothing stops a
17th gate being added to the controller and never reaching Studio. It would not fail anything —
the Studio would simply apply a weaker correction chain than the pipeline, silently, and the
operator's re-run would stop matching what the export actually contains. That is the bug class
this file exists to make impossible.

So the controller is treated as the source of truth and parsed, not duplicated: every gate it
calls must be accounted for in Studio as either runnable or explicitly provider-blocked. The
test cannot be satisfied by editing a list of expected names here, because there isn't one —
both sides are read from the code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from poseannot.rerun import _GATES, _PROVIDER_GATES

CONTROLLER = Path(__file__).resolve().parents[2] / "src" / "pitch3d" / "app" / "controller.py"

#: The one gate whose function is not named ``*_gate``. Kept as data so a rename is a
#: one-line fix here rather than a silently-dropped gate.
_ALIASES = {"add_temporal_coherence": "coherence"}


def _controller_gate_ids() -> set[str]:
    """Gate ids the live pipeline applies, read out of ``run_reconstruction`` itself."""
    fn = next(
        node
        for node in ast.walk(ast.parse(CONTROLLER.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == "run_reconstruction"
    )
    called = {
        node.func.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return {
        _ALIASES.get(name, name[: -len("_gate")])
        for name in called
        if name.endswith("_gate") or name in _ALIASES
    }


def test_the_parse_still_finds_the_chain():
    """Guards the guard. If ``run_reconstruction`` is renamed or the chain is extracted, this
    file starts asserting over an empty set and would pass no matter what drifted."""
    ids = _controller_gate_ids()
    assert len(ids) >= 12, f"only found {len(ids)} gates in the controller — has the chain moved?"
    assert "foot_floor" in ids and "jerk_clamp" in ids


def test_studio_accounts_for_every_gate_the_pipeline_runs():
    """Each controller gate is either runnable in Studio or listed as provider-blocked."""
    runnable = {g[0] for g in _GATES}
    blocked = {g[0] for g in _PROVIDER_GATES}
    missing = _controller_gate_ids() - runnable - blocked
    assert not missing, (
        f"{sorted(missing)} run in app/controller.run_reconstruction but Studio neither applies "
        f"them nor declares them provider-blocked. A Studio re-run would quietly apply a weaker "
        f"chain than the export. Add each to _GATES in poseannot/rerun.py, or to _PROVIDER_GATES "
        f"with the reason it cannot run from scene.json."
    )


def test_studio_does_not_invent_gates_the_pipeline_never_runs():
    """The mirror points one way too: Studio must not apply corrections the export will not."""
    declared = {g[0] for g in _GATES} | {g[0] for g in _PROVIDER_GATES}
    extra = declared - _controller_gate_ids()
    assert not extra, (
        f"{sorted(extra)} are declared in poseannot/rerun.py but not run by "
        f"app/controller.run_reconstruction. Studio would show the operator a correction the "
        f"real pipeline never applies."
    )


def test_a_gate_is_never_both_runnable_and_blocked():
    overlap = {g[0] for g in _GATES} & {g[0] for g in _PROVIDER_GATES}
    assert not overlap, f"{sorted(overlap)} appear in both _GATES and _PROVIDER_GATES"


@pytest.mark.parametrize("gate_id,_label,reason", _PROVIDER_GATES)
def test_every_blocked_gate_says_why(gate_id, _label, reason):
    """``available: false`` reaches the operator's screen, so the reason has to be a reason."""
    assert "provider" in reason.lower(), (
        f"{gate_id} is hidden from Studio with reason {reason!r}, which does not say which "
        f"provider is missing — the operator cannot act on that."
    )
