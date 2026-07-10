"""Studio correction re-run — run correction-family gates as an ephemeral layer.

The correction stages (temporal coherence + the physics gate chain) are pure and
torch-free: each takes a whole :class:`Scene`, APPENDS ``Correction`` objects, and
returns a new scene (never mutates poses in place). This module runs a config-driven
subset of that chain on the *baseline* scene (the scene minus any prior Studio
re-run corrections), isolates the newly-added corrections by id so they can be
reverted, and rebuilds SMPL-X FK only for the subjects those corrections touch.

Why this is the glass-box seam: the same layered-override model powers manual joint
edits (see ``scene_state.apply_and_persist_edit``). A physics re-run is just another
layer of ``Correction`` objects — so the existing ``/api/subject/.../joints`` path
serves the corrected poses with no special-casing. Toggle a gate → re-run → the 3D
and 2D overlays move.

Ephemeral by design: these corrections live in memory only and are NEVER written to
``edits.json`` (that file is reserved for the operator's manual joint edits).
"Revert to baseline" drops exactly the tracked ids and rebuilds.

Fidelity: the registry order and per-gate configs mirror
``app/controller.run_reconstruction`` so a Studio re-run matches what the real
pipeline would do. Four gates in that chain (foot_plant, momentum_smooth,
contact_lock, gravity_project) need live-pipeline providers (pelvis-target /
foot-position callables that come from the running reconstruction, not scene.json)
and cannot run here — they are surfaced as ``available: false`` rather than faked.
"""

from __future__ import annotations

import time
from dataclasses import fields as _dc_fields
from dataclasses import replace as _dc_replace
from typing import Any

from pitch3d.core.config.physics import load_physics_config
from pitch3d.core.correction.coherence import add_temporal_coherence
from pitch3d.core.correction.collision import collision_gate
from pitch3d.core.correction.facing_align import facing_align_gate
from pitch3d.core.correction.foot_floor import foot_floor_gate
from pitch3d.core.correction.inertia_smooth import inertia_smooth_gate
from pitch3d.core.correction.jerk_clamp import jerk_clamp_gate
from pitch3d.core.correction.joint_kinematics import joint_kinematic_gate
from pitch3d.core.correction.joint_smooth import joint_smooth_gate
from pitch3d.core.correction.kinematics import kinematic_gate
from pitch3d.core.correction.orient_verticality import orient_verticality_gate
from pitch3d.core.correction.orientation import orientation_gate
from pitch3d.core.correction.pose_motion_sync import pose_motion_sync_gate

from .config import PoseAnnotConfig
from .config import load as load_config
from .scene_state import SceneState, rebuild_subject_cache

# ── gate registry — order + configs mirror controller.run_reconstruction ─────────
# Each entry: (id, label, cfg_attr, fn, needs_fps, default_on)
#   cfg_attr    → attribute on the loaded PhysicsConfig (e.g. cfg.orient_verticality)
#   needs_fps   → the gate takes fps as a keyword
#   default_on  → True/False forces the default; None means "use cfg.<attr>.enabled"
#                 (coherence + kinematic always run in the pipeline, hence True)
_GATES: list[tuple[str, str, str, Any, bool, bool | None]] = [
    ("coherence",          "Coherence",           "coherence",          add_temporal_coherence, True,  True),
    ("kinematic",          "Kinematic",           "kinematic",          kinematic_gate,         True,  True),
    ("foot_floor",         "Foot floor",          "foot_floor",         foot_floor_gate,        False, None),
    ("joint_kinematic",    "Joint kinematic",     "joint",              joint_kinematic_gate,   True,  None),
    ("orientation",        "Orientation",         "orientation",        orientation_gate,       True,  None),
    ("collision",          "Collision",           "collision",          collision_gate,         False, None),
    ("orient_verticality", "Orient verticality",  "orient_verticality", orient_verticality_gate, False, None),
    ("pose_motion_sync",   "Pose-motion sync",    "pose_motion_sync",   pose_motion_sync_gate,  True,  None),
    ("facing_align",       "Facing align",        "facing_align",       facing_align_gate,      True,  None),
    ("inertia_smooth",     "Inertia smooth",      "inertia_smooth",     inertia_smooth_gate,    True,  None),
    ("jerk_clamp",         "Jerk clamp",          "jerk_clamp",         jerk_clamp_gate,        True,  None),
    ("joint_smooth",       "Joint smooth",        "joint_smooth",       joint_smooth_gate,      True,  None),
]

#: Chain gates that need a live-pipeline provider — cannot run from scene.json alone.
_PROVIDER_GATES: list[tuple[str, str, str]] = [
    ("foot_plant",      "Foot plant",      "needs a pelvis-target provider (live pipeline)"),
    ("momentum_smooth", "Momentum smooth", "needs a foot-position provider (live pipeline)"),
    ("contact_lock",    "Contact lock",    "needs a foot-position provider (live pipeline)"),
    ("gravity_project", "Gravity project", "needs a foot-position provider (live pipeline)"),
]

#: The flagship demo gate — off in 'default', dramatic when toggled on.
FLAGSHIP_GATE = "orient_verticality"


def _effective_enabled(gid: str, gate_cfg: Any, default_on: bool | None,
                       overrides: dict[str, bool]) -> bool:
    """Whether a gate runs: request override wins, else the profile's intent."""
    if gid in overrides:
        return bool(overrides[gid])
    if default_on is not None:
        return default_on
    return bool(getattr(gate_cfg, "enabled", True)) if gate_cfg is not None else False


def _dedup_last_wins(corrections: list) -> list:
    """Collapse duplicate ids keeping the LAST occurrence, preserving first-seen order.

    Gates use deterministic ids (``auto-orient-vertical-7``); on an already-corrected
    scene a re-run regenerates an existing id. The freshly-computed correction is the
    authority, but it must appear once (resolve applies every entry in order). We keep
    the original slot but the new payload.
    """
    order: list[str] = []
    latest: dict[str, object] = {}
    for c in corrections:
        if c.id not in latest:
            order.append(c.id)
        latest[c.id] = c
    return [latest[i] for i in order]


def _editable_params(gate_cfg: Any) -> list[dict]:
    """Numeric/bool fields of a gate's config, as editable specs for the UI.

    Sourced from the SAME config object the gate runs off (``getattr(phys, attr)``),
    so switching profile updates the shown defaults. ``enabled`` is omitted — the
    gate on/off checkbox already owns it. String/enum fields (e.g. ``smooth_method``,
    ``teleport_policy``) are skipped: the editor is numeric+bool only, so a typo
    can't produce an invalid config.
    """
    if gate_cfg is None:
        return []
    out: list[dict] = []
    for f in _dc_fields(gate_cfg):
        if f.name == "enabled":
            continue
        v = getattr(gate_cfg, f.name)
        # bool is a subclass of int — test it first.
        if isinstance(v, bool):
            typ = "bool"
        elif isinstance(v, int):
            typ = "int"
        elif isinstance(v, float):
            typ = "float"
        else:
            continue
        out.append({"key": f.name, "value": v, "type": typ})
    return out


def _clean_param_overrides(gate_cfg: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Cast incoming param values to each field's declared type; drop unknown keys.

    JSON carries no int/float distinction, so coerce to the field's CURRENT type
    (bool checked before int — ``isinstance(True, int)`` is True). Anything that
    doesn't cast, or isn't a known numeric/bool field, is silently dropped so a
    bad value can't wedge the run.
    """
    clean: dict[str, Any] = {}
    for k, v in (params or {}).items():
        if not hasattr(gate_cfg, k):
            continue
        cur = getattr(gate_cfg, k)
        try:
            if isinstance(cur, bool):
                clean[k] = bool(v)
            elif isinstance(cur, int):
                clean[k] = int(v)
            elif isinstance(cur, float):
                clean[k] = float(v)
            else:
                continue
        except (TypeError, ValueError):
            continue
    return clean


def gate_catalog(profile: str = "default") -> dict:
    """Ordered gate list + profile default-enabled flags, for the UI toggles."""
    try:
        phys = load_physics_config(profile=profile)
    except Exception as exc:  # noqa: BLE001 — bad profile shouldn't 500 the panel
        return {"profile": profile, "error": str(exc), "gates": [], "profiles": physics_profiles()}
    gates: list[dict] = []
    for gid, label, attr, _fn, _needs_fps, default_on in _GATES:
        gate_cfg = getattr(phys, attr, None)
        default_enabled = (
            default_on if default_on is not None
            else bool(getattr(gate_cfg, "enabled", True))
        )
        gates.append({
            "id": gid, "label": label, "available": True,
            "default_enabled": default_enabled,
            "flagship": gid == FLAGSHIP_GATE,
            "params": _editable_params(gate_cfg),
        })
    for gid, label, why in _PROVIDER_GATES:
        gates.append({
            "id": gid, "label": label, "available": False,
            "default_enabled": False, "reason": why,
        })
    return {"profile": profile, "gates": gates, "profiles": physics_profiles()}


def physics_profiles() -> list[str]:
    """Named profiles from config/physics.yaml (default first)."""
    try:
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parent.parent
        data = yaml.safe_load((root / "config" / "physics.yaml").read_text())
        names = list((data.get("profiles") or {}).keys())
        names.sort(key=lambda n: (n != "default", n))
        return names
    except Exception:  # noqa: BLE001
        return ["default"]


def run_corrections(
    st: SceneState,
    *,
    profile: str = "default",
    overrides: dict[str, bool] | None = None,
    params: dict[str, dict[str, Any]] | None = None,
    cfg: PoseAnnotConfig | None = None,
) -> dict:
    """Run the enabled correction gates on the baseline scene; rebuild affected FK.

    ``overrides`` decides which gates run (on/off); ``params`` carries per-gate
    field overrides (``{gate_id: {field: value}}``) applied on top of the profile
    defaults via :func:`dataclasses.replace` — the same config object the gate runs
    off, so tuning a knob is exactly what the real pipeline would do with that value.

    Returns a report: which gates ran, per-gate timing + corrections added + any
    param overrides that took effect, the affected subject ids, and FK-rebuild
    timing. The new corrections are tracked in ``st.studio_correction_ids``
    (ephemeral) and layered onto ``st.scene`` so the existing joint/mesh endpoints
    serve the corrected poses.
    """
    cfg = cfg or load_config()
    overrides = overrides or {}
    params = params or {}
    fps = float(cfg.fps or 25.0)
    t_start = time.perf_counter()

    phys = load_physics_config(profile=profile)

    with st.lock:
        # Freeze the true baseline once (before any studio re-run), then always
        # re-run from it — repeated re-runs are independent, never cumulative.
        if st.studio_baseline_corrections is None:
            st.studio_baseline_corrections = list(st.scene.corrections)
        baseline_corrs = st.studio_baseline_corrections
        n_baseline = len(baseline_corrs)
        scene = _dc_replace(st.scene, corrections=list(baseline_corrs))
        applied: list[str] = []
        gate_reports: dict[str, dict] = {}
        for gid, _label, attr, fn, needs_fps, default_on in _GATES:
            gate_cfg = getattr(phys, attr, None)
            if not _effective_enabled(gid, gate_cfg, default_on, overrides):
                continue
            # We already decided this gate runs; build its live config in one
            # replace: (a) force ``enabled=True`` — each gate ALSO short-circuits
            # on its own ``cfg.enabled`` (the 'default' profile ships most False),
            # so a toggled-on gate would no-op; (b) apply any per-gate param
            # overrides on top of the profile defaults.
            run_cfg = gate_cfg
            param_ovr: dict[str, Any] = {}
            if run_cfg is not None:
                replacements: dict[str, Any] = {}
                if getattr(run_cfg, "enabled", True) is False:
                    replacements["enabled"] = True
                param_ovr = _clean_param_overrides(run_cfg, params.get(gid, {}))
                replacements.update(param_ovr)
                if replacements:
                    run_cfg = _dc_replace(run_cfg, **replacements)
            t0 = time.perf_counter()
            try:
                if needs_fps:
                    scene, rep = fn(scene, run_cfg, fps=fps)
                else:
                    scene, rep = fn(scene, run_cfg)
            except Exception as exc:  # noqa: BLE001 — one bad gate must not kill the run
                gate_reports[gid] = {"error": str(exc)}
                continue
            gate_reports[gid] = {
                "ms": round((time.perf_counter() - t0) * 1000, 1),
                "corrections_added": int(getattr(rep, "corrections_added", 0) or 0),
            }
            if param_ovr:
                gate_reports[gid]["params"] = param_ovr
            applied.append(gid)

        # Gates append; everything past the baseline slice is gate-generated.
        generated = scene.corrections[n_baseline:]
        affected = sorted({
            int(c.target.subject_track_id) for c in generated
            if c.target.subject_track_id is not None
        })
        st.scene = _dc_replace(scene, corrections=_dedup_last_wins(scene.corrections))
        st.studio_correction_ids = {c.id for c in generated}

    fk_ms = _rebuild_affected(st, affected, cfg)

    return {
        "profile": profile,
        "applied": applied,
        "gates": gate_reports,
        "corrections_added": len(st.studio_correction_ids),
        "affected_subjects": affected,
        "fk_ms": fk_ms,
        "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
    }


def clear_corrections(st: SceneState, *, cfg: PoseAnnotConfig | None = None) -> dict:
    """Drop the ephemeral Studio corrections and rebuild the affected subjects."""
    cfg = cfg or load_config()
    with st.lock:
        ids = st.studio_correction_ids or set()
        if not ids or st.studio_baseline_corrections is None:
            st.studio_correction_ids = set()
            st.studio_baseline_corrections = None
            return {"cleared": 0, "affected_subjects": []}
        # Subjects that any studio correction touched must be rebuilt from baseline.
        affected = sorted({
            int(c.target.subject_track_id) for c in st.scene.corrections
            if c.id in ids and c.target.subject_track_id is not None
        })
        st.scene = _dc_replace(st.scene, corrections=list(st.studio_baseline_corrections))
        st.studio_correction_ids = set()
        st.studio_baseline_corrections = None

    fk_ms = _rebuild_affected(st, affected, cfg)
    return {"cleared": len(ids), "affected_subjects": affected, "fk_ms": fk_ms}


def _rebuild_affected(st: SceneState, track_ids: list[int],
                     cfg: PoseAnnotConfig) -> dict[int, float]:
    """Rebuild FK caches for the given subjects (outside st.lock — rebuild locks)."""
    fk_ms: dict[int, float] = {}
    for tid in track_ids:
        t0 = time.perf_counter()
        try:
            rebuild_subject_cache(st, tid, cfg)
        except KeyError:
            continue
        fk_ms[tid] = round((time.perf_counter() - t0) * 1000, 1)
    return fk_ms
