"""poseannot Pipeline Studio — per-stage manifest (Phase 0/1, read-only).

Exposes the pipeline's stage DAG (docs/pipeline-studio.md) as an inspectable
manifest so the UI can render every step, its temporal shape, its parameters,
and — per loaded clip — whether that stage's data is available to visualise yet.

Torch-free by construction: it imports only pure-half configs (``core/``) for the
real default knobs and reads ``config/physics.yaml``; it never touches an adapter
(the heavy half), so the manifest endpoint stays fast and cannot pull in models.

This is the backbone of the Studio UI. Editing/re-running stages (Phases 2-6) hangs
off the same stage ids; here everything is read-only introspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── real default knobs, pulled from the pure-half configs (never adapters) ──────
def _pure_param_defaults() -> dict[str, list[dict]]:
    """Params sourced from the actual ``core/`` config dataclasses + physics.yaml.

    Wrapped in try/except so a future refactor that moves a class can't 500 the
    manifest — we degrade to the documented static defaults instead.
    """
    out: dict[str, list[dict]] = {}
    try:
        from pitch3d.core.orchestration.continuity import StitchConfig

        c = StitchConfig()
        out["stitch"] = [
            _p("max_gap", c.max_gap, "int", "StitchConfig"),
            _p("max_center_dist", c.max_center_dist, "float", "StitchConfig"),
            _p("max_size_ratio", c.max_size_ratio, "float", "StitchConfig"),
            _p("min_track_frames", c.min_track_frames, "int", "StitchConfig"),
            _p("require_same_team", c.require_same_team, "bool", "StitchConfig"),
            _p("velocity_window", c.velocity_window, "int", "StitchConfig"),
        ]
    except Exception:  # noqa: BLE001
        pass
    try:
        from pitch3d.core.correction.coherence import CoherenceConfig

        c = CoherenceConfig()
        out["coherence"] = [
            _p("max_fill_gap", c.max_fill_gap, "int", "CoherenceConfig"),
            _p("smooth_window", c.smooth_window, "int", "CoherenceConfig"),
            _p("smooth_root_translation", c.smooth_root_translation, "bool", "CoherenceConfig"),
            _p("smooth_root_orientation", c.smooth_root_orientation, "bool", "CoherenceConfig"),
            _p("filled_confidence", c.filled_confidence, "float", "CoherenceConfig"),
            _p("extrapolated_confidence", c.extrapolated_confidence, "float", "CoherenceConfig"),
            _p("extrapolate_decay", c.extrapolate_decay, "float", "CoherenceConfig"),
        ]
    except Exception:  # noqa: BLE001
        pass
    try:
        from pitch3d.core.correction import kinematics as kin

        out["physics_kinematic"] = [
            _p("HUMAN_MAX_SPEED", kin.HUMAN_MAX_SPEED, "float", "kinematics"),
            _p("HUMAN_MAX_ACCEL", kin.HUMAN_MAX_ACCEL, "float", "kinematics"),
            _p("teleport_factor", kin.teleport_factor if hasattr(kin, "teleport_factor") else 2.0,
               "float", "kinematics"),
        ]
    except Exception:  # noqa: BLE001
        pass
    out["physics_profiles"] = _physics_profiles()
    return out


def _physics_profiles() -> list[dict]:
    """The named physics profiles from config/physics.yaml (default first)."""
    try:
        import yaml

        data = yaml.safe_load((_REPO_ROOT / "config" / "physics.yaml").read_text())
        names = list((data.get("profiles") or {}).keys())
        names.sort(key=lambda n: (n != "default", n))
        return [_p("profile", names, "enum", "config/physics.yaml")] if names else []
    except Exception:  # noqa: BLE001
        return []


def _p(key: str, value: Any, typ: str, source: str, note: str = "") -> dict:
    return {"key": key, "value": value, "type": typ, "source": source, "note": note}


# ── static DAG description (structure never changes per clip) ────────────────────
# family: which backend family runs the stage today (honesty about the substrate):
#   cached      → core/orchestration/stages.run_cached (content-addressed)
#   continuity  → core/orchestration/continuity (runs around the cached stages)
#   correction  → app/controller.run_reconstruction (emits layered Corrections)
#   tail        → resolve/export/render scripts
# layer: which frontend overlay the stage's OUTPUT drives (Phase-1 renderers).
def _stage_defs() -> list[dict]:
    pd = _pure_param_defaults()
    detect_params = [
        _p("score_threshold", 0.3, "float", "default (RF-DETR)"),
        _p("classes", ["ball", "goalkeeper", "player", "referee"], "enum", "default"),
    ]
    track_params = [
        _p("team_clusters", 2, "int", "default (k-means)"),
        _p("class_vote", "majority", "enum", "default (ByteTrack)"),
    ]
    calib_params = [
        _p("model", "PnLCalib-HRNet", "str", "default"),
        _p("input_size", "960x540", "str", "default"),
    ]
    pose_params = [
        _p("backend", ["smplestx", "sam3dbody"], "enum", "--pose-backend"),
        _p("bbox_expand", 1.25, "float", "default"),
    ]
    ball_params = [
        _p("backend", ["tracknet", "wasb"], "enum", "default"),
        _p("score_threshold", 0.1, "float", "default"),
    ]
    ball3d_params = [
        _p("max_speed_mps", 35.0, "float", "ball_lift"),
        _p("contact_px", 140, "int", "ball_lift"),
    ]
    return [
        _s("decode", "1 · Decode", "whole_clip", [], "instant", "cached",
           "source video uri", "video frames + ClipRef", "frame",
           [_p("fps", 29.97, "float", "ffprobe")]),
        _s("detect", "2 · Detect", "per_frame", ["decode"], "seconds", "cached",
           "frame image", "boxes {cls, score, xyxy}", "boxes", detect_params),
        _s("track", "3 · Track", "per_track", ["detect"], "seconds", "cached",
           "per-frame boxes", "tracklets {id, frames, bboxes, team}", "tracks", track_params),
        _s("stitch", "4 · Stitch", "per_track", ["track"], "instant", "continuity",
           "tracklets", "re-linked tracklets + StitchReport", "tracks",
           pd.get("stitch", [])),
        _s("identity", "5 · Identity", "per_track", ["stitch"], "seconds", "continuity",
           "tracklets", "jersey #, role", "labels",
           [_p("enabled", False, "bool", "default (off)")]),
        _s("calibrate", "6 · Calibrate", "per_frame", ["decode"], "seconds", "cached",
           "frame image", "homography (3,3) → pitch↔world", "pitch", calib_params),
        _s("pose", "7 · Pose", "per_track", ["track", "calibrate"], "gpu_minutes", "cached",
           "tracklet crop + calibration", "SMPL-X {global_orient*, body_pose, transl, betas}",
           "pose", pose_params,
           note="*global_orient stays in the CAMERA frame (pose.py:188) — inverted bodies are "
                "expected, not a UI bug; toggle orient_verticality in Physics to fix."),
        _s("ball2d", "8 · Ball 2D", "per_frame", ["decode"], "seconds", "cached",
           "frame image", "ball pixel + score", "ball", ball_params),
        _s("ball3d", "9 · Ball 3D", "per_frame", ["ball2d", "calibrate"], "instant", "cached",
           "ball 2D + homography", "world xyz + on_ground", "ball", ball3d_params),
        _s("coherence", "10 · Coherence", "per_track", ["pose"], "instant", "correction",
           "poses", "gap-filled/extended poses + smoothing correction", "confidence",
           pd.get("coherence", []),
           note="Confidence ribbon: measured=1.0, gap-fill=0.3, edge-coast=0.2."),
        _s("physics", "11 · Physics", "per_track", ["coherence"], "instant", "correction",
           "poses", "corrected poses + layered Corrections", "confidence",
           pd.get("physics_kinematic", []) + pd.get("physics_profiles", []),
           note="14 opt-in gates ship enabled:false in the 'default' profile — corrections "
                "resolve/bake at export, so scene.json shows 0."),
        _s("assemble", "12 · Assemble", "whole_clip", ["pose", "ball3d"], "instant", "cached",
           "all stage outputs", "Scene + ConfidenceMap", "summary", []),
        _s("export", "13 · Resolve/Export", "whole_clip", ["assemble"], "instant", "tail",
           "Scene ⊕ Corrections", "scene.json / glTF", "summary",
           [_p("format", ["json", "gltf"], "enum", "default")]),
        _s("render", "14 · Render", "whole_clip", ["export"], "gpu_minutes", "tail",
           "scene.json → npz", "Blender Cycles → mp4 per camera", "summary",
           [_p("cameras", ["broadcast", "sideline", "top", "goal"], "enum", "video_defaults"),
            _p("samples", 32, "int", "blender_animate")]),
    ]


def _s(sid, label, temporal, deps, cost, family, in_desc, out_desc, layer, params, note=""):
    return {
        "id": sid,
        "label": label,
        "temporal": temporal,
        "depends_on": deps,
        "rerun_cost": cost,
        "family": family,
        "input": {"desc": in_desc},
        "output": {"desc": out_desc, "layer": layer},
        "params": params,
        "note": note,
    }


# ── per-clip availability: what the loaded scene.json can actually show today ────
def _availability(sid: str, scene: Any) -> tuple[str, str]:
    """Return (status, note) for a stage given the live scene.

    status ∈ {"live", "partial", "unmaterialized"}:
      live           → the frontend can render this stage's layer from scene.json now
      partial        → some data present (e.g. team/role) but not the full per-frame layer
      unmaterialized → needs a Phase-0 run bundle (raw boxes/tracklets/homographies)
    """
    has_cam = getattr(scene, "camera", None) is not None
    has_field = getattr(scene, "field", None) is not None
    has_subjects = bool(getattr(scene, "subjects", None))
    has_ball = getattr(scene, "ball", None) is not None
    has_conf = getattr(scene, "confidence", None) is not None

    if sid == "decode":
        return "live", ""
    if sid == "calibrate":
        return ("live", "") if (has_cam and has_field) else (
            "unmaterialized", "no camera/field in scene")
    if sid == "pose":
        return ("live", "") if has_subjects else ("unmaterialized", "no subjects in scene")
    if sid in ("ball2d", "ball3d"):
        return ("live", "") if has_ball else (
            "unmaterialized", "no ball track in this scene.json")
    if sid == "coherence":
        return ("live", "") if has_conf else (
            "partial", "confidence map not in scene — ribbon unavailable")
    if sid == "track":
        return "partial", "teams/roles/spans available in sidebar; raw boxes need a run bundle"
    if sid in ("assemble", "export"):
        return "live", ""
    if sid == "physics":
        return "partial", "corrections resolve at export (baked); live diff needs a run bundle"
    # detect, stitch, identity, render
    return "unmaterialized", "raw artifact not in scene.json — needs a Phase-0 run bundle"


def build_manifest(scene: Any, *, n_frames: int, active_clip: str) -> dict:
    """Assemble the full Studio stage manifest for the currently loaded clip."""
    stages = []
    for st in _stage_defs():
        status, note = _availability(st["id"], scene)
        entry = dict(st)
        entry["status"] = status
        # keep the stage's own honest note and append the availability note
        entry["avail_note"] = note
        stages.append(entry)
    return {
        "active_clip": active_clip,
        "n_frames": int(n_frames),
        "stages": stages,
        "families": {
            "cached": "content-addressed run_cached stage",
            "continuity": "runs around the cached stages (stitch/identity)",
            "correction": "emits layered, resolvable Corrections (coherence/physics)",
            "tail": "resolve/export/render",
        },
    }
