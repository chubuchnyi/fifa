#!/usr/bin/env python3
"""Compare physics profiles on the same scene — the research harness.

Runs the M3-9 kinematic gate (and optionally the coherence pass) for every named
profile in ``config/physics.yaml`` (or a custom config path), reports a table of
metrics side-by-side, and points at the WHERE-does-each-number-come-from lineage.

Usage:

    # all profiles from the shipped config
    python scripts/physics_compare.py --scene out/anim_adr11/export/scene.json

    # only a subset
    python scripts/physics_compare.py --scene <scene> --profiles default,conservative,strict

    # a custom config file
    python scripts/physics_compare.py --scene <scene> --config /tmp/exp.yaml

    # print lineage for each field
    python scripts/physics_compare.py --scene <scene> --show-lineage

The intent is research-mode: sweep profiles, measure the delta, pick the one
you'll ship — no arguing over hidden constants.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pitch3d.core.config import load_physics_config
from pitch3d.core.correction.coherence import add_temporal_coherence
from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.correction.foot_floor import foot_floor_gate
from pitch3d.core.correction.kinematics import kinematic_gate
from pitch3d.core.scene.serialization import load_scene


def _kin_stats(frames: np.ndarray, transl: np.ndarray, fps: float) -> dict:
    """XY speed/accel/turn stats — same convention as ``scripts/motion_stats.py``."""
    xy = np.asarray(transl, float)[:, :2]
    f = np.asarray(frames, float)
    if len(f) < 3:
        return {"n": len(f), "sp_p50": 0.0, "sp_p95": 0.0, "sp_max": 0.0, "ac_max": 0.0}
    dt = np.diff(f) / fps
    ok = dt > 0
    vel = np.diff(xy, axis=0)[ok] / dt[ok, None]
    speed = np.linalg.norm(vel, axis=1)
    if len(speed) < 2:
        return {"n": len(f), "sp_p50": 0.0, "sp_p95": 0.0, "sp_max": 0.0, "ac_max": 0.0}
    accel = np.linalg.norm(np.diff(vel, axis=0), axis=1) / dt[ok][1:]
    return {
        "n": len(f),
        "sp_p50": float(np.median(speed)),
        "sp_p95": float(np.percentile(speed, 95)),
        "sp_max": float(speed.max()),
        "ac_max": float(accel.max()) if len(accel) else 0.0,
    }


def _scene_stats(scene, fps: float, cfg) -> dict:
    """Aggregate resolved-motion stats across all subjects."""
    speeds_over = 0
    accels_over = 0
    total_intervals = 0
    max_sp = 0.0
    max_ac = 0.0
    per_subject = []
    for s in scene.subjects:
        motion = resolve_subject_motion(s.proposal, scene.corrections_for(s.track_id))
        frames = np.asarray(motion.pose.frames, dtype=int)
        transl = np.asarray(motion.pose.transl, dtype=float)
        st = _kin_stats(frames, transl, fps)
        st["track_id"] = int(s.track_id)
        per_subject.append(st)
        if len(frames) < 3:
            continue
        dt = np.diff(frames.astype(float)) / fps
        ok = dt > 0
        vel = np.diff(transl[:, :2], axis=0)[ok] / dt[ok, None]
        speed = np.linalg.norm(vel, axis=1)
        speeds_over += int((speed > cfg.kinematic.max_speed).sum())
        max_sp = max(max_sp, float(speed.max()) if speed.size else 0.0)
        if len(speed) > 1:
            accel = np.linalg.norm(np.diff(vel, axis=0), axis=1) / dt[ok][1:]
            accels_over += int((accel > cfg.kinematic.max_accel).sum())
            max_ac = max(max_ac, float(accel.max()) if accel.size else 0.0)
            total_intervals += int(accel.shape[0])
    return {
        "speed_viol": speeds_over,
        "accel_viol": accels_over,
        "sp_max_mps": max_sp,
        "ac_max_mps2": max_ac,
        "intervals": total_intervals,
        "per_subject": per_subject,
    }


def run_profile(scene_path: str, profile: str, config: str | None,
                fps: float, do_coherence: bool) -> dict:
    """Load scene, apply gate (± coherence) under the named profile, return metrics."""
    scene = load_scene(scene_path)
    cfg = load_physics_config(path=config, profile=profile, env={})
    before = _scene_stats(scene, fps, cfg)

    corrections_added = 0
    if do_coherence:
        scene, cr = add_temporal_coherence(scene, cfg.coherence, fps=fps)
        corrections_added += cr.corrections_added
    scene, kr = kinematic_gate(scene, cfg.kinematic, fps=fps)
    corrections_added += kr.corrections_added
    scene, ffr = foot_floor_gate(scene, cfg.foot_floor)
    corrections_added += ffr.corrections_added

    after = _scene_stats(scene, fps, cfg)
    return {
        "profile": profile,
        "description": cfg.profile_description,
        "cfg": cfg,
        "corrections_added": corrections_added,
        "teleports": len(kr.teleports),
        "teleport_events": [
            {"track_id": t.track_id, "frame": t.frame, "jump_m": t.jump_m,
             "speed_mps": t.speed_mps, "n_intervals": t.n_intervals}
            for t in kr.teleports
        ],
        "max_dev_m": kr.max_dev_m,
        "foot_below_floor": ffr.subjects_below_floor,
        "foot_plateau": ffr.subjects_plateau,
        "foot_hovering": ffr.subjects_hovering,
        "foot_corrected": ffr.subjects_corrected,
        "before": before,
        "after": after,
        "kin_max_speed": cfg.kinematic.max_speed,
        "kin_max_accel": cfg.kinematic.max_accel,
    }


def _print_table(results: list[dict]) -> None:
    hdr = (f"{'profile':>14} {'lim_sp':>7} {'lim_ac':>7} "
           f"{'sp>lim':>7} {'ac>lim':>7} {'tele':>5} "
           f"{'sp_max':>7} {'ac_max':>8} {'max_dev':>8} "
           f"{'blwFl':>5} {'plat':>4} {'hovr':>4} {'ffFix':>5} "
           f"{'corrs':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['profile']:>14} "
              f"{r['kin_max_speed']:>7.2f} {r['kin_max_accel']:>7.2f} "
              f"{r['after']['speed_viol']:>7} {r['after']['accel_viol']:>7} "
              f"{r['teleports']:>5} "
              f"{r['after']['sp_max_mps']:>7.2f} {r['after']['ac_max_mps2']:>8.1f} "
              f"{r['max_dev_m']:>8.3f} "
              f"{r['foot_below_floor']:>5} {r['foot_plateau']:>4} "
              f"{r['foot_hovering']:>4} {r['foot_corrected']:>5} "
              f"{r['corrections_added']:>6}")


def _print_lineage(results: list[dict]) -> None:
    if not results:
        return
    print("\nlineage (profile:field → source):")
    for r in results:
        print(f"  [{r['profile']}] {r['description']}")
        cfg = r["cfg"]
        for key in sorted(cfg.lineage):
            if cfg.lineage[key] != "base":
                print(f"    {key:38s} {cfg.lineage[key]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True, help="path to a scene.json")
    ap.add_argument("--config", default=None, help="physics config path (default: repo)")
    ap.add_argument("--profiles", default=None,
                    help="comma-separated profile names (default: all in the config)")
    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--no-coherence", action="store_true",
                    help="skip the coherence pass (gate only)")
    ap.add_argument("--show-lineage", action="store_true",
                    help="print per-profile field-lineage table")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of a table")
    args = ap.parse_args()

    if args.profiles:
        profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    else:
        # discover all profiles from the config
        cfg_path = Path(args.config) if args.config else None
        import yaml
        from pitch3d.core.config.physics import DEFAULT_CONFIG_PATH
        raw = yaml.safe_load((cfg_path or DEFAULT_CONFIG_PATH).read_text())
        profiles = list(raw.get("profiles", {}))

    results = []
    for p in profiles:
        try:
            results.append(run_profile(
                args.scene, p, args.config, args.fps, do_coherence=not args.no_coherence,
            ))
        except Exception as exc:  # keep going — one bad profile doesn't kill the sweep
            print(f"!! profile {p!r} failed: {exc}", file=sys.stderr)

    if args.json:
        # strip the un-serialisable cfg dataclass; keep the summary + lineage subset
        for r in results:
            cfg = r.pop("cfg")
            r["config_summary"] = cfg.summary()
            r["lineage_nonbase"] = {k: v for k, v in cfg.lineage.items() if v != "base"}
        print(json.dumps({"scene": args.scene, "fps": args.fps, "results": results},
                         indent=2))
        return 0

    print(f"== physics_compare: {args.scene}  fps={args.fps}  "
          f"coherence={'off' if args.no_coherence else 'on'}")
    _print_table(results)
    if args.show_lineage:
        _print_lineage(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
