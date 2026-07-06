#!/usr/bin/env python3
"""Kinematic plausibility probe for a reconstructed scene (R-6 fidelity, T0 lever).

Measures per-subject motion against thresholds pulled from ``config/physics.yaml``
(never hard-coded here — see the "parametric stays parametric" rule). Reports both
LAYERS (raw proposal vs resolved = proposal ⊕ corrections) so you can see what the
gates removed.

Six categories, one per user complaint:

* XY   — root translation speed/accel/turn rate (§B, D of the physics playbook)
* Z    — foot floor (min/max/hover fraction, §A: "hovering in the air")
* orient — root ``global_orient`` angular rate (§D)
* joint  — per-joint ``body_pose`` angular rate (§C: "poses change too fast")
* ball  — ball speed & accel (§F reference: ball is already clean)
* teleport count — jumps above ``teleport_factor * max_speed`` (§E)

Usage::

    python scripts/motion_stats.py --scene out/anim_adr11/export/scene.json
    python scripts/motion_stats.py --scene <s> --profile strict
    python scripts/motion_stats.py --scene <s> --json > probe.json

Every threshold is documented on stdout with lineage so the run log answers
"why did that flag fire?" without leaving the terminal.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from pitch3d.core.config import PhysicsConfig, load_physics_config
from pitch3d.core.correction.engine import resolve_ball, resolve_subject_motion
from pitch3d.core.scene.serialization import load_scene


def _axis_angle_rates(rot_seq: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Per-interval angular rate (deg/s) between axis-angle rotations.

    ``rot_seq`` shape ``(T, 3)`` axis-angle in radians. Composes the group delta
    ``R_i+1 · R_i^-1`` and returns its magnitude divided by dt — the correct
    angular velocity, not the componentwise diff (which is only valid for tiny
    rotations and lies for anything else).
    """
    if rot_seq.shape[0] < 2:
        return np.zeros(0)
    r = Rotation.from_rotvec(rot_seq)
    delta = r[1:] * r[:-1].inv()
    angle_rad = np.linalg.norm(delta.as_rotvec(), axis=1)
    return np.degrees(angle_rad) / dt


def xy_stats(frames: np.ndarray, transl: np.ndarray, fps: float,
             turn_min_speed: float, max_speed: float, max_accel: float,
             teleport_factor: float) -> dict:
    """XY speed/accel/turn — root translation."""
    frames = np.asarray(frames, float)
    xy = np.asarray(transl, float)[:, :2]
    if len(frames) < 3:
        return {"n": len(frames)}
    dt = np.diff(frames) / fps
    ok = dt > 0
    vel = np.diff(xy, axis=0)[ok] / dt[ok, None]
    speed = np.linalg.norm(vel, axis=1)
    if len(speed) < 2:
        return {"n": len(frames)}
    accel = np.linalg.norm(np.diff(vel, axis=0), axis=1) / dt[ok][1:]
    moving = (speed[:-1] > turn_min_speed) & (speed[1:] > turn_min_speed)
    if moving.any():
        u, w = vel[:-1][moving], vel[1:][moving]
        cosang = np.clip(
            np.sum(u * w, axis=1)
            / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1) + 1e-9),
            -1.0, 1.0,
        )
        turn = np.degrees(np.arccos(cosang)) / dt[ok][1:][moving]
    else:
        turn = np.zeros(1)
    return {
        "n": len(frames),
        "sp_p50": float(np.median(speed)),
        "sp_p95": float(np.percentile(speed, 95)),
        "sp_max": float(speed.max()),
        "ac_max": float(accel.max()) if len(accel) else 0.0,
        "turn_max": float(turn.max()),
        "viol_sp": int((speed > max_speed).sum()),
        "viol_ac": int((accel > max_accel).sum()),
        "teleport_intervals": int((speed > teleport_factor * max_speed).sum()),
    }


def foot_z_stats(frames: np.ndarray, transl: np.ndarray,
                 floor_m: float, hover_m: float) -> dict:
    """Root Z (foot proxy) min/max + hovering fraction.

    ``pelvis_above_foot ≈ 0.92 m`` is expected — a constant plateau there means
    the foot IS resting on Z=0 with only shape variation. ``hover_frac`` counts
    frames where the pelvis sits high above the floor for its whole visible
    stretch — the "helicopter" symptom.
    """
    z = np.asarray(transl, float)[:, 2]
    if z.size == 0:
        return {"n": 0}
    hover = z > (floor_m + hover_m + 0.92)  # pelvis+hover above floor
    return {
        "n": int(z.size),
        "z_min": float(z.min()),
        "z_median": float(np.median(z)),
        "z_max": float(z.max()),
        "hover_frac": float(hover.mean()),
        "below_floor_frac": float((z < floor_m).mean()),
    }


def orient_stats(frames: np.ndarray, global_orient: np.ndarray,
                 fps: float, flag_dps: float) -> dict:
    """Root ``global_orient`` angular rate — degrees per second."""
    frames = np.asarray(frames, float)
    if len(frames) < 2:
        return {"n": len(frames)}
    dt = np.diff(frames) / fps
    ok = dt > 0
    if not ok.any():
        return {"n": len(frames)}
    rates = _axis_angle_rates(np.asarray(global_orient, float), dt)
    return {
        "n": len(frames),
        "orient_p95_dps": float(np.percentile(rates, 95)) if rates.size else 0.0,
        "orient_max_dps": float(rates.max()) if rates.size else 0.0,
        "orient_viol": int((rates > flag_dps).sum()),
    }


def joint_stats(frames: np.ndarray, body_pose: np.ndarray,
                fps: float, flag_dps: float) -> dict:
    """Per-joint ``body_pose`` angular rate — max over joints.

    ``body_pose`` is ``(T, K, 3)`` axis-angle per joint. For each joint we compute
    the angular rate between consecutive frames, then aggregate: max/p95 over ALL
    (frame × joint) rate samples.
    """
    frames = np.asarray(frames, float)
    if len(frames) < 2 or body_pose.size == 0:
        return {"n": len(frames)}
    dt = np.diff(frames) / fps
    ok = dt > 0
    if not ok.any():
        return {"n": len(frames)}
    K = body_pose.shape[1]
    all_rates = []
    per_joint_max = np.zeros(K)
    for j in range(K):
        rates = _axis_angle_rates(np.asarray(body_pose[:, j, :], float), dt)
        all_rates.append(rates)
        per_joint_max[j] = float(rates.max()) if rates.size else 0.0
    flat = np.concatenate(all_rates) if all_rates else np.zeros(0)
    return {
        "n": len(frames),
        "joint_p95_dps": float(np.percentile(flat, 95)) if flat.size else 0.0,
        "joint_max_dps": float(flat.max()) if flat.size else 0.0,
        "joint_viol_samples": int((flat > flag_dps).sum()),
        "hottest_joint_idx": int(per_joint_max.argmax()),
        "hottest_joint_max_dps": float(per_joint_max.max()),
    }


def _flag(v: bool) -> str:
    return " <<<" if v else ""


def probe_scene(scene, cfg: PhysicsConfig, fps: float) -> dict:
    """Run every stat category on both layers of every subject; return a dict."""
    kin, coh, probe, ball_cfg = cfg.kinematic, cfg.coherence, cfg.probe, cfg.ball
    out = {
        "scene_limits": {
            "max_speed": kin.max_speed, "max_accel": kin.max_accel,
            "teleport_factor": kin.teleport_factor,
            "foot_floor_m": cfg.foot_floor.floor_m,
            "foot_hover_m": probe.foot_hover_m,
            "orient_flag_dps": probe.orient_min_dps,
            "joint_flag_dps": probe.joint_min_omega_dps,
            "ball_max_speed": ball_cfg.max_speed,
        },
        "profile": cfg.profile_name,
        "profile_description": cfg.profile_description,
        "config_path": cfg.source_path,
        "subjects": [],
        "totals": {"proposal": {}, "resolved": {}},
    }

    totals = {
        "proposal": {"viol_sp": 0, "viol_ac": 0, "orient_viol": 0, "joint_viol": 0,
                     "hover_frac": [], "teleport_intervals": 0},
        "resolved": {"viol_sp": 0, "viol_ac": 0, "orient_viol": 0, "joint_viol": 0,
                     "hover_frac": [], "teleport_intervals": 0},
    }

    for sub in scene.subjects:
        layers = {
            "proposal": sub.proposal,
            "resolved": resolve_subject_motion(
                sub.proposal, scene.corrections_for(sub.track_id)),
        }
        subj = {"track_id": int(sub.track_id), "layers": {}}
        for name, motion in layers.items():
            frames = np.asarray(motion.pose.frames, dtype=int)
            xy = xy_stats(frames, motion.pose.transl, fps,
                          probe.turn_min_speed, kin.max_speed, kin.max_accel,
                          kin.teleport_factor)
            zst = foot_z_stats(frames, motion.pose.transl,
                               cfg.foot_floor.floor_m, probe.foot_hover_m)
            ost = orient_stats(frames, motion.pose.global_orient, fps,
                               probe.orient_min_dps)
            jst = joint_stats(frames, motion.pose.body_pose, fps,
                              probe.joint_min_omega_dps)
            subj["layers"][name] = {"xy": xy, "z": zst, "orient": ost, "joint": jst}
            totals[name]["viol_sp"] += xy.get("viol_sp", 0)
            totals[name]["viol_ac"] += xy.get("viol_ac", 0)
            totals[name]["orient_viol"] += ost.get("orient_viol", 0)
            totals[name]["joint_viol"] += jst.get("joint_viol_samples", 0)
            totals[name]["hover_frac"].append(zst.get("hover_frac", 0.0))
            totals[name]["teleport_intervals"] += xy.get("teleport_intervals", 0)
        out["subjects"].append(subj)

    # aggregate the hover fraction as mean over subjects (bounded [0, 1])
    for name, t in totals.items():
        t["hover_frac_mean"] = float(np.mean(t.pop("hover_frac"))) if t["hover_frac"] is not None else 0.0
    out["totals"] = totals

    if scene.ball is not None:
        ball = resolve_ball(scene.ball, scene.corrections_for(None))
        bf = np.asarray(ball.frames, float)
        bp = np.asarray(ball.positions_3d, float)
        if len(bf) >= 2:
            dt = np.diff(bf) / fps
            ok = dt > 0
            bs = np.linalg.norm(np.diff(bp, axis=0)[ok], axis=1) / dt[ok]
            ba = np.linalg.norm(np.diff(np.diff(bp, axis=0)[ok], axis=0), axis=1) / dt[ok][1:] if len(bs) > 1 else np.zeros(0)
            out["ball"] = {
                "n": int(len(bf)),
                "sp_p50": float(np.median(bs)) if bs.size else 0.0,
                "sp_p95": float(np.percentile(bs, 95)) if bs.size else 0.0,
                "sp_max": float(bs.max()) if bs.size else 0.0,
                "ac_max": float(ba.max()) if ba.size else 0.0,
                "viol_sp": int((bs > ball_cfg.max_speed).sum()),
                "viol_ac": int((ba > ball_cfg.max_accel).sum()),
            }
    return out


def print_report(report: dict) -> None:
    lim = report["scene_limits"]
    print(f"== motion stats: {report['config_path']} "
          f"profile={report['profile']!r} — {report['profile_description']}")
    print(f"== limits: XY speed {lim['max_speed']} m/s, accel {lim['max_accel']} m/s² "
          f"(teleport ×{lim['teleport_factor']}) | orient {lim['orient_flag_dps']}°/s | "
          f"joint {lim['joint_flag_dps']}°/s | foot hover {lim['foot_hover_m']} m | "
          f"ball {lim['ball_max_speed']} m/s")

    hdr = (f"{'subj':>6} {'layer':>8} {'n':>4} "
           f"{'sp_p95':>7} {'sp_max':>7} {'ac_max':>7} {'turn':>6} "
           f"{'z_min':>6} {'z_max':>6} {'hover':>6} "
           f"{'orient':>7} {'joint':>7} "
           f"{'>spd':>4} {'>acc':>4} {'>ori':>4} {'>jnt':>4} {'tel':>3}")
    print(hdr)
    print("-" * len(hdr))
    for subj in report["subjects"]:
        for name, layer in subj["layers"].items():
            xy = layer["xy"]; z = layer["z"]; o = layer["orient"]; j = layer["joint"]
            if not xy or "sp_max" not in xy:
                continue
            row = (
                f"{subj['track_id']:>6} {name:>8} {xy['n']:>4} "
                f"{xy['sp_p95']:>7.2f} {xy['sp_max']:>7.2f} {xy['ac_max']:>7.1f} "
                f"{xy['turn_max']:>6.0f} "
                f"{z.get('z_min', 0):>6.2f} {z.get('z_max', 0):>6.2f} "
                f"{z.get('hover_frac', 0)*100:>5.0f}% "
                f"{o.get('orient_max_dps', 0):>7.0f} "
                f"{j.get('joint_max_dps', 0):>7.0f} "
                f"{xy['viol_sp']:>4} {xy['viol_ac']:>4} "
                f"{o.get('orient_viol', 0):>4} "
                f"{j.get('joint_viol_samples', 0):>4} "
                f"{xy['teleport_intervals']:>3}"
            )
            flags = (xy['viol_sp'] or xy['viol_ac'] or o.get('orient_viol', 0)
                     or j.get('joint_viol_samples', 0) or xy['teleport_intervals'])
            print(row + _flag(bool(flags) and name == "resolved"))

    for name, t in report["totals"].items():
        print(f"== TOTAL {name}: speed_viol={t['viol_sp']} accel_viol={t['viol_ac']} "
              f"orient_viol={t['orient_viol']} joint_viol_samples={t['joint_viol']} "
              f"teleport_intervals={t['teleport_intervals']} "
              f"hover_frac_mean={t['hover_frac_mean']*100:.0f}%")
    if "ball" in report:
        b = report["ball"]
        print(f"== ball: n={b['n']} sp_p95={b['sp_p95']:.2f} sp_max={b['sp_max']:.2f} "
              f"ac_max={b['ac_max']:.1f} viol_sp={b['viol_sp']} viol_ac={b['viol_ac']}")
    print("MOTION_STATS_OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True, help="scene.json path")
    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--profile", default="default",
                    help="physics profile from config/physics.yaml")
    ap.add_argument("--config", default=None, help="alternate physics config path")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of a table")
    args = ap.parse_args(argv)

    scene = load_scene(args.scene)
    cfg = load_physics_config(path=args.config, profile=args.profile)
    report = probe_scene(scene, cfg, args.fps)

    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
