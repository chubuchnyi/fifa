#!/usr/bin/env python3
"""One-command physics diagnostic — runs every new probe on a scene.

Complements ``scripts/motion_stats.py`` (M3-9 baseline probe) and
``scripts/physics_compare.py`` (correction-gate A/B). This tool answers
the operator's question "how bad is the physics, ALL dimensions?" on
any exported scene.json.

Reports (all thresholds pulled from ``config/physics.yaml``):

* contact — foot slide magnitude during stance
* momentum — CoM jerk
* pose_motion — root moves without joint activity ("still pose walking")
* inertia — torso angular acceleration

Requires the SMPL-X model dir (env PITCH3D_SMPLX_MODEL/MODELS) for the
contact/momentum probes; the pose_motion / inertia probes are pure-numpy
and work without it.

Usage:
    python scripts/physics_diagnose.py --scene <path> [--fps 29.97]
    python scripts/physics_diagnose.py --scene <path> --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace

from pitch3d.core.config import load_physics_config
from pitch3d.core.correction.contact_probe import contact_probe
from pitch3d.core.correction.inertia_probe import InertiaConfig, inertia_probe
from pitch3d.core.correction.momentum_probe import (
    MomentumProbeConfig,
    momentum_probe,
)
from pitch3d.core.correction.pose_motion_probe import (
    PoseMotionConfig,
    pose_motion_probe,
)
from pitch3d.core.scene.serialization import load_scene


def run_all(scene_path: str, fps: float) -> dict:
    scene = load_scene(scene_path)
    cfg = load_physics_config(env={})

    try:
        from pitch3d.adapters.models.smplx_foot_pos import (
            make_smplx_foot_position_provider,
        )
        foot_pos_provider = make_smplx_foot_position_provider()
    except Exception:
        foot_pos_provider = None

    contact = None
    if foot_pos_provider is not None:
        contact = contact_probe(
            scene, replace(cfg.contact_probe, enabled=True),
            foot_pos_provider,
        )

    mom = momentum_probe(
        scene, MomentumProbeConfig(enabled=True, jerk_threshold_mps3=100.0),
        fps=fps,
    )
    pm = pose_motion_probe(
        scene, PoseMotionConfig(enabled=True,
                                velocity_threshold_mps=1.0,
                                joint_activity_threshold=0.10,
                                desync_fraction_threshold=0.30),
        fps=fps,
    )
    inr = inertia_probe(
        scene, InertiaConfig(enabled=True, max_alpha_rad_s2=15.0),
        fps=fps,
    )

    return {
        "scene": scene_path, "fps": fps,
        "n_subjects": len(scene.subjects),
        "smplx_available": foot_pos_provider is not None,
        "contact": None if contact is None else {
            "total_runs": contact.total_runs,
            "total_slides": contact.total_slides,
            "subjects_with_slides": contact.subjects_with_slides,
            "mean_slide_m": contact.mean_slide_m,
            "max_slide_m": contact.max_slide_m,
        },
        "momentum": {
            "subjects_chatty": mom.subjects_chatty,
            "max_jerk_mps3": mom.max_jerk_mps3,
            "max_accel_mps2": mom.max_accel_mps2,
        },
        "pose_motion": {
            "subjects_desynced": pm.subjects_desynced,
            "max_desync_fraction": pm.max_desync_fraction,
        },
        "inertia": {
            "subjects_flagged": inr.subjects_flagged,
            "max_alpha_rad_s2": inr.max_alpha_rad_s2,
            "total_alpha_viol": inr.total_alpha_viol,
        },
    }


def print_summary(r: dict) -> None:
    n = r["n_subjects"]
    print(f"== physics diagnose: {r['scene']} ({n} subjects @ {r['fps']} fps)")
    print(f"   SMPL-X FK provider: {'available' if r['smplx_available'] else 'MISSING (contact probe skipped)'}")
    print()
    if r["contact"] is not None:
        c = r["contact"]
        print(f"[contact]      slides={c['total_slides']}/{c['total_runs']} "
              f"subjects={c['subjects_with_slides']}/{n} "
              f"mean={c['mean_slide_m']:.2f}m max={c['max_slide_m']:.2f}m")
    else:
        print("[contact]      SKIPPED (no SMPL-X model)")
    m = r["momentum"]
    print(f"[momentum]     chatty={m['subjects_chatty']}/{n} "
          f"max_jerk={m['max_jerk_mps3']:.0f}m/s³ "
          f"max_accel={m['max_accel_mps2']:.1f}m/s²")
    pm = r["pose_motion"]
    print(f"[pose_motion]  desynced={pm['subjects_desynced']}/{n} "
          f"max_desync_frac={pm['max_desync_fraction']:.2f}")
    inr = r["inertia"]
    print(f"[inertia]      flagged={inr['subjects_flagged']}/{n} "
          f"max_α={inr['max_alpha_rad_s2']:.0f}rad/s² "
          f"total_viol={inr['total_alpha_viol']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    report = run_all(args.scene, args.fps)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
