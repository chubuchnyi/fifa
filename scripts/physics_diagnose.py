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
from pitch3d.core.correction.body_scale_probe import BodyScaleConfig, body_scale_probe
from pitch3d.core.correction.contact_probe import contact_probe
from pitch3d.core.correction.gravity_probe import GravityConfig, gravity_probe
from pitch3d.core.correction.inertia_probe import InertiaConfig, inertia_probe
from pitch3d.core.correction.interpen_probe import InterpenConfig, interpen_probe
from pitch3d.core.correction.momentum_probe import (
    MomentumProbeConfig,
    momentum_probe,
)
from pitch3d.core.correction.pose_motion_probe import (
    PoseMotionConfig,
    pose_motion_probe,
)
from pitch3d.core.correction.stride_probe import StrideProbeConfig, stride_probe
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
    try:
        from pitch3d.adapters.models.smplx_foot_z import make_smplx_foot_z_provider
        scale_provider = make_smplx_foot_z_provider()
    except Exception:
        scale_provider = None

    contact = None
    gravity = None
    body_scale = None
    if foot_pos_provider is not None:
        contact = contact_probe(
            scene, replace(cfg.contact_probe, enabled=True),
            foot_pos_provider,
        )
        gravity = gravity_probe(
            scene, GravityConfig(enabled=True), foot_pos_provider, fps=fps,
        )
    if scale_provider is not None:
        body_scale = body_scale_probe(
            scene, BodyScaleConfig(enabled=True), scale_provider,
        )

    interpen = interpen_probe(scene, InterpenConfig(enabled=True))
    stride = stride_probe(scene, StrideProbeConfig(enabled=True), fps=fps)

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
        "gravity": None if gravity is None else {
            "subjects_violating": gravity.subjects_violating,
            "total_airborne_frames": gravity.total_airborne_frames,
            "max_deviation_mps2": gravity.max_deviation_mps2,
        },
        "body_scale": None if body_scale is None else {
            "n_pairs_flagged": body_scale.n_pairs_flagged,
            "max_pair_diff_m": body_scale.max_pair_diff_m,
        },
        "stride": {
            "subjects_off": stride.subjects_off,
        },
        "interpen": {
            "subjects_with_overlap": interpen.subjects_with_overlap,
            "total_overlap_frames": interpen.total_overlap_frames,
            "max_overlap_m": interpen.max_overlap_m,
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
    if r["gravity"] is not None:
        g = r["gravity"]
        print(f"[gravity]      violating={g['subjects_violating']}/{n} "
              f"airborne_frames={g['total_airborne_frames']} "
              f"max_dev={g['max_deviation_mps2']:.1f}m/s²")
    if r["body_scale"] is not None:
        bs = r["body_scale"]
        print(f"[body_scale]   pairs_flagged={bs['n_pairs_flagged']} "
              f"max_diff={bs['max_pair_diff_m']:.2f}m")
    st = r["stride"]
    print(f"[stride]       off={st['subjects_off']}/{n}")
    ip = r["interpen"]
    print(f"[interpen]     overlap_subj={ip['subjects_with_overlap']}/{n} "
          f"frames={ip['total_overlap_frames']} max={ip['max_overlap_m']:.2f}m")


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
