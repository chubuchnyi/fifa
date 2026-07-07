"""Apply the current full_realism correction stack on an existing scene.json.

``out/anim_full_realism/scene.json`` was exported with ``corrections=[]``,
so its transl/pose reflect raw HMR — none of the physics gates ran. This
script replays the current controller order (post 2026-07-07 reorder)
locally, so we can compare RAW HMR vs POST-corrections without a pod
pipeline pass.

Order matches ``app/controller.py`` (2026-07-07):

    momentum_smooth → pose_motion_sync → facing_align → inertia_smooth
    → jerk_clamp → contact_lock → gravity_project → joint_smooth
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pitch3d.core.config import load_physics_config
from pitch3d.core.correction.contact_lock import contact_lock_gate
from pitch3d.core.correction.contact_probe import contact_probe
from pitch3d.core.correction.facing_align import facing_align_gate
from pitch3d.core.correction.foot_floor import foot_floor_gate
from pitch3d.core.correction.foot_plant import foot_plant_gate
from pitch3d.core.correction.gravity_project import gravity_project_gate
from pitch3d.core.correction.inertia_smooth import inertia_smooth_gate
from pitch3d.core.correction.jerk_clamp import jerk_clamp_gate
from pitch3d.core.correction.joint_kinematics import joint_kinematic_gate
from pitch3d.core.correction.joint_smooth import joint_smooth_gate
from pitch3d.core.correction.momentum_smooth import momentum_smooth_gate
from pitch3d.core.correction.orient_verticality import orient_verticality_gate
from pitch3d.core.correction.orientation import orientation_gate
from pitch3d.core.correction.pose_motion_sync import pose_motion_sync_gate
from pitch3d.core.scene.serialization import load_scene, to_json
from pitch3d.env import load_env

load_env()


def _try(fn):
    try:
        return fn()
    except Exception as e:
        print(f"warn: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="out/anim_full_realism/scene.json")
    ap.add_argument("--out",   default="out/physics_debug/scene_replayed.json")
    ap.add_argument("--profile", default="full_realism")
    ap.add_argument("--fps", type=float, default=29.97)
    args = ap.parse_args()

    scene = load_scene(args.scene)
    cfg = load_physics_config(profile=args.profile, env={})
    print(f"loaded {len(scene.subjects)} subjects, "
          f"{len(scene.corrections)} pre-existing corrections; "
          f"profile={args.profile}")

    from pitch3d.adapters.models.smplx_foot_pos import make_smplx_foot_position_provider
    from pitch3d.adapters.models.smplx_foot_z import make_smplx_foot_z_provider
    foot_pos = _try(make_smplx_foot_position_provider)
    pelvis_target = _try(make_smplx_foot_z_provider)

    n_before = len(scene.corrections)

    if cfg.foot_floor.enabled:
        scene, _ = foot_floor_gate(scene, cfg.foot_floor)
        print(f"  ↳ foot_floor applied, corrs={len(scene.corrections)}")
    if cfg.foot_plant.enabled:
        scene, _ = foot_plant_gate(
            scene, cfg.foot_plant, pelvis_target_provider=pelvis_target,
        )
        print(f"  ↳ foot_plant applied, corrs={len(scene.corrections)}")
    if cfg.joint.enabled:
        scene, _ = joint_kinematic_gate(scene, cfg.joint, fps=args.fps)
        print(f"  ↳ joint_kinematic applied, corrs={len(scene.corrections)}")
    if cfg.orientation.enabled:
        scene, _ = orientation_gate(scene, cfg.orientation, fps=args.fps)
        print(f"  ↳ orientation applied, corrs={len(scene.corrections)}")
    if cfg.momentum_smooth.enabled:
        scene, _ = momentum_smooth_gate(
            scene, cfg.momentum_smooth,
            foot_position_provider=foot_pos, fps=args.fps,
        )
        print(f"  ↳ momentum_smooth applied, corrs={len(scene.corrections)}")
    if cfg.orient_verticality.enabled:
        scene, ovr = orient_verticality_gate(scene, cfg.orient_verticality)
        print(f"  ↳ orient_verticality applied — subj_corrected={ovr.subjects_corrected}/"
              f"{ovr.n_subjects}, corrs={len(scene.corrections)}")
    if cfg.pose_motion_sync.enabled:
        scene, _ = pose_motion_sync_gate(scene, cfg.pose_motion_sync, fps=args.fps)
        print(f"  ↳ pose_motion_sync applied, corrs={len(scene.corrections)}")
    if cfg.facing_align.enabled:
        scene, _ = facing_align_gate(scene, cfg.facing_align, fps=args.fps)
        print(f"  ↳ facing_align applied, corrs={len(scene.corrections)}")
    if cfg.inertia_smooth.enabled:
        scene, _ = inertia_smooth_gate(scene, cfg.inertia_smooth, fps=args.fps)
        print(f"  ↳ inertia_smooth applied, corrs={len(scene.corrections)}")
    if cfg.jerk_clamp.enabled:
        scene, _ = jerk_clamp_gate(scene, cfg.jerk_clamp, fps=args.fps)
        print(f"  ↳ jerk_clamp applied, corrs={len(scene.corrections)}")
    if cfg.contact_probe.enabled and foot_pos is not None:
        scene, cl = contact_lock_gate(scene, cfg.contact_probe, foot_pos)
        c_rep = contact_probe(scene, cfg.contact_probe, foot_pos)
        print(f"  ↳ contact_lock applied — runs_locked={cl.runs_locked}  "
              f"post_slides={c_rep.total_slides}/{c_rep.total_runs} "
              f"max={c_rep.max_slide_m:.2f}m, corrs={len(scene.corrections)}")
    if cfg.gravity_project.enabled and foot_pos is not None:
        scene, _ = gravity_project_gate(
            scene, cfg.gravity_project, foot_pos, fps=args.fps,
        )
        print(f"  ↳ gravity_project applied, corrs={len(scene.corrections)}")
    if cfg.joint_smooth.enabled:
        scene, _ = joint_smooth_gate(scene, cfg.joint_smooth, fps=args.fps)
        print(f"  ↳ joint_smooth applied, corrs={len(scene.corrections)}")

    n_after = len(scene.corrections)
    print(f"\ncorrections: {n_before} → {n_after} (Δ = +{n_after - n_before})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(to_json(scene), encoding="utf-8")
    print(f"REPLAY_OK → {args.out}")


if __name__ == "__main__":
    main()
