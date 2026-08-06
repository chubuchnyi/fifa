"""Do the joint/orientation gates fix the poses, or just flatten them? — the T1b/T1c question.

`config/physics.yaml` ships `joint.enabled` and `orientation.enabled` **false** under a profile
that calls itself "no future gates", while the same file carries a `safe_new` profile described as
"Recommended for pod runs". Measured on the 2026-08-05 pod scene, the exported poses violate the
joint ceiling **118** times and the orientation ceiling **11** times — so the limits are being
measured and deliberately not enforced.

    .venv/bin/python scripts/pose_gate_ab.py --scene out/.../scene.json --profile safe_new

Turning a clamp on is easy and is not the question. The question is what it costs, because this
repo has already been bitten once: an iterative moving average on HMR yaw removed 90 % of the
jitter *and* flattened 100°+ real turns, which is why `facing_align` exists instead. So this
reports two things side by side:

* **violations removed** — what the gate is for.
* **angular travel kept** — the sum of per-frame absolute rotation change, per joint and for the
  root, before vs after. A clamp that removes jitter keeps nearly all of it; a clamp that flattens
  real motion shows up here as travel falling well below 100 %.

A gate that removes violations while keeping ~all the travel is fixing noise. One that removes
violations by eating travel is the yaw low-pass again, and should be rejected the same way.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from pitch3d.core.config import load_physics_config  # noqa: E402
from pitch3d.core.correction.engine import resolve_subject_motion  # noqa: E402
from pitch3d.core.correction.joint_kinematics import joint_kinematic_gate  # noqa: E402
from pitch3d.core.correction.orientation import orientation_gate  # noqa: E402
from pitch3d.core.scene.serialization import load_scene, save_scene  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", required=True)
parser.add_argument("--profile", default="safe_new")
parser.add_argument("--fps", type=float, default=29.97)
parser.add_argument("--out", default=None, help="write the gated scene here for motion_stats")
args = parser.parse_args()


def travel(scene):
    """Total absolute per-frame rotation change, radians, summed over frames.

    Root and body are reported apart because they fail differently: the root is where a yaw
    low-pass flattens a turn, the body joints are where per-frame HMR jitter lives.
    """
    root = body = 0.0
    for s in scene.subjects:
        m = resolve_subject_motion(s.proposal, list(scene.corrections_for(s.track_id)))
        g = np.asarray(m.pose.global_orient, dtype=float)
        b = np.asarray(m.pose.body_pose, dtype=float)
        if g.shape[0] > 1:
            root += float(np.abs(np.diff(g, axis=0)).sum())
        if b.shape[0] > 1:
            body += float(np.abs(np.diff(b, axis=0)).sum())
    return root, body


scene = load_scene(args.scene)
cfg = load_physics_config(profile=args.profile)
print(f"scene {args.scene}: {len(scene.subjects)} subjects, "
      f"{len(scene.corrections)} correction(s) on entry")
print(f"profile '{args.profile}': joint.enabled={cfg.joint.enabled} "
      f"(<= {cfg.joint.max_omega_dps} deg/s), orientation.enabled={cfg.orientation.enabled} "
      f"(<= {cfg.orientation.max_turn_rate_dps} deg/s)")

r0, b0 = travel(scene)
scene, jrep = joint_kinematic_gate(scene, cfg.joint, fps=args.fps)
scene, orep = orientation_gate(scene, cfg.orientation, fps=args.fps)
r1, b1 = travel(scene)

# The reports carry every violation; print the shape of the fix, not the list.
print(f"\njoint gate      : {jrep.subjects_corrected}/{jrep.n_subjects} subjects, "
      f"{jrep.joints_corrected} joints, {jrep.intervals_over_limit} intervals over limit, "
      f"max rate {jrep.max_rate_before_dps:.0f} -> {jrep.max_rate_after_dps:.0f} deg/s")
print(f"orientation gate: {orep.subjects_corrected}/{orep.n_subjects} subjects, "
      f"{orep.intervals_over_limit} intervals over limit, "
      f"max rate {orep.max_rate_before_dps:.0f} -> {orep.max_rate_after_dps:.0f} deg/s")
print("\nangular travel kept (100% = nothing removed, well under = real motion flattened)")
print(f"  root orientation: {r0:9.2f} -> {r1:9.2f} rad   {100 * r1 / r0 if r0 else 100:6.1f}%")
print(f"  body joints     : {b0:9.2f} -> {b1:9.2f} rad   {100 * b1 / b0 if b0 else 100:6.1f}%")
print(f"  corrections now : {len(scene.corrections)}")

if args.out:
    save_scene(scene, args.out)
    print(f"\nwrote {args.out}  — re-measure with:")
    print(f"  .venv/bin/python scripts/motion_stats.py --scene {args.out} --profile {args.profile}")
