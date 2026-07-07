"""Compare raw HMR vs post-corrections physics side-by-side.

Loads two scenes (before / after physics replay) and produces a single
overlay figure showing each subject's ``|Δbody_pose|`` and yaw over time
in both states. Immediately reveals how much the current gate stack
removes from HMR noise.

Usage:
    .venv/bin/python scripts/pipeline_before_after.py \\
        --before out/anim_full_realism/scene.json \\
        --after  out/physics_debug/scene_replayed.json \\
        --out    out/physics_debug/before_after.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.scene.serialization import load_scene


def _stats(scene):
    subs = sorted(scene.subjects, key=lambda s: s.track_id)
    per_sub = {}
    for s in subs:
        r = resolve_subject_motion(s.proposal, scene.corrections_for(s.track_id))
        bp = np.asarray(r.pose.body_pose)
        o = np.asarray(r.pose.global_orient)
        t = np.asarray(r.pose.transl)
        dpose = np.linalg.norm(np.diff(bp, axis=0), axis=(1, 2))
        yaw_unwrapped = np.unwrap(o[:, 2])
        per_sub[s.track_id] = {
            "dpose": dpose, "yaw": yaw_unwrapped, "transl": t,
        }
    return per_sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after",  required=True)
    ap.add_argument("--out",    required=True)
    args = ap.parse_args()

    s_bef = load_scene(args.before)
    s_aft = load_scene(args.after)
    before = _stats(s_bef)
    after = _stats(s_aft)
    tids = sorted(set(before) & set(after))
    n = len(tids)
    print(f"comparing {n} subjects")

    colors = plt.cm.tab20(np.linspace(0, 1, n))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for i, tid in enumerate(tids):
        c = colors[i]
        # Δbody_pose
        axes[0, 0].plot(before[tid]["dpose"], "-", color=c, lw=0.7, alpha=0.7)
        axes[0, 1].plot(after[tid]["dpose"],  "-", color=c, lw=0.7, alpha=0.7)
        # yaw (degrees, unwrapped)
        axes[1, 0].plot(np.degrees(before[tid]["yaw"]), "-", color=c, lw=0.7, alpha=0.7)
        axes[1, 1].plot(np.degrees(after[tid]["yaw"]),  "-", color=c, lw=0.7, alpha=0.7)

    for ax, title in zip(axes.ravel(), [
        "BEFORE — |Δbody_pose| (rad/frame)",
        "AFTER  — |Δbody_pose| (rad/frame)",
        "BEFORE — yaw (deg, unwrapped)",
        "AFTER  — yaw (deg, unwrapped)",
    ]):
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("frame")

    for ax in axes[0]:
        ax.set_ylim(0, 1.5)
    for ax in axes[1]:
        ax.set_ylim(-500, 500)

    fig.suptitle("Physics stack impact — raw HMR vs post-corrections", fontsize=12)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    plt.close(fig)
    print(f"wrote {args.out}")

    # aggregate numbers
    def agg(d):
        return np.array([np.max(d[tid]["dpose"]) for tid in tids])
    b_peaks = agg(before); a_peaks = agg(after)
    print()
    print("Per-subject peak |Δbody_pose| (rad):")
    print(f"  BEFORE: mean={b_peaks.mean():.3f} max={b_peaks.max():.3f}")
    print(f"  AFTER : mean={a_peaks.mean():.3f} max={a_peaks.max():.3f}")
    print(f"  reduction: mean={-100*(1 - a_peaks.mean()/b_peaks.mean()):+.0f}%  "
          f"max={-100*(1 - a_peaks.max()/b_peaks.max()):+.0f}%")


if __name__ == "__main__":
    main()
