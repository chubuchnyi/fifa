"""Stage-by-stage debug of a scene.json's raw HMR proposal.

Since ``scene.json`` ships with ``corrections=[]`` in the anim path, the
subject proposals ARE the raw output of the pose stage (SMPLest-X /
SMART). This script prints per-subject stats + saves per-stage matplotlib
diagnostic images so we can pinpoint WHERE the twitch enters.

Stages inspected:
    (1) HMR transl trajectory — jumps? out-of-pitch?
    (2) HMR global_orient   — 180° flips? unwrap issues?
    (3) HMR body_pose norm  — twitchy per-joint noise?
    (4) Anomaly frame       — for a chosen frame, dump each subject's
                              pose stats side-by-side.

Outputs:
    out/pipeline_debug/transl_trajectories.png
    out/pipeline_debug/orient_yaw_per_subject.png
    out/pipeline_debug/body_pose_activity.png
    out/pipeline_debug/frame_anomaly_report.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pitch3d.core.scene.serialization import load_scene


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="out/anim_full_realism/scene.json")
    ap.add_argument("--out", default="out/pipeline_debug/")
    ap.add_argument("--anomaly-frame", type=int, default=15)
    ap.add_argument("--fps", type=float, default=29.97)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    scene = load_scene(args.scene)
    fps = args.fps
    dt = 1.0 / fps

    n_sub = len(scene.subjects)
    print(f"loaded {n_sub} subjects from {args.scene}")

    subs = sorted(scene.subjects, key=lambda s: s.track_id)
    tids = [s.track_id for s in subs]
    colors = plt.cm.tab20(np.linspace(0, 1, n_sub))

    # ─── (1) transl trajectories in top-down view ─────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7))
    # pitch outline
    px, py = 105 / 2, 68 / 2
    ax.plot([-px, px, px, -px, -px], [-py, -py, py, py, -py], "w-", lw=1.5)
    ax.plot([0, 0], [-py, py], "w-", lw=0.5)
    ax.set_facecolor("#2a4a2a")
    for i, s in enumerate(subs):
        t = np.asarray(s.proposal.pose.transl)
        ax.plot(t[:, 0], t[:, 1], "-", color=colors[i], lw=1.2, alpha=0.9)
        ax.plot(t[0, 0], t[0, 1], "o", color=colors[i], ms=4)
        ax.text(t[0, 0], t[0, 1] + 0.5, str(s.track_id),
                color="white", fontsize=6, ha="center")
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"HMR raw transl trajectories — {n_sub} subjects "
                 f"({t.shape[0]} frames)")
    fig.tight_layout()
    fig.savefig(Path(args.out) / "transl_trajectories.png", dpi=110)
    plt.close(fig)
    print("  wrote transl_trajectories.png")

    # ─── (2) yaw per subject over time ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, s in enumerate(subs):
        o = np.asarray(s.proposal.pose.global_orient)
        yaw = np.unwrap(o[:, 2])
        frames = np.arange(o.shape[0])
        ax.plot(frames, np.degrees(yaw), "-", color=colors[i], lw=0.8, alpha=0.8,
                label=f"t{s.track_id}")
    ax.set_xlabel("frame"); ax.set_ylabel("yaw (deg, unwrapped)")
    ax.set_title("HMR raw global_orient yaw per subject")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(args.out) / "orient_yaw_per_subject.png", dpi=110)
    plt.close(fig)
    print("  wrote orient_yaw_per_subject.png")

    # ─── (3) body_pose per-frame activity |dpose/dt| ────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, s in enumerate(subs):
        bp = np.asarray(s.proposal.pose.body_pose)
        if bp.shape[0] < 2:
            continue
        dpose = np.linalg.norm(np.diff(bp, axis=0), axis=(1, 2))
        ax.plot(range(1, dpose.shape[0] + 1), dpose, "-",
                color=colors[i], lw=0.8, alpha=0.75, label=f"t{s.track_id}")
    ax.axvline(x=args.anomaly_frame, color="red", ls="--", alpha=0.5,
               label=f"anomaly f{args.anomaly_frame}")
    ax.set_xlabel("frame"); ax.set_ylabel("|Δbody_pose| (rad)")
    ax.set_title("Per-frame body pose delta magnitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(args.out) / "body_pose_activity.png", dpi=110)
    plt.close(fig)
    print("  wrote body_pose_activity.png")

    # ─── (4) anomaly-frame report ───────────────────────────────────────
    lines = [
        f"# Anomaly-frame report — frame {args.anomaly_frame}",
        f"# scene: {args.scene}",
        f"# fps: {fps}",
        "",
        "track_id | transl (x,y,z)                | yaw_deg | roll_deg | pitch_deg | |body_pose|_max_joint_delta_rad",
    ]
    f = args.anomaly_frame
    for s in subs:
        o = np.asarray(s.proposal.pose.global_orient)
        t = np.asarray(s.proposal.pose.transl)
        bp = np.asarray(s.proposal.pose.body_pose)
        n = o.shape[0]
        if f >= n:
            lines.append(f"t{s.track_id} SKIPPED (only {n} frames)")
            continue
        # axis-angle → yaw is approx z component only for upright poses.
        rx, ry, rz = o[f]
        yaw = np.degrees(rz); roll = np.degrees(rx); pitch = np.degrees(ry)
        if f > 0:
            dpj = np.abs(bp[f] - bp[f - 1]).max()
        else:
            dpj = 0.0
        lines.append(
            f"t{s.track_id:3d} | ({t[f,0]:+7.2f},{t[f,1]:+7.2f},{t[f,2]:+5.2f}) "
            f"| {yaw:+7.1f} | {roll:+7.1f} | {pitch:+7.1f} | {dpj:.3f}"
        )
    txt = "\n".join(lines) + "\n"
    Path(args.out, "frame_anomaly_report.txt").write_text(txt, encoding="utf-8")
    print("  wrote frame_anomaly_report.txt")

    # ─── (5) which subjects have BIG jumps? ─────────────────────────────
    print()
    print("=== Twitch culprits (per-subject peak jumps) ===")
    print("track | transl jump (m) | yaw jump (deg) | dpose_max (rad) | at frame")
    for s in subs:
        t = np.asarray(s.proposal.pose.transl)
        o = np.asarray(s.proposal.pose.global_orient)
        bp = np.asarray(s.proposal.pose.body_pose)
        n = t.shape[0]
        if n < 2:
            continue
        dt_norm = np.linalg.norm(np.diff(t, axis=0), axis=1)
        dyaw = np.abs(np.mod(np.diff(np.unwrap(o[:, 2])) + np.pi, 2*np.pi) - np.pi)
        dpose = np.linalg.norm(np.diff(bp, axis=0), axis=(1, 2))
        tmax = dt_norm.argmax(); ymax = dyaw.argmax(); pmax = dpose.argmax()
        print(f"t{s.track_id:3d} | {dt_norm[tmax]:5.2f} m @ f{tmax+1:2d} | "
              f"{np.degrees(dyaw[ymax]):6.1f}° @ f{ymax+1:2d} | "
              f"{dpose[pmax]:5.2f} @ f{pmax+1:2d}")


if __name__ == "__main__":
    main()
