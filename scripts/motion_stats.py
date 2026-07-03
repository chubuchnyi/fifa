#!/usr/bin/env python3
"""Kinematic plausibility probe for a reconstructed scene (R-6 fidelity).

Prints per-subject root speed / acceleration / turn-rate stats against human
limits, for BOTH layers (raw proposal vs resolved = proposal ⊕ corrections),
plus the ball. Run under the pipeline venv:

    python scripts/motion_stats.py --scene out/anim_adr11/export/scene.json
"""

import argparse

import numpy as np

from pitch3d.core.correction.engine import resolve_ball, resolve_subject_motion
from pitch3d.core.scene.serialization import load_scene

HUMAN_MAX_SPEED = 10.5   # m/s — elite sprint ceiling
HUMAN_MAX_ACCEL = 8.0    # m/s^2 — elite acceleration ceiling
BALL_MAX_SPEED = 36.0    # m/s — hardest shots
TURN_MIN_SPEED = 2.0     # m/s — turn rate only meaningful when moving


def kin_stats(frames, transl, fps):
    """Speed/accel/turn stats over the horizontal (XY) root track."""
    frames = np.asarray(frames, float)
    xy = np.asarray(transl, float)[:, :2]
    if len(frames) < 3:
        return None
    dt = np.diff(frames) / fps
    ok = dt > 0
    vel = np.diff(xy, axis=0)[ok] / dt[ok, None]
    speed = np.linalg.norm(vel, axis=1)
    if len(speed) < 2:
        return None
    dv = np.diff(vel, axis=0)
    accel = np.linalg.norm(dv, axis=1) / dt[ok][1:]
    moving = (speed[:-1] > TURN_MIN_SPEED) & (speed[1:] > TURN_MIN_SPEED)
    if moving.any():
        u, w = vel[:-1][moving], vel[1:][moving]
        cosang = np.clip(
            np.sum(u * w, axis=1)
            / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1) + 1e-9),
            -1.0, 1.0)
        turn = np.degrees(np.arccos(cosang)) / dt[ok][1:][moving]  # deg/s
    else:
        turn = np.zeros(1)
    return {
        "n": len(frames),
        "sp_p50": float(np.median(speed)),
        "sp_p95": float(np.percentile(speed, 95)),
        "sp_max": float(speed.max()),
        "ac_max": float(accel.max()) if len(accel) else 0.0,
        "turn_max": float(turn.max()),
        "viol_sp": int((speed > HUMAN_MAX_SPEED).sum()),
        "viol_ac": int((accel > HUMAN_MAX_ACCEL).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--fps", type=float, default=29.97)
    args = ap.parse_args()

    scene = load_scene(args.scene)
    fps = args.fps

    header = (f"{'subj':>6} {'layer':>8} {'n':>4} {'sp_p50':>7} {'sp_p95':>7} "
              f"{'sp_max':>7} {'ac_max':>8} {'turn_max':>9} {'>spd':>5} {'>acc':>5}")
    print(f"== motion stats: {args.scene} fps={fps}")
    print(f"== limits: speed {HUMAN_MAX_SPEED} m/s, accel {HUMAN_MAX_ACCEL} m/s^2, "
          f"ball {BALL_MAX_SPEED} m/s")
    print(header)

    totals = {"proposal": [0, 0], "resolved": [0, 0]}
    for sub in scene.subjects:
        layers = {
            "proposal": sub.proposal,
            "resolved": resolve_subject_motion(
                sub.proposal, scene.corrections_for(sub.track_id)),
        }
        for name, motion in layers.items():
            st = kin_stats(motion.pose.frames, motion.pose.transl, fps)
            if st is None:
                continue
            totals[name][0] += st["viol_sp"]
            totals[name][1] += st["viol_ac"]
            flag = " <<<" if (st["viol_sp"] or st["viol_ac"]) and name == "resolved" else ""
            print(f"{sub.track_id:>6} {name:>8} {st['n']:>4} {st['sp_p50']:>7.2f} "
                  f"{st['sp_p95']:>7.2f} {st['sp_max']:>7.2f} {st['ac_max']:>8.1f} "
                  f"{st['turn_max']:>9.0f} {st['viol_sp']:>5} {st['viol_ac']:>5}{flag}")

    if scene.ball is not None:
        ball = resolve_ball(scene.ball, scene.corrections_for(None))
        bf = np.asarray(ball.frames, float)
        bp = np.asarray(ball.positions_3d, float)
        dt = np.diff(bf) / fps
        ok = dt > 0
        bs = np.linalg.norm(np.diff(bp, axis=0)[ok], axis=1) / dt[ok]
        print(f"==  ball: n={len(bf)} sp_p50={np.median(bs):.2f} "
              f"sp_p95={np.percentile(bs, 95):.2f} sp_max={bs.max():.2f} "
              f"viol(> {BALL_MAX_SPEED}): {(bs > BALL_MAX_SPEED).sum()}")

    for name, (vs, va) in totals.items():
        print(f"== TOTAL {name}: speed-violation frames={vs} accel-violation frames={va}")
    print("MOTION_STATS_OK")


if __name__ == "__main__":
    main()
