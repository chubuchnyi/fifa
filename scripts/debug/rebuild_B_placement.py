"""Reconstruct variant B's world placement from a reference calibration.

Diagnosis (scripts/debug/pose_probe + ab_transl_diff + this clip, 2026-07-08):
scene_B.json's per-subject root ``transl`` holds raw FOOT-PIXEL coordinates
(x,y in [0,W]x[0,H]) with only ``z`` = pelvis height in metres — the placement
stage never ran image->world, and scene_B's own field homography is identity.
So B's bodies project thousands of px off-screen even though the SMPL-X
*articulation* (body_pose / global_orient) is fine.

Because A and B are the SAME clip and frame, variant A's real FieldCalibration
is the correct camera+homography for B too.  This re-places every B subject by
running its stored foot pixel through A's ``image_to_world`` (keeping z), then
projects with A's camera.  It's the approximation that unblocks B's overlay; the
durable fix belongs in the pipeline placement stage that emitted scene_B.

Usage:
    python scripts/debug/rebuild_B_placement.py --b out/bakeoff/scene_B.json \
        --ref out/bakeoff/scene_A.json --frames 0 16 --out out/bakeoff
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from poseannot.camera import frame_projector, project_points
from poseannot.config import load as load_cfg
from poseannot.scene_state import BODY_JOINT_NAMES, build_scene_state
from poseannot.video import frame_size, read_frame

DEFAULT_VIDEO = "samples/video/Colombia-1-0-Congo-DR1080p.mp4"
# Body skeleton connectivity (same as overlay_from_scene / index.html).
BONES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9),
         (7, 10), (8, 11), (9, 12), (12, 15), (12, 13), (12, 14), (13, 16),
         (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]


def _load(scene: str, video: str):
    cfg = replace(load_cfg(), scene_json=Path(scene).resolve(),
                  source_video=Path(video).resolve(),
                  corrections_out=Path("/tmp/pose_probe_noedits.json").resolve())
    return build_scene_state(cfg)


def _pitch_ok(x, y) -> bool:
    return abs(x) <= 53 and abs(y) <= 35


def rebuild(b: str, ref: str, video: str, frame: int, out_dir: str) -> None:
    B = _load(b, video)
    A = _load(ref, video)
    cal = A.scene.field.calibration
    W, H = frame_size(video)
    proj = frame_projector(A.scene.camera, frame, video_size=(W, H))
    bgr = read_frame(video, frame)

    placed = dropped = 0
    for tid, sub in sorted(B.subjects.items()):
        hit = np.where(sub.frames == frame)[0]
        if hit.size == 0:
            continue
        k = int(hit[0])
        t_wrong = sub.transl[k].astype(float)
        u, v, z = float(t_wrong[0]), float(t_wrong[1]), float(t_wrong[2])
        wx, wy = cal.image_to_world(frame, np.array([[u, v]], dtype=float))[0]
        if not _pitch_ok(wx, wy):
            dropped += 1
            continue
        placed += 1
        t_correct = np.array([wx, wy, z], dtype=float)
        joints = sub.joints[k] - t_wrong + t_correct        # re-anchor articulation
        pix = project_points(joints, proj)
        for a, c in BONES:
            pa, pb = pix[a], pix[c]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                cv2.line(bgr, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 220, 0), 2)
        for p in pix:
            if np.isfinite(p).all():
                cv2.circle(bgr, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
        pu, pv = pix[0]
        if np.isfinite(pu):
            cv2.putText(bgr, str(tid), (int(pu) + 4, int(pv) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    banner = f"B re-placed via A homography  f{frame}  placed={placed} dropped={dropped}"
    cv2.putText(bgr, banner, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(bgr, banner, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    fn = outp / f"overlay_Bfixed_f{frame}.png"
    cv2.imwrite(str(fn), bgr)
    print(f"  frame {frame}: placed={placed} dropped={dropped} -> {fn}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--frames", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="out/bakeoff")
    args = ap.parse_args()
    _ = BODY_JOINT_NAMES
    for fr in args.frames:
        rebuild(args.b, args.ref, args.video, fr, args.out)


if __name__ == "__main__":
    main()
