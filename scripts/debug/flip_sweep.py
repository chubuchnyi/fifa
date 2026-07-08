"""Sweep the four camera-frame flip hypotheses and score each objectively.

The calibrated CameraTrack is solved for a 180°-rolled frame (memory
project_camera_180_roll), so the overlay path composes a mirror ``D`` before
projecting.  Which ``D``?  A pure left-right mirror diag(-1,1,1) fixes the
horizontal landing but — as the pose_probe harness showed — leaves every body
vertically inverted (feet above head).  A true 180° optical-axis roll is
diag(-1,-1,1) which flips BOTH image axes.

This bypasses ``frame_projector``'s auto-flip, rebuilds the projector from the
raw camera for each candidate ``D``, and reports, per candidate:

  * upright%  : players with foot pixel-v > head pixel-v (gravity ground truth)
  * inframe%  : pelvis inside the image
  * pitch anchors: where the centre mark and the far / near touchline land
                   (v small = top of image).  A sane camera puts the pitch
                   markings inside the frame, not in the crowd or off screen.

The winning ``D`` is the one that makes players upright AND keeps the pitch
anchors on the field.  Feed that back into ``poseannot/camera.py``.

Usage:
    python scripts/debug/flip_sweep.py --scene out/bakeoff/scene_A.json --frame 0
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from poseannot.camera import ProjectedFrame, project_points
from poseannot.config import load as load_cfg
from poseannot.scene_state import build_scene_state
from poseannot.video import frame_size

DEFAULT_VIDEO = "samples/video/Colombia-1-0-Congo-DR1080p.mp4"
PELVIS, HEAD = 0, 15
FEET = (7, 8, 10, 11)

CANDIDATES = {
    "I    (none)":       np.diag([1.0, 1.0, 1.0]),
    "X    diag(-1,1,1)": np.diag([-1.0, 1.0, 1.0]),
    "Y    diag(1,-1,1)": np.diag([1.0, -1.0, 1.0]),
    "XY   diag(-1,-1,1)": np.diag([-1.0, -1.0, 1.0]),
}


def _raw_projector(camera_track, frame_index: int, W: int, H: int, D: np.ndarray) -> ProjectedFrame:
    idx = int(frame_index)
    q = np.asarray(camera_track.rotation_quat[idx], dtype=float)
    R = Rotation.from_quat(np.roll(q, -1)).as_matrix()
    t = np.asarray(camera_track.translation[idx], dtype=float)
    K = camera_track.intrinsics
    fx, fy, cx, cy = K.fx, K.fy, K.cx, K.cy
    cal_w, cal_h = 2.0 * cx, 2.0 * cy
    if abs(cal_w - W) > 1 or abs(cal_h - H) > 1:
        sx, sy = W / cal_w, H / cal_h
        fx *= sx; fy *= sy; cx *= sx; cy *= sy
    R = D @ R
    t = D @ t
    return ProjectedFrame(fx=fx, fy=fy, cx=cx, cy=cy, R=R, t=t, frame_index=idx)


def sweep(scene: str, video: str, frame: int) -> None:
    base = load_cfg()
    cfg = replace(base, scene_json=Path(scene).resolve(), source_video=Path(video).resolve(),
                  corrections_out=Path("/tmp/pose_probe_noedits.json").resolve())
    st = build_scene_state(cfg)
    W, H = frame_size(str(cfg.source_video))
    from pitch3d.core.scene.pitch import pitch_line_world_points
    field = st.scene.field
    anchors = np.array([
        [0.0, 0.0, field.plane_z],                                   # centre mark
        [-field.dimensions.length / 2, -field.dimensions.width / 2, field.plane_z],  # far corner
        [-field.dimensions.length / 2,  field.dimensions.width / 2, field.plane_z],  # near corner
    ], dtype=float)
    pitch = pitch_line_world_points(field.dimensions, plane_z=field.plane_z, spacing=1.0)

    print(f"\n=== flip sweep  {Path(scene).name}  frame {frame}  {W}x{H} ===")
    print("  candidate           upright   inframe   pitch_inframe   centre(u,v)   farC(v) nearC(v)")
    for name, D in CANDIDATES.items():
        proj = _raw_projector(st.scene.camera, frame, W, H, D)
        up = infr = present = 0
        for sub in st.subjects.values():
            hit = np.where(sub.frames == frame)[0]
            if hit.size == 0:
                continue
            k = int(hit[0])
            pix = project_points(sub.joints[k], proj)
            pu, pv = pix[PELVIS]
            if not (np.isfinite(pu) and np.isfinite(pv)):
                continue
            present += 1
            if 0 <= pu < W and 0 <= pv < H:
                infr += 1
            head_v = pix[HEAD, 1]
            foot_vs = [pix[i, 1] for i in FEET if np.isfinite(pix[i, 1])]
            if foot_vs and np.isfinite(head_v) and max(foot_vs) > head_v:
                up += 1
        auv = project_points(anchors, proj)
        puv = project_points(pitch, proj)
        pin = int((np.isfinite(puv).all(axis=1) & (puv[:, 0] >= 0) & (puv[:, 0] < W)
                   & (puv[:, 1] >= 0) & (puv[:, 1] < H)).sum())
        c, farc, nearc = auv
        print(
            f"  {name:<18} {up:>2}/{present:<4}  {infr:>2}/{present:<4}  {pin:>4}/{len(pitch):<5}  "
            f"({c[0]:6.0f},{c[1]:5.0f})   {farc[1]:6.0f}  {nearc[1]:6.0f}"
        )
    print("  (want: high upright, high inframe, pitch_inframe>0, farC(v) < nearC(v) = far side higher up)")


def render(scene: str, video: str, frame: int, flip: str, out_dir: str) -> None:
    """Draw pitch (green) + per-player head(red)/pelvis(yellow)/feet(cyan) under one D,
    so orientation (head-up vs head-down) and pitch placement are visible at a glance."""
    import cv2
    from pitch3d.core.scene.pitch import pitch_line_world_points
    from poseannot.video import read_frame

    D = np.diag([float(s) for s in flip.split(",")])
    base = load_cfg()
    cfg = replace(base, scene_json=Path(scene).resolve(), source_video=Path(video).resolve(),
                  corrections_out=Path("/tmp/pose_probe_noedits.json").resolve())
    st = build_scene_state(cfg)
    W, H = frame_size(str(cfg.source_video))
    proj = _raw_projector(st.scene.camera, frame, W, H, D)
    field = st.scene.field
    bgr = read_frame(video, frame)
    puv = project_points(pitch_line_world_points(field.dimensions, plane_z=field.plane_z, spacing=0.5), proj)
    for u, v in puv:
        if np.isfinite(u) and 0 <= u < W and 0 <= v < H:
            cv2.circle(bgr, (int(u), int(v)), 1, (0, 200, 0), -1)
    for sub in st.subjects.values():
        hit = np.where(sub.frames == frame)[0]
        if hit.size == 0:
            continue
        pix = project_points(sub.joints[int(hit[0])], proj)
        for j, col in ((PELVIS, (0, 255, 255)), (HEAD, (0, 0, 255))):
            u, v = pix[j]
            if np.isfinite(u):
                cv2.circle(bgr, (int(u), int(v)), 3, col, -1)
        for j in FEET:
            u, v = pix[j]
            if np.isfinite(u):
                cv2.circle(bgr, (int(u), int(v)), 2, (255, 200, 0), -1)
        # head→pelvis line shows body axis direction
        hu, hv = pix[HEAD]; pu, pv = pix[PELVIS]
        if np.isfinite(hu) and np.isfinite(pu):
            cv2.line(bgr, (int(hu), int(hv)), (int(pu), int(pv)), (0, 0, 255), 1)
    banner = f"{Path(scene).stem} f{frame} flip={flip}  red=head yellow=pelvis cyan=feet green=pitch"
    cv2.putText(bgr, banner, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(bgr, banner, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    fn = outp / f"flip_{Path(scene).stem}_f{frame}_{flip.replace(',', '')}.png"
    cv2.imwrite(str(fn), bgr)
    print(f"  wrote {fn}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--render", default=None, help="draw markers for one D, e.g. -1,-1,1")
    ap.add_argument("--out", default="out/bakeoff")
    args = ap.parse_args()
    if args.render:
        render(args.scene, args.video, args.frame, args.render, args.out)
    else:
        sweep(args.scene, args.video, args.frame)


if __name__ == "__main__":
    main()
