#!/usr/bin/env python
"""Diagnose the #61 camera-scale error by overlaying the pitch three ways.

The stored 3D camera (rotation_quat/translation/intrinsics) projects players
~3x too small (task #61). The per-frame homography ``H`` (image->world plane)
is a *separate* solve and is what actually grounds the feet. Question this
probe answers visually: **is H trustworthy?** If the homography-projected pitch
lands on the painted lines but the 3D-camera pitch does not, then the fix is to
*re-decompose* H with a correct focal prior (no pod needed).

For a ground point (Z=0):  [u v 1]^T ~ K [r1 r2 t] [x y 1]^T  =  H^{-1} [x y 1]^T
so given the trusted H and a chosen focal f we can recover a consistent (R, t).

Outputs (look at them):
  /tmp/calib_probe_homography.png  — pitch via H.world_to_image  (GREEN)
  /tmp/calib_probe_current.png     — pitch via stored 3D camera  (RED)
  /tmp/calib_probe_redecomp.png    — pitch via H re-decomposed at several focals
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
A_PATH = ROOT / "poseannot/clips/A_smplestx/scene.json"
VIDEO = ROOT / "samples/video/Colombia-1-0-Congo-DR1080p.mp4"
FRAME = 0


def redecompose(hinv: np.ndarray, f: float, cx: float, cy: float):
    """Recover (R, t) from world->image homography ``hinv`` at focal ``f``."""
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float)
    M = np.linalg.inv(K) @ hinv          # [r1 r2 t] up to scale
    lam = 1.0 / np.linalg.norm(M[:, 0])
    r1, r2, t = M[:, 0] * lam, M[:, 1] * lam, M[:, 2] * lam
    if t[2] < 0:                          # camera must sit in front of the plane
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R)           # nearest orthonormal
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R, t, K


def project_rt(world: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    cam = world @ R.T + t
    z = np.where(cam[:, 2] > 1e-6, cam[:, 2], np.nan)
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1)


def draw(frame: np.ndarray, uv: np.ndarray, color, label: str) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    n = 0
    for u, v in uv:
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        iu, iv = int(round(u)), int(round(v))
        if -50 <= iu < w + 50 and -50 <= iv < h + 50:
            cv2.circle(img, (iu, iv), 3, color, -1)
            n += 1
    cv2.putText(img, f"{label}  ({n} pts on/near frame)", (24, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return img


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    from poseannot.camera import frame_projector, project_points
    from pitch3d.core.scene.pitch import pitch_line_world_points
    from pitch3d.core.scene.serialization import load_scene

    scene = load_scene(str(A_PATH))
    cal = scene.field.calibration
    world = pitch_line_world_points(scene.field.dimensions, plane_z=scene.field.plane_z, spacing=1.0)
    print(f"pitch world points: {world.shape[0]}")

    frame = cv2.imread(str(_ensure_frame()))
    vh, vw = frame.shape[:2]
    cx, cy = vw / 2.0, vh / 2.0
    print(f"frame {FRAME}: {vw}x{vh}")

    # (a) pure homography (world->image); this is what grounds the feet
    uv_h = cal.world_to_image(FRAME, world[:, :2])
    cv2.imwrite("/tmp/calib_probe_homography.png",
                draw(frame, uv_h, (0, 255, 0), "HOMOGRAPHY world_to_image"))

    # (b) current stored 3D camera (the one the GUI uses)
    proj = frame_projector(scene.camera, FRAME, video_size=(vw, vh))
    uv_c = project_points(world, proj)
    print(f"current cam: fx={proj.fx:.0f} flipped={proj.frame_flipped}")
    cv2.imwrite("/tmp/calib_probe_current.png",
                draw(frame, uv_c, (0, 0, 255), "CURRENT 3D CAMERA"))

    # (c) re-decompose H^-1 at a sweep of focals (1920-space pixels)
    hinv = np.linalg.inv(cal.homographies[cal._frame_row(FRAME)])
    combo = frame.copy()
    colors = [(255, 200, 0), (0, 220, 255), (255, 0, 255), (0, 255, 0)]
    for i, f in enumerate([1158, 2300, 3500, 4600]):
        R, t, K = redecompose(hinv, f, cx, cy)
        uv = project_rt(world, R, t, K)
        combo = draw(combo, uv, colors[i % len(colors)], "")
        good = np.isfinite(uv).all(axis=1)
        print(f"  f={f:5.0f}: t={np.round(t,1)}  on-frame "
              f"{int(((uv[:,0]>=0)&(uv[:,0]<vw)&(uv[:,1]>=0)&(uv[:,1]<vh)&good).sum())}/{world.shape[0]}")
    cv2.putText(combo, "REDECOMP f=1158(gold) 2300(cyan) 3500(mag) 4600(grn)",
                (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite("/tmp/calib_probe_redecomp.png", combo)
    print("wrote /tmp/calib_probe_{homography,current,redecomp}.png")


def _ensure_frame() -> Path:
    out = Path("/tmp/calib_probe_frame.png")
    import sys
    sys.path.insert(0, str(ROOT))
    from poseannot.video import read_frame
    bgr = read_frame(str(VIDEO), FRAME)
    cv2.imwrite(str(out), bgr)
    return out


if __name__ == "__main__":
    main()
