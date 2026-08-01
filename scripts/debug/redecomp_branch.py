#!/usr/bin/env python
"""Find the CORRECT planar-homography decomposition branch (#61 head-down fix).

`scripts/recalibrate_camera.py`'s direct r3=r1xr2 decomposition projects the
GROUND correctly but renders every body HEAD-DOWN (flip_sweep: 0/23 upright,
yet pitch far/near depth is right). That is the signature of the wrong plane-
normal branch: the camera is reconstructed on the far side of the pitch plane,
so world +Z projects DOWN. A pure optical-axis flip can't fix it (it inverts the
pitch too).

cv2.decomposeHomographyMat returns up to 4 physical (R, t, n). This probe scores
each on frame 0 of the default scene: bodies upright% (foot_v>head_v via real
SMPL-X joints) AND pitch depth (far corner above near). The winner is the branch
whose plane normal faces the camera and puts the camera ABOVE the pitch. Renders
/tmp/branch_<i>.png for eyeball confirmation and prints the selection rule.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "out/physics_debug/scene_replayed_v2.json"
VIDEO = ROOT / "samples/video/Colombia-1-0-Congo-DR1080p.mp4"
FRAME = 0
FOCAL = 3500.0
PELVIS, HEAD = 0, 15
FEET = (7, 8, 10, 11)


def project(world, R, t, K):
    cam = world @ R.T + t
    z = np.where(cam[:, 2] > 1e-6, cam[:, 2], np.nan)
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1)


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
    from poseannot.config import load as load_cfg
    from poseannot.scene_state import build_scene_state
    from poseannot.video import frame_size, read_frame

    from pitch3d.core.scene.pitch import pitch_line_world_points

    cfg = replace(load_cfg(), scene_json=SCENE.resolve(), source_video=VIDEO.resolve(),
                  corrections_out=Path("/tmp/branch_noedits.json").resolve())
    st = build_scene_state(cfg)
    W, H = frame_size(str(VIDEO))
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[FOCAL, 0, cx], [0, FOCAL, cy], [0, 0, 1.0]])
    field = st.scene.field
    cal = field.calibration
    hinv = np.linalg.inv(cal.homographies[cal._frame_row(FRAME)])   # world->image

    # normalise hinv so K^-1 hinv has unit first column (metric scale)
    M = np.linalg.inv(K) @ hinv
    hinv = hinv / np.linalg.norm(M[:, 0])

    Mn = np.linalg.inv(K) @ hinv          # [s*r1 s*r2 s*t]; recover metric t per branch
    s = np.linalg.norm(Mn[:, 0])
    n_sol, Rs, Ts, Ns = cv2.decomposeHomographyMat(hinv, K)
    pitch = pitch_line_world_points(field.dimensions, plane_z=field.plane_z, spacing=1.0)
    far = np.array([[-field.dimensions.length / 2, 0, field.plane_z]])
    near = np.array([[field.dimensions.length / 2, 0, field.plane_z]])
    bgr0 = read_frame(str(VIDEO), FRAME)

    print(f"=== cv2 decomposeHomographyMat: {n_sol} solutions  f={FOCAL:.0f} ===")
    print("  i  upright  inframe  pitchIn  farV  nearV   C_z    n=[..]   t=[..]")
    for i in range(n_sol):
        R, n = Rs[i], Ns[i].reshape(3)
        sgn = np.sign(float(R[:, 0] @ (Mn[:, 0] / s)))    # match r1 direction to homography
        t = sgn * Mn[:, 2] / s                            # metric translation for THIS rotation
        C = -R.T @ t                      # camera centre in world
        up = infr = present = 0
        img = bgr0.copy()
        puv = project(pitch, R, t, K)
        for u, v in puv:
            if np.isfinite(u) and 0 <= u < W and 0 <= v < H:
                cv2.circle(img, (int(u), int(v)), 1, (0, 200, 0), -1)
        for sub in st.subjects.values():
            hit = np.where(sub.frames == FRAME)[0]
            if hit.size == 0:
                continue
            pix = project(sub.joints[int(hit[0])], R, t, K)
            pu, pv = pix[PELVIS]
            if not (np.isfinite(pu) and np.isfinite(pv)):
                continue
            present += 1
            if 0 <= pu < W and 0 <= pv < H:
                infr += 1
            head_v = pix[HEAD, 1]
            foot_vs = [pix[j, 1] for j in FEET if np.isfinite(pix[j, 1])]
            if foot_vs and np.isfinite(head_v) and max(foot_vs) > head_v:
                up += 1
            hu, hv = pix[HEAD]; du, dv = pix[PELVIS]
            if np.isfinite(hu) and np.isfinite(du):
                cv2.circle(img, (int(hu), int(hv)), 3, (0, 0, 255), -1)     # head red
                cv2.circle(img, (int(du), int(dv)), 3, (0, 255, 255), -1)   # pelvis yellow
                cv2.line(img, (int(hu), int(hv)), (int(du), int(dv)), (0, 0, 255), 1)
        fv = project(far, R, t, K)[0, 1]
        nv = project(near, R, t, K)[0, 1]
        pin = int((np.isfinite(puv).all(axis=1) & (puv[:, 0] >= 0) & (puv[:, 0] < W)
                   & (puv[:, 1] >= 0) & (puv[:, 1] < H)).sum())
        cv2.putText(img, f"branch {i}  upright {up}/{present}  farV {fv:.0f} nearV {nv:.0f}  C_z {C[2]:.1f}",
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, f"branch {i}  upright {up}/{present}  farV {fv:.0f} nearV {nv:.0f}  C_z {C[2]:.1f}",
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(f"/tmp/branch_{i}.png", img)
        print(f"  {i}  {up:>2}/{present:<3}  {infr:>2}/{present:<3}  {pin:>4}   {fv:6.0f} {nv:6.0f}  "
              f"{C[2]:6.1f}  n={np.round(n,2)}  t={np.round(t.reshape(3),1)}")
    print("wrote /tmp/branch_*.png")
    print("want: upright≈full, farV<nearV (far above near), C_z>0 (cam above pitch)")


if __name__ == "__main__":
    main()
