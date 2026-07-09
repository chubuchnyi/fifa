#!/usr/bin/env python
"""Pick the #61 focal prior by matching projected player *height* to reality.

The pitch alignment is focal-invariant under re-decomposition (see calib_probe),
so the pitch cannot choose the focal. Player height can: a standing footballer
is ~1.8 m. At each candidate focal we erect a 1.8 m vertical stick at every
grounded foot (clip A's ``transl`` XY, which already sits on real players) and
project it through the re-decomposed camera. The focal whose sticks match the
real players' pixel height is the calibration's true focal.

Output: /tmp/focal_size_f<f>.png per focal — look and compare stick vs player.
Also prints median stick pixel-height per focal.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
A_PATH = ROOT / "poseannot/clips/A_smplestx/scene.json"
VIDEO = ROOT / "samples/video/Colombia-1-0-Congo-DR1080p.mp4"
FRAME = 0
STATURE = 1.8
FOCALS = [1158, 2300, 3500, 4600]


def redecompose(hinv, f, cx, cy):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float)
    M = np.linalg.inv(K) @ hinv
    lam = 1.0 / np.linalg.norm(M[:, 0])
    r1, r2, t = M[:, 0] * lam, M[:, 1] * lam, M[:, 2] * lam
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R, t, K


def project_rt(world, R, t, K):
    cam = world @ R.T + t
    z = np.where(cam[:, 2] > 1e-6, cam[:, 2], np.nan)
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1)


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    from poseannot.video import read_frame
    from pitch3d.core.scene.serialization import load_scene

    scene = load_scene(str(A_PATH))
    cal = scene.field.calibration
    hinv = np.linalg.inv(cal.homographies[cal._frame_row(FRAME)])

    feet = []
    for sub in scene.subjects:
        pose = sub.proposal.pose
        frames = np.asarray(pose.frames).reshape(-1)
        rows = np.where(frames == FRAME)[0]
        if rows.size == 0:
            continue
        x, y = np.asarray(pose.transl, dtype=float)[rows[0], :2]
        feet.append((x, y))
    feet = np.asarray(feet)
    print(f"subjects on frame {FRAME}: {len(feet)}")

    bgr = read_frame(str(VIDEO), FRAME)
    vh, vw = bgr.shape[:2]
    cx, cy = vw / 2.0, vh / 2.0

    for f in FOCALS:
        R, t, K = redecompose(hinv, f, cx, cy)
        img = bgr.copy()
        heights = []
        for (x, y) in feet:
            foot = np.array([[x, y, 0.0]])
            head = np.array([[x, y, STATURE]])
            fu = project_rt(foot, R, t, K)[0]
            hu = project_rt(head, R, t, K)[0]
            if not (np.isfinite(fu).all() and np.isfinite(hu).all()):
                continue
            if not (0 <= fu[0] < vw and 0 <= fu[1] < vh):
                continue
            px_h = abs(fu[1] - hu[1])
            heights.append(px_h)
            cv2.line(img, (int(fu[0]), int(fu[1])), (int(hu[0]), int(hu[1])), (0, 255, 255), 2)
            cv2.circle(img, (int(fu[0]), int(fu[1])), 4, (0, 128, 255), -1)
        med = float(np.median(heights)) if heights else 0.0
        cv2.putText(img, f"f={f}  1.8m stick median = {med:.0f}px  (n={len(heights)})",
                    (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(f"/tmp/focal_size_f{f}.png", img)
        print(f"  f={f:5.0f}: median 1.8m height = {med:5.1f}px  n={len(heights)}")
    print("wrote /tmp/focal_size_f*.png")


if __name__ == "__main__":
    main()
