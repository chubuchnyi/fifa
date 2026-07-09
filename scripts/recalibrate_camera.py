#!/usr/bin/env python
"""Rebuild the scene camera from the trusted homography — task #61.

Measured 2026-07-09 (scripts/debug/calib_probe.py): the stored PnLCalib
extrinsics place the 3D camera up in the crowd — the projected pitch lands in
the stands and players render ~3x too small. The per-frame homography ``H``
(image->world plane), by contrast, lands the pitch exactly on the painted lines
and is what already grounds the feet (``FieldCalibration.image_to_world``). It
is the trustworthy calibration.

For a ground point (Z=0):  ``[u v 1]^T ~ K [r1 r2 t] [x y 1]^T = H^{-1} [x y 1]^T``
so given the trusted ``H`` and a focal ``f`` we recover a *consistent* ``(R, t)``
by decomposing ``H^{-1}``. With the right focal (~3500 px @1920x1080, measured
against real standing-player pixel heights in scripts/debug/focal_size_probe.py)
BOTH the pitch and the players project correctly — the camera is no longer a
guess, it is the homography turned into a pinhole.

The rebuilt track is flagged ``raw_frame_aligned=True`` so poseannot's legacy
180-roll workaround (poseannot.camera.frame_projector) is skipped for it.

Auto default focal 3500; ``--focal`` overrides (auto + manual, per project rule).
Idempotent: always re-decomposes from the stored homography, never from a prior
rebuild. Run on A and B (same clip -> same real camera)::

    python scripts/recalibrate_camera.py \\
        poseannot/clips/A_smplestx/scene.json \\
        poseannot/clips/B_sam3dbody/scene.json --focal 3500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_FOCAL = 3500.0
VW, VH = 1920, 1080  # the homography's pixel space == video native size


def redecompose(hinv: np.ndarray, f: float, cx: float, cy: float):
    """Recover world->camera ``(R, t)`` from world->image homography ``hinv``.

    ``hinv = K [r1 r2 t]`` up to scale for the ground plane; invert K, normalise
    on the first column, complete the rotation with ``r3 = r1 x r2`` and snap to
    the nearest orthonormal matrix. ``t[2] > 0`` keeps the camera in front of
    the pitch (resolves the two-fold planar-homography sign ambiguity).
    """
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
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
    return R, t


def rebuild_camera(scene, focal: float):
    from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack

    cal = scene.field.calibration
    homs = cal.homographies
    frames = np.asarray(cal.frames, dtype=int)
    cx, cy = VW / 2.0, VH / 2.0

    quats = np.zeros((homs.shape[0], 4))
    transl = np.zeros((homs.shape[0], 3))
    for i in range(homs.shape[0]):
        R, t = redecompose(np.linalg.inv(homs[i]), focal, cx, cy)
        quats[i] = np.roll(Rotation.from_matrix(R).as_quat(), 1)  # (x,y,z,w)->(w,x,y,z)
        transl[i] = t

    intr = CameraIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy, width=VW, height=VH)
    return CameraTrack(
        intrinsics=intr, frames=frames, rotation_quat=quats,
        translation=transl, estimated=True, raw_frame_aligned=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenes", nargs="+", help="scene.json paths to recalibrate")
    ap.add_argument("--focal", type=float, default=DEFAULT_FOCAL,
                    help=f"focal length in 1920-space px (default {DEFAULT_FOCAL:.0f})")
    args = ap.parse_args()

    from pitch3d.core.scene.serialization import load_scene, save_scene

    for path in args.scenes:
        scene = load_scene(path)
        if scene.field is None or scene.field.calibration is None:
            print(f"SKIP {path}: no field calibration to decompose")
            continue
        cam = rebuild_camera(scene, args.focal)
        scene.camera = cam
        save_scene(scene, path)
        t = cam.translation
        print(
            f"OK   {path}\n"
            f"     focal={args.focal:.0f}  frames={cam.n_frames}  "
            f"raw_frame_aligned={cam.raw_frame_aligned}\n"
            f"     cam dist t[z] median={np.median(t[:,2]):.1f}m  "
            f"t[x,y] median=({np.median(t[:,0]):.1f},{np.median(t[:,1]):.1f})"
        )


if __name__ == "__main__":
    main()
