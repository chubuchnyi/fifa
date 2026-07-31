#!/usr/bin/env python
"""Put the one fitted camera (#119) into a scene — the fix for #61.

``scripts/fit_rigid_camera.py --out camera.npz`` measures ONE focal, ONE camera centre and one
rotation per frame against two independent instruments (the painted lines, and the raw pixel
motion). This turns that measurement into a :class:`CameraTrack` and a matching
:class:`FieldCalibration`, so every consumer — the pitch overlay, the goal frames, the 3D
editor, Blender — sees the same camera.

It supersedes ``scripts/recalibrate_camera.py``, whose two recorded failures are exactly the
two things #118 and #119 measured instead of assuming:

* *"camera lands BELOW the pitch at every focal"* — the world frame was mirrored (#118).
* *"A single plane cannot determine the focal"* — true, so #119 does not use a single plane; the
  focal comes from image→image homographies over gaps of up to 59 frames, where ``K Rⱼ Rᵢᵀ K⁻¹``
  is no longer degenerate in f.

**The frame flip.** The fit works in the honest right-handed Z-up world; the stored scene is in
PnLCalib's mirrored top-down template (#118, #120). A camera cannot be written in a mirrored
world at all — ``M·R`` with ``M = diag(1, −1, 1)`` has determinant −1 and is not a rotation — so
writing an honest camera *forces* the scene right-handed, and the subjects have to come along:
``transl → M·t`` and ``global_orient → M·R·M`` (a conjugation, and therefore a real rotation).

What that does NOT do is mirror each body's own left/right, which needs the SMPL-X joint
permutation and is #120's remaining half. Every body is placed and facing correctly and is its
own mirror image internally — at 40-70 m that is a limb-swap, not a pose error, but it is
unfixed and this script says so rather than pretending the flip was complete.

Writes to a NEW scene by default: the user judges the camera by eye against the old one
(memory: the user is ground truth on pixel alignment), so both have to exist at once.

    python scripts/apply_rigid_camera.py /tmp/rigid_pan.npz \\
        --scene out/carry_off/export/scene.json --out out/carry_off/export/scene_rigid.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

#: World-Y mirror. Its own inverse, and improper — which is the whole difficulty above.
M = np.diag([1.0, -1.0, 1.0])


def rot_from_rvec(rvec: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.asarray(rvec, dtype=float)).as_matrix()


def camera_track(blob, width: int, height: int):
    """The fitted parameters as a CameraTrack, in the right-handed world."""
    from scipy.spatial.transform import Rotation

    from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack

    focal = float(blob["focal"])
    centre = np.asarray(blob["centre"], dtype=float)
    rot = np.stack([rot_from_rvec(r) for r in blob["rvecs"]])
    quat = np.roll(Rotation.from_matrix(rot).as_quat(), 1, axis=1)  # (x,y,z,w) → (w,x,y,z)
    return CameraTrack(
        intrinsics=CameraIntrinsics(
            fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0, width=width, height=height
        ),
        frames=np.asarray(blob["frames"], dtype=int),
        rotation_quat=quat,
        translation=-np.einsum("fij,j->fi", rot, centre),
        estimated=True,
        raw_frame_aligned=True,
    )


def mirror_subjects(scene) -> int:
    """Carry the subjects from the mirrored template world into the right-handed one."""
    from scipy.spatial.transform import Rotation

    for subject in scene.subjects:
        pose = subject.proposal.pose
        pose.transl = pose.transl @ M  # M is diagonal, so this is M·t per row
        rot = rot_from_rvec(pose.global_orient)
        pose.global_orient = Rotation.from_matrix(M @ rot @ M).as_rotvec()
    return len(scene.subjects)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("camera", type=Path, help="npz from fit_rigid_camera.py --out")
    ap.add_argument("--scene", type=Path, default=ROOT / "out/carry_off/export/scene.json")
    ap.add_argument("--out", type=Path, default=None, help="defaults to <scene>_rigid.json")
    ap.add_argument("--focal", type=float, default=None,
                    help="override the measured focal (auto npz → this flag, per project rule); "
                         "the rotations and centre are the fit's either way")
    ap.add_argument("--no-mirror", action="store_true",
                    help="leave the subjects in the stored frame — for isolating whether a visual "
                         "regression came from the camera or from the flip")
    args = ap.parse_args()

    from pitch3d.core.scene.serialization import load_scene, save_scene

    blob = dict(np.load(args.camera))
    if args.focal is not None:
        print(f"focal overridden: {float(blob['focal']):.1f} -> {args.focal:.1f}")
        blob["focal"] = np.array(args.focal)

    scene = load_scene(str(args.scene))
    assert scene.field is not None and scene.field.calibration is not None, "scene has no field"
    cal = scene.field.calibration
    frames = np.asarray(blob["frames"], dtype=int)
    assert (np.asarray(cal.frames, dtype=int) == frames).all(), (
        f"the fit covers frames {frames.min()}-{frames.max()} ({len(frames)}), the scene "
        f"{len(cal.frames)} — refit with --frames {len(cal.frames)}"
    )

    width, height = int(blob.get("width", 1920)), int(blob.get("height", 1080))
    scene.camera = camera_track(blob, width, height)
    # The calibration has to be the SAME camera, or the pitch overlay and the players would be
    # drawn through two different ones — the split #61 is a symptom of.
    cal.homographies = np.stack([np.linalg.inv(h) for h in blob["world_to_image"]])
    if not args.no_mirror:
        print(f"mirrored {mirror_subjects(scene)} subjects into the right-handed world")

    out = args.out or args.scene.with_name(args.scene.stem + "_rigid.json")
    save_scene(scene, str(out))

    cam = scene.camera
    print(
        f"OK   {out}\n"
        f"     focal {cam.intrinsics.fx:.0f} px @ {width}x{height}   frames {cam.n_frames}\n"
        f"     centre {np.asarray(blob['centre'], dtype=float).round(1)} m (fixed, by construction)"
    )


if __name__ == "__main__":
    main()
