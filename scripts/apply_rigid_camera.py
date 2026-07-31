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
writing an honest camera *forces* the scene right-handed, and the subjects have to come along.

Mirroring a *human* is the awkward part: ``M·(FK output)`` is not a pose SMPL-X can represent,
because no setting of the parameters produces a left-handed body. The way through is to mirror
the body about its own sagittal plane as well, which SMPL-X *can* represent, and let the two
improper maps cancel. See :func:`mirror_subjects` for the three parameters that changes and the
measurement that pins each one.

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
#: SMPL-X's own sagittal (left↔right) mirror. Measured, not assumed: in the rest pose
#: ``left_hip − right_hip`` is ``(0.122, 0.011, −0.005)``, so ``x`` is the left-right axis.
S = np.diag([-1.0, 1.0, 1.0])

#: ``body_pose`` rows (SMPL-X joints 1-21) under the left↔right swap. Rows absent from a pair
#: (spines, neck, head) are on the midline and map to themselves.
_LR_PAIRS = ((0, 1), (3, 4), (6, 7), (9, 10), (12, 13), (15, 16), (17, 18), (19, 20))
_BP_SWAP = np.arange(21)
for _a, _b in _LR_PAIRS:
    _BP_SWAP[_a], _BP_SWAP[_b] = _b, _a


def rot_from_rvec(rvec: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.asarray(rvec, dtype=float)).as_matrix()


def flip_body_pose(body_pose: np.ndarray) -> np.ndarray:
    """Mirror ``(T, 21, 3)`` axis-angle body pose about the body's sagittal plane.

    Swap the left/right joints, then negate each rotation's ``y`` and ``z``. The negation looks
    arbitrary and is not: axis-angle is a *pseudo*vector, so under an improper map ``A`` it
    transforms as ``ω → det(A)·A·ω``, and for ``A = S`` that is ``(ωx, −ωy, −ωz)``.
    """
    out = np.asarray(body_pose, dtype=float)[:, _BP_SWAP, :].copy()
    out[..., 1:] *= -1
    return out


def local_mirror() -> np.ndarray:
    """The world-Y mirror written in the frame ``global_orient`` is actually expressed in.

    ``global_orient`` is a **camera-frame** rotation (``core.scene.frames``): world vertices are
    ``R_cam→world · R · (body − pelvis) + transl``. So mirroring the *world* means conjugating
    ``M`` into that frame — it comes out ``diag(1, 1, −1)``, because world Y is SMPL-X's Z.
    Derived from the shared constant rather than hard-coded, so the two cannot drift apart.
    """
    from pitch3d.core.scene.frames import R_SMPLX_CAMERA_TO_WORLD as r_c2w

    return r_c2w.T @ M @ r_c2w


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
    """Carry the subjects from the mirrored template world into the right-handed one.

    The requirement is exactly ``world(mirrored params) == M · world(stored params)``, and it
    takes two body parameters, not one — the old code changed only ``global_orient`` and left
    every body 1.01 m out and its own mirror image internally:

    * ``body_pose`` is mirrored about the sagittal plane (:func:`flip_body_pose`). A mirrored
      human is not a pose SMPL-X can represent, so the only way through is to flip the body too
      and let the two improper maps cancel.
    * ``global_orient → Mˡ·R·S`` with ``Mˡ`` from :func:`local_mirror`. Both factors are improper,
      so ``det(Mˡ·R·S) = +1`` and it is a real rotation; ``S`` cancels the sagittal flip above,
      leaving the world mirror. The old ``M·R·M`` was wrong twice over — a conjugation leaves a
      body-local 180° yaw, and it used the *world* ``M`` on a *camera-frame* rotation.

    ``transl`` needs only ``M·t``: it is the world pelvis, and SMPL-X's root turns about the
    pelvis, so no rest-pose offset rides along.

    Verified against real SMPL-X FK over all 23 subjects × 4 frames of the target clip: worst
    world-joint error 0.041 m, against 1.01 m for the old code. That floor is the neutral
    template's own left/right asymmetry (its rest hips differ by 0.011 m off-axis), not a bug.
    """
    from scipy.spatial.transform import Rotation

    m_local = local_mirror()
    for subject in scene.subjects:
        pose = subject.proposal.pose
        rot = rot_from_rvec(pose.global_orient)  # (T, 3, 3)
        pose.transl = pose.transl @ M  # M is diagonal, so this is M·t per row
        pose.global_orient = Rotation.from_matrix(m_local @ rot @ S).as_rotvec()
        pose.body_pose = flip_body_pose(pose.body_pose)
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
