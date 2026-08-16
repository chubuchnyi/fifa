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

**Why every scene needs this, not just the one it was fitted on.** PnLCalib solves each frame as a
free 8-DOF DLT, with nothing tying the result to a pinhole. On this clip the result is not merely
an inaccurate camera — it is *no camera at all*: swept over every focal from 200 to 12000 px, the
closest realizable pinhole is still **525 px** away from the stored homographies (#119's are
1.4 px). So ``camera_from_calibration`` rightly refuses, ``app/controller.py`` falls back to an
invented ``Viewpoint.BROADCAST`` camera, and the scene ends up drawing its pitch through the
measured homography and its players through a synthetic camera 3.9x too small — which is #61,
and exactly what "the ground marks are right but the players are not" looks like from the UI.
The homographies still fit the *visible paint* (that is why the marks land), they just extrapolate
to a pitch 9157 px wide in a 1920 px image. There is no cheaper repair than a real camera.

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

The fit itself is kept in ``calib/`` — a few kB, minutes to recompute, and only reproducible from
a video that is not in the repo. One camera serves every scene of that clip, so pass them all::

    python scripts/apply_rigid_camera.py calib/Colombia-1-0-Congo-DR1080p.npz \\
        --scene out/carry_off/export/scene.json out/anim_full_realism/scene.json \\
                out/fresh60/export/scene.json out/physics_debug/scene_replayed_v2.json \\
                poseannot/clips/A_smplestx/scene.json poseannot/clips/B_sam3dbody/scene.json
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


def scene_is_mirrored(calibration, width: int, height: int) -> bool:
    """Is this scene's world the mirrored template one (#118), rather than right-handed Z-up?

    Measured off the scene's own calibration, because it is not a property of the file format or
    of the producer version — ``out/fresh60`` was written after the #118 fix and is right-handed,
    ``out/anim_full_realism`` before it and is not, and both otherwise look identical. Mirroring a
    scene that is already right-handed is just as wrong as not mirroring one that is not.
    """
    from poseannot.camera import plane_orientation

    w2i = np.linalg.inv(np.asarray(calibration.homographies, dtype=float))
    signs = np.array([plane_orientation(h, width, height) for h in w2i])
    assert (signs > 0).all() or (signs < 0).all(), (
        f"the clip changes frame mid-way: {int((signs < 0).sum())}/{len(signs)} mirrored"
    )
    return bool(signs[0] < 0)


def read_fit(blob) -> dict:
    """Normalise either npz schema into ``focal (T,)``, ``centre (T,3)``, ``cx``, ``cy``.

    **Two schemas, and the second is camlab's.** This repo's own
    ``scripts/fit_rigid_camera.py`` writes schema 1 — ONE ``focal`` and ONE ``centre`` for the
    whole clip, with no principal point, because it assumed the image centre. camlab's
    ``export_camera.py`` writes schema 2: ``focal_px`` and ``position`` **per frame**, plus the
    ``cx, cy`` the camera was actually solved under.

    Reading both is deliberate and is not the compatibility branch ADR-0013 §4 forbids. That rule
    is about not asking camlab to keep writing an old shape; schema 1 is *ours*, still produced by
    our own fitter, and pinned by `tests/e2e/test_golden_real_camera.py` against a committed 7 kB
    file. Dropping it would delete the only test in this repo backed by a real measurement.

    What schema 2 buys, measured by camlab: collapsing a zooming clip to one focal costs
    **1.65 → 4.56 px** on `fan` (zoom 1.59×) and drops 2 of 12 frames out of the 20 px band. And
    ``cx, cy`` end the guess — on a cropped clip the optical axis is not the image centre
    (`stadium_a`'s is at −204, not 304), which is the landmine both repos have hit.
    """
    # `np.load` gives an NpzFile (keys in `.files`), `main` then turns it into a plain dict to
    # apply --focal. Ask both the same way, or the dict silently reads as schema 1 and dies on a
    # key that only schema 1 has.
    keys = set(getattr(blob, "files", None) or blob.keys())
    schema = int(blob["schema"]) if "schema" in keys else 1
    n = len(np.asarray(blob["frames"]))
    if schema >= 2:
        focal = np.asarray(blob["focal_px"], dtype=float).reshape(-1)
        centre = np.asarray(blob["position"], dtype=float).reshape(-1, 3)
        cx = float(blob["cx"]) if "cx" in keys else None
        cy = float(blob["cy"]) if "cy" in keys else None
    else:
        focal = np.full(n, float(blob["focal"]))
        centre = np.tile(np.asarray(blob["centre"], dtype=float), (n, 1))
        cx = cy = None
    if focal.shape != (n,) or centre.shape != (n, 3):
        raise SystemExit(f"schema {schema}: focal {focal.shape} / centre {centre.shape} "
                         f"do not match {n} frames")
    return {"schema": schema, "focal": focal, "centre": centre, "cx": cx, "cy": cy}


def camera_track(blob, width: int, height: int):
    """The fitted parameters as a CameraTrack, in the right-handed world."""
    from scipy.spatial.transform import Rotation

    from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack

    fit = read_fit(blob)
    focal, centre = fit["focal"], fit["centre"]
    cx = fit["cx"] if fit["cx"] is not None else width / 2.0
    cy = fit["cy"] if fit["cy"] is not None else height / 2.0
    rot = np.stack([rot_from_rvec(r) for r in blob["rvecs"]])
    quat = np.roll(Rotation.from_matrix(rot).as_quat(), 1, axis=1)  # (x,y,z,w) → (w,x,y,z)
    # Per frame on both counts: `translation` always was, and `focal_px` now is. A track whose
    # focal never changes carries the array anyway — `intrinsics_at` returns the shared intrinsics
    # when it is None, and this is never None, so the nominal `fx` below is the median rather than
    # a value that pretends to hold for every frame.
    return CameraTrack(
        intrinsics=CameraIntrinsics(
            fx=float(np.median(focal)), fy=float(np.median(focal)),
            cx=cx, cy=cy, width=width, height=height
        ),
        focal_px=focal,
        frames=np.asarray(blob["frames"], dtype=int),
        rotation_quat=quat,
        translation=-np.einsum("fij,fj->fi", rot, centre),
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


def mirror_corrections(scene) -> int:
    """Carry the *corrections* across the world flip too — :func:`mirror_subjects` is only half.

    A replayed scene keeps its motion here, not in ``proposal.pose``: ``resolve_subject_motion``
    lays these absolute keyframes over the proposal, so mirroring the proposal alone moves nothing
    a viewer sees. That is how the default clip ended up holding right-handed proposals, a
    right-handed camera and 677 left-handed corrections, and drew every player mirrored about the
    halfway line — the visible face of #120.

    ``KEYFRAME_INTERP`` values are absolute, so each takes exactly the map its target takes in
    :func:`mirror_subjects`. Anything else raises rather than passing through unmirrored.
    """
    from scipy.spatial.transform import Rotation

    from pitch3d.core.scene.layers import CorrectionMode, TargetKind

    m_local = local_mirror()
    for corr in scene.corrections:
        kind = corr.target.kind
        if corr.mode != CorrectionMode.KEYFRAME_INTERP or kind not in (
            TargetKind.ROOT_TRANSLATION, TargetKind.ROOT_ORIENTATION, TargetKind.POSE_BODY_JOINT
        ):
            raise NotImplementedError(
                f"correction {corr.id} is {kind.value}/{corr.mode.value}, which the world flip "
                f"has no rule for. Letting it through unmirrored is the bug this function fixes."
            )
        v = np.asarray(corr.payload.key_values, dtype=float)
        if kind is TargetKind.ROOT_TRANSLATION:
            corr.payload.key_values = v @ M
        elif kind is TargetKind.ROOT_ORIENTATION:
            corr.payload.key_values = Rotation.from_matrix(
                m_local @ Rotation.from_rotvec(v).as_matrix() @ S
            ).as_rotvec()
        else:
            # The sagittal flip, so the joint swaps side as well as negating its y and z.
            j = corr.target.joint_index
            assert j is not None and 0 <= j < len(_BP_SWAP), f"joint_index {j} out of body_pose"
            corr.payload.key_values = v * (1.0, -1.0, -1.0)
            corr.target.joint_index = int(_BP_SWAP[j])
    return len(scene.corrections)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("camera", type=Path, nargs="?",
                    default=ROOT / "calib/Colombia-1-0-Congo-DR1080p.npz",
                    help="npz from fit_rigid_camera.py --out")
    ap.add_argument("--scene", type=Path, nargs="+",
                    default=[ROOT / "out/carry_off/export/scene.json"],
                    help="one or more scenes of the SAME clip and frame range — the camera is a "
                         "property of the video, so one fit serves every variant of it")
    ap.add_argument("--out", type=Path, default=None,
                    help="single --scene only; otherwise each goes to <scene>_rigid.json")
    ap.add_argument("--focal", type=float, default=None,
                    help="override the measured focal (auto npz → this flag, per project rule); "
                         "the rotations and centre are the fit's either way")
    ap.add_argument("--mirror", choices=("auto", "on", "off"), default="auto",
                    help="mirror the subjects into the right-handed world. 'auto' measures the "
                         "scene's own frame (see scene_is_mirrored); 'off' isolates whether a "
                         "visual regression came from the camera or from the flip")
    args = ap.parse_args()

    from pitch3d.core.orchestration.pipeline import require_solved_calibration
    from pitch3d.core.scene.serialization import load_scene, save_scene

    assert args.out is None or len(args.scene) == 1, "--out takes a single --scene"

    blob = np.load(args.camera)
    fit = read_fit(blob)
    blob = dict(blob)
    if args.focal is not None:
        # The override collapses the clip to ONE focal, which is what schema 1 always was. Say so
        # rather than let it look like a nudge: on `fan` that is 2875..4592 px replaced by a
        # constant, 36 % of the value at the extremes.
        print(f"focal overridden: {fit['focal'].min():.0f}..{fit['focal'].max():.0f} -> "
              f"{args.focal:.1f} px, one focal for the whole clip")
        blob["schema"], blob["focal"] = np.array(1), np.array(args.focal)
        blob["centre"] = fit["centre"][0]
        fit = read_fit(blob)
    width, height = int(blob.get("width", 1920)), int(blob.get("height", 1080))
    frames = np.asarray(blob["frames"], dtype=int)
    spread = float(np.linalg.norm(fit["centre"] - fit["centre"][0], axis=1).max())
    zoom = float(fit["focal"].max() / max(fit["focal"].min(), 1e-9))
    print(f"fit: schema {fit['schema']}, focal {fit['focal'].min():.0f}.."
          f"{fit['focal'].max():.0f} px (zoom {zoom:.3f}x) @ {width}x{height}   frames "
          f"{frames.min()}-{frames.max()} ({len(frames)})   centre "
          f"{fit['centre'][0].round(1)} m (spread {spread:.2f} m)   "
          f"principal point {'from the fit' if fit['cx'] is not None else 'assumed at the centre'}")

    for path in args.scene:
        scene = load_scene(str(path))
        assert scene.field is not None and scene.field.calibration is not None, f"{path}: no field"
        cal = scene.field.calibration
        # This script is how a dead scene gets to look alive (#125): it replaces the calibration
        # outright, so a run that solved nothing — identity homographies, confidence 0 — comes out
        # the far side scoring a healthy 1.0 px against the paint, and every subject in it is still
        # placed by the fiction that one pixel is one metre. Refuse rather than launder.
        require_solved_calibration(cal)
        # A subset is fine and is not a special case: the fit is one focal, one centre and a
        # rotation per frame OF THE VIDEO, so a scene covering frames 0-47 of a 0-59 fit wants
        # exactly those 48 rotations. Only frames the fit never saw are a refit.
        want = np.asarray(cal.frames, dtype=int)
        idx = np.searchsorted(frames, want)
        assert (idx < len(frames)).all() and (frames[np.minimum(idx, len(frames) - 1)] == want).all(), (
            f"{path}: needs frames {want.min()}-{want.max()} ({len(want)}), the fit covers "
            f"{frames.min()}-{frames.max()} ({len(frames)}) — refit with --frames {want.max() + 1}"
        )
        # Every per-frame array, not just the two schema 1 had. `read_fit`'s shape check caught
        # this the first time it ran: a schema-2 blob cut to 60 frames still carried 120 focals.
        cut = {**blob, "frames": want, "rvecs": np.asarray(blob["rvecs"])[idx],
               "world_to_image": np.asarray(blob["world_to_image"])[idx]}
        for key in ("focal_px", "position"):
            if key in cut and np.ndim(cut[key]) and len(np.asarray(cut[key])) == len(frames):
                cut[key] = np.asarray(cut[key])[idx]

        mirrored = scene_is_mirrored(cal, width, height)
        do_mirror = mirrored if args.mirror == "auto" else args.mirror == "on"
        scene.camera = camera_track(cut, width, height)
        # The calibration has to be the SAME camera, or the pitch overlay and the players would be
        # drawn through two different ones — the split #61 is a symptom of.
        cal.homographies = np.stack([np.linalg.inv(h) for h in cut["world_to_image"]])
        n = mirror_subjects(scene) if do_mirror else 0
        n_corr = mirror_corrections(scene) if do_mirror else 0

        out = args.out or path.with_name(path.stem + "_rigid.json")
        save_scene(scene, str(out))
        print(f"OK   {out}\n"
              f"     stored frame {'mirrored template' if mirrored else 'right-handed'} "
              f"({args.mirror}) -> mirrored {n} subjects, {n_corr} corrections")


if __name__ == "__main__":
    main()
