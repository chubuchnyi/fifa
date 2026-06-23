"""Load a 3DPW sequence into a :class:`PoseEvalScene` for the real-GT pose bake-off.

3DPW (``imageFiles/<seq>/*.jpg`` + ``sequenceFiles/<seq>.pkl``) is the *accessible* stand-in for
the gated WorldPose video: it ships, per sequence, world-frame 3D SMPL joints, a **moving**
camera (one extrinsic per frame), and intrinsics — exactly what condition A of the harness needs
to place a backend's camera-space prediction into world and score Global/Local MPJPE in metres.

**Joint convention.** 3DPW's ``jointPositions`` are the 24-joint SMPL set. SMPL's body skeleton
(indices 0–21) is *identical* to SMPL-X's, so we select the 16 canonical joints with the **same**
:data:`SMPLX_TO_CANONICAL` tuple the SMPL-X FK seam uses — GT and a SMPL-X backend's prediction
then index the same anatomical joints, keeping the MPJPE a like-for-like comparison (in
particular ``spine`` = SMPL/SMPL-X joint 6 on both sides).

**Camera.** ``cam_poses`` are world→camera extrinsics; the camera moves, so the scene carries a
per-frame ``(T, 3, 3)`` rotation / ``(T, 3)`` translation (handled by :class:`SyntheticScene`'s
generalised maps). Condition B (foot-plane grounding) is *not* defined here — there is no fixed
``Z = 0`` pitch plane — so 3DPW is a **condition-A-only** benchmark.

Validation status (honest): the *parsing + projection geometry* below is unit-tested end-to-end
against a synthetic 3DPW-shaped pickle (``tests/unit/test_eval_3dpw.py``). What a real run must
still confirm is the *field semantics* of an actual 3DPW pickle — chiefly that ``cam_poses`` is
world→camera as assumed; :func:`diagnose_3dpw_scene` exposes the depth/in-frame fractions that
catch an inverted convention, and :func:`load_3dpw_sequence` hard-fails when GT projects behind
the camera. Run it for the first number once the data is downloaded (see
``docs/pose-bakeoff-runbook.md``).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..core.scene.camera import CameraIntrinsics
from .bodymodel import SMPLX_TO_CANONICAL, PlaceholderJointModel
from .dataset import PoseEvalScene

if TYPE_CHECKING:
    from .bodymodel import JointModel

#: SMPL 24-joint → 16 canonical. Identical to :data:`SMPLX_TO_CANONICAL` (shared body skeleton),
#: aliased for call-site clarity at 3DPW load sites.
SMPL24_TO_CANONICAL: tuple[int, ...] = SMPLX_TO_CANONICAL

_DEFAULT_W, _DEFAULT_H = 1920, 1080  # 3DPW imageFiles are 1920x1080


def _project_perframe(joints_world: np.ndarray, rot: np.ndarray, transl: np.ndarray,
                      k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame project world joints ``(T,N,J,3)`` → depth ``(T,N,J)`` + pixels ``(T,N,J,2)``."""
    cam = np.einsum("tij,tnkj->tnki", rot, joints_world) + transl[:, None, None, :]
    img = cam @ k.T
    return cam[..., 2], img[..., :2] / img[..., 2:3]


def load_3dpw_sequence(
    pkl_path: str | Path,
    images_dir: str | Path | None = None,
    *,
    width: int = _DEFAULT_W,
    height: int = _DEFAULT_H,
    joint_model: JointModel | None = None,
    box_margin_px: float = 40.0,
    stride: int = 1,
) -> PoseEvalScene:
    """Read a 3DPW ``sequenceFiles/*.pkl`` into a condition-A :class:`PoseEvalScene`.

    Keeps only frames valid for *every* actor (``campose_valid``), optionally subsampled by
    ``stride``. Person boxes are the GT joints' image extent padded by ``box_margin_px`` and
    clipped to frame (the crop the backend sees). ``images_dir`` becomes the scene's ``clip_uri``
    so a real backend decodes those frames; pass ``joint_model=SmplxJointModel()`` when scoring a
    SMPL-X backend so its params FK into the canonical joints (the asset-free default
    :class:`PlaceholderJointModel` is fine for loading + geometry checks).

    Raises ``ValueError`` if the GT projects behind the camera for most joints — the loud signal
    that ``cam_poses`` is not the assumed world→camera convention.
    """
    with open(pkl_path, "rb") as fh:
        seq = pickle.load(fh, encoding="latin1")  # 3DPW pickles are Python-2 (latin1)

    joints_pa = [np.asarray(a, dtype=float).reshape(-1, 24, 3) for a in seq["jointPositions"]]
    cam_poses = np.asarray(seq["cam_poses"], dtype=float)          # (T, 4, 4) world→camera
    k = np.asarray(seq["cam_intrinsics"], dtype=float)             # (3, 3)
    valid = np.logical_and.reduce(
        [np.asarray(v, dtype=bool) for v in seq["campose_valid"]]  # per-actor → all-actor AND
    )
    frames = np.nonzero(valid)[0][:: max(1, int(stride))]
    if frames.size == 0:
        raise ValueError(f"{pkl_path}: no frames valid for all actors")

    rot = cam_poses[frames, :3, :3]                               # (T, 3, 3)
    transl = cam_poses[frames, :3, 3]                            # (T, 3)
    sel = list(SMPL24_TO_CANONICAL)
    joints_world = np.stack(
        [a[frames][:, sel, :] for a in joints_pa], axis=1
    )                                                            # (T, N, 16, 3)
    if not np.isfinite(joints_world).all():
        raise ValueError(f"{pkl_path}: non-finite GT joints after frame filtering")

    depth, joints_image = _project_perframe(joints_world, rot, transl, k)
    if float((depth > 0).mean()) < 0.5:
        raise ValueError(
            f"{pkl_path}: GT projects behind the camera for most joints "
            f"(depth>0 only {float((depth > 0).mean()):.0%}) — cam_poses is likely NOT the assumed "
            "world→camera extrinsic; transpose/invert it before retrying."
        )

    u, v = joints_image[..., 0], joints_image[..., 1]
    boxes = np.stack(
        [u.min(-1) - box_margin_px, v.min(-1) - box_margin_px,
         u.max(-1) + box_margin_px, v.max(-1) + box_margin_px], axis=-1)   # (T, N, 4)
    boxes[..., 0::2] = boxes[..., 0::2].clip(0, width)
    boxes[..., 1::2] = boxes[..., 1::2].clip(0, height)

    jm = joint_model if joint_model is not None else PlaceholderJointModel()
    n_subjects = joints_world.shape[1]
    return PoseEvalScene(
        intrinsics=CameraIntrinsics(
            fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]),
            width=width, height=height,
        ),
        frames=frames,
        rotation=rot,
        translation=transl,
        joints_world=joints_world,
        joints_image=joints_image,
        boxes_xyxy=boxes,
        pelvis_height_m=0.0,  # unused: condition B (foot-plane grounding) is N/A for moving cam
        gt_global_orient=np.zeros((frames.size, n_subjects, 3)),       # no FK-oracle for 3DPW —
        gt_body_pose=np.zeros((frames.size, n_subjects, jm.n_pose_joints, 3)),  # geometry-checked
        gt_betas=np.zeros((n_subjects, 10)),                          # only (see module docstring)
        joint_model=jm,
        clip_uri=f"file://{Path(images_dir).resolve()}" if images_dir else "memory://3dpw",
        source_id=str(seq.get("sequence", Path(pkl_path).stem)),
        fps=30.0,  # 3DPW is 30 fps
    )


def diagnose_3dpw_scene(scene: PoseEvalScene) -> dict[str, float]:
    """Geometry sanity for a loaded 3DPW scene — the cheap convention check before a real run.

    Returns the fraction of GT joints that (a) sit in front of the camera and (b) project inside
    the image. Both near ``1.0`` mean ``cam_poses``/intrinsics were read correctly; a low value is
    the fingerprint of an inverted extrinsic or a wrong image size.
    """
    depth, _ = _project_perframe(
        scene.joints_world, scene.rotation, scene.translation, scene.intrinsics.matrix()
    )
    u, v = scene.joints_image[..., 0], scene.joints_image[..., 1]
    in_frame = (u >= 0) & (u < scene.intrinsics.width) & (v >= 0) & (v < scene.intrinsics.height)
    return {
        "depth_positive_fraction": float((depth > 0).mean()),
        "in_frame_fraction": float(in_frame.mean()),
    }
