"""Bake-off harness — turn predicted joints into a Global/Local MPJPE grid.

This is the OUR-side glue of ``docs/pose-bakeoff-runbook.md`` §2–3. A pose backend
yields camera-space joints per subject; the harness places them in world coordinates
and scores them against ground truth, **per camera condition**:

* **Condition A — GT camera** (:func:`place_under_gt_camera`): use the scene's true
  ``R, t``. Isolates pose-net / articulation quality. Implemented here (pure).
* **Condition B — our calibration** (PnLCalib + foot-plane anchor): estimate the camera
  from pitch lines, then place. This is the *product* number and the A→B gap is the cost
  our calibration adds. It needs the heavy keypoint backend, so it is wired on the box —
  this module intentionally stops at the seam (the GT homography from
  :meth:`SyntheticScene.field_calibration` is the perfect-calibration stand-in until then).

For real WorldPose, :func:`evaluate` is a *sanity cross-check* only — the runbook defers
headline numbers to the challenge's official evaluator. On synthetic GT it is authoritative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.ports.io import ClipRef
from ..core.ports.perception import Tracklet, Tracks
from .metrics import mpjpe_global, mpjpe_local

if TYPE_CHECKING:
    from ..adapters.models.pose import HMRBackend, RawBodyMotion
    from ..core.scene.field import FieldCalibration
    from .synthetic import SyntheticScene


def place_under_gt_camera(scene: SyntheticScene, joints_cam: np.ndarray) -> np.ndarray:
    """Condition A: place camera-space joints ``(..., 3)`` into world via the GT camera."""
    return scene.camera_to_world(joints_cam)


def _place_subject(
    scene: SyntheticScene, raw: RawBodyMotion, root_world: np.ndarray
) -> np.ndarray:
    """FK one subject's articulation and seat it at ``root_world (T,3)`` under the GT camera.

    Shared by both conditions; they differ only in *where the root comes from* (GT 3D root vs a
    homography-grounded foot point), so the A→B gap is exactly the grounding/calibration cost.
    """
    fk_cam = scene.joint_model.joints(raw.global_orient, raw.body_pose, raw.betas)  # (T, J, 3)
    root_cam = scene.world_to_camera(root_world)                                    # (T, 3)
    return scene.camera_to_world(root_cam[:, None, :] + fk_cam)


def _ground_root_from_feet(
    tl: Tracklet, calibration: FieldCalibration, pelvis_height_m: float
) -> np.ndarray:
    """Root world position from the bbox foot point via the calibration homography (FR-8).

    Mirrors :meth:`GVHMRPoseEstimator._ground_root` — bbox bottom-centre → world plane (XY), with
    Z pinned at the nominal pelvis height — so condition B exercises the product's real grounding.
    """
    foot_uv = np.column_stack(
        [(tl.bboxes_xyxy[:, 0] + tl.bboxes_xyxy[:, 2]) / 2.0, tl.bboxes_xyxy[:, 3]]
    )
    world_xy = np.stack(
        [calibration.image_to_world(int(f), foot_uv[i])[0] for i, f in enumerate(tl.frames)]
    )
    return np.column_stack([world_xy, np.full(tl.frames.shape[0], pelvis_height_m)])


def _clip_and_tracks(scene: SyntheticScene) -> tuple[ClipRef, Tracks]:
    """Wrap the synthetic GT as the ``ClipRef`` + ``Tracks`` an ``HMRBackend`` consumes.

    Each subject ``n`` becomes a player tracklet with ``track_id = n`` carrying the GT boxes,
    so the backend's per-crop contract is exercised exactly as in the product.
    """
    clip = ClipRef(
        source_id="synthetic",
        uri="memory://synthetic",
        frames=scene.frames,
        width=scene.intrinsics.width,
        height=scene.intrinsics.height,
        fps=25.0,
    )
    tracklets = [
        Tracklet(
            track_id=n,
            frames=scene.frames,
            bboxes_xyxy=scene.boxes_xyxy[:, n],
            cls="player",
        )
        for n in range(scene.n_subjects)
    ]
    return clip, Tracks(tracklets=tracklets)


def run_backend(
    scene: SyntheticScene, backend: HMRBackend, root_joint: int = 0, visible_only: bool = False
) -> dict[str, float]:
    """Run an HMR backend over the synthetic scene → condition-A Global/Local MPJPE grid.

    The backend yields camera-space SMPL-X articulation per subject (the
    :class:`~pitch3d.adapters.models.pose.RawBodyMotion` contract); the harness runs the scene's
    FK to joints, places them at the GT root through the **GT camera** (condition A — isolates
    pose-net / articulation quality), and scores against GT. A backend returning the scene's GT
    params scores ~0; a zero-pose backend gives the finite Local-MPJPE sanity floor.

    ``visible_only`` scores only joints the scene marks visible (:attr:`SyntheticScene.visibility`),
    mirroring the official evaluator's occlusion masking.

    Assumes the backend returns one :class:`RawBodyMotion` per tracklet whose ``frames`` match
    the scene's (true for the synthetic fixtures); real WorldPose alignment is the product's job.
    """
    clip, tracks = _clip_and_tracks(scene)
    bodies = backend.estimate_bodies(clip, tracks)
    pred = np.empty_like(scene.joints_world)
    for n, tl in enumerate(tracks.tracklets):
        pred[:, n] = _place_subject(scene, bodies[tl.track_id], scene.root_world[:, n])
    mask = scene.visibility if visible_only else None
    return evaluate(pred, scene.joints_world, root_joint, mask)


def run_backend_grounded(
    scene: SyntheticScene,
    backend: HMRBackend,
    calibration: FieldCalibration,
    root_joint: int = 0,
    pelvis_height_m: float | None = None,
    visible_only: bool = False,
) -> dict[str, float]:
    """Condition B: place each subject at a root **grounded from its bbox foot point** through
    ``calibration`` (homography), instead of the GT 3D root — the product's real grounding path.

    Holds the GT camera *rotation* for the camera→world articulation map (synthetic does not run
    PnLCalib), so this isolates the grounding/translation cost: the **A→B gap is Global-MPJPE
    only** (Local is root-relative and so is identical to condition A). Feed
    :meth:`SyntheticScene.field_calibration` (perfect GT homography) for the methodology floor, or
    a perturbed homography to model real calibration error. ``pelvis_height_m`` defaults to the
    scene's grounded height; ``visible_only`` masks occluded joints as in :func:`run_backend`.
    """
    clip, tracks = _clip_and_tracks(scene)
    bodies = backend.estimate_bodies(clip, tracks)
    ph = scene.pelvis_height_m if pelvis_height_m is None else float(pelvis_height_m)
    pred = np.empty_like(scene.joints_world)
    for n, tl in enumerate(tracks.tracklets):
        root_world = _ground_root_from_feet(tl, calibration, ph)
        pred[:, n] = _place_subject(scene, bodies[tl.track_id], root_world)
    mask = scene.visibility if visible_only else None
    return evaluate(pred, scene.joints_world, root_joint, mask)


def run_conditions(
    scene: SyntheticScene,
    backend: HMRBackend,
    calibration: FieldCalibration | None = None,
    root_joint: int = 0,
    visible_only: bool = False,
) -> dict[str, dict[str, float] | None]:
    """Both bake-off conditions for one backend → ``{'A': grid, 'B': grid | None}``.

    ``A`` is the GT camera (pose-net only); ``B`` grounds via ``calibration`` (the product number).
    ``B`` is ``None`` when no calibration is given — the GT homography
    (:meth:`SyntheticScene.field_calibration`) is the perfect-calibration stand-in on synthetic
    until PnLCalib runs on the box. ``visible_only`` masks occluded joints in both grids
    (:attr:`SyntheticScene.visibility`), mirroring the official evaluator's occlusion handling.
    """
    return {
        "A": run_backend(scene, backend, root_joint, visible_only=visible_only),
        "B": (
            run_backend_grounded(scene, backend, calibration, root_joint, visible_only=visible_only)
            if calibration is not None
            else None
        ),
    }


def evaluate(
    pred_world: np.ndarray,
    gt_world: np.ndarray,
    root_joint: int = 0,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Score predicted vs GT world joints → ``{global_mpjpe_m, local_mpjpe_m}`` (metres).

    ``mask`` (``(..., J)`` bool, optional) restricts scoring to visible joints — feed
    :attr:`SyntheticScene.visibility` to mirror the official evaluator's occlusion masking.
    """
    return {
        "global_mpjpe_m": mpjpe_global(pred_world, gt_world, mask),
        "local_mpjpe_m": mpjpe_local(pred_world, gt_world, root_joint, mask),
    }
