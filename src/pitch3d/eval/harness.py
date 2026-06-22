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
    from ..adapters.models.pose import HMRBackend
    from .synthetic import SyntheticScene


def place_under_gt_camera(scene: SyntheticScene, joints_cam: np.ndarray) -> np.ndarray:
    """Condition A: place camera-space joints ``(..., 3)`` into world via the GT camera."""
    return scene.camera_to_world(joints_cam)


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
    scene: SyntheticScene, backend: HMRBackend, root_joint: int = 0
) -> dict[str, float]:
    """Run an HMR backend over the synthetic scene → condition-A Global/Local MPJPE grid.

    The backend yields camera-space SMPL-X articulation per subject (the
    :class:`~pitch3d.adapters.models.pose.RawBodyMotion` contract); the harness runs the scene's
    FK to joints, places them at the GT root through the **GT camera** (condition A — isolates
    pose-net / articulation quality), and scores against GT. A backend returning the scene's GT
    params scores ~0; a zero-pose backend gives the finite Local-MPJPE sanity floor.

    Assumes the backend returns one :class:`RawBodyMotion` per tracklet whose ``frames`` match
    the scene's (true for the synthetic fixtures); real WorldPose alignment is the product's job.
    """
    clip, tracks = _clip_and_tracks(scene)
    bodies = backend.estimate_bodies(clip, tracks)
    jm = scene.joint_model
    pred = np.empty_like(scene.joints_world)
    for n, tl in enumerate(tracks.tracklets):
        raw = bodies[tl.track_id]
        fk_cam = jm.joints(raw.global_orient, raw.body_pose, raw.betas)   # (T, J, 3)
        root_cam = scene.world_to_camera(scene.root_world[:, n])          # (T, 3)
        pred[:, n] = scene.camera_to_world(root_cam[:, None, :] + fk_cam)
    return evaluate(pred, scene.joints_world, root_joint)


def evaluate(pred_world: np.ndarray, gt_world: np.ndarray, root_joint: int = 0) -> dict[str, float]:
    """Score predicted vs GT world joints → ``{global_mpjpe_m, local_mpjpe_m}`` (metres)."""
    return {
        "global_mpjpe_m": mpjpe_global(pred_world, gt_world),
        "local_mpjpe_m": mpjpe_local(pred_world, gt_world, root_joint),
    }
