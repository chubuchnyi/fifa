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

from .metrics import mpjpe_global, mpjpe_local

if TYPE_CHECKING:
    from .synthetic import SyntheticScene


def place_under_gt_camera(scene: SyntheticScene, joints_cam: np.ndarray) -> np.ndarray:
    """Condition A: place camera-space joints ``(..., 3)`` into world via the GT camera."""
    return scene.camera_to_world(joints_cam)


def evaluate(pred_world: np.ndarray, gt_world: np.ndarray, root_joint: int = 0) -> dict[str, float]:
    """Score predicted vs GT world joints → ``{global_mpjpe_m, local_mpjpe_m}`` (metres)."""
    return {
        "global_mpjpe_m": mpjpe_global(pred_world, gt_world),
        "local_mpjpe_m": mpjpe_local(pred_world, gt_world, root_joint),
    }
