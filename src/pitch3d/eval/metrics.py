"""MPJPE in metres — the bake-off metric, no Procrustes, no PA.

These are the *sanity / synthetic-path* metrics described in
``docs/pose-bakeoff-runbook.md`` §5. For real WorldPose headline numbers the
runbook defers to the challenge starter-kit's official evaluator (exact joint
correspondence, root choice and visibility masking live there); on our own
synthetic ground truth — where we generate the joints in a known order — these
two functions *are* authoritative.

Both accept any leading shape ``(..., J, 3)`` (e.g. ``(T, N, J, 3)`` or
``(T, J, 3)``) and return a scalar in metres.
"""

from __future__ import annotations

import numpy as np


def mpjpe_global(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-joint position error in **world** coordinates (placement + articulation)."""
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=-1).mean())


def mpjpe_local(pred: np.ndarray, gt: np.ndarray, root: int = 0) -> float:
    """Root-relative MPJPE (articulation only): subtract each frame's ``root`` joint first."""
    p = np.asarray(pred)
    g = np.asarray(gt)
    p = p - p[..., root : root + 1, :]
    g = g - g[..., root : root + 1, :]
    return float(np.linalg.norm(p - g, axis=-1).mean())
