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


def _masked_mean(err: np.ndarray, mask: np.ndarray | None) -> float:
    """Mean of per-joint errors ``(..., J)``; over ``mask``-True joints only when a mask is given.

    Mirrors the official evaluator's **visibility masking**: occluded / out-of-frame joints are
    excluded from the average. Returns NaN if nothing is visible.
    """
    if mask is None:
        return float(err.mean())
    m = np.asarray(mask, dtype=bool)
    if m.shape != err.shape:
        raise ValueError(f"visibility mask {m.shape} does not match per-joint errors {err.shape}")
    visible = int(m.sum())
    if visible == 0:
        return float("nan")
    return float(err[m].sum() / visible)


def mpjpe_global(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean per-joint position error in **world** coordinates (placement + articulation).

    ``mask`` (``(..., J)`` bool, optional) scores visible joints only (occlusion-aware).
    """
    err = np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=-1)
    return _masked_mean(err, mask)


def mpjpe_local(
    pred: np.ndarray, gt: np.ndarray, root: int = 0, mask: np.ndarray | None = None
) -> float:
    """Root-relative MPJPE (articulation only): subtract each frame's ``root`` joint first.

    The root is subtracted from the real joint values regardless of its visibility; ``mask``
    (``(..., J)`` bool, optional) only restricts which joints enter the average.
    """
    p = np.asarray(pred)
    g = np.asarray(gt)
    p = p - p[..., root : root + 1, :]
    g = g - g[..., root : root + 1, :]
    err = np.linalg.norm(p - g, axis=-1)
    return _masked_mean(err, mask)
