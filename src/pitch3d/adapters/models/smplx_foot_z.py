"""SMPL-X FK-based pelvis-above-foot provider for :mod:`.foot_plant` T6a v2.

The foot_plant gate ships with a constant target of 0.92 m as the nominal
pelvis-above-foot; this adapter replaces that constant with a **measured**
per-subject offset via SMPL-X forward kinematics on the subject's
betas + pose. Different players have different standing heights (betas)
and different crouch profiles at any given frame (pose), so a shared
0.92 m systematically hovers or sinks entire subjects.

Approach: run SMPL-X FK per frame with ``transl=0`` (pelvis anchored at
origin), take ``-min(verts.y)`` (SMPL-X native y-up; the lowest vertex is
the foot in a standing/running pose). Median across frames gives the
robust per-subject standing offset; foot_plant's median_lock then centres
the track on that value while preserving stride variance.

Requires the SMPL-X ``.npz`` model; located via
:func:`pitch3d.adapters.models.smplx_lbs.locate_smplx_model`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.correction.foot_plant import PelvisTargetProvider
from ...core.scene.scene import Subject
from .smplx_lbs import SmplxModel, locate_smplx_model


@dataclass(frozen=True)
class SmplxFootZConfig:
    """Knobs for :func:`make_smplx_foot_z_provider`."""

    max_frames_sampled: int = 20        # cap FK cost per subject
    fallback_when_unavailable: bool = True  # None on load failure so gate falls back to cfg


def _lowest_vertex_offset(
    model: SmplxModel, betas: np.ndarray,
    global_orient: np.ndarray, body_pose: np.ndarray,
) -> float:
    """Height of the pelvis above the lowest posed vertex (metres), pelvis at origin.

    SMPL-X native frame is y-up; the pelvis is at origin when ``transl=0``.
    The lowest vertex is typically a foot in a standing/running pose, so
    ``pelvis_above_foot = 0 - min(y) = -min(y)``.
    """
    verts = model.pose(
        betas=betas,
        global_orient=np.asarray(global_orient, dtype=float).reshape(3),
        body_pose=np.asarray(body_pose, dtype=float).reshape(-1, 3),
        transl=None,   # pelvis-at-origin
    )
    return float(-verts[:, 1].min())


def make_smplx_foot_z_provider(
    model_path: str | None = None,
    cfg: SmplxFootZConfig | None = None,
) -> PelvisTargetProvider | None:
    """Build a :data:`PelvisTargetProvider` that measures per-subject offsets via SMPL-X FK.

    * ``model_path`` — if ``None``, resolved via
      :func:`locate_smplx_model` (env vars ``PITCH3D_SMPLX_MODEL`` /
      ``PITCH3D_SMPLX_MODELS`` or a repo-local ``SMPL-X/models`` fallback).
    * Returns ``None`` when the model is unavailable — the gate then falls
      back to its shared ``cfg.target_pelvis_m`` per T6a defaults.

    The returned callable computes ``pelvis_above_foot`` on up to
    ``cfg.max_frames_sampled`` evenly-spaced frames per subject, returning a
    ``(k,)`` array. foot_plant_gate takes the median so a bad frame doesn't
    move the whole target.
    """
    cfg = cfg or SmplxFootZConfig()
    path = model_path or locate_smplx_model()
    if path is None:
        return None
    try:
        model = SmplxModel.load(path)
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return None

    def provider(subject: Subject) -> np.ndarray | None:
        motion = subject.proposal
        if motion is None:
            return None
        pose = motion.pose
        frames = np.asarray(pose.frames, dtype=int).reshape(-1)
        if frames.shape[0] == 0:
            return None
        betas = np.asarray(motion.shape.betas, dtype=float).reshape(-1)
        n = frames.shape[0]
        k = min(cfg.max_frames_sampled, n)
        sampled_rows = np.unique(
            np.linspace(0, n - 1, k).round().astype(int)
        )
        offsets = np.zeros(sampled_rows.shape[0], dtype=float)
        for i, row in enumerate(sampled_rows):
            try:
                offsets[i] = _lowest_vertex_offset(
                    model, betas,
                    pose.global_orient[row],
                    pose.body_pose[row],
                )
            except (ValueError, IndexError):
                offsets[i] = np.nan
        offsets = offsets[np.isfinite(offsets)]
        if offsets.size == 0:
            return None
        return offsets

    return provider


__all__ = [
    "SmplxFootZConfig",
    "make_smplx_foot_z_provider",
]
