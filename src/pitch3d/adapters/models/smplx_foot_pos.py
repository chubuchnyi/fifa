"""SMPL-X FK foot-position provider for :mod:`.contact_probe`.

Sister adapter to :mod:`.smplx_foot_z`. Where that one returns a scalar
per-subject pelvis offset (used by ``foot_plant_gate`` for median-lock),
this one returns a full ``(T, 3)`` world-space foot-position per frame,
which the ``contact_probe`` needs to detect contact frames and measure
foot slide.

The "foot" is the lowest posed vertex; positions include the subject's
world root translation, so a resolved motion (proposal ⊕ corrections)
yields foot XY in the pitch's own metric frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.correction.contact_probe import FootPositionProvider
from ...core.scene.frames import SourceFrame, detect_source_frame, smplx_to_world
from ...core.scene.scene import Subject
from .smplx_lbs import SmplxModel, locate_smplx_model


@dataclass(frozen=True)
class SmplxFootPosConfig:
    """Knobs for :func:`make_smplx_foot_position_provider`."""

    #: Cap FK cost per subject. Set generously — the held-between-samples
    #: interpolation creates fake stance detections when downsampled
    #: (contact_probe reads constant XY across held rows as a planted foot).
    max_frames_sampled: int = 240
    #: Override the SMPL-X source frame; ``None`` detects it per subject.
    source_frame: SourceFrame | None = None


def make_smplx_foot_position_provider(
    model_path: str | None = None,
    cfg: SmplxFootPosConfig | None = None,
) -> FootPositionProvider | None:
    """Return a :data:`FootPositionProvider` that samples SMPL-X foot pos per frame.

    Returns ``None`` when the SMPL-X model isn't available (same fallback
    contract as :mod:`.smplx_foot_z`).
    """
    cfg = cfg or SmplxFootPosConfig()
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
        n = frames.shape[0]
        if n == 0:
            return None
        betas = np.asarray(motion.shape.betas, dtype=float).reshape(-1)
        transl = np.asarray(pose.transl, dtype=float)
        # `transl` is the world pelvis, so re-origin on the pelvis joint before rotating.
        pelvis = (model.j_regressor @ model.shaped(betas))[0]
        k = min(cfg.max_frames_sampled, n)
        sampled = np.unique(np.linspace(0, n - 1, k).round().astype(int))
        # dense output — hold last known between sampled frames
        out = np.zeros((n, 3), dtype=float)
        last: np.ndarray | None = None
        for i in range(n):
            if i in sampled:
                try:
                    verts = model.pose(
                        betas=betas,
                        global_orient=pose.global_orient[i],
                        body_pose=pose.body_pose[i],
                        transl=None,   # world root added after the frame remap
                    )
                    frame = cfg.source_frame or detect_source_frame(verts, pelvis)
                    world = smplx_to_world(
                        verts, pelvis=pelvis, frame=frame, transl=transl[i]
                    )
                    foot_world = world[int(np.argmin(world[:, 2]))]
                    out[i] = foot_world
                    last = foot_world
                except (ValueError, IndexError):
                    if last is not None:
                        out[i] = last
            elif last is not None:
                out[i] = last
        return out

    return provider


__all__ = [
    "SmplxFootPosConfig",
    "make_smplx_foot_position_provider",
]
