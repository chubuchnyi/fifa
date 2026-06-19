"""FakePoseEstimator — rest-pose SMPL-X with roots anchored on the pitch.

Implements :class:`~pitch3d.core.ports.pose.PoseEstimator` without HMR: every subject gets
a rest pose whose root translation is the bbox foot point projected to world meters through
the field homography (FR-8's "root from homography", honestly, in core math). ``refit``
returns a deterministically nudged copy so the REFIT correction path has something real to
splice. No model, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitch3d.core.ports.io import ClipRef
from pitch3d.core.ports.perception import Tracks
from pitch3d.core.ports.pose import PoseEstimator
from pitch3d.core.scene.field import FieldCalibration
from pitch3d.core.scene.motion import (
    N_SMPLX_BODY_JOINTS,
    PoseSequence,
    SmplxShape,
    SubjectMotion,
)
from pitch3d.core.scene.provenance import Backend, ModelInfo

_PELVIS_HEIGHT_M = 0.92  # rough SMPL-X root height above the ground plane


@dataclass
class FakePoseEstimator(PoseEstimator):
    """Rest-pose proposals placed on the pitch; deterministic refit."""

    n_betas: int = 10

    def info(self) -> ModelInfo:
        return ModelInfo(name="FakePoseEstimator", backend=Backend.FAKE)

    def estimate(
        self,
        clip: ClipRef,
        tracks: Tracks,
        calibration: FieldCalibration,
    ) -> dict[int, SubjectMotion]:
        out: dict[int, SubjectMotion] = {}
        for tl in tracks.tracklets:
            if tl.cls == "ball":
                continue
            foot_uv = np.column_stack(
                [(tl.bboxes_xyxy[:, 0] + tl.bboxes_xyxy[:, 2]) / 2.0, tl.bboxes_xyxy[:, 3]]
            )
            world_xy = np.stack(
                [
                    calibration.image_to_world(int(f), foot_uv[i])[0]
                    for i, f in enumerate(tl.frames)
                ]
            )
            transl = np.column_stack([world_xy, np.full(tl.frames.shape[0], _PELVIS_HEIGHT_M)])
            pose = PoseSequence(
                frames=tl.frames,
                global_orient=np.zeros((tl.frames.shape[0], 3)),
                body_pose=np.zeros((tl.frames.shape[0], N_SMPLX_BODY_JOINTS, 3)),
                transl=transl,
            )
            out[tl.track_id] = SubjectMotion(shape=SmplxShape(betas=np.zeros(self.n_betas)), pose=pose)
        return out

    def refit(
        self,
        clip: ClipRef,
        motion: SubjectMotion,
        constraints: dict,
        frames: np.ndarray,
    ) -> SubjectMotion:
        refined = motion.copy()
        frames = np.asarray(frames, dtype=int).reshape(-1)
        rows = np.isin(refined.pose.frames, frames)
        # Deterministic "refinement": relax body pose toward rest on the selected frames.
        refined.pose.body_pose[rows] *= 0.5
        nudge = float(constraints.get("root_z_nudge", 0.0))
        if nudge:
            refined.pose.transl[rows, 2] += nudge
        return refined
