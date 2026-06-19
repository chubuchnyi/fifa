"""Human Mesh Recovery port: SMPL-X estimation and constraint-guided re-fit.

Central to the whole tool (pose estimation is the emphasis of the TZ). Two methods:

* :meth:`estimate` — first-pass HMR producing proposal SMPL-X motion per subject
  (FR-8). Root translation is expected to be anchored by the field homography and
  foot-contact handled by the adapter (anti-foot-sliding).
* :meth:`refit` — constraint-guided re-fit on a frame subset (FR-22c). This is the
  ONLY model the correction engine calls, and it does so through this abstraction, so
  the engine stays pure. Default backend: PromptHMR-class.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from ..scene.field import FieldCalibration
from ..scene.motion import SubjectMotion
from .base import ModelProvider
from .io import ClipRef
from .perception import Tracks


class PoseEstimator(ModelProvider):
    """SMPL-X HMR with constraint-guided re-fit (FR-8, FR-22c).

    Default backends: GVHMR / WHAM / TRAM (gravity-view, SMPL-X), HumanMM for stitching.
    """

    @abstractmethod
    def estimate(
        self,
        clip: ClipRef,
        tracks: Tracks,
        calibration: FieldCalibration,
    ) -> dict[int, SubjectMotion]:
        """Estimate proposal SMPL-X motion per subject, keyed by ``track_id`` (FR-8).

        Root translation should be placed on the pitch via ``calibration`` (the world
        anchor); foot-ground contact should be respected to avoid foot-sliding.
        """
        raise NotImplementedError

    @abstractmethod
    def refit(
        self,
        clip: ClipRef,
        motion: SubjectMotion,
        constraints: dict,
        frames: np.ndarray,
    ) -> SubjectMotion:
        """Re-fit ``motion`` on ``frames`` under operator ``constraints`` (FR-22c).

        ``constraints`` is adapter-defined (2D keypoint hints, foot locks, …). Returns a
        new :class:`SubjectMotion`; the correction engine wraps the result as a REFIT
        correction so it remains non-destructive.
        """
        raise NotImplementedError
