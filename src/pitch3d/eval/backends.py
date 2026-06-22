"""Synthetic-only HMR backends for the bake-off: the GT oracle and the zero-pose floor.

These are **not** perception nets — they fabricate ``RawBodyMotion`` straight from a
:class:`~pitch3d.eval.synthetic.SyntheticScene` (its GT articulation, or zeros) so the
harness/driver can be exercised end to end with no GPU and no frames. Real candidates (SMPLest-X,
SAM 3D Body) implement the same ``HMRBackend`` seam and drop into the same grid on the box; see
``docs/pose-bakeoff-runbook.md``.

Kept out of :mod:`pitch3d.eval`'s ``__init__`` so importing the eval package stays light (these are
the only pieces that pull in the adapters layer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..adapters.models.pose import RawBodyMotion

if TYPE_CHECKING:
    from ..core.ports.io import ClipRef
    from ..core.ports.perception import Tracks
    from .synthetic import SyntheticScene


class GtOracleBackend:
    """Oracle backend: replays the scene's own GT articulation per tracklet.

    Reconstructs the GT exactly under condition A (Global/Local MPJPE ~0) — the methodology
    self-check that the harness placement/scoring is faithful.
    """

    def __init__(self, scene: SyntheticScene) -> None:
        self.scene = scene

    def estimate_bodies(self, clip: ClipRef, tracks: Tracks) -> dict[int, RawBodyMotion]:
        s = self.scene
        return {
            tl.track_id: RawBodyMotion(
                track_id=tl.track_id,
                frames=s.frames,
                global_orient=s.gt_global_orient[:, n],
                body_pose=s.gt_body_pose[:, n],
                betas=s.gt_betas[n],
            )
            for n, tl in enumerate(tracks.tracklets)
        }


class ZeroPoseBackend:
    """Floor backend: zero articulation (T-pose) at the grounded root.

    The finite Local-MPJPE sanity baseline (runbook §3). Sizes ``body_pose`` to the scene FK's
    ``n_pose_joints`` so it is valid for both the placeholder (16) and SMPL-X (21) backends.
    """

    def __init__(self, scene: SyntheticScene) -> None:
        self.scene = scene

    def estimate_bodies(self, clip: ClipRef, tracks: Tracks) -> dict[int, RawBodyMotion]:
        s = self.scene
        p = s.joint_model.n_pose_joints
        return {
            tl.track_id: RawBodyMotion(
                track_id=tl.track_id,
                frames=s.frames,
                global_orient=np.zeros((s.n_frames, 3)),
                body_pose=np.zeros((s.n_frames, p, 3)),
                betas=np.zeros(10),
            )
            for tl in tracks.tracklets
        }
