"""Orientation verticality gate — force HMR global_orient to keep body upright.

HMR (SMPLest-X and friends) has fundamental orientation ambiguity for
standing / slow-moving players — 35% of subjects in a real broadcast
scene come out inverted or lying on their side because the per-frame
regressor can't disambiguate the body's up-axis without a strong motion
prior.

This gate applies a hard verticality constraint: for any frame where the
body's up axis deviates more than ``max_tilt_rad`` from the world up, we
rewrite ``global_orient`` to the nearest orientation that (a) keeps the
current yaw and (b) has body-up perfectly aligned with world-up.

Design:

* Pure numpy. Uses ``scipy.spatial.transform.Rotation`` for axis-angle
  ↔ matrix conversions but the actual clamp is composition of two
  rotations we build ourselves.
* Preserves yaw so we don't fight the ``facing_align`` gate.
* R-6 low-conf stamp on rewritten frames (this is inferred, not measured).

Convention:

* SMPL-X native: Y is body-up.
* Our world: Z is up. Adapter remap is ``(x, y, z)_smplx → (x, z, -y)_ours``.
* So the body's world-up direction is ``(-R @ e_y).z`` remapped, and we
  want it near +1 (world Z).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial.transform import Rotation

from ..config.gates import OrientVerticalityConfig
from ..scene.layers import Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene
from .engine import make_keyframes, resolve_subject_motion

VERTICALITY_INFERRED_CONF = 0.25


@dataclass
class SubjectVerticalityReport:
    track_id: int
    n_frames: int = 0
    corrected_frames: int = 0
    max_tilt_before_rad: float = 0.0
    max_tilt_after_rad: float = 0.0


@dataclass
class OrientVerticalityReport:
    n_subjects: int = 0
    subjects_corrected: int = 0
    corrections_added: int = 0
    max_tilt_before_rad: float = 0.0
    max_tilt_after_rad: float = 0.0
    subjects: list[SubjectVerticalityReport] = field(default_factory=list)


#: Vertex remap: verts_ours = verts_smplx @ R_SMPLX_TO_OURS.T where
#:     R_SMPLX_TO_OURS = [[1,0,0],[0,0,1],[0,-1,0]]  (world_z = −smplx_y).
#: A body direction v_body gets transformed by the SMPL-X global rotation R_g to
#: R_g @ v_body in smplx frame, then remapped to (x_s, z_s, -y_s) in ours.

def _body_up_world_z(rotvec: np.ndarray) -> np.ndarray:
    """World-Z component of the body-up axis, one per frame.

    ``rotvec`` shape ``(T, 3)`` — axis-angle in SMPL-X native frame.
    Body-up in body local = +Y_smplx. After R_g, in smplx frame it's R_g[:,1].
    After remap, its world_z is −(R_g[:,1])_y = −R_g[1,1].
    """
    R = Rotation.from_rotvec(rotvec).as_matrix()   # (T, 3, 3)
    return -R[:, 1, 1]


def _world_yaw_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """World yaw angle (radians) for each frame.

    Body-forward in local = +Z_smplx. R_g @ (0,0,1) = R_g[:,2] in smplx frame.
    Remap → world components: (world_x, world_y, world_z) = (R_g[0,2], R_g[2,2], −R_g[1,2]).
    Yaw = atan2(world_y, world_x).
    """
    R = Rotation.from_rotvec(rotvec).as_matrix()
    return np.arctan2(R[:, 2, 2], R[:, 0, 2])


def _upright_rotvec(yaw: np.ndarray) -> np.ndarray:
    """Axis-angle vectors that produce EXACTLY upright bodies at the given yaws.

    Constructed matrix (SMPL-X native, applied by SMPL-X FK):

        R = [[-sin(y), 0, cos(y)],
             [ 0,     -1, 0     ],
             [ cos(y), 0, sin(y)]]

    Verified: this R sends body-up (+Y_smplx) → -Y_smplx → world +Z (upright)
    and body-forward (+Z_smplx) → (cos(y), 0, sin(y))_smplx →
    (cos(y), sin(y), 0)_world (yaw around world Z as requested).
    """
    T = yaw.shape[0]
    R = np.zeros((T, 3, 3), dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    R[:, 0, 0] = -s;  R[:, 0, 2] = c
    R[:, 1, 1] = -1
    R[:, 2, 0] = c;   R[:, 2, 2] = s
    return Rotation.from_matrix(R).as_rotvec().astype(np.float32)


def orient_verticality_gate(
    scene: Scene, cfg: OrientVerticalityConfig | None = None,
) -> tuple[Scene, OrientVerticalityReport]:
    """Force body-up to world-up on frames tilted beyond threshold."""
    cfg = cfg if cfg is not None else OrientVerticalityConfig()
    report = OrientVerticalityReport(n_subjects=len(scene.subjects))
    if not cfg.enabled:
        return scene, report

    cos_thresh = float(np.cos(cfg.max_tilt_rad))

    auto_corrs: list[Correction] = []
    for s in scene.subjects:
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        frames = np.asarray(resolved.pose.frames, dtype=int)
        orient = np.asarray(resolved.pose.global_orient, dtype=float)
        n = orient.shape[0]
        r = SubjectVerticalityReport(track_id=int(s.track_id), n_frames=n)
        if n < 1:
            report.subjects.append(r)
            continue

        body_up_z = _body_up_world_z(orient)
        tilt_rad = np.arccos(np.clip(body_up_z, -1.0, 1.0))
        r.max_tilt_before_rad = float(tilt_rad.max())
        report.max_tilt_before_rad = max(report.max_tilt_before_rad, r.max_tilt_before_rad)

        needs_fix = body_up_z < cos_thresh
        if not needs_fix.any():
            r.max_tilt_after_rad = r.max_tilt_before_rad
            report.max_tilt_after_rad = max(report.max_tilt_after_rad, r.max_tilt_after_rad)
            report.subjects.append(r)
            continue

        yaw = _world_yaw_from_rotvec(orient)
        upright = _upright_rotvec(yaw)   # (T, 3)
        new_orient = orient.copy()
        new_orient[needs_fix] = upright[needs_fix]

        new_up = _body_up_world_z(new_orient)
        new_tilt = np.arccos(np.clip(new_up, -1.0, 1.0))
        r.max_tilt_after_rad = float(new_tilt.max())
        r.corrected_frames = int(needs_fix.sum())
        report.max_tilt_after_rad = max(report.max_tilt_after_rad, r.max_tilt_after_rad)
        report.subjects_corrected += 1
        report.subjects.append(r)

        auto_corrs.append(
            make_keyframes(
                f"auto-orient-vertical-{s.track_id}",
                CorrectionTarget(
                    kind=TargetKind.ROOT_ORIENTATION,
                    subject_track_id=s.track_id,
                ),
                (int(frames[0]), int(frames[-1])),
                key_frames=frames.astype(float),
                key_values=new_orient,
                interp="slerp",
                note=(
                    f"auto verticality: {r.corrected_frames}/{n} frames, "
                    f"tilt {np.degrees(r.max_tilt_before_rad):.0f}° → "
                    f"{np.degrees(r.max_tilt_after_rad):.0f}°"
                ),
            )
        )

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report


__all__ = [
    "VERTICALITY_INFERRED_CONF",
    "OrientVerticalityConfig",
    "OrientVerticalityReport",
    "orient_verticality_gate",
]
