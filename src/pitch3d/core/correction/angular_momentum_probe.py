"""Angular-momentum probe — trunk & limb angular activity should co-vary.

When a player twists their trunk sharply, their arms swing to conserve
angular momentum (in the absence of external torque). If body_pose shows
trunk rotation without limb reaction — or arm swing without trunk rotation
— the reconstruction lacks angular-momentum coherence (HMR-style
independent-joint noise).

Metric per subject: correlation between spine-joint angular velocity
and arm/leg-joint angular velocity. Low or negative correlation on
active frames = uncoordinated motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scene.scene import Scene
from .engine import resolve_subject_motion

#: SMPL-X body joints roughly grouped (indices in body_pose[T, 21, 3]).
JOINT_SPINE = (3, 6, 9)              # spine1/spine2/spine3
JOINT_LIMBS = (16, 17, 18, 19, 20)    # shoulders + arms + head end


@dataclass(frozen=True)
class AngularMomentumConfig:
    enabled: bool = False
    activity_threshold: float = 0.05    # rad/frame — ignore quiescent frames
    correlation_threshold: float = 0.2  # below this = uncoordinated


@dataclass
class SubjectAngMomReport:
    track_id: int
    n_frames: int = 0
    active_frames: int = 0
    correlation: float = 0.0
    is_uncoordinated: bool = False


@dataclass
class AngularMomentumReport:
    n_subjects: int = 0
    subjects_uncoordinated: int = 0
    subjects: list[SubjectAngMomReport] = field(default_factory=list)


def angular_momentum_probe(
    scene: Scene, cfg: AngularMomentumConfig | None = None,
    *, fps: float = 25.0,
) -> AngularMomentumReport:
    cfg = cfg if cfg is not None else AngularMomentumConfig()
    report = AngularMomentumReport(n_subjects=len(scene.subjects))
    if not cfg.enabled or fps <= 0:
        return report

    for s in scene.subjects:
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        body = np.asarray(resolved.pose.body_pose, dtype=float)
        n = body.shape[0]
        r = SubjectAngMomReport(track_id=int(s.track_id), n_frames=n)
        if n < 3 or body.ndim < 3 or body.shape[1] < 21:
            report.subjects.append(r)
            continue

        spine_active = np.linalg.norm(
            np.diff(body[:, list(JOINT_SPINE), :], axis=0), axis=(1, 2),
        )
        limbs_active = np.linalg.norm(
            np.diff(body[:, list(JOINT_LIMBS), :], axis=0), axis=(1, 2),
        )
        active = (spine_active > cfg.activity_threshold) | (
            limbs_active > cfg.activity_threshold
        )
        r.active_frames = int(active.sum())
        if r.active_frames < 3:
            report.subjects.append(r)
            continue

        s_a = spine_active[active]
        l_a = limbs_active[active]
        if s_a.std() < 1e-6 or l_a.std() < 1e-6:
            report.subjects.append(r)
            continue
        r.correlation = float(np.corrcoef(s_a, l_a)[0, 1])
        r.is_uncoordinated = r.correlation < cfg.correlation_threshold
        if r.is_uncoordinated:
            report.subjects_uncoordinated += 1
        report.subjects.append(r)

    return report


__all__ = [
    "AngularMomentumConfig",
    "AngularMomentumReport",
    "SubjectAngMomReport",
    "angular_momentum_probe",
    "JOINT_SPINE",
    "JOINT_LIMBS",
]
