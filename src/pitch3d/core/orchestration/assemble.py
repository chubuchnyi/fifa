"""Assemble a proposal :class:`Scene` and resolve it — pure composition, no adapter.

``assemble_scene`` folds the reconstruction stage outputs (tracks, calibration, per-subject
motion, ball) into the canonical :class:`Scene` (proposal layer + confidence). ``resolve_scene``
returns a fully *resolved* copy (proposal ⊕ corrections, ADR-0002) so render/observe consume a
single source of truth and never touch the correction stack themselves. Both are pure core
functions, unit-testable without any model/renderer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ..correction.engine import resolve_ball, resolve_subject_motion
from ..scene.field import FieldModel
from ..scene.layers import ConfidenceMap
from ..scene.scene import Scene
from ..scene.subject import Role, Subject, Team
from .pipeline import ReconstructionResult

if TYPE_CHECKING:
    from ..ports.io import ClipRef
    from ..ports.motion_prior import MotionPrior
    from ..ports.pose import PoseEstimator

_CLS_TO_ROLE = {
    "player": Role.PLAYER,
    "goalkeeper": Role.GOALKEEPER,
    "referee": Role.REFEREE,
}


def assemble_scene(
    result: ReconstructionResult,
    *,
    scene_id: str,
    episode_id: str,
    source_id: str,
    camera=None,
    teams: list[Team] | None = None,
) -> Scene:
    """Fold reconstruction outputs into a proposal :class:`Scene` (FR-5..9)."""
    meta = {tl.track_id: tl for tl in result.tracks.tracklets}
    subjects: list[Subject] = []
    for track_id, motion in result.motions.items():
        tl = meta.get(track_id)
        role = _CLS_TO_ROLE.get(tl.cls if tl else "player", Role.PLAYER)
        subjects.append(
            Subject(
                track_id=track_id,
                proposal=motion,
                role=role,
                team_id=tl.team_id if tl else None,
            )
        )
    subjects.sort(key=lambda s: s.track_id)

    confidence = ConfidenceMap(field_homography_conf=result.calibration.confidence)

    return Scene(
        id=scene_id,
        episode_id=episode_id,
        source_id=source_id,
        field=FieldModel(calibration=result.calibration),
        camera=camera,
        subjects=subjects,
        teams=teams if teams is not None else list(result.tracks.teams),
        ball=result.ball_3d,
        confidence=confidence,
    )


def resolve_scene(
    scene: Scene,
    *,
    refit_port: PoseEstimator | None = None,
    clip: ClipRef | None = None,
    motion_prior: MotionPrior | None = None,
) -> Scene:
    """Return a copy of ``scene`` with every subject/ball resolved and the stack baked empty.

    REFIT corrections call ``refit_port`` (the :class:`PoseEstimator`) over ``clip``; learned
    TEMPORAL_SMOOTHING corrections call ``motion_prior`` (the :class:`MotionPrior`). Pass them when
    the scene may contain those corrections. The input scene is never mutated.
    """
    resolved_subjects = [
        replace(
            s,
            proposal=resolve_subject_motion(
                s.proposal, scene.corrections_for(s.track_id),
                refit_port=refit_port, clip=clip, motion_prior=motion_prior,
            ),
        )
        for s in scene.subjects
    ]
    resolved_ball = (
        resolve_ball(scene.ball, scene.corrections_for(None)) if scene.ball is not None else None
    )
    return replace(scene, subjects=resolved_subjects, ball=resolved_ball, corrections=[])
