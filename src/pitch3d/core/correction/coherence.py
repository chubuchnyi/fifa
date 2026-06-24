"""Temporal coherence: bridge short pose gaps + auto-smooth, honestly (FR-21/FR-22, R-6).

After continuity stitching (``orchestration.continuity``) a recovered identity may still
have *interior* frame gaps — the frames where the player was occluded and nobody was posed.
The renderer would simply blink the body out for those frames. This module closes that gap
in two honest, separable ways:

* **Gap fill is STRUCTURAL** (:func:`fill_pose_gaps`): the proposal is densified by
  interpolating the missing rows — linear for translation, *slerp* for every rotation
  (never axis-angle componentwise). Only gaps ``<= max_fill_gap`` are bridged; a long
  occlusion is left as a true gap rather than inventing a second of motion. Bridged frames
  are flagged with a low ``subject_frame_conf`` so the attention list surfaces them as
  inferred, not measured.
* **Smoothing is a CORRECTION** (:func:`coherence_corrections`): a normal, inspectable,
  disableable ``TEMPORAL_SMOOTHING`` correction (ADR-0002), zero-phase (centered window),
  layered on top — never baked into the proposal.

:func:`add_temporal_coherence` composes both over a whole :class:`Scene` and returns a new
scene (the input is never mutated) plus a :class:`CoherenceReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..scene.layers import ConfidenceMap, Correction, CorrectionTarget, TargetKind
from ..scene.motion import PoseSequence, SubjectMotion
from ..scene.scene import Scene
from .engine import interp_rotation, interp_vector, make_smoothing


@dataclass(frozen=True)
class CoherenceConfig:
    """Knobs for the gap-fill + auto-smooth pass."""

    max_fill_gap: int = 12               # bridge interior gaps up to this many missing frames
    smooth_window: int = 5               # centered (zero-phase) smoothing window, frames
    smooth_method: str = "moving_average"  # "moving_average" | "gaussian"
    smooth_sigma: float = 1.0            # for the gaussian kernel
    smooth_root_translation: bool = True  # smooth the world root path (main jitter source)
    smooth_root_orientation: bool = False  # off by default — can over-flatten fast turns
    filled_confidence: float = 0.3       # subject_frame_conf assigned to bridged frames
    real_confidence: float = 1.0         # subject_frame_conf for measured frames


@dataclass
class CoherenceReport:
    """What the pass did, for logging / inspection (R-6 transparency)."""

    filled_frames: int = 0       # total interior frames bridged across all subjects
    subjects_filled: int = 0     # how many subjects had at least one bridged gap
    corrections_added: int = 0   # auto smoothing corrections appended
    n_subjects: int = 0


def fill_pose_gaps(pose: PoseSequence, max_gap: int) -> tuple[PoseSequence, np.ndarray]:
    """Densify interior gaps ``<= max_gap`` in a pose; return the new pose + bridged frames.

    Measured rows are copied **verbatim**; only the inserted rows are interpolated (vectors
    linearly, rotations via slerp). Gaps larger than ``max_gap`` are left intact. The input
    pose is not mutated.
    """
    ef = np.asarray(pose.frames, dtype=int).reshape(-1)
    if ef.shape[0] <= 1:
        return pose.copy(), np.empty(0, dtype=int)

    filled: list[int] = []
    for a, b in zip(ef[:-1], ef[1:], strict=True):
        missing = int(b - a - 1)
        if 1 <= missing <= max_gap:
            filled.extend(range(int(a) + 1, int(b)))
    if not filled:
        return pose.copy(), np.empty(0, dtype=int)

    ff = np.asarray(sorted(filled), dtype=int)
    nf = np.sort(np.concatenate([ef, ff]))
    epos = np.searchsorted(nf, ef)
    fpos = np.searchsorted(nf, ff)
    t_new = nf.shape[0]

    def _vec(values: np.ndarray) -> np.ndarray:
        out = np.empty((t_new, values.shape[1]))
        out[epos] = values
        out[fpos] = interp_vector(ff, ef, values)
        return out

    def _rot(values: np.ndarray) -> np.ndarray:  # (T, 3) axis-angle
        out = np.empty((t_new, 3))
        out[epos] = values
        out[fpos] = interp_rotation(ff, ef, values)
        return out

    def _rot_joints(values: np.ndarray) -> np.ndarray:  # (T, K, 3)
        out = np.empty((t_new, values.shape[1], 3))
        out[epos] = values
        for j in range(values.shape[1]):
            out[fpos, j, :] = interp_rotation(ff, ef, values[:, j, :])
        return out

    new_pose = PoseSequence(
        frames=nf,
        global_orient=_rot(pose.global_orient),
        body_pose=_rot_joints(pose.body_pose),
        transl=_vec(pose.transl),
        left_hand_pose=None if pose.left_hand_pose is None else _rot_joints(pose.left_hand_pose),
        right_hand_pose=None if pose.right_hand_pose is None else _rot_joints(pose.right_hand_pose),
        jaw_pose=None if pose.jaw_pose is None else _rot(pose.jaw_pose),
    )
    return new_pose, ff


def fill_motion_gaps(motion: SubjectMotion, max_gap: int) -> tuple[SubjectMotion, np.ndarray]:
    """:func:`fill_pose_gaps` lifted to a :class:`SubjectMotion` (shape is gap-invariant)."""
    new_pose, filled = fill_pose_gaps(motion.pose, max_gap)
    return SubjectMotion(shape=motion.copy().shape, pose=new_pose), filled


def coherence_corrections(
    track_id: int,
    frame_range: tuple[int, int],
    cfg: CoherenceConfig,
    *,
    id_prefix: str = "auto-coh",
) -> list[Correction]:
    """Build the auto temporal-smoothing corrections for one subject (root path / orient)."""
    out: list[Correction] = []
    if cfg.smooth_root_translation:
        out.append(
            make_smoothing(
                f"{id_prefix}-transl-{track_id}",
                CorrectionTarget(TargetKind.ROOT_TRANSLATION, subject_track_id=track_id),
                frame_range,
                window=cfg.smooth_window,
                method=cfg.smooth_method,
                sigma=cfg.smooth_sigma,
                note="auto temporal coherence (root translation)",
            )
        )
    if cfg.smooth_root_orientation:
        out.append(
            make_smoothing(
                f"{id_prefix}-orient-{track_id}",
                CorrectionTarget(TargetKind.ROOT_ORIENTATION, subject_track_id=track_id),
                frame_range,
                window=cfg.smooth_window,
                method=cfg.smooth_method,
                sigma=cfg.smooth_sigma,
                note="auto temporal coherence (root orientation)",
            )
        )
    return out


def add_temporal_coherence(
    scene: Scene, cfg: CoherenceConfig | None = None
) -> tuple[Scene, CoherenceReport]:
    """Densify subject gaps + append auto-smoothing corrections; return a NEW scene + report.

    The input ``scene`` is never mutated. Bridged frames get a low ``subject_frame_conf`` so
    the attention list flags them as inferred (R-6). Smoothing is layered as corrections, so a
    later ``resolve_scene`` applies it non-destructively over the (now dense) proposal.
    """
    cfg = cfg or CoherenceConfig()
    base_conf = scene.confidence or ConfidenceMap()
    frame_conf = dict(base_conf.subject_frame_conf)
    new_subjects = []
    auto_corrs: list[Correction] = []
    report = CoherenceReport(n_subjects=len(scene.subjects))

    for s in scene.subjects:
        motion, filled = fill_motion_gaps(s.proposal, cfg.max_fill_gap)
        new_subjects.append(replace(s, proposal=motion))
        frames = motion.pose.frames
        conf = np.full(frames.shape[0], cfg.real_confidence, dtype=float)
        if filled.size:
            conf[np.isin(frames, filled)] = cfg.filled_confidence
            report.filled_frames += int(filled.size)
            report.subjects_filled += 1
        frame_conf[s.track_id] = conf
        corrs = coherence_corrections(s.track_id, (int(frames[0]), int(frames[-1])), cfg)
        auto_corrs.extend(corrs)
        report.corrections_added += len(corrs)

    new_scene = replace(
        scene,
        subjects=new_subjects,
        corrections=[*scene.corrections, *auto_corrs],
        confidence=replace(base_conf, subject_frame_conf=frame_conf),
    )
    return new_scene, report
