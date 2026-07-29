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
* **Edge extension is STRUCTURAL** (:func:`extend_pose_to_span`): a subject the tracker
  acquired late or lost early is still physically on the pitch, so instead of letting the
  renderer blink it out at the clip edges we extend it to the full clip span — posture held,
  root coasting with a decaying edge velocity ("running keeps running, then eases; standing
  stays standing"). Extrapolated frames get an even lower ``subject_frame_conf``.
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
from ..scene.motion import PoseSequence, Provenance, SubjectMotion
from ..scene.scene import Scene
from .engine import interp_rotation, interp_vector, make_smoothing
from .kinematics import HUMAN_MAX_SPEED


@dataclass(frozen=True)
class CoherenceConfig:
    """Knobs for the gap-fill + edge-extend + auto-smooth pass."""

    max_fill_gap: int = 12               # bridge interior gaps up to this many missing frames
    smooth_window: int = 5               # centered (zero-phase) smoothing window, frames
    smooth_method: str = "moving_average"  # "moving_average" | "gaussian"
    smooth_sigma: float = 1.0            # for the gaussian kernel
    smooth_root_translation: bool = True  # smooth the world root path (main jitter source)
    smooth_root_orientation: bool = False  # off by default — can over-flatten fast turns
    filled_confidence: float = 0.3       # subject_frame_conf assigned to bridged frames
    real_confidence: float = 1.0         # subject_frame_conf for measured frames
    # Edge extension: a tracker-lost player is still physically present, so rather than let the
    # renderer blink it out at the clip edges we extend each subject to the full clip span —
    # hold its posture, coast its root with a decaying velocity ("running keeps running, then
    # eases; standing stays standing"). Bridged interior frames are interpolated as before.
    extend_to_span: bool = True          # extend every subject to cover the whole clip span
    extrapolate_decay: float = 0.9       # per-frame geometric velocity decay at the edges
    extrapolated_confidence: float = 0.2  # subject_frame_conf for extrapolated edge frames
    extrapolate_velocity_window: int = 3  # frames used to estimate the edge velocity
    # A dying track often slides off the body BEFORE the tracker loses it, so the measured
    # edge velocity can be garbage (#207: 43 m/s inherited by the coast → a ghost slid 10.9m).
    # Coasting is inference, not measurement — inferring at inhuman speed is fabrication (R-6),
    # so the coast velocity is capped at the shared human sprint ceiling.
    coast_max_speed: float = HUMAN_MAX_SPEED  # m/s cap on the extrapolation edge velocity


@dataclass
class CoherenceReport:
    """What the pass did, for logging / inspection (R-6 transparency)."""

    filled_frames: int = 0       # total interior frames bridged across all subjects
    subjects_filled: int = 0     # how many subjects had at least one bridged gap
    extended_frames: int = 0     # total edge frames extrapolated across all subjects
    subjects_extended: int = 0   # how many subjects were extended to the clip span
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

    prov = np.empty(t_new, dtype=pose.provenance.dtype)
    prov[epos] = pose.provenance
    prov[fpos] = Provenance.INTERPOLATED.value

    new_pose = PoseSequence(
        frames=nf,
        global_orient=_rot(pose.global_orient),
        body_pose=_rot_joints(pose.body_pose),
        transl=_vec(pose.transl),
        left_hand_pose=None if pose.left_hand_pose is None else _rot_joints(pose.left_hand_pose),
        right_hand_pose=None if pose.right_hand_pose is None else _rot_joints(pose.right_hand_pose),
        jaw_pose=None if pose.jaw_pose is None else _rot(pose.jaw_pose),
        provenance=prov,
    )
    return new_pose, ff


def fill_motion_gaps(motion: SubjectMotion, max_gap: int) -> tuple[SubjectMotion, np.ndarray]:
    """:func:`fill_pose_gaps` lifted to a :class:`SubjectMotion` (shape is gap-invariant)."""
    new_pose, filled = fill_pose_gaps(motion.pose, max_gap)
    return SubjectMotion(shape=motion.copy().shape, pose=new_pose), filled


def _geom_steps(k: np.ndarray, decay: float) -> np.ndarray:
    """Cumulative decayed displacement after ``k`` steps: ``sum_{i=1..k} decay**(i-1)``.

    ``decay == 1`` → ``k`` (constant velocity, never eases); ``decay < 1`` saturates at
    ``1/(1-decay)`` so an extrapolated runner coasts a *bounded* distance and eases to a stop
    instead of sliding away forever.
    """
    k = np.asarray(k, dtype=float)
    if abs(decay - 1.0) < 1e-12:
        return k
    return (1.0 - np.power(decay, k)) / (1.0 - decay)


def _edge_velocity(
    frames: np.ndarray, values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame leading/trailing velocity of ``values`` over a short edge ``window``.

    Returns ``(v_lead, v_trail)``, each shape ``(D,)``. A subject seen only once (``n < 2``)
    has no measurable motion → zero velocity → the edges hold its position.
    """
    n = frames.shape[0]
    d = values.shape[1]
    if n < 2:
        return np.zeros(d), np.zeros(d)
    w = max(1, min(int(window), n - 1))
    df_lead = float(frames[w] - frames[0])
    df_trail = float(frames[-1] - frames[-1 - w])
    v_lead = (values[w] - values[0]) / df_lead if df_lead else np.zeros(d)
    v_trail = (values[-1] - values[-1 - w]) / df_trail if df_trail else np.zeros(d)
    return v_lead, v_trail


def extend_pose_to_span(
    pose: PoseSequence,
    first: int,
    last: int,
    *,
    decay: float = 0.9,
    vel_window: int = 3,
    max_step: float | None = None,
) -> tuple[PoseSequence, np.ndarray]:
    """Extend a pose to cover ``[first, last]`` by motion-aware edge extrapolation.

    A subject the tracker lost is still physically on the pitch; rather than let the renderer
    blink it out, we *hold its posture* (every rotation clamps to the nearest measured pose —
    "standing stays standing") and let the world translation *coast* with a decaying edge
    velocity ("running keeps running, then eases"). Only leading (``< frames[0]``) and trailing
    (``> frames[-1]``) frames are added; interior rows are untouched and the input is not
    mutated. Returns the new pose + the array of added (extrapolated) frame indices.
    """
    ef = np.asarray(pose.frames, dtype=int).reshape(-1)
    if ef.shape[0] == 0:
        return pose.copy(), np.empty(0, dtype=int)
    f0, fn = int(ef[0]), int(ef[-1])
    lead = np.arange(int(first), f0, dtype=int)
    trail = np.arange(fn + 1, int(last) + 1, dtype=int)
    added = np.concatenate([lead, trail])
    if added.size == 0:
        return pose.copy(), np.empty(0, dtype=int)

    ln, tn = lead.shape[0], trail.shape[0]
    nf = np.concatenate([lead, ef, trail])

    def _hold(v: np.ndarray) -> np.ndarray:  # posture frozen at the measured edge pose
        head = np.repeat(v[:1], ln, axis=0)
        tail = np.repeat(v[-1:], tn, axis=0)
        return np.concatenate([head, v, tail], axis=0)

    # translation coasts with a decaying edge velocity (held position when velocity is zero);
    # max_step (m/frame) caps that velocity — a dying track's slid-off edge must not launch
    # the coast at inhuman speed (#207)
    v_lead, v_trail = _edge_velocity(ef, pose.transl, vel_window)
    if max_step is not None:
        for v in (v_lead, v_trail):
            m = float(np.linalg.norm(v))
            if m > max_step:
                v *= max_step / m
    tr_head = pose.transl[0][None, :] - v_lead[None, :] * _geom_steps(f0 - lead, decay)[:, None]
    tr_tail = pose.transl[-1][None, :] + v_trail[None, :] * _geom_steps(trail - fn, decay)[:, None]
    transl = np.concatenate([tr_head, pose.transl, tr_tail], axis=0)

    # Coasted edges are IMPUTED, not INTERPOLATED: there is no measurement on the far side to
    # bridge to, so the position is inference all the way down.
    prov = np.concatenate([
        np.full(ln, Provenance.IMPUTED.value, dtype=pose.provenance.dtype),
        pose.provenance,
        np.full(tn, Provenance.IMPUTED.value, dtype=pose.provenance.dtype),
    ])

    new_pose = PoseSequence(
        frames=nf,
        global_orient=_hold(pose.global_orient),
        body_pose=_hold(pose.body_pose),
        transl=transl,
        left_hand_pose=None if pose.left_hand_pose is None else _hold(pose.left_hand_pose),
        right_hand_pose=None if pose.right_hand_pose is None else _hold(pose.right_hand_pose),
        jaw_pose=None if pose.jaw_pose is None else _hold(pose.jaw_pose),
        provenance=prov,
    )
    return new_pose, added


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
    scene: Scene, cfg: CoherenceConfig | None = None, *, fps: float = 25.0
) -> tuple[Scene, CoherenceReport]:
    """Densify subject gaps + append auto-smoothing corrections; return a NEW scene + report.

    The input ``scene`` is never mutated. Bridged frames get a low ``subject_frame_conf`` so
    the attention list flags them as inferred (R-6). Smoothing is layered as corrections, so a
    later ``resolve_scene`` applies it non-destructively over the (now dense) proposal.
    ``fps`` converts ``cfg.coast_max_speed`` (m/s) into the per-frame edge-coast cap.
    """
    cfg = cfg or CoherenceConfig()
    base_conf = scene.confidence or ConfidenceMap()
    frame_conf = dict(base_conf.subject_frame_conf)
    new_subjects = []
    auto_corrs: list[Correction] = []
    report = CoherenceReport(n_subjects=len(scene.subjects))

    # Clip span = union of every present frame (subjects + ball), matching the range
    # anim_export.py / blender_animate.py iterate. Extending each subject to this span keeps the
    # on-screen population stable: a tracker-lost player is reconstructed (held posture + coasting
    # root, R-6 low confidence) instead of evaporating at the clip edges.
    span: tuple[int, int] | None = None
    if cfg.extend_to_span:
        present = [np.asarray(s.proposal.pose.frames, dtype=int) for s in scene.subjects]
        present = [f for f in present if f.size]
        if scene.ball is not None and np.asarray(scene.ball.frames).size:
            present.append(np.asarray(scene.ball.frames, dtype=int))
        if present:
            span = (
                int(min(int(f.min()) for f in present)),
                int(max(int(f.max()) for f in present)),
            )

    # When extending we commit to full presence, so bridge interior gaps of ANY length (both
    # endpoints are real observations); otherwise respect the conservative max_fill_gap cap.
    interior_cap = (
        max(cfg.max_fill_gap, span[1] - span[0]) if span is not None else cfg.max_fill_gap
    )

    for s in scene.subjects:
        motion, filled = fill_motion_gaps(s.proposal, interior_cap)
        extended = np.empty(0, dtype=int)
        if span is not None:
            new_pose, extended = extend_pose_to_span(
                motion.pose,
                span[0],
                span[1],
                decay=cfg.extrapolate_decay,
                vel_window=cfg.extrapolate_velocity_window,
                max_step=cfg.coast_max_speed / fps,
            )
            motion = SubjectMotion(shape=motion.shape, pose=new_pose)
        new_subjects.append(replace(s, proposal=motion))
        frames = motion.pose.frames
        conf = np.full(frames.shape[0], cfg.real_confidence, dtype=float)
        if filled.size:
            conf[np.isin(frames, filled)] = cfg.filled_confidence
            report.filled_frames += int(filled.size)
            report.subjects_filled += 1
        if extended.size:
            conf[np.isin(frames, extended)] = cfg.extrapolated_confidence
            report.extended_frames += int(extended.size)
            report.subjects_extended += 1
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
