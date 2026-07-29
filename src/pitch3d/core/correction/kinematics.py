"""Kinematic plausibility gate: clamp impossible player motion, mark teleports (M3-9, R-6).

Measured need (#207): a real-clip reconstruction carries root tracks with speeds up to
~70 m/s and accelerations >3000 m/s² — tracker jitter and ID-swap teleports that the
whole-track MA(5) coherence smoothing is structurally too weak to fix (a 1-frame 1.8 m
jump stays ~70× over the human accel limit after averaging). This module closes the gap
in two honest, separable ways:

* **Physically impossible jitter is CORRECTED**: per subject, the root XY track (the same
  horizontal signal ``scripts/motion_stats.py`` measures) is projected onto the feasible
  set ``|v| <= max_speed``, ``|dv|/dt <= max_accel`` by velocity clamping + bounded
  forward/backward acceleration sweeps, then reintegrated with both endpoints anchored to
  the measured positions. The result is layered as ONE dense ``KEYFRAME_INTERP``
  correction per subject through the normal Correction seam (ADR-0002) — inspectable,
  disableable, never baked into the proposal. Z (body height) is left untouched.
* **Teleports are MARKED, not erased**: a single-interval speed above
  ``teleport_factor * max_speed`` is an identity-class error (ID swap / mis-stitch), not
  noise. Inventing a sprint that never happened would be dishonest (R-6), so the gate
  preserves the jump exactly, splits the clamp at the boundary, and reports a
  :class:`TeleportEvent` for identity/stitch review.

The gate runs on the *resolved-so-far* track (proposal ⊕ earlier corrections, e.g. the
auto coherence smoothing), so the emitted keyframes capture and supersede that basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from ..scene.layers import Correction, CorrectionMode, CorrectionTarget, TargetKind
from ..scene.player_profile import PlayerProfile, ProfileUpdateProposal
from ..scene.scene import Scene, Subject
from .engine import make_keyframes, resolve_subject_motion

#: Shared human-motion ceilings — single source of truth with scripts/motion_stats.py.
HUMAN_MAX_SPEED = 10.5  # m/s — elite sprint ceiling
HUMAN_MAX_ACCEL = 8.0   # m/s² — elite acceleration ceiling


#: Valid values for :attr:`KinematicConfig.teleport_policy`.
TELEPORT_POLICIES = ("hold", "interpolate")

#: Confidence stamped on frames whose XY was interpolated across a teleport
#: region — below coherence's ``extrapolated_confidence=0.2`` so the operator
#: can still tell "reconstructed across an ID swap" apart from "coast-extended
#: past the tracker" in the attention list (R-6).
TELEPORT_INTERPOLATED_CONF = 0.15


@dataclass(frozen=True)
class KinematicConfig:
    """Physical limits + gate knobs (fps is passed at call time, like motion_stats)."""

    max_speed: float = HUMAN_MAX_SPEED
    max_accel: float = HUMAN_MAX_ACCEL
    teleport_factor: float = 2.0   # one-interval speed > factor*max_speed → teleport candidate
    # A candidate is demoted to a jitter SPIKE (and clamped as noise) only on clear
    # out-and-back evidence: a neighbouring interval of comparable speed pointing the
    # opposite way. When ambiguous we keep the jump and mark it — clamping a real ID swap
    # would fabricate a sprint that never happened (R-6: mark, don't invent).
    spike_neighbor_frac: float = 0.5   # neighbour speed must be > frac*candidate speed
    spike_reversal_cos: float = -0.5   # ...and point back (cosine below this)
    max_passes: int = 50           # bounded accel-clamp sweeps per segment
    min_correction_m: float = 1e-6  # skip emitting a correction below this max deviation
    #: What to do with detected teleport regions (T2.b):
    #: * ``"hold"`` (default, R-6 strict): preserve the jump verbatim in the emitted
    #:   correction. The renderer shows a discontinuity at the ID-swap frame.
    #: * ``"interpolate"``: replace the region with a linear XY path between the
    #:   two anchor rows (the entry and exit of the region). ``TeleportEvent`` is
    #:   still recorded so the audit trail is preserved, and interpolated frames
    #:   are stamped with :data:`TELEPORT_INTERPOLATED_CONF` in the confidence
    #:   map so the attention list flags them as inferred (never silently trusted).
    teleport_policy: str = "hold"

    def __post_init__(self) -> None:
        if self.teleport_policy not in TELEPORT_POLICIES:
            raise ValueError(
                f"teleport_policy={self.teleport_policy!r} not in {TELEPORT_POLICIES}"
            )


@dataclass(frozen=True)
class TeleportEvent:
    """One preserved identity-class discontinuity (marked for stitch review, R-6).

    ``n_intervals == 1`` is an instantaneous ID-swap jump; ``> 1`` is a run of
    consecutive impossible intervals (tracker sliding off / smeared swap) preserved as
    one region — there is no feasible path covering its displacement, so inventing one
    would fabricate motion.
    """

    track_id: int
    frame: int      # frame index where the discontinuity starts landing
    jump_m: float   # net XY displacement across the whole region
    speed_mps: float  # fastest interval inside the region
    n_intervals: int = 1


@dataclass
class KinematicReport:
    """What the gate measured and did, for logging / inspection (R-6 transparency)."""

    n_subjects: int = 0
    subjects_corrected: int = 0
    corrections_added: int = 0
    speed_viol_before: int = 0
    accel_viol_before: int = 0
    # counted EXCLUDING intervals adjacent to preserved teleports — those are marked as
    # identity errors (``teleports``), deliberately not "fixed" into invented motion
    speed_viol_after: int = 0
    accel_viol_after: int = 0
    max_dev_m: float = 0.0  # largest positional change the clamp introduced anywhere
    teleports: list[TeleportEvent] = field(default_factory=list)
    #: Per-subject observations for the auto-tuner (T4). Empty unless a
    #: ``profile_provider`` was passed. Consumed by :func:`apply_profile_updates`.
    profile_updates: list[ProfileUpdateProposal] = field(default_factory=list)
    #: Track ids for which the per-subject ceilings came from a stored profile.
    subjects_using_profile: list[int] = field(default_factory=list)
    #: Total frames whose XY was replaced with a low-confidence interpolant
    #: across a teleport region (T2.b `teleport_policy="interpolate"`).
    teleport_interpolated_frames: int = 0


def _speeds_accels(
    frames: np.ndarray, xy: np.ndarray, fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-interval speed + per-junction accel magnitudes (motion_stats convention).

    Returns ``(dt, speed, accel)`` with ``dt``/``speed`` of length ``n-1`` and ``accel``
    of length ``n-2`` (``accel[j]`` compares intervals ``j`` and ``j+1`` over ``dt[j+1]``).
    """
    dt = np.diff(np.asarray(frames, dtype=float)) / fps
    vel = np.diff(np.asarray(xy, dtype=float), axis=0) / dt[:, None]
    speed = np.linalg.norm(vel, axis=1)
    accel = (
        np.linalg.norm(np.diff(vel, axis=0), axis=1) / dt[1:]
        if len(vel) > 1
        else np.empty(0)
    )
    return dt, speed, accel


def clamp_track_xy(
    frames: np.ndarray, xy: np.ndarray, fps: float, cfg: KinematicConfig
) -> np.ndarray:
    """Project one contiguous XY segment onto the speed/accel-feasible set.

    Velocity clamp → bounded forward/backward accel sweeps → reintegration from the first
    measured position, with the endpoint error redistributed proportionally to elapsed
    time so BOTH segment ends stay at their measured positions. Deterministic and
    bounded; the input is not mutated.
    """
    p = np.asarray(xy, dtype=float)
    n = p.shape[0]
    if n < 3:
        return p.copy()
    dt = np.diff(np.asarray(frames, dtype=float)) / fps
    v = np.diff(p, axis=0) / dt[:, None]

    # clamp a hair INSIDE the limits: landing exactly on the boundary reads as a violation
    # after reintegration round-off (measured 8.0000000000006 vs an 8.0 limit)
    safety = 1.0 - 1e-6
    max_speed = cfg.max_speed * safety
    max_accel = cfg.max_accel * safety

    def _forward() -> bool:
        clean = True
        for i in range(1, len(v)):  # pull v[i] toward v[i-1]
            dv = v[i] - v[i - 1]
            lim = max_accel * dt[i]
            m = float(np.linalg.norm(dv))
            if m > lim:
                v[i] = v[i - 1] + dv * (lim / m)
                clean = False
        return clean

    def _backward() -> bool:
        clean = True
        for i in range(len(v) - 2, -1, -1):
            dv = v[i] - v[i + 1]
            lim = max_accel * dt[i + 1]
            m = float(np.linalg.norm(dv))
            if m > lim:
                v[i] = v[i + 1] + dv * (lim / m)
                clean = False
        return clean

    speed = np.linalg.norm(v, axis=1)
    over = speed > max_speed
    if over.any():
        v[over] *= (max_speed / speed[over])[:, None]
    # Alternating sweeps symmetrize the bias but each direction can re-break the other's
    # constraints, so convergence alone is not guaranteed within the budget. The FINAL
    # forward sweep guarantees feasibility outright: each clamped v[i] is a convex
    # combination of speed-feasible velocities (norm <= max of the pair <= max_speed), and
    # every consecutive pair is explicitly limited — so the loop budget only buys symmetry.
    for _ in range(max(0, int(cfg.max_passes))):
        if _forward() & _backward():
            break
    _forward()

    q = np.empty_like(p)
    q[0] = p[0]
    q[1:] = p[0] + np.cumsum(v * dt[:, None], axis=0)
    # anchor the far end: distribute the residual by elapsed time (adds only err/T velocity)
    err = p[-1] - q[-1]
    t = np.concatenate([[0.0], np.cumsum(dt)])
    q += err[None, :] * (t / t[-1])[:, None]
    return q


def _is_jitter_spike(vel: np.ndarray, speed: np.ndarray, k: int, cfg: KinematicConfig) -> bool:
    """True when candidate interval ``k`` is an out-and-back noise spike, not a relocation.

    A spike's outlier row produces TWO fast intervals in nearly opposite directions (out,
    then straight back); a true ID-swap teleport is one isolated fast interval between
    normal-speed neighbours. Only the clear reversal demotes a candidate — ambiguity stays
    a teleport (marked, preserved) because clamping a real identity jump would invent
    motion (R-6).
    """
    for j in (k - 1, k + 1):
        if 0 <= j < len(vel) and speed[j] > cfg.spike_neighbor_frac * speed[k]:
            cos = float(np.dot(vel[j], vel[k])) / (speed[j] * speed[k] + 1e-12)
            if cos < cfg.spike_reversal_cos:
                return True
    return False


def gate_subject_xy(
    frames: np.ndarray, xy: np.ndarray, fps: float, cfg: KinematicConfig
) -> tuple[np.ndarray, list[tuple[int, int]], list[int]]:
    """Clamp one subject's XY track. Returns ``(new_xy, regions, interpolated_rows)``.

    Consecutive candidate intervals are grouped into ONE region ``(k0, k1)`` (inclusive
    interval indices). Behaviour then depends on ``cfg.teleport_policy``:

    * ``"hold"`` (R-6 default): the jump between rows ``k0`` and ``k1+1`` is kept as
      measured; ``interpolated_rows`` is empty. Candidate-free stretches are clamped
      independently between regions.
    * ``"interpolate"``: the whole track is clamped as a single anchored segment, which
      linearly interpolates across the region — no invented sprint, just a straight
      path between the two anchors. ``interpolated_rows`` lists the frame indices
      whose XY changed from the raw measurement so the caller can stamp them with a
      low confidence in the attention list.
    """
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]
    if n < 3:
        return xy.copy(), [], []
    dt = np.diff(np.asarray(frames, dtype=float)) / fps
    vel = np.diff(xy, axis=0) / dt[:, None]
    speed = np.linalg.norm(vel, axis=1)
    cand = [
        int(k)
        for k in np.nonzero(speed > cfg.teleport_factor * cfg.max_speed)[0]
        if not _is_jitter_spike(vel, speed, int(k), cfg)
    ]
    regions: list[tuple[int, int]] = []
    for k in cand:  # group consecutive interval indices into inclusive runs
        if regions and k == regions[-1][1] + 1:
            regions[-1] = (regions[-1][0], k)
        else:
            regions.append((k, k))

    out = xy.copy()
    interpolated_rows: list[int] = []

    if cfg.teleport_policy == "interpolate" and regions:
        # Clamp the whole track as one anchored segment. The velocity/accel sweeps
        # linearly interpolate across the region (no invented sprint — a straight
        # line between the anchors). Rows inside each region are tagged R-6 for
        # the confidence map so the attention list flags them as inferred.
        clamped = clamp_track_xy(frames, xy, fps, cfg)
        out = clamped
        for k0, k1 in regions:
            # region intervals are k0..k1 inclusive; the "after-jump" rows are
            # k0+1 through k1+1 inclusive (n_intervals+1 rows).
            for r in range(k0 + 1, k1 + 2):
                if r < n:
                    interpolated_rows.append(int(r))
    else:
        cuts = [0, *(i for k0, k1 in regions for i in (k0 + 1, k1 + 1)), n]
        for b, e in zip(cuts[::2], cuts[1::2], strict=True):  # candidate-free [b, e) stretches
            if e - b >= 3:
                out[b:e] = clamp_track_xy(frames[b:e], xy[b:e], fps, cfg)

    return out, regions, interpolated_rows


def _count_viols(
    frames: np.ndarray,
    xy: np.ndarray,
    fps: float,
    cfg: KinematicConfig,
    exclude_intervals: list[int] | None = None,
) -> tuple[int, int]:
    """Speed/accel violation counts; ``exclude_intervals`` drops teleport-adjacent entries."""
    _, speed, accel = _speeds_accels(frames, xy, fps)
    sp_mask = speed > cfg.max_speed
    ac_mask = accel > cfg.max_accel
    for k in exclude_intervals or []:
        sp_mask[k] = False
        for j in (k - 1, k):  # accel[j] touches intervals j and j+1
            if 0 <= j < ac_mask.shape[0]:
                ac_mask[j] = False
    return int(sp_mask.sum()), int(ac_mask.sum())


#: Callable that returns the stored profile for a subject, or None if unknown.
ProfileProvider = Callable[[Subject], PlayerProfile | None]


def _mark_low_conf(
    scene: Scene, track_id: int, frames: np.ndarray,
    row_indices: list[int], conf: float,
) -> None:
    """Stamp ``subject_frame_conf[track_id]`` at the given row indices with ``conf``.

    Mutates the scene's confidence map in place. If no map exists yet, one is
    created. Missing per-track entries are seeded with 1.0 (measured) so only
    the interpolated rows get the low value — the rest stay honest.

    Also stamps the matching pose rows ``INTERPOLATED``. The gate emits its result as a
    *correction* rather than rewriting the proposal, but the frame is fabricated either way
    once the stack resolves, and provenance describes the frame, not which array holds it.
    """
    from ..scene.layers import ConfidenceMap
    from ..scene.motion import Provenance
    for s in scene.subjects:
        if s.track_id == track_id and s.proposal.pose.n_frames == len(frames):
            s.proposal.pose.mark(row_indices, Provenance.INTERPOLATED)
    if scene.confidence is None:
        scene.confidence = ConfidenceMap()
    frame_conf = dict(scene.confidence.subject_frame_conf)
    existing = frame_conf.get(track_id)
    if existing is None or len(existing) != len(frames):
        existing = np.ones(len(frames), dtype=float)
    else:
        existing = np.asarray(existing, dtype=float).copy()
    for r in row_indices:
        if 0 <= r < existing.shape[0]:
            existing[r] = float(conf)
    frame_conf[track_id] = existing
    scene.confidence = replace(scene.confidence, subject_frame_conf=frame_conf)


def _subject_cfg(cfg: KinematicConfig, profile: PlayerProfile | None
                 ) -> tuple[KinematicConfig, bool]:
    """Return ``(cfg_for_this_subject, used_profile)``.

    When ``profile`` carries a measured/default value for ``peak_speed_mps`` or
    ``peak_accel_mps2``, replace the shared ``cfg`` value with it. Untouched
    fields (spike_neighbor_frac, teleport_factor, etc.) stay global. This is
    the per-subject ceiling from §4.1 without touching downstream code.
    """
    if profile is None:
        return cfg, False
    used = False
    new_speed = cfg.max_speed
    new_accel = cfg.max_accel
    speed_field = profile.kinematics.get("peak_speed_mps")
    accel_field = profile.kinematics.get("peak_accel_mps2")
    if speed_field is not None:
        new_speed = float(speed_field.value)
        used = True
    if accel_field is not None:
        new_accel = float(accel_field.value)
        used = True
    if not used:
        return cfg, False
    return replace(cfg, max_speed=new_speed, max_accel=new_accel), True


def _emit_profile_updates(
    track_id: int, resolved_xy: np.ndarray, frames: np.ndarray, fps: float,
    profile: PlayerProfile | None, min_conf: float,
) -> list[ProfileUpdateProposal]:
    """Compute p95 speed/accel on the resolved track and emit auto-tune proposals.

    Follows the §4.4 rules that live INSIDE the gate:

    * layer 1 (resolved-only) — we already pass the CLAMPED xy in.
    * layer 2 (robust estimator) — p95 over the whole visible window, not max.

    The other five filters run inside ``update_field`` when the store consumes
    these proposals. If ``profile`` is ``None`` there is no target to write to
    — return empty.
    """
    if profile is None or resolved_xy.shape[0] < 3:
        return []
    _, speed, accel = _speeds_accels(frames, resolved_xy, fps)
    updates: list[ProfileUpdateProposal] = []
    if speed.size >= 2:
        obs = float(np.percentile(speed, 95))
        default_v = None
        f = profile.kinematics.get("peak_speed_mps")
        if f is not None:
            default_v = f.value
        updates.append(ProfileUpdateProposal(
            track_id=int(track_id), domain="player",
            field_key="peak_speed_mps", observation=obs,
            confidence=float(min_conf), default_value=default_v,
        ))
    if accel.size >= 2:
        obs = float(np.percentile(accel, 95))
        default_v = None
        f = profile.kinematics.get("peak_accel_mps2")
        if f is not None:
            default_v = f.value
        updates.append(ProfileUpdateProposal(
            track_id=int(track_id), domain="player",
            field_key="peak_accel_mps2", observation=obs,
            confidence=float(min_conf), default_value=default_v,
        ))
    return updates


def kinematic_gate(
    scene: Scene, cfg: KinematicConfig | None = None, *, fps: float,
    profile_provider: ProfileProvider | None = None,
) -> tuple[Scene, KinematicReport]:
    """Append per-subject kinematic-clamp corrections to a scene; return NEW scene + report.

    For each subject the gate resolves the track through its existing non-REFIT
    corrections (so it corrects the residual violations the coherence smoothing left),
    clamps the XY root path, and — when anything actually changed — emits one dense
    ``KEYFRAME_INTERP`` ``ROOT_TRANSLATION`` correction whose keyframes are every measured
    frame. Resolving later reproduces the gated track exactly and supersedes the earlier
    smoothing on that target. The input scene is never mutated; the ball is out of scope
    (measured clean, and ball physics differ).

    ``profile_provider``: optional per-subject :class:`PlayerProfile` lookup. When
    present, each subject's ``peak_speed_mps`` / ``peak_accel_mps2`` from the
    profile override the shared ``cfg`` values (T4.b). The gate also emits
    :class:`ProfileUpdateProposal`s on ``report.profile_updates`` — the store
    consumer feeds each through :func:`update_field` so the seven-filter policy
    runs at the persistence seam (never in the gate).
    """
    cfg = cfg or KinematicConfig()
    report = KinematicReport(n_subjects=len(scene.subjects))
    auto_corrs: list[Correction] = []

    for s in scene.subjects:
        corrs = [
            c for c in scene.corrections_for(s.track_id) if c.mode != CorrectionMode.REFIT
        ]
        resolved = resolve_subject_motion(s.proposal, corrs)
        frames = np.asarray(resolved.pose.frames, dtype=int)
        transl = np.asarray(resolved.pose.transl, dtype=float)
        if frames.shape[0] < 3:
            continue
        xy = transl[:, :2]

        profile = profile_provider(s) if profile_provider is not None else None
        subject_cfg, used_profile = _subject_cfg(cfg, profile)
        if used_profile:
            report.subjects_using_profile.append(int(s.track_id))

        sp_b, ac_b = _count_viols(frames, xy, fps, subject_cfg)
        report.speed_viol_before += sp_b
        report.accel_viol_before += ac_b

        # confidence for auto-tune: use the min subject_frame_conf over the visible
        # window (falls back to 1.0 when no confidence map is attached)
        min_conf = 1.0
        conf_map = scene.confidence
        if conf_map is not None and s.track_id in conf_map.subject_frame_conf:
            cvals = np.asarray(conf_map.subject_frame_conf[s.track_id], dtype=float)
            if cvals.size:
                min_conf = float(cvals.min())

        if sp_b == 0 and ac_b == 0:
            # nothing to clamp, but still feed the auto-tuner with the resolved p95
            report.profile_updates.extend(
                _emit_profile_updates(s.track_id, xy, frames, fps, profile, min_conf)
            )
            continue

        new_xy, regions, interpolated_rows = gate_subject_xy(
            frames, xy, fps, subject_cfg,
        )
        _, speed, _ = _speeds_accels(frames, xy, fps)
        for k0, k1 in regions:
            report.teleports.append(
                TeleportEvent(
                    track_id=s.track_id,
                    frame=int(frames[k0 + 1]),
                    jump_m=float(np.linalg.norm(xy[k1 + 1] - xy[k0])),
                    speed_mps=float(speed[k0: k1 + 1].max()),
                    n_intervals=k1 - k0 + 1,
                )
            )
        preserved = [k for k0, k1 in regions for k in range(k0, k1 + 1)]
        # In "interpolate" policy the region rows are no longer preserved
        # verbatim, so accel violations from the smooth interpolant count
        # against the "after" tally (they'll be near zero anyway).
        if subject_cfg.teleport_policy == "interpolate":
            preserved = []
        sp_a, ac_a = _count_viols(
            frames, new_xy, fps, subject_cfg, exclude_intervals=preserved,
        )
        report.speed_viol_after += sp_a
        report.accel_viol_after += ac_a

        dev = float(np.linalg.norm(new_xy - xy, axis=1).max())
        report.max_dev_m = max(report.max_dev_m, dev)

        # T2.b R-6 tag: stamp interpolated teleport frames with a low
        # subject_frame_conf so the attention list flags them as inferred.
        if interpolated_rows:
            report.teleport_interpolated_frames += len(interpolated_rows)
            _mark_low_conf(
                scene, s.track_id, frames, interpolated_rows,
                TELEPORT_INTERPOLATED_CONF,
            )
        # auto-tune fed from the CLAMPED track (§4.4 layer 1)
        report.profile_updates.extend(
            _emit_profile_updates(s.track_id, new_xy, frames, fps, profile, min_conf)
        )
        if dev < subject_cfg.min_correction_m:
            continue

        new_transl = transl.copy()
        new_transl[:, :2] = new_xy
        auto_corrs.append(
            make_keyframes(
                f"auto-kin-transl-{s.track_id}",
                CorrectionTarget(TargetKind.ROOT_TRANSLATION, subject_track_id=s.track_id),
                (int(frames[0]), int(frames[-1])),
                key_frames=frames.astype(float),
                key_values=new_transl,
                note=(
                    f"auto kinematic gate (root translation): "
                    f"speed<={subject_cfg.max_speed}m/s, "
                    f"accel<={subject_cfg.max_accel}m/s² @ {fps:.3g}fps"
                    + (f" [profile #{s.track_id}]" if used_profile else "")
                ),
            )
        )
        report.subjects_corrected += 1

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report
