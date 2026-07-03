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

import numpy as np

from ..scene.layers import Correction, CorrectionMode, CorrectionTarget, TargetKind
from ..scene.scene import Scene
from .engine import make_keyframes, resolve_subject_motion

#: Shared human-motion ceilings — single source of truth with scripts/motion_stats.py.
HUMAN_MAX_SPEED = 10.5  # m/s — elite sprint ceiling
HUMAN_MAX_ACCEL = 8.0   # m/s² — elite acceleration ceiling


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
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Clamp one subject's XY track, preserving teleport regions verbatim.

    Consecutive candidate intervals are grouped into ONE region ``(k0, k1)`` (inclusive
    interval indices): the jump between rows ``k0`` and ``k1+1`` is kept exactly as
    measured (an instantaneous ID swap when ``k0 == k1``, a slid-off/smeared swap when
    longer — no feasible path covers its displacement, so clamping would invent motion).
    The candidate-free stretches between regions are clamped independently.
    """
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]
    if n < 3:
        return xy.copy(), []
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
    cuts = [0, *(i for k0, k1 in regions for i in (k0 + 1, k1 + 1)), n]
    for b, e in zip(cuts[::2], cuts[1::2], strict=True):  # candidate-free [b, e) stretches
        if e - b >= 3:
            out[b:e] = clamp_track_xy(frames[b:e], xy[b:e], fps, cfg)
    return out, regions


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


def kinematic_gate(
    scene: Scene, cfg: KinematicConfig | None = None, *, fps: float
) -> tuple[Scene, KinematicReport]:
    """Append per-subject kinematic-clamp corrections to a scene; return NEW scene + report.

    For each subject the gate resolves the track through its existing non-REFIT
    corrections (so it corrects the residual violations the coherence smoothing left),
    clamps the XY root path, and — when anything actually changed — emits one dense
    ``KEYFRAME_INTERP`` ``ROOT_TRANSLATION`` correction whose keyframes are every measured
    frame. Resolving later reproduces the gated track exactly and supersedes the earlier
    smoothing on that target. The input scene is never mutated; the ball is out of scope
    (measured clean, and ball physics differ).
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

        sp_b, ac_b = _count_viols(frames, xy, fps, cfg)
        report.speed_viol_before += sp_b
        report.accel_viol_before += ac_b
        if sp_b == 0 and ac_b == 0:
            continue

        new_xy, regions = gate_subject_xy(frames, xy, fps, cfg)
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
        sp_a, ac_a = _count_viols(frames, new_xy, fps, cfg, exclude_intervals=preserved)
        report.speed_viol_after += sp_a
        report.accel_viol_after += ac_a

        dev = float(np.linalg.norm(new_xy - xy, axis=1).max())
        report.max_dev_m = max(report.max_dev_m, dev)
        if dev < cfg.min_correction_m:
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
                    f"auto kinematic gate (root translation): speed<={cfg.max_speed}m/s, "
                    f"accel<={cfg.max_accel}m/s² @ {fps:.3g}fps"
                ),
            )
        )
        report.subjects_corrected += 1

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report
