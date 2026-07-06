"""Capsule collision post-process (T3, R-6): soft-repulse overlapping players.

Measured need (§F of ``docs/research/2026-07-06-player-physics.md``): the M3-9
gate is per-subject and cannot see subject↔subject interpenetration. Two
players whose reconstructed XY drifts within ``2r`` of each other look like
they're merged into one blob.

Not a physics simulation: at each frame the subjects are modelled as vertical
capsules of radius ``CollisionConfig.capsule_radius_m`` on the pitch plane.
Overlapping pairs get a soft push apart — one Jacobi iteration per pass,
``cfg.n_passes`` passes total per frame. Each subject's per-frame net push is
capped at ``cfg.max_push_per_frame_m`` so a stack of ten never launches a body
across the pitch (R-6: mark, don't fabricate).

The gate emits ONE ``KEYFRAME_INTERP`` ``ROOT_TRANSLATION`` correction per
subject that actually moved. Layered through the ADR-0002 seam, so a later
``resolve_scene`` sees the deconflicted XY without touching the proposal.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..config.gates import CollisionConfig
from ..scene.layers import Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene
from .engine import make_keyframes, resolve_subject_motion


@dataclass
class CollisionReport:
    """What the collision gate saw and did (R-6 transparency)."""

    n_subjects: int = 0
    subjects_moved: int = 0
    corrections_added: int = 0
    frames_with_overlap: int = 0
    pairs_resolved: int = 0
    max_push_m: float = 0.0
    max_overlap_before_m: float = 0.0


def _jacobi_pass(xy: np.ndarray, radius: float, strength: float
                 ) -> tuple[np.ndarray, int, float]:
    """Push overlapping pairs apart; return (delta, pairs, max_overlap_before).

    ``xy`` shape ``(N, 2)`` — one row per subject present at this frame.
    ``delta`` shape ``(N, 2)`` — per-subject net displacement to add.
    Every pair contributes ``strength * overlap / 2`` to each subject (equal
    Jacobi split).
    """
    n = xy.shape[0]
    delta = np.zeros_like(xy)
    two_r = 2.0 * radius
    if n < 2:
        return delta, 0, 0.0
    pairs = 0
    max_over = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = xy[j] - xy[i]
            dist = float(np.linalg.norm(d))
            if dist >= two_r:
                continue
            overlap = two_r - dist
            max_over = max(max_over, overlap)
            if dist < 1e-9:
                # coincident points — split along +X so passes converge
                axis = np.array([1.0, 0.0])
            else:
                axis = d / dist
            half = 0.5 * strength * overlap
            delta[i] -= axis * half
            delta[j] += axis * half
            pairs += 1
    return delta, pairs, max_over


def _resolve_frame(xy: np.ndarray, cfg: CollisionConfig
                   ) -> tuple[np.ndarray, int, float, float]:
    """Iterate the Jacobi pass ``cfg.n_passes`` times; return (new_xy, pairs, max_over_before, max_push)."""
    total_delta = np.zeros_like(xy)
    max_over_before = 0.0
    pairs_before = 0
    cur = xy.copy()
    for pass_i in range(cfg.n_passes):
        delta, pairs, max_over = _jacobi_pass(cur, cfg.capsule_radius_m, cfg.strength)
        if pass_i == 0:
            pairs_before = pairs
            max_over_before = max_over
        if pairs == 0:
            break
        cur = cur + delta
        total_delta = total_delta + delta
    # per-subject safety cap on the net push
    lengths = np.linalg.norm(total_delta, axis=1)
    max_push = float(lengths.max()) if lengths.size else 0.0
    over_cap = lengths > cfg.max_push_per_frame_m
    if over_cap.any():
        scale = np.where(
            over_cap, cfg.max_push_per_frame_m / np.maximum(lengths, 1e-12), 1.0
        )
        total_delta = total_delta * scale[:, None]
        max_push = float(np.linalg.norm(total_delta, axis=1).max())
    return xy + total_delta, pairs_before, max_over_before, max_push


def collision_gate(
    scene: Scene, cfg: CollisionConfig | None = None,
) -> tuple[Scene, CollisionReport]:
    """Soft-repulse overlapping subjects frame-by-frame; return NEW scene + report.

    * ``cfg is None`` or ``cfg.enabled is False``: measure-only path — counts
      frames with overlap and the largest overlap, no corrections emitted.
    * Enabled: emits ONE ``KEYFRAME_INTERP`` ``ROOT_TRANSLATION`` correction
      per subject whose per-frame push accumulated above
      ``cfg.min_correction_m``.
    """
    cfg = cfg if cfg is not None else CollisionConfig()
    report = CollisionReport(n_subjects=len(scene.subjects))

    subj_data: list[dict] = []
    for s in scene.subjects:
        corrs = list(scene.corrections_for(s.track_id))
        resolved = resolve_subject_motion(s.proposal, corrs)
        frames = np.asarray(resolved.pose.frames, dtype=int)
        transl = np.asarray(resolved.pose.transl, dtype=float)
        subj_data.append({
            "subject": s, "frames": frames, "transl": transl,
            "new_transl": transl.copy(),
        })

    # union of frames across all subjects
    if not subj_data:
        return scene, report
    all_frames = np.unique(np.concatenate([sd["frames"] for sd in subj_data]))

    for f in all_frames:
        present = [
            (i, np.searchsorted(sd["frames"], f))
            for i, sd in enumerate(subj_data)
            if f in sd["frames"]
        ]
        # filter to rows that actually hit f (searchsorted may return >= boundary)
        present = [
            (i, row) for i, row in present
            if row < subj_data[i]["frames"].shape[0] and subj_data[i]["frames"][row] == f
        ]
        if len(present) < 2:
            continue
        xy = np.stack([subj_data[i]["transl"][row, :2] for i, row in present])
        new_xy, pairs, max_over_before, max_push = _resolve_frame(xy, cfg)
        if pairs == 0:
            continue
        report.frames_with_overlap += 1
        report.pairs_resolved += pairs
        report.max_overlap_before_m = max(report.max_overlap_before_m, max_over_before)
        if cfg.enabled:
            report.max_push_m = max(report.max_push_m, max_push)
            for k, (i, row) in enumerate(present):
                subj_data[i]["new_transl"][row, :2] = new_xy[k]

    if not cfg.enabled:
        return scene, report

    auto_corrs: list[Correction] = []
    for sd in subj_data:
        dev = float(np.linalg.norm(
            sd["new_transl"][:, :2] - sd["transl"][:, :2], axis=1
        ).max())
        if dev < cfg.min_correction_m:
            continue
        s = sd["subject"]
        frames = sd["frames"]
        auto_corrs.append(
            make_keyframes(
                f"auto-collision-{s.track_id}",
                CorrectionTarget(
                    TargetKind.ROOT_TRANSLATION, subject_track_id=s.track_id,
                ),
                (int(frames[0]), int(frames[-1])),
                key_frames=frames.astype(float),
                key_values=sd["new_transl"],
                note=(
                    f"auto collision push: capsule r={cfg.capsule_radius_m}m, "
                    f"strength={cfg.strength}, n_passes={cfg.n_passes}, "
                    f"cap={cfg.max_push_per_frame_m}m/frame"
                ),
            )
        )
        report.subjects_moved += 1

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report
