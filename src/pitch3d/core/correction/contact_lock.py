"""Step 3b — foot contact-lock: freeze foot XY during stance phases.

Real-scene measurement (contact_probe on out/anim_safe_new_plant_v2/scene.json):
23/23 subjects slide 2-6m during their stance phases. This module emits
one dense ``KEYFRAME_INTERP`` ``ROOT_TRANSLATION`` correction per subject
that zeros the foot XY drift over each detected stance run, preserving
the foot's initial world XY within the run.

Algorithm (frame-by-frame Jacobi, deterministic, no L-BFGS dep):

* For each contact run ``[a, b]``:
  * target foot XY = position at row ``a`` (the first stance frame — the
    anchor). During the run the foot MUST stay there.
  * per-frame slide = foot_xy(t) - foot_xy(a). To zero it we shift root
    by ``-slide``: ``new_root_xy(t) = root_xy(t) - (foot_xy(t) - foot_xy(a))``.

* Between stance runs (swing / air-borne frames) root XY is untouched
  (subject legitimately moves through the air).

* Runs where the anchor drift is below ``min_correction_m`` don't emit —
  micro-jitter isn't worth a correction.

The emit is R-6-clean: TeleportEvent-class jumps that the M3-9 gate
already preserved are NOT locked (contact_probe won't see a stationary
foot for those anyway — the huge XY jump breaks the contact window).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..config.gates import ContactProbeConfig
from ..scene.layers import Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene
from .contact_probe import FootPositionProvider, _find_runs
from .engine import make_keyframes, resolve_subject_motion


@dataclass
class ContactLockReport:
    """What the lock did per subject + aggregate."""

    n_subjects: int = 0
    subjects_corrected: int = 0
    runs_locked: int = 0
    corrections_added: int = 0
    total_slide_before_m: float = 0.0
    total_slide_after_m: float = 0.0
    max_shift_m: float = 0.0


def contact_lock_gate(
    scene: Scene,
    cfg: ContactProbeConfig | None = None,
    foot_position_provider: FootPositionProvider | None = None,
    floor_z: float = 0.0,
    min_correction_m: float = 1e-3,
) -> tuple[Scene, ContactLockReport]:
    """Zero foot slide during stance phases; return NEW scene + report.

    * ``cfg is None`` / ``cfg.enabled is False`` / ``foot_position_provider
      is None`` → passthrough (no corrections).
    * Emits ONE dense ``KEYFRAME_INTERP`` per subject whose combined slide
      exceeds ``min_correction_m``.
    """
    cfg = cfg if cfg is not None else ContactProbeConfig()
    report = ContactLockReport(n_subjects=len(scene.subjects))
    if not cfg.enabled or foot_position_provider is None:
        return scene, report

    auto_corrs: list[Correction] = []
    for s in scene.subjects:
        corrs = list(scene.corrections_for(s.track_id))
        resolved = resolve_subject_motion(s.proposal, corrs)
        resolved_subject = replace(s, proposal=resolved)
        pos = foot_position_provider(resolved_subject)
        if pos is None:
            continue
        pos = np.asarray(pos, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            continue
        frames = np.asarray(resolved.pose.frames, dtype=int)
        transl = np.asarray(resolved.pose.transl, dtype=float)
        if pos.shape[0] != transl.shape[0]:
            continue
        contact_mask = pos[:, 2] <= floor_z + cfg.contact_z_threshold_m
        runs = _find_runs(contact_mask, cfg.min_contact_run_frames)
        if not runs:
            continue

        new_transl = transl.copy()
        subject_slide_before = 0.0
        subject_slide_after = 0.0
        n_locked = 0
        for a, b in runs:
            anchor_xy = pos[a, :2].copy()
            slide_xy = pos[a:b + 1, :2] - anchor_xy   # (L, 2) per-frame drift
            slide_max = float(np.linalg.norm(slide_xy, axis=1).max())
            subject_slide_before += slide_max
            if slide_max < cfg.slide_threshold_m:
                subject_slide_after += slide_max
                continue
            # shift root by -slide over the run rows
            new_transl[a:b + 1, :2] -= slide_xy
            n_locked += 1

        dev = float(np.linalg.norm(new_transl - transl, axis=1).max())
        if dev < min_correction_m or n_locked == 0:
            continue
        report.subjects_corrected += 1
        report.runs_locked += n_locked
        report.total_slide_before_m += subject_slide_before
        report.total_slide_after_m += subject_slide_after
        report.max_shift_m = max(report.max_shift_m, dev)
        auto_corrs.append(
            make_keyframes(
                f"auto-contact-lock-{s.track_id}",
                CorrectionTarget(
                    TargetKind.ROOT_TRANSLATION, subject_track_id=s.track_id,
                ),
                (int(frames[0]), int(frames[-1])),
                key_frames=frames.astype(float),
                key_values=new_transl,
                note=(
                    f"auto contact-lock: {n_locked} stance run(s), "
                    f"max shift {dev:.3f}m"
                ),
            )
        )

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report


__all__ = [
    "ContactLockReport",
    "contact_lock_gate",
]
