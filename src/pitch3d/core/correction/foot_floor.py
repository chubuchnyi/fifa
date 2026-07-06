"""Foot-floor gate: clamp root Z above a physical floor, flag constant-Z plateaus (T1a, R-6).

Measured need (§A of ``docs/research/2026-07-06-player-physics.md``): user reports
"players float in the air." Two distinct failure modes hide behind that eye-note:

1. **Below-floor drift** — the resolved root Z dips under the pitch plane. Any
   occluded-frame extrapolation, a homography with metric bias, or a raw HMR
   output with negative foot depth can produce this. Clamp is the right fix.
2. **Constant-Z plateau** — the root Z is a flat plateau at ``pelvis_height_m``
   for the whole run. This is what happens when the backend's
   ``pelvis_above_foot`` is ``None`` (the fake HMR path today, plus any real
   backend that doesn't emit foot-plane FK). We can't invent oscillation, but
   the plateau IS the "helicopter" symptom the user sees — so we surface it as
   an R-6 warning in :class:`FootFloorReport`, never silently fix it.

The gate emits ONE dense ``KEYFRAME_INTERP`` ``ROOT_TRANSLATION`` correction
per subject through the ADR-0002 seam — inspectable, disable-able, layered on
top of the proposal + earlier corrections. Read-once ``FootFloorConfig`` comes
from ``config/physics.yaml`` (never a hidden constant).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..config.gates import FootFloorConfig
from ..scene.layers import Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene
from .engine import make_keyframes, resolve_subject_motion

#: Pelvis-above-floor nominal — mirrors :mod:`pitch3d.adapters.models.pose`.
#: This is the height a standing player's SMPL-X root sits above their feet;
#: for the floor clamp we require Z ≥ floor_m + this so feet stay on the pitch.
_PELVIS_HEIGHT_M = 0.92

#: A run whose Z std is below this reads as a plateau (constant Z, no stride/crouch).
_PLATEAU_STD_M = 0.02


@dataclass
class SubjectFootReport:
    """Per-subject foot-floor diagnostics."""

    track_id: int
    n_frames: int = 0
    below_floor_frames: int = 0     # frames with root Z < floor_m + pelvis_height
    max_below_m: float = 0.0        # deepest sink below the floor
    hover_frames: int = 0           # frames with root Z > floor + pelvis + warn_hover
    z_min: float = 0.0
    z_max: float = 0.0
    z_std: float = 0.0
    plateau: bool = False           # z_std < _PLATEAU_STD_M → constant-Z symptom
    corrected: bool = False


@dataclass
class FootFloorReport:
    """Aggregate + per-subject foot-floor gate report."""

    n_subjects: int = 0
    subjects_corrected: int = 0
    subjects_below_floor: int = 0
    subjects_hovering: int = 0
    subjects_plateau: int = 0
    corrections_added: int = 0
    subjects: list[SubjectFootReport] = field(default_factory=list)


def _clamp_z_track(z: np.ndarray, floor_z: float) -> np.ndarray:
    """Project Z onto ``z >= floor_z``. Below-floor rows lift to the floor exactly."""
    return np.maximum(np.asarray(z, dtype=float), float(floor_z))


def foot_floor_gate(
    scene: Scene, cfg: FootFloorConfig | None = None
) -> tuple[Scene, FootFloorReport]:
    """Clamp root Z below the floor, flag plateaus; return NEW scene + report.

    * When ``cfg is None`` or ``cfg.enabled is False``: measure only, emit no
      corrections. The report still surfaces plateau / hover / below-floor
      counts so the operator can decide whether to enable the clamp.
    * When ``cfg.enabled is True``: emit ONE ``KEYFRAME_INTERP`` per subject
      whose resolved Z sinks under ``cfg.floor_m + _PELVIS_HEIGHT_M`` (the
      pelvis-above-floor floor), covering that subject's whole visible span.
    """
    cfg = cfg if cfg is not None else FootFloorConfig()
    report = FootFloorReport(n_subjects=len(scene.subjects))
    auto_corrs: list[Correction] = []
    floor_z = float(cfg.floor_m) + _PELVIS_HEIGHT_M
    hover_z = floor_z + float(cfg.warn_hover_m)

    for s in scene.subjects:
        corrs = list(scene.corrections_for(s.track_id))
        resolved = resolve_subject_motion(s.proposal, corrs)
        frames = np.asarray(resolved.pose.frames, dtype=int)
        transl = np.asarray(resolved.pose.transl, dtype=float)
        z = transl[:, 2] if transl.size else np.zeros(0)
        r = SubjectFootReport(track_id=int(s.track_id), n_frames=int(z.size))
        if z.size == 0:
            report.subjects.append(r)
            continue
        r.z_min = float(z.min())
        r.z_max = float(z.max())
        r.z_std = float(z.std())
        below = z < floor_z
        r.below_floor_frames = int(below.sum())
        r.max_below_m = float((floor_z - z)[below].max()) if r.below_floor_frames else 0.0
        r.hover_frames = int((z > hover_z).sum())
        r.plateau = r.z_std < _PLATEAU_STD_M and r.n_frames >= 3

        if r.below_floor_frames:
            report.subjects_below_floor += 1
        if r.hover_frames:
            report.subjects_hovering += 1
        if r.plateau:
            report.subjects_plateau += 1

        if cfg.enabled and r.below_floor_frames > 0:
            new_z = _clamp_z_track(z, floor_z)
            if not np.array_equal(new_z, z):
                new_transl = transl.copy()
                new_transl[:, 2] = new_z
                auto_corrs.append(
                    make_keyframes(
                        f"auto-foot-floor-{s.track_id}",
                        CorrectionTarget(
                            TargetKind.ROOT_TRANSLATION,
                            subject_track_id=s.track_id,
                        ),
                        (int(frames[0]), int(frames[-1])),
                        key_frames=frames.astype(float),
                        key_values=new_transl,
                        note=(
                            f"auto foot-floor clamp: Z >= {floor_z:.3f}m "
                            f"(floor {cfg.floor_m:.3f} + pelvis {_PELVIS_HEIGHT_M:.2f})"
                        ),
                    )
                )
                r.corrected = True
                report.subjects_corrected += 1
        report.subjects.append(r)

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report
