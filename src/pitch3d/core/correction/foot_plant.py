"""Foot plant gate (T6.a, R-6): recenter root Z so subjects don't hover.

Measured need (user report 2026-07-06 after safe_new pod run): "all humans
are hovering." T1a ``foot_floor`` clamps ``Z >= floor + pelvis_height``, but
on the real SMPLest-X path the reported ``pelvis_above_foot`` is a
systematically-high value for the *whole* track — so no frame is under the
floor and T1a never fires. The eye still reads "helicopter." The proper fix
is either per-frame SMPL-X FK (T6.a v2, deferred) or, cheaply, a
**median-lock** that removes only the constant bias while preserving the
stride/jump variance we DO want.

Design mirrors :mod:`.foot_floor` and :mod:`.kinematics`:

* ``mode="off"`` — measure only, no corrections.
* ``mode="median_lock"`` — per subject: shift the whole Z track by
  ``target_pelvis_m - median(z)`` when the offset is > ``bias_threshold_m``.
  Stride amplitude is preserved (relative Z variation unchanged).
* ``mode="hard_lock"`` — clamp every frame's Z to ``target_pelvis_m``.
  Kills legitimate jumps; use for debug only.

Emits ONE dense ``KEYFRAME_INTERP`` ``ROOT_TRANSLATION`` correction per
subject whose max deviation exceeds ``min_correction_m``. Layered through
ADR-0002. Reads its config from ``PhysicsConfig.foot_plant`` — no hidden
constants here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from ..config.gates import FootPlantConfig
from ..scene.layers import Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene, Subject
from .engine import make_keyframes, resolve_subject_motion

VALID_MODES = ("off", "median_lock", "hard_lock")

#: Callable returning per-frame ``pelvis_above_foot`` (m) for a subject, or ``None``.
#: When present the gate uses ``median(offsets)`` as the per-subject target,
#: replacing the shared ``cfg.target_pelvis_m`` for that subject. This is what
#: closes the hover complaint properly — each player's standing offset comes
#: from SMPL-X FK on their betas + pose, not a nominal 0.92 m constant (T6a v2).
PelvisTargetProvider = Callable[[Subject], "np.ndarray | float | None"]


@dataclass
class SubjectPlantReport:
    track_id: int
    n_frames: int = 0
    z_min_before: float = 0.0
    z_max_before: float = 0.0
    z_median_before: float = 0.0
    z_bias_m: float = 0.0            # target - median (signed shift applied)
    target_used_m: float = 0.0       # per-subject target chosen (measured or cfg)
    target_source: str = "cfg"       # "cfg" | "provider"
    corrected: bool = False


@dataclass
class FootPlantReport:
    n_subjects: int = 0
    subjects_corrected: int = 0
    subjects_using_provider: int = 0
    corrections_added: int = 0
    max_shift_m: float = 0.0
    max_abs_bias_m: float = 0.0
    subjects: list[SubjectPlantReport] = field(default_factory=list)


def foot_plant_gate(
    scene: Scene, cfg: FootPlantConfig | None = None,
    *, pelvis_target_provider: PelvisTargetProvider | None = None,
) -> tuple[Scene, FootPlantReport]:
    """Recenter root Z to remove the systematic hover bias; return NEW scene + report.

    * ``cfg is None`` or ``cfg.enabled is False`` or ``cfg.mode == "off"``:
      measure-only — the report still surfaces each subject's ``z_bias_m``
      so the operator can decide whether to enable.
    * Enabled: emits ONE ``KEYFRAME_INTERP`` per subject whose net shift
      exceeds ``cfg.min_correction_m``.
    * ``pelvis_target_provider`` (T6.a stage A): callable ``Subject → offset``
      returning per-subject measured pelvis-above-foot (either a scalar or
      a per-frame ``(T,)`` array whose median is used). When set, each
      subject's target is that measurement instead of the shared
      ``cfg.target_pelvis_m`` — closes the hover complaint per-player,
      accounting for different betas / standing heights.
    """
    cfg = cfg if cfg is not None else FootPlantConfig()
    if cfg.mode not in VALID_MODES:
        raise ValueError(f"foot_plant mode={cfg.mode!r} not in {VALID_MODES}")
    report = FootPlantReport(n_subjects=len(scene.subjects))
    auto_corrs: list[Correction] = []
    cfg_target = float(cfg.target_pelvis_m)

    for s in scene.subjects:
        # per-subject target: measured (provider) or the shared cfg default
        target_source = "cfg"
        target = cfg_target
        if pelvis_target_provider is not None:
            offset = pelvis_target_provider(s)
            if offset is not None:
                arr = np.asarray(offset, dtype=float).reshape(-1)
                if arr.size:
                    target = float(np.median(arr))
                    target_source = "provider"
                    report.subjects_using_provider += 1
        corrs = list(scene.corrections_for(s.track_id))
        resolved = resolve_subject_motion(s.proposal, corrs)
        frames = np.asarray(resolved.pose.frames, dtype=int)
        transl = np.asarray(resolved.pose.transl, dtype=float)
        if transl.shape[0] == 0:
            report.subjects.append(SubjectPlantReport(track_id=int(s.track_id)))
            continue
        z = transl[:, 2]
        med = float(np.median(z))
        bias = target - med
        r = SubjectPlantReport(
            track_id=int(s.track_id),
            n_frames=int(z.size),
            z_min_before=float(z.min()),
            z_max_before=float(z.max()),
            z_median_before=med,
            z_bias_m=float(bias),
            target_used_m=float(target),
            target_source=target_source,
        )
        report.max_abs_bias_m = max(report.max_abs_bias_m, abs(bias))

        if not cfg.enabled or cfg.mode == "off":
            report.subjects.append(r)
            continue

        if cfg.mode == "median_lock":
            if abs(bias) < cfg.bias_threshold_m:
                report.subjects.append(r)
                continue
            new_transl = transl.copy()
            new_transl[:, 2] = z + bias
        elif cfg.mode == "hard_lock":
            new_transl = transl.copy()
            new_transl[:, 2] = target
        else:
            report.subjects.append(r)
            continue

        dev = float(np.linalg.norm(new_transl - transl, axis=1).max())
        if dev < cfg.min_correction_m:
            report.subjects.append(r)
            continue

        auto_corrs.append(
            make_keyframes(
                f"auto-foot-plant-{s.track_id}",
                CorrectionTarget(
                    TargetKind.ROOT_TRANSLATION, subject_track_id=s.track_id,
                ),
                (int(frames[0]), int(frames[-1])),
                key_frames=frames.astype(float),
                key_values=new_transl,
                note=(
                    f"auto foot plant ({cfg.mode}): target Z {target:.3f}m "
                    f"(median before {med:.3f}m, bias {bias:+.3f}m)"
                ),
            )
        )
        r.corrected = True
        report.subjects_corrected += 1
        report.max_shift_m = max(report.max_shift_m, abs(bias))
        report.subjects.append(r)

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report
