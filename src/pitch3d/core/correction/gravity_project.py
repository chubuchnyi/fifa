"""Gravity projection — force airborne subjects onto a ballistic Z parabola.

Correction sibling of :mod:`.gravity_probe`. For each airborne run
detected via the foot-position provider, we solve the parabolic
trajectory ``z(t) = z0 + vz0·t - 0.5·g·t²`` from the two run endpoints
and rewrite the root Z inside the run to that parabola. Preserves entry
and exit heights (no visible teleport at the edges).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from ..config.gates import GravityProjectConfig
from ..scene.layers import Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene, Subject
from .contact_probe import _find_runs
from .engine import make_keyframes, resolve_subject_motion

FootPositionProvider = Callable[[Subject], "np.ndarray | None"]

G_MPS2 = 9.81


@dataclass
class GravityProjectReport:
    n_subjects: int = 0
    subjects_corrected: int = 0
    corrections_added: int = 0
    runs_projected: int = 0
    max_shift_m: float = 0.0


def gravity_project_gate(
    scene: Scene,
    cfg: GravityProjectConfig | None = None,
    foot_position_provider: FootPositionProvider | None = None,
    *,
    fps: float = 25.0,
    floor_z: float = 0.0,
) -> tuple[Scene, GravityProjectReport]:
    cfg = cfg if cfg is not None else GravityProjectConfig()
    report = GravityProjectReport(n_subjects=len(scene.subjects))
    if not cfg.enabled or fps <= 0 or foot_position_provider is None:
        return scene, report

    auto_corrs: list[Correction] = []
    for s in scene.subjects:
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        resolved_subject = replace(s, proposal=resolved)
        pos = foot_position_provider(resolved_subject)
        if pos is None:
            continue
        pos = np.asarray(pos, dtype=float)
        frames = np.asarray(resolved.pose.frames, dtype=int)
        transl = np.asarray(resolved.pose.transl, dtype=float)
        n = pos.shape[0]
        if n != transl.shape[0] or n < 4:
            continue

        airborne_mask = pos[:, 2] > floor_z + cfg.airborne_z_threshold_m
        runs = _find_runs(airborne_mask, cfg.min_airborne_run_frames)
        if not runs:
            continue

        new_transl = transl.copy()
        n_projected = 0
        dt = 1.0 / fps
        for a, b in runs:
            # z0 = start, zf = end; solve vz0 s.t. z(t) parabola fits both endpoints
            L = b - a  # number of intervals
            if L < 2:
                continue
            z0 = float(transl[a, 2])
            zf = float(transl[b, 2])
            T_air = L * dt
            # z(T) = z0 + vz0 * T - 0.5 * g * T² = zf
            # → vz0 = (zf - z0 + 0.5 * g * T²) / T
            vz0 = (zf - z0 + 0.5 * G_MPS2 * T_air ** 2) / T_air
            ts = np.arange(L + 1) * dt
            new_z = z0 + vz0 * ts - 0.5 * G_MPS2 * ts ** 2
            new_transl[a:b + 1, 2] = new_z
            n_projected += 1

        dev = float(np.linalg.norm(new_transl - transl, axis=1).max())
        if dev < cfg.min_correction_m or n_projected == 0:
            continue
        report.subjects_corrected += 1
        report.runs_projected += n_projected
        report.max_shift_m = max(report.max_shift_m, dev)
        auto_corrs.append(
            make_keyframes(
                f"auto-gravity-project-{s.track_id}",
                CorrectionTarget(
                    TargetKind.ROOT_TRANSLATION, subject_track_id=s.track_id,
                ),
                (int(frames[0]), int(frames[-1])),
                key_frames=frames.astype(float),
                key_values=new_transl,
                note=(
                    f"auto gravity project: {n_projected} airborne run(s), "
                    f"max shift {dev:.3f}m"
                ),
            )
        )

    report.corrections_added = len(auto_corrs)
    if not auto_corrs:
        return scene, report
    return replace(scene, corrections=[*scene.corrections, *auto_corrs]), report


__all__ = [
    "GravityProjectConfig",
    "GravityProjectReport",
    "gravity_project_gate",
]
