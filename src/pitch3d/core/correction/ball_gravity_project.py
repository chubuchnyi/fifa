"""Ball gravity project — force airborne ball Z segments onto a parabola.

Between contact_lift anchor points, the ball is in free flight and Z must
follow ``z(t) = z0 + vz0·t − 0.5·g·t²``. This gate detects airborne
segments (Z above a floor threshold) and rewrites the interior Z to match
the parabola between the two endpoints of each segment.

Preserves the endpoints so the contact-anchored points stay fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..scene.layers import Correction, CorrectionTarget, TargetKind
from ..scene.scene import Scene
from .contact_probe import _find_runs
from .engine import make_keyframes, resolve_ball


G_MPS2 = 9.81


@dataclass(frozen=True)
class BallGravityProjectConfig:
    enabled: bool = False
    airborne_z_threshold_m: float = 0.15   # ball airborne when Z above this
    min_airborne_run_frames: int = 3
    min_correction_m: float = 1e-3


@dataclass
class BallGravityProjectReport:
    n_frames: int = 0
    n_airborne_runs: int = 0
    runs_projected: int = 0
    max_shift_m: float = 0.0
    correction_added: bool = False


def ball_gravity_project_gate(
    scene: Scene, cfg: BallGravityProjectConfig | None = None,
    *, fps: float = 25.0,
) -> tuple[Scene, BallGravityProjectReport]:
    cfg = cfg if cfg is not None else BallGravityProjectConfig()
    report = BallGravityProjectReport()
    if not cfg.enabled or scene.ball is None or fps <= 0:
        return scene, report

    ball = resolve_ball(scene.ball, scene.corrections_for(None))
    frames = np.asarray(ball.frames, dtype=int)
    pos = np.asarray(ball.positions_3d, dtype=float)
    n = frames.shape[0]
    report.n_frames = n
    if n < 4:
        return scene, report

    airborne = pos[:, 2] > cfg.airborne_z_threshold_m
    runs = _find_runs(airborne, cfg.min_airborne_run_frames)
    report.n_airborne_runs = len(runs)
    if not runs:
        return scene, report

    new_pos = pos.copy()
    n_projected = 0
    dt = 1.0 / fps
    for a, b in runs:
        L = b - a
        if L < 2:
            continue
        z0 = float(pos[a, 2])
        zf = float(pos[b, 2])
        T = L * dt
        # z(T) = z0 + vz0·T − 0.5·g·T² = zf → vz0 = (zf − z0 + 0.5·g·T²) / T
        vz0 = (zf - z0 + 0.5 * G_MPS2 * T ** 2) / T
        ts = np.arange(L + 1) * dt
        new_z = z0 + vz0 * ts - 0.5 * G_MPS2 * ts ** 2
        new_pos[a:b + 1, 2] = new_z
        n_projected += 1

    dev = float(np.linalg.norm(new_pos - pos, axis=1).max())
    if dev < cfg.min_correction_m or n_projected == 0:
        return scene, report
    report.runs_projected = n_projected
    report.max_shift_m = dev

    # Emit a KEYFRAME_INTERP on the BALL_POSITION target.
    from .engine import make_keyframes
    new_corr = make_keyframes(
        "auto-ball-gravity-project",
        CorrectionTarget(TargetKind.BALL_POSITION, subject_track_id=None),
        (int(frames[0]), int(frames[-1])),
        key_frames=frames.astype(float),
        key_values=new_pos,
        note=(
            f"auto ball gravity project: {n_projected} airborne run(s), "
            f"max shift {dev:.3f}m"
        ),
    )
    report.correction_added = True
    return replace(scene, corrections=[*scene.corrections, new_corr]), report


__all__ = [
    "BallGravityProjectConfig",
    "BallGravityProjectReport",
    "ball_gravity_project_gate",
]
