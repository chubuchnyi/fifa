"""Ball gravity probe — airborne ball must obey ~ -9.81 m/s² vertical accel.

The ball_lift.py contact-anchoring pins the ball to feet during contact;
between contacts we get a "ballistic" segment. This probe measures whether
that segment obeys gravity — i.e. z(t) is a downward parabola.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scene.scene import Scene
from .engine import resolve_ball


G_MPS2 = 9.81


@dataclass(frozen=True)
class BallGravityConfig:
    enabled: bool = False
    tolerance_mps2: float = 5.0     # tol around -9.81


@dataclass
class BallGravityReport:
    n_frames: int = 0
    mean_vertical_accel_mps2: float = 0.0
    max_deviation_mps2: float = 0.0
    is_violating: bool = False


def ball_gravity_probe(
    scene: Scene, cfg: BallGravityConfig | None = None, *, fps: float = 25.0,
) -> BallGravityReport:
    cfg = cfg if cfg is not None else BallGravityConfig()
    report = BallGravityReport()
    if not cfg.enabled or scene.ball is None or fps <= 0:
        return report

    ball = resolve_ball(scene.ball, scene.corrections_for(None))
    z = np.asarray(ball.positions_3d, dtype=float)[:, 2]
    n = z.shape[0]
    report.n_frames = n
    if n < 4:
        return report
    dt = 1.0 / fps
    vz = np.diff(z) / dt
    az = np.diff(vz) / dt
    if az.size == 0:
        return report
    report.mean_vertical_accel_mps2 = float(az.mean())
    report.max_deviation_mps2 = float(
        np.abs(az - (-G_MPS2)).max(),
    )
    report.is_violating = report.max_deviation_mps2 > cfg.tolerance_mps2
    return report


__all__ = [
    "BallGravityConfig",
    "BallGravityReport",
    "ball_gravity_probe",
]
