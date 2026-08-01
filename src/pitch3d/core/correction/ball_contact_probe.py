"""Ball-contact probe — how well does the ball touch player feet.

Physical constraint: the ball moves only when a player kicks it (or the
occasional collision). If the ball changes direction abruptly WITHOUT a
nearby player foot, the reconstruction is off — the ball is teleporting
or the tracking is wrong.

Metric per ball direction-change frame:

* nearest player's foot XY distance to the ball XY at that frame;
* if > ``contact_radius_m`` — flag as "orphan hit."

Measurement-only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from ..scene.scene import Scene, Subject
from .engine import resolve_ball, resolve_subject_motion

FootPositionProvider = Callable[[Subject], "np.ndarray | None"]


@dataclass(frozen=True)
class BallContactConfig:
    enabled: bool = False
    direction_change_deg: float = 45.0     # angle threshold for a "hit"
    contact_radius_m: float = 0.60         # foot-to-ball threshold
    min_speed_before_mps: float = 3.0      # ignore direction changes at rest


@dataclass
class BallContactReport:
    n_direction_changes: int = 0
    n_orphan_hits: int = 0
    total_players_checked: int = 0
    max_orphan_distance_m: float = 0.0
    orphan_frames: list[int] = field(default_factory=list)


def ball_contact_probe(
    scene: Scene,
    cfg: BallContactConfig | None = None,
    foot_position_provider: FootPositionProvider | None = None,
    *,
    fps: float = 25.0,
) -> BallContactReport:
    """Measure how many ball direction changes lack a nearby player foot."""
    cfg = cfg if cfg is not None else BallContactConfig()
    report = BallContactReport()
    if not cfg.enabled or scene.ball is None or foot_position_provider is None or fps <= 0:
        return report

    ball = resolve_ball(scene.ball, scene.corrections_for(None))
    b_frames = np.asarray(ball.frames, dtype=int)
    b_pos = np.asarray(ball.positions_3d, dtype=float)
    if b_frames.shape[0] < 3:
        return report

    dt = np.diff(b_frames.astype(float)) / fps
    ok = dt > 0
    vel = np.zeros((b_frames.shape[0] - 1, 3))
    vel[ok] = np.diff(b_pos, axis=0)[ok] / dt[ok, None]
    speed = np.linalg.norm(vel[:, :2], axis=1)

    # per-frame angle change between consecutive velocity vectors
    for i in range(1, vel.shape[0]):
        if speed[i - 1] < cfg.min_speed_before_mps or speed[i] < 0.1:
            continue
        cos_theta = float(np.dot(vel[i - 1, :2], vel[i, :2]) / (speed[i - 1] * speed[i] + 1e-9))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_deg = np.degrees(np.arccos(cos_theta))
        if angle_deg < cfg.direction_change_deg:
            continue
        report.n_direction_changes += 1
        # find closest player foot XY at this ball frame
        ball_frame = b_frames[i]
        ball_xy = b_pos[i, :2]
        nearest_dist = np.inf
        for s in scene.subjects:
            pos = foot_position_provider(replace(
                s, proposal=resolve_subject_motion(
                    s.proposal, scene.corrections_for(s.track_id),
                ),
            ))
            if pos is None:
                continue
            pose_frames = np.asarray(s.proposal.pose.frames, dtype=int)
            match = np.where(pose_frames == ball_frame)[0]
            if match.size == 0:
                continue
            player_xy = pos[int(match[0]), :2]
            d = float(np.linalg.norm(player_xy - ball_xy))
            nearest_dist = min(nearest_dist, d)
        report.total_players_checked += 1
        if nearest_dist > cfg.contact_radius_m:
            report.n_orphan_hits += 1
            report.orphan_frames.append(int(ball_frame))
            report.max_orphan_distance_m = max(
                report.max_orphan_distance_m, nearest_dist,
            )
    return report


__all__ = [
    "BallContactConfig",
    "BallContactReport",
    "FootPositionProvider",
    "ball_contact_probe",
]
