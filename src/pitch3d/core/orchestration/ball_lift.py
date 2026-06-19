"""Ball 2D→3D lift — honest mono ballistics (FR-9, R-4).

Recovering a ball's 3D position from a single view is genuinely under-determined in the
vertical axis (R-4): the image gives a ray, not a depth. We resolve it the only honest way
in mono:

* **On the ground** (``on_ground``): the field homography maps the 2D point straight onto
  the pitch plane ``Z = 0`` (a pure projective fact, no model). Confidence is high.
* **Airborne, bracketed by two ground contacts**: fit the unique gravity parabola pinned to
  ``Z = 0`` at both contacts, and interpolate world XY linearly between them (constant
  horizontal velocity — gravity acts only on Z). Confidence dips toward the apex, where the
  mono guess is weakest.
* **Airborne with no bracketing contact** (leading/trailing flight): we cannot pin a
  parabola, so we fall back to the ground projection with low confidence and flag it.

The per-frame ``height_confidence`` on :class:`BallTrack` is therefore a first-class output,
not an afterthought — it is exactly where this mono ambiguity is surfaced to the editor.
"""

from __future__ import annotations

import numpy as np

from ..scene.field import FieldCalibration
from ..scene.motion import Ball2DTrack, BallTrack
from ..scene.units import GRAVITY


def ballistic_z(elapsed_s: np.ndarray, flight_s: float, gravity: float = GRAVITY) -> np.ndarray:
    """Height of a projectile pinned to ``Z = 0`` at ``t=0`` and ``t=flight_s``.

    Solves ``Z(t) = v0·t − ½g t²`` with ``Z(0)=Z(flight_s)=0`` ⇒ ``v0 = ½ g·flight_s``.
    Peak is at the midpoint with height ``g·flight_s²/8``.
    """
    t = np.asarray(elapsed_s, dtype=float)
    if flight_s <= 0:
        return np.zeros_like(t)
    v0 = 0.5 * gravity * flight_s
    return np.clip(v0 * t - 0.5 * gravity * t * t, 0.0, None)


def _ground_projection(ball2d: Ball2DTrack, calibration: FieldCalibration) -> np.ndarray:
    """World-plane XY for every 2D sample (valid as-is only for grounded frames)."""
    xy = np.empty((ball2d.frames.shape[0], 2))
    for i, f in enumerate(ball2d.frames):
        xy[i] = calibration.image_to_world(int(f), ball2d.positions_2d[i])[0]
    return xy


def lift_ball_to_3d(
    ball2d: Ball2DTrack,
    calibration: FieldCalibration,
    *,
    on_ground: np.ndarray | None = None,
    fps: float = 25.0,
    gravity: float = GRAVITY,
    airborne_confidence: float = 0.5,
) -> BallTrack:
    """Lift a 2D ball track to a 3D :class:`BallTrack` (Z-up, meters).

    Args:
        ball2d: Raw 2D track from a :class:`~pitch3d.core.ports.perception.BallTracker`.
        calibration: Per-frame image→pitch-plane homography (the world anchor).
        on_ground: Optional per-frame bool of ground contact. ``None`` ⇒ treat every frame
            as grounded (the honest default when no contact segmentation is available: the
            system believes the ball is on the pitch and says so via confidence).
        fps: Frame rate, for converting frame gaps to seconds in the ballistics.
        gravity: m/s² (world ``-Z``).
        airborne_confidence: Confidence at the apex of a bracketed flight ``[0, 1]``.

    Returns:
        A :class:`BallTrack` with ``positions_3d``, per-frame ``height_confidence``
        (detection confidence × geometric confidence), the original 2D track, and the
        resolved ``on_ground`` mask.
    """
    frames = ball2d.frames
    n = frames.shape[0]
    xy = _ground_projection(ball2d, calibration)
    og = (
        np.ones(n, dtype=bool)
        if on_ground is None
        else np.asarray(on_ground, dtype=bool).reshape(n)
    )

    pos = np.zeros((n, 3))
    pos[:, :2] = xy  # ground projection is the XY default everywhere
    geo_conf = np.where(og, 1.0, airborne_confidence)

    contacts = np.nonzero(og)[0]
    if contacts.size >= 2:
        for a, b in zip(contacts[:-1], contacts[1:], strict=False):
            if b <= a + 1:
                continue  # adjacent contacts: no airborne frames between
            seg = np.arange(a + 1, b)
            f0, f1 = float(frames[a]), float(frames[b])
            u = (frames[seg].astype(float) - f0) / (f1 - f0)  # 0..1 across the arc
            pos[seg, :2] = (1.0 - u)[:, None] * xy[a] + u[:, None] * xy[b]
            flight_s = (f1 - f0) / fps
            pos[seg, 2] = ballistic_z((frames[seg].astype(float) - f0) / fps, flight_s, gravity)
            # confidence dips to airborne_confidence at the apex, 1 at the contacts
            geo_conf[seg] = 1.0 - (1.0 - airborne_confidence) * (4.0 * u * (1.0 - u))

    # leading / trailing airborne frames have no bracketing contact: cannot pin a parabola
    if contacts.size:
        lead, trail = slice(0, contacts[0]), slice(contacts[-1] + 1, n)
        geo_conf[lead] = np.minimum(geo_conf[lead], airborne_confidence * 0.5)
        geo_conf[trail] = np.minimum(geo_conf[trail], airborne_confidence * 0.5)
    elif not og.any():
        geo_conf[:] = airborne_confidence * 0.5  # no contact anywhere: all guesses are weak

    conf = np.clip(geo_conf * ball2d.confidence, 0.0, 1.0)
    return BallTrack(
        frames=frames,
        positions_3d=pos,
        height_confidence=conf,
        track_2d=ball2d.positions_2d,
        on_ground=og,
    )
