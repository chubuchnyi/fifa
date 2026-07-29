"""Ball 2D→3D lift — honest mono ballistics (FR-9, R-4).

Recovering a ball's 3D position from a single view is genuinely under-determined in the
vertical axis (R-4): the image gives a ray, not a depth. An airborne ball appears *higher*
in the frame than its true ground point, so projecting that 2D sample onto the pitch plane
overshoots — it lands the ball metres beyond where it really is, often clean off the pitch
(#206). We resolve the depth the only honest ways available in mono:

* **Contact-anchored** (preferred, when player motions are supplied): a ball is only ever
  at a *known* world position when a player plays it — at that instant ball and foot share
  the same point. We find those contacts (the ball's 2D detection landing on a player's
  projected foot), anchor the ball's world XY to that foot, interpolate XY linearly between
  consecutive contacts (constant horizontal velocity) and pin a gravity parabola for Z. This
  trades the unknowable mono depth for *measured* player anchors (R-6), so the ball stays on
  the pitch and follows the actual kick→receive play instead of overshooting.
* **On the ground** (``on_ground``, no motions): the homography maps the 2D point straight
  onto plane ``Z = 0`` (a pure projective fact). Confidence is high.
* **Airborne, bracketed by two ground contacts**: fit the unique gravity parabola pinned to
  ``Z = 0`` at both contacts, interpolating world XY linearly between them. Confidence dips
  toward the apex, where the mono guess is weakest.
* **Airborne with no bracketing contact** (leading/trailing flight): we cannot pin a
  parabola, so we fall back to the ground projection with low confidence and flag it.

The per-frame ``height_confidence`` on :class:`BallTrack` is therefore a first-class output,
not an afterthought — it is exactly where this mono ambiguity is surfaced to the editor.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..scene.field import FieldCalibration
from ..scene.motion import Ball2DTrack, BallMode, BallTrack, SubjectMotion
from ..scene.units import GRAVITY

# Defaults tuned for ~1080p broadcast. ``CONTACT_PX`` is the image-space radius (px) within
# which the ball's 2D detection is taken to be *on* a player's projected foot — i.e. a
# contact; it scales with frame resolution. ``MAX_CONTACT_SPEED`` (m/s) rejects anchors that
# would require an impossible horizontal ball speed between contacts — these are spurious
# "nearest player" matches caused by a high airborne ball lining up with a distant player's
# image column (the depth ambiguity), not real touches.
CONTACT_PX = 140.0
MAX_CONTACT_SPEED = 35.0


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


def _stale_mask(positions_2d: np.ndarray) -> np.ndarray:
    """Frames whose 2D sample exactly repeats the previous one — the tracker froze.

    A frozen detection is not a real observation (the tracker lost the ball and held its
    last pixel), so it must never seed a contact: its ground projection is meaningless.
    """
    n = positions_2d.shape[0]
    stale = np.zeros(n, dtype=bool)
    if n > 1:
        stale[1:] = np.all(positions_2d[1:] == positions_2d[:-1], axis=1)
    return stale


def _nearest_player_per_frame(
    ball2d: Ball2DTrack,
    calibration: FieldCalibration,
    motions: Mapping[int, SubjectMotion],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each ball frame, the nearest player foot in *image* space.

    Returns ``(dist_px, track_id, world_xy)`` arrays over the ball's frames. A foot is the
    player's root XY (grounded by the homography) projected back into the image; the contact
    test compares it to the ball's 2D detection, so it is robust to the mono depth ambiguity
    at the instant of a touch (where ball and foot coincide in *both* image and world).
    """
    frames = ball2d.frames
    n = frames.shape[0]
    dist_px = np.full(n, np.inf)
    track_id = np.full(n, -1, dtype=int)
    world_xy = np.full((n, 2), np.nan)
    for i, f in enumerate(frames):
        uv = ball2d.positions_2d[i]
        for tid, motion in motions.items():
            ps = motion.pose
            hits = np.nonzero(ps.frames == int(f))[0]
            if hits.size == 0:
                continue
            foot_xy = ps.transl[int(hits[0]), :2]
            foot_uv = calibration.world_to_image(int(f), foot_xy)[0]
            d = float(np.hypot(foot_uv[0] - uv[0], foot_uv[1] - uv[1]))
            if d < dist_px[i]:
                dist_px[i] = d
                track_id[i] = int(tid)
                world_xy[i] = foot_xy
    return dist_px, track_id, world_xy


def detect_ball_contacts(
    ball2d: Ball2DTrack,
    calibration: FieldCalibration,
    motions: Mapping[int, SubjectMotion],
    *,
    contact_px: float = CONTACT_PX,
    max_speed_mps: float = MAX_CONTACT_SPEED,
    fps: float = 25.0,
) -> list[tuple[int, np.ndarray, int]]:
    """Find the ordered ball→player contacts that anchor the trajectory (#206).

    A contact is a non-stale frame whose ball detection lands within ``contact_px`` of a
    player's projected foot. We keep at most one anchor per player (its closest frame), then
    seed from the single strongest contact and extend outward in time, accepting a further
    contact only if the implied horizontal ball speed from the last accepted one is
    physically plausible (``≤ max_speed_mps``). That speed gate drops spurious matches where
    a high airborne ball merely lines up with a distant player's image column.

    Returns a time-ordered list of ``(frame_pos, world_xy, track_id)`` where ``frame_pos`` is
    the index into ``ball2d.frames`` — empty if no player is ever in contact.
    """
    frames = ball2d.frames
    n = frames.shape[0]
    if not motions or n == 0:
        return []

    stale = _stale_mask(ball2d.positions_2d)
    dist_px, track_id, world_xy = _nearest_player_per_frame(ball2d, calibration, motions)

    # one strongest (closest) contact per player
    best: dict[int, int] = {}
    for i in range(n):
        if stale[i] or dist_px[i] >= contact_px:
            continue
        tid = int(track_id[i])
        if tid not in best or dist_px[i] < dist_px[best[tid]]:
            best[tid] = i
    cand = sorted(best.values())  # frame positions, time-ordered
    if not cand:
        return []

    def speed(i: int, j: int) -> float:
        d = float(np.hypot(*(world_xy[i] - world_xy[j])))
        dt = abs(int(frames[i]) - int(frames[j])) / fps
        return d / dt if dt > 0 else np.inf

    seed = min(cand, key=lambda i: dist_px[i])
    s = cand.index(seed)
    accepted = {seed}
    last = seed
    for k in range(s - 1, -1, -1):  # extend backward in time
        if speed(cand[k], last) <= max_speed_mps:
            accepted.add(cand[k])
            last = cand[k]
    last = seed
    for k in range(s + 1, len(cand)):  # extend forward in time
        if speed(cand[k], last) <= max_speed_mps:
            accepted.add(cand[k])
            last = cand[k]
    return [(i, world_xy[i].copy(), int(track_id[i])) for i in sorted(accepted)]


def _lift_from_contacts(
    ball2d: Ball2DTrack,
    anchors: list[tuple[int, np.ndarray, int]],
    *,
    fps: float,
    gravity: float,
    airborne_confidence: float,
) -> BallTrack:
    """Build a 3D ball track from measured player-contact anchors (the #206 fix).

    World XY is pinned to each contacting player's foot, interpolated linearly between
    contacts and held flat before the first / after the last (those tails are genuinely
    unknown, so they carry the lowest confidence). Z is the gravity parabola pinned to
    ``Z = 0`` at each contact.
    """
    frames = ball2d.frames
    n = frames.shape[0]
    pos = np.zeros((n, 3))
    geo_conf = np.full(n, airborne_confidence * 0.5)  # lead/trail: unknown ⇒ low
    # The held lead/trail tails stay UNMEASURED: they have no bracketing contact, so their
    # height is not a ballistic estimate — it is a hold. Only a fitted arc earns BALLISTIC.
    mode = np.full(n, BallMode.UNMEASURED.value, dtype=object)

    aks = [a[0] for a in anchors]
    axy = {a[0]: a[1] for a in anchors}
    pos[: aks[0], :2] = axy[aks[0]]  # hold before first contact
    pos[aks[-1] + 1 :, :2] = axy[aks[-1]]  # hold after last contact
    for a in aks:
        pos[a, :2] = axy[a]
        geo_conf[a] = 1.0
        mode[a] = BallMode.ON_GROUND.value

    for a, b in zip(aks[:-1], aks[1:], strict=False):
        if b <= a + 1:
            continue  # adjacent contacts: nothing airborne between
        seg = np.arange(a + 1, b)
        mode[seg] = BallMode.BALLISTIC.value
        f0, f1 = float(frames[a]), float(frames[b])
        u = (frames[seg].astype(float) - f0) / (f1 - f0)  # 0..1 across the arc
        pos[seg, :2] = (1.0 - u)[:, None] * axy[a] + u[:, None] * axy[b]
        flight_s = (f1 - f0) / fps
        pos[seg, 2] = ballistic_z((frames[seg].astype(float) - f0) / fps, flight_s, gravity)
        geo_conf[seg] = 1.0 - (1.0 - airborne_confidence) * (4.0 * u * (1.0 - u))

    conf = np.clip(geo_conf * ball2d.confidence, 0.0, 1.0)
    return BallTrack(
        frames=frames,
        positions_3d=pos,
        height_confidence=conf,
        track_2d=ball2d.positions_2d,
        mode=mode,
    )


def lift_ball_to_3d(
    ball2d: Ball2DTrack,
    calibration: FieldCalibration,
    *,
    on_ground: np.ndarray | None = None,
    motions: Mapping[int, SubjectMotion] | None = None,
    fps: float = 25.0,
    gravity: float = GRAVITY,
    airborne_confidence: float = 0.5,
    contact_px: float = CONTACT_PX,
    max_speed_mps: float = MAX_CONTACT_SPEED,
) -> BallTrack:
    """Lift a 2D ball track to a 3D :class:`BallTrack` (Z-up, meters).

    Args:
        ball2d: Raw 2D track from a :class:`~pitch3d.core.ports.perception.BallTracker`.
        calibration: Per-frame image→pitch-plane homography (the world anchor).
        on_ground: Optional per-frame bool of ground contact. ``None`` ⇒ treat every frame
            as grounded (the honest default when no contact segmentation is available: the
            system believes the ball is on the pitch and says so via confidence).
        motions: Optional per-subject motions. When supplied and at least one ball→player
            contact is found, the ball is **contact-anchored** (the #206 fix): its world XY
            is pinned to the contacting players' feet rather than projected onto the plane,
            which keeps an airborne ball on the pitch instead of overshooting. Falls back to
            the mono ``on_ground`` path when no contact is found.
        fps: Frame rate, for converting frame gaps to seconds in the ballistics.
        gravity: m/s² (world ``-Z``).
        airborne_confidence: Confidence at the apex of a bracketed flight ``[0, 1]``.
        contact_px: Image-space contact radius (px); see :func:`detect_ball_contacts`.
        max_speed_mps: Plausible-speed gate for contacts; see :func:`detect_ball_contacts`.

    Returns:
        A :class:`BallTrack` with ``positions_3d``, per-frame ``height_confidence``
        (detection confidence × geometric confidence), the original 2D track, and a
        per-frame :class:`~pitch3d.core.scene.motion.BallMode`. Frames with no bracketing
        contact stay ``UNMEASURED`` — their Z is a hold, not an estimate.
    """
    if motions:
        anchors = detect_ball_contacts(
            ball2d,
            calibration,
            motions,
            contact_px=contact_px,
            max_speed_mps=max_speed_mps,
            fps=fps,
        )
        if anchors:
            return _lift_from_contacts(
                ball2d,
                anchors,
                fps=fps,
                gravity=gravity,
                airborne_confidence=airborne_confidence,
            )

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
    mode = np.where(og, BallMode.ON_GROUND.value, BallMode.UNMEASURED.value).astype(object)

    contacts = np.nonzero(og)[0]
    if contacts.size >= 2:
        for a, b in zip(contacts[:-1], contacts[1:], strict=False):
            if b <= a + 1:
                continue  # adjacent contacts: no airborne frames between
            seg = np.arange(a + 1, b)
            mode[seg] = BallMode.BALLISTIC.value
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
        mode=mode,
    )
