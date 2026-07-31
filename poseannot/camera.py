"""Camera projection utilities — world (z-up) → pixel.

Wraps ``scene.camera``:
    intrinsics: CameraIntrinsics(fx, fy, cx, cy, ...)
    rotation_quat: (T, 4) quaternion (world→camera)
    translation: (T, 3)

Also handles the 2026-06-30 finding recorded in memory: the calibrated
CameraTrack produces coords for a 180°-rolled frame; if we detect the
"camera-flipped" case we compose a 180° roll before projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ProjectedFrame:
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray             # (3, 3) world → camera
    t: np.ndarray             # (3,)
    frame_index: int
    frame_flipped: bool = False


@dataclass(frozen=True)
class CameraAdjust:
    """Manual overlay-camera nudge (the GUI's camera-controls panel).

    The solved PnLCalib camera is only approximately right (per-player offset +
    ~3× scale, task #61); these deltas let the user hand-align the projected
    overlay to the video without touching the stored calibration. All applied on
    top of the auto-solved (and 180-flip-corrected) projection:

      zoom  — focal multiplier about the principal point (>1 = overlay bigger)
      panx/pany — shift the principal point in pixels (translate the overlay)
      yaw/pitch/roll — degrees, rotate the camera about its down / right / view axes
      dolly — metres along the view axis (+ = push the scene farther / smaller)
    """

    zoom: float = 1.0
    panx: float = 0.0
    pany: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    dolly: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (
            self.zoom == 1.0 and self.panx == 0.0 and self.pany == 0.0
            and self.yaw == 0.0 and self.pitch == 0.0 and self.roll == 0.0
            and self.dolly == 0.0
        )


def frame_projector(
    camera_track, frame_index: int,
    video_size: tuple[int, int] | None = None,
    adjust: "CameraAdjust | None" = None,
) -> ProjectedFrame:
    """Return the per-frame projection package for a given clip frame.

    ``video_size`` — the size ``(width, height)`` of the frame the caller will
    overlay onto. When given and it differs from the calibration size
    (inferred as 2·cx by 2·cy), intrinsics are scaled so projected pixels
    land in the caller's coordinate system.  This handles the common case
    of PnLCalib being run at 1280×720 while the video is 1920×1080.
    """
    idx = int(frame_index)
    q = np.asarray(camera_track.rotation_quat[idx], dtype=float)
    R = Rotation.from_quat(np.roll(q, -1)).as_matrix()   # our (w, x, y, z) → (x, y, z, w)
    t = np.asarray(camera_track.translation[idx], dtype=float)
    K = camera_track.intrinsics
    fx, fy, cx, cy = K.fx, K.fy, K.cx, K.cy
    if video_size is not None:
        vw, vh = video_size
        cal_w = 2.0 * cx
        cal_h = 2.0 * cy
        if abs(cal_w - vw) > 1 or abs(cal_h - vh) > 1:
            sx = vw / cal_w
            sy = vh / cal_h
            fx *= sx; fy *= sy; cx *= sx; cy *= sy
    # Auto-detect the camera-frame mismatch (memory project_camera_180_roll). The
    # solved CameraTrack is self-consistent only on the frame turned upside-down,
    # so a no-roll projection lands every body HEAD-DOWN on the as-decoded frame.
    # Detect via the validated gate ``-R[1,2] < 0`` (⟺ R[1,2] > 0).
    # A camera rebuilt from the raw-frame homography (recalibrate_camera.py) is
    # already frame-aligned and trips this gate as a false positive, so skip it.
    aligned = bool(getattr(camera_track, "raw_frame_aligned", False))
    flipped = (not aligned) and bool(-R[1, 2] < 0)
    if flipped:
        # The correction is a full 180° roll about the optical axis:
        # D=diag(-1,-1,1) maps (u,v) -> (W-u, H-v) (cx=W/2, cy=H/2 exactly).
        # A prior X-only mirror diag(-1,1,1) was validated by eye on 2026-07-07,
        # but the objective harness (scripts/debug/pose_probe.py, 2026-07-08) then
        # showed it leaves EVERY body vertically inverted (foot pixel-v < head,
        # upright 0/18) — invisible at ~22 px, which is why the eye missed it.
        # Negating cam-y too restores upright (18/18, scripts/debug/flip_sweep.py)
        # and moves the projected pitch far-line off the crowd onto the boards
        # (pitch-only visual check). We never rotate the frame the user sees.
        D = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        R = D @ R
        t = D @ t
    if adjust is not None and not adjust.is_identity:
        # Manual overlay-camera nudge, applied in the (flip-corrected) camera
        # frame: intrinsics for size/position, an extrinsic rotation about the
        # camera centre for the angles, and a view-axis translation for dolly.
        fx *= adjust.zoom
        fy *= adjust.zoom
        cx += adjust.panx
        cy += adjust.pany
        if adjust.yaw or adjust.pitch or adjust.roll:
            Rc = Rotation.from_euler(
                "xyz", [adjust.pitch, adjust.yaw, adjust.roll], degrees=True,
            ).as_matrix()
            R = Rc @ R
            t = Rc @ t
        if adjust.dolly:
            t = t + np.array([0.0, 0.0, adjust.dolly])
    return ProjectedFrame(
        fx=fx, fy=fy, cx=cx, cy=cy,
        R=R, t=t, frame_index=idx, frame_flipped=flipped,
    )


def project_points(pts_world: np.ndarray, proj: ProjectedFrame) -> np.ndarray:
    """Project ``(N, 3)`` world points to ``(N, 2)`` pixel coords.

    Returns floats; caller can round for pixel-level plotting. Points behind
    the camera get ``NaN`` so the client can hide them.
    """
    pts_cam = pts_world @ proj.R.T + proj.t
    z = pts_cam[:, 2]
    x = pts_cam[:, 0] / np.where(z > 1e-6, z, np.nan)
    y = pts_cam[:, 1] / np.where(z > 1e-6, z, np.nan)
    u = proj.fx * x + proj.cx
    v = proj.fy * y + proj.cy
    return np.stack([u, v], axis=-1)


# ── the ground-plane path: project through the SOLVED calibration, not scene.camera ──
# Everything above projects through ``scene.camera``. That is a *synthetic* frozen pose
# (#107: AppController overwrites the solved CameraTrack with a tiled BROADCAST viewpoint
# before export), so it does not describe the clip at all — measured 2026-07-30, the pitch
# markings projected through it land a median ~1300 px away from the same markings projected
# through the per-frame homography, on a 1920x1080 frame. That is the whole reason
# hand-aligning the overlay never converged: the user was being asked to hand-fit the wrong
# view with global sliders, frame after frame.
#
# ``field.calibration.homographies`` is the real thing and it is accurate — it lands within
# 1.1-1.7 px of the painted lines it is scored against. It is only a PLANE map, so it cannot
# project a 3D body; but it is EXACT for anything on the pitch surface, which covers the
# markings and every player's ground contact, and it needs no focal (#61's open problem).


def world_to_image(calibration, frame_index: int) -> np.ndarray:
    """The solved image→world homography for ``frame_index``, inverted for drawing.

    ``FieldCalibration.frames`` need not be 0..T-1, so look the frame up rather than
    indexing positionally.
    """
    frames = np.asarray(calibration.frames, dtype=int)
    hit = np.nonzero(frames == int(frame_index))[0]
    if hit.size == 0:
        raise KeyError(f"frame {frame_index} is not in the calibration ({frames.size} frames)")
    return np.linalg.inv(np.asarray(calibration.homographies[int(hit[0])], dtype=float))


def project_ground(xy: np.ndarray, w2i: np.ndarray) -> np.ndarray:
    """Project ``(N, 2)`` pitch-plane world XY to ``(N, 2)`` pixels; NaN where behind camera."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    p = np.column_stack([xy, np.ones(len(xy))]) @ w2i.T
    w = p[:, 2]
    ok = w > 1e-9
    uv = np.full((len(xy), 2), np.nan)
    uv[ok] = p[ok, :2] / w[ok, None]
    return uv


# ── off the plane: what it costs to draw something that stands up ──
# A ground homography is a map of the lawn. Anything at Z > 0 needs the full camera, and the one
# thing separating the two is the focal — so drawing a goalpost is a bet on a number the plane
# cannot supply. Measured on the target clip: the two constraints a single homography puts on the
# focal (r1 ⊥ r2, and |r1| == |r2|) disagree by a factor of 1.5-1.8, and sweeping the focal to
# hold the recovered camera centre still across all 60 frames has a minimum so flat that 2500 and
# 3500 px score within 4% of each other. The focal is genuinely not observable from the lawn.
#
#
# It IS observable from the goal frame — 2.44 m of known height standing in shot — but only by eye,
# and the automated version of that check is a trap worth recording. Fitting the focal so the drawn
# crossbar sits on the brightest pixels of the real one reads 4350-4585 px across the frames where
# the goal is unoccluded, ~15% above what the homographies imply. It is wrong: the Laws put 2.44 m
# at the crossbar's LOWER edge, so the line belongs under the white band, not through the middle of
# it, and at this distance the bar is most of the disagreement. Rendered at 9x the clip-median focal
# lands on the lower edge and the "measured" one rides the top. Hence the design here — the estimate
# is a starting point for a control the user sets by eye, not an answer.


def focal_from_homography(w2i: np.ndarray, width: int, height: int) -> float | None:
    """Focal in pixels from one ground-plane homography, or ``None`` if it has no real solution.

    Assumes square pixels and the principal point at the image centre, which leaves ``H = sK[r1
    r2 t]`` with two constraints on ``K`` — ``r1·r2 == 0`` and ``|r1| == |r2|``. Solved together
    in least squares because on a broadcast's grazing view of the pitch they disagree badly, and
    picking either one alone silently commits to its bias.
    """
    h = np.asarray(w2i, dtype=float)
    (a1, b1, c1), (a2, b2, c2) = h[:, 0], h[:, 1]
    a1, a2 = a1 - c1 * width / 2.0, a2 - c2 * width / 2.0
    b1, b2 = b1 - c1 * height / 2.0, b2 - c2 * height / 2.0
    coef = np.array([a1 * a2 + b1 * b2, (a1**2 + b1**2) - (a2**2 + b2**2)])
    rhs = np.array([c1 * c2, c1**2 - c2**2])
    denom = float(coef @ coef)
    if denom < 1e-30:
        return None
    inv_f_sq = -float(coef @ rhs) / denom
    return float(1.0 / np.sqrt(inv_f_sq)) if inv_f_sq > 1e-12 else None


def lift_homography(
    w2i: np.ndarray, focal_px: float, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split the homography into ``(ground map, image direction of world +Z)``.

    A world point ``(X, Y, Z)`` then images at ``ground @ (X, Y, 1) + Z * up``, homogeneous, with a
    usable depth in the third component. A homography is only defined up to scale — sign included
    — so both halves need one fixed by hand, and each is fixed by something that cannot be argued
    with rather than by algebra:

    ``ground``  the patch of grass under the image centre is *visible*, so it is in front of the
                camera and its depth is positive.
    ``up``      the camera is above the pitch, not under it, so lifting a point off the grass
                moves it UP the image. Taking the algebraic branch instead draws goalposts buried.

    Two separate decisions on purpose. On a homography that really is ``sK[r1 r2 t]`` they always
    agree — asserted against a synthetic camera in the golden tests. On this clip's solved
    homography they do **not**: it is a least-squares fit to keypoints, not a camera, and it misses
    the pinhole form badly enough (``|r1|/|r2| = 0.91``, ``r1·r2 = -0.21`` where an honest
    decomposition needs 1 and 0) that deriving one sign from the other blanks the overlay.
    """
    k = np.array([[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]])
    h = np.asarray(w2i, dtype=float)
    m = np.linalg.inv(k) @ h
    scale = (float(np.linalg.norm(m[:, 0])) + float(np.linalg.norm(m[:, 1]))) / 2.0

    centre = np.linalg.solve(h, [width / 2.0, height / 2.0, 1.0])  # grass under the image centre
    ground = (1.0 if centre[2] > 0.0 else -1.0) / scale * h
    up = k @ np.cross(m[:, 0] / scale, m[:, 1] / scale)
    seen = ground @ (centre / centre[2])
    if float(up[1] * seen[2] - seen[1] * up[2]) > 0.0:  # d(v)/d(Z) — must be negative
        up = -up
    return ground, up


def camera_centre(w2i: np.ndarray, focal_px: float, width: int, height: int) -> np.ndarray:
    """Where the camera is standing, in world metres — the sanity check on a chosen focal.

    A broadcast main camera is ~15-25 m up and tens of metres beyond the touchline, and it does
    not move between frames. A focal that puts it underground or on the pitch is wrong whatever
    the overlay looks like.
    """
    k = np.array([[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]])
    m = np.linalg.inv(k) @ np.asarray(w2i, dtype=float)
    scale = (float(np.linalg.norm(m[:, 0])) + float(np.linalg.norm(m[:, 1]))) / 2.0
    r1, r2, t = m[:, 0] / scale, m[:, 1] / scale, m[:, 2] / scale
    if float(np.cross(r1, r2) @ t) > 0.0:
        r1, r2, t = -r1, -r2, -t
    u, _s, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    return -(u @ vt).T @ t


def project_world(
    xyz: np.ndarray, w2i: np.ndarray, focal_px: float, width: int, height: int
) -> np.ndarray:
    """Project ``(N, 3)`` world points to ``(N, 2)`` pixels; NaN where behind the camera."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    ground, up = lift_homography(w2i, focal_px, width, height)
    p = np.column_stack([xyz[:, :2], np.ones(len(xyz))]) @ ground.T + xyz[:, 2:3] * up
    ok = p[:, 2] > 1e-9
    uv = np.full((len(xyz), 2), np.nan)
    uv[ok] = p[ok, :2] / p[ok, 2, None]
    return uv


def image_to_ground(uv: np.ndarray, calibration, frame_index: int) -> np.ndarray:
    """Inverse of :func:`project_ground` — pixels back to pitch-plane world XY.

    This is what turns a user's drag into a correction: wherever they drop a player on the
    frame is a point on the pitch, and the homography says exactly which one.
    """
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    frames = np.asarray(calibration.frames, dtype=int)
    hit = np.nonzero(frames == int(frame_index))[0]
    if hit.size == 0:
        raise KeyError(f"frame {frame_index} is not in the calibration ({frames.size} frames)")
    h = np.asarray(calibration.homographies[int(hit[0])], dtype=float)
    p = np.column_stack([uv, np.ones(len(uv))]) @ h.T
    return p[:, :2] / p[:, 2:3]
