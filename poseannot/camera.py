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

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from pitch3d.core.correction.rotations import matrix_to_quat
from pitch3d.core.scene.layers import TargetKind
from pitch3d.core.scene.projection import quat_to_rotation_matrix


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
    adjust: CameraAdjust | None = None,
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


# ── which way is up? the frame the solved homography actually lives in (#118) ──
# PnLCalib's world table is a top-down pitch TEMPLATE: X across the image, Y *down* it. Read those
# two axes as our X and Y and then call the third one "up", and the labelling is left-handed — so a
# homography that maps the lawn perfectly still decomposes to a camera looking upward from under the
# grass. Measured on the target clip, at every candidate focal and on all 60 frames: optical axis
# +0.175 (up), centre 18 m below. Mirroring world Y turns it into a real broadcast gantry —
# -0.175, 18 m up, 72 m beyond the touchline — and moves not one pixel, because the pitch is
# symmetric about Y = 0. Nothing drawn on the lawn can ever catch this; only something with height.
#
# So the frame is not a constant to hardcode, it is a property of the homography in hand, and it is
# cheap to measure: `plane_orientation` reads it off the sign of `det(H)` alone. The synthetic
# golden camera (`pitch3d.eval.synthetic`, honestly right-handed Z-up) and this clip's solve sit on
# opposite sides of it, which is exactly why the two must be told apart rather than assumed.


def plane_orientation(w2i: np.ndarray, width: int, height: int) -> float:
    """``+1`` if this homography's world is right-handed with ``Z`` up, ``-1`` if it is mirrored.

    A camera above a right-handed ``Z``-up world sees the plane *reversed*: world ``+Y`` runs UP
    the image while pixel ``v`` counts down it, so the map from lawn to pixels flips orientation
    and its Jacobian determinant is negative. That determinant is ``det(H)/(h₃ᵀx)³``, so its sign
    is ``sign(det(H) · depth)`` at any visible point — free of the focal, and unchanged by the
    arbitrary scale (and sign) a homography is only defined up to, since ``λH`` scales it by ``λ⁴``.

    The point used is the grass under the image centre, which is visible by definition and so has
    positive depth whatever sign convention the caller's ``H`` arrived in.
    """
    h = np.asarray(w2i, dtype=float)
    centre = np.linalg.solve(h, [width / 2.0, height / 2.0, 1.0])
    depth = float(h[2] @ (centre / centre[2]))
    return -float(np.sign(np.linalg.det(h) * depth))


def lift_homography(
    w2i: np.ndarray, focal_px: float, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split the homography into ``(ground map, image direction of world +Z)``.

    A world point ``(X, Y, Z)`` then images at ``ground @ (X, Y, 1) + Z * up``, homogeneous, with a
    usable depth in the third component. A homography is only defined up to scale — sign included —
    so both halves need one fixed, and neither is fixed by algebra alone:

    ``ground``  the patch of grass under the image centre is *visible*, so it is in front of the
                camera and its depth is positive.
    ``up``      ``r1 × r2`` is the image of the world's third axis, which points up in a
                right-handed frame and down in a mirrored one — :func:`plane_orientation` says
                which this homography is in. Taking the algebraic branch unconditionally draws
                goalposts buried, and that is not a quirk of a noisy solve: on this clip it is the
                correct answer to the wrong question.
    """
    k = np.array([[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]])
    h = np.asarray(w2i, dtype=float)
    m = np.linalg.inv(k) @ h
    scale = (float(np.linalg.norm(m[:, 0])) + float(np.linalg.norm(m[:, 1]))) / 2.0

    centre = np.linalg.solve(h, [width / 2.0, height / 2.0, 1.0])  # grass under the image centre
    ground = (1.0 if centre[2] > 0.0 else -1.0) / scale * h
    up = plane_orientation(h, width, height) * (k @ np.cross(m[:, 0] / scale, m[:, 1] / scale))
    return ground, up


def camera_centre(w2i: np.ndarray, focal_px: float, width: int, height: int) -> np.ndarray:
    """Where the camera is standing, in world metres — the sanity check on a chosen focal.

    A broadcast main camera is ~15-25 m up and tens of metres beyond the touchline, and it does not
    move between frames. A focal that puts it on the pitch is wrong whatever the overlay looks like.

    The height is signed by :func:`plane_orientation`, for the same reason :func:`lift_homography`
    needs it: in a mirrored frame the decomposition's ``+Z`` is physically down. ``X`` and ``Y``
    stay in the caller's labels, so the answer is directly comparable with anything else drawn
    through the same homography.
    """
    k = np.array([[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]])
    h = np.asarray(w2i, dtype=float)
    m = np.linalg.inv(k) @ h
    scale = (float(np.linalg.norm(m[:, 0])) + float(np.linalg.norm(m[:, 1]))) / 2.0
    # Same branch `lift_homography` puts the ground on: the grass under the image centre is seen.
    front = 1.0 if float(np.linalg.solve(h, [width / 2.0, height / 2.0, 1.0])[2]) > 0.0 else -1.0
    r1, r2, t = front * m[:, 0] / scale, front * m[:, 1] / scale, front * m[:, 2] / scale
    u, _s, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    rot = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))]) @ vt
    centre = -rot.T @ t
    return centre * np.array([1.0, 1.0, plane_orientation(h, width, height)])


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


# ── hand-registering the pitch: #112 ───────────────────────────────────────────────────
# The solved homography can be a good camera and still put the pitch model in the wrong place
# on its own plane — a metre of drift or a few degrees of yaw look exactly like a correct
# calibration to every residual we compute, because the residual is measured against the same
# lines that placed it. The user's eye is the only instrument that sees it, so they get to move
# the layout, and the move is stored as a plane similarity that composes with the solve.


def plane_similarity(
    *, anchor: np.ndarray, src: np.ndarray, dst: np.ndarray, turn: bool
) -> np.ndarray:
    """The pitch-plane transform ``B`` that carries world point ``src`` onto ``dst``.

    ``turn=False`` translates; ``turn=True`` rotates and uniformly scales about ``anchor``,
    which stays where it is. Together the two gestures span the full 4-DOF similarity without
    a modifier key: grab the middle to slide the layout, grab the outer handle to spin/size it.

    Returned in the sense :class:`PlaneTransformPayload` documents — ``H'_w2i = H_w2i @ B`` —
    so ``B`` maps *model* coordinates into the coordinates the current solve draws.
    """
    src = np.asarray(src, dtype=float).reshape(2)
    dst = np.asarray(dst, dtype=float).reshape(2)
    b = np.eye(3)
    if not turn:
        b[:2, 2] = dst - src
        return b
    a = np.asarray(anchor, dtype=float).reshape(2)
    v, w = src - a, dst - a
    n2 = float(v @ v)
    if n2 < 1e-12:  # dragging the turn handle onto the anchor has no rotation to report
        return b
    # One complex division does rotation and scale at once: w/v is s·e^{iθ}.
    c = float(v @ w) / n2
    s = float(v[0] * w[1] - v[1] * w[0]) / n2
    m = np.array([[c, -s], [s, c]])
    b[:2, :2] = m
    b[:2, 2] = a - m @ a
    return b


def plane_similarity_params(dx: float, dy: float, deg: float, scale: float) -> np.ndarray:
    """The typed panel's ``B``: slide, yaw and size about the **pitch centre** (#127).

    The drag builds ``B`` from a dropped pixel and so can only rotate about whichever handle is
    under the cursor; the panel has no cursor, and rotating about a moving handle would make its
    own readout drift — turn the layout and the "slide" number would change even though nothing
    slid. Anchoring at the world origin instead makes the four numbers a *decomposition* of
    ``B`` rather than a history of gestures, which is what lets the panel show totals that hold
    still. It is the same subgroup either way, so the ``PlaneTransformPayload`` guarantee is
    untouched: this is a similarity, hence ``K[r₁ r₂ t] @ B`` is again a legal camera.
    """
    th = np.radians(float(deg))
    c, s = float(scale) * np.cos(th), float(scale) * np.sin(th)
    return np.array([[c, -s, float(dx)], [s, c, float(dy)], [0.0, 0.0, 1.0]])


def decompose_similarity(b: np.ndarray) -> dict[str, float]:
    """``plane_similarity_params`` read backwards, so the panel can rehydrate from the scene."""
    b = np.asarray(b, dtype=float)
    return {
        "dx": float(b[0, 2]),
        "dy": float(b[1, 2]),
        "deg": float(np.degrees(np.arctan2(b[1, 0], b[0, 0]))),
        "scale": float(np.hypot(b[0, 0], b[1, 0])),
    }


def plane_adjustment(corrections, frame: int) -> np.ndarray:
    """The composed ``B`` for ``frame`` from every enabled FIELD_CALIBRATION correction.

    Applied in insertion order on the right, because each drag was measured against the layout
    as it stood *after* the previous ones — the same order the user made them in.
    """
    b = np.eye(3)
    for c in corrections:
        if c.target.kind is not TargetKind.FIELD_CALIBRATION or not c.enabled:
            continue
        if frame not in c.frame_range:
            continue
        b = b @ np.asarray(c.payload.matrix, dtype=float)
    return b


def adjusted_camera(camera, corrections):
    """``camera`` re-expressed after the user re-registered the pitch plane (#112).

    A scene holds two descriptions of one camera, and #107 exists because they were allowed to
    drift apart. Moving the pitch under a fixed camera would split them again — measured live at
    2500 px on the first drag — so the camera moves too, and it moves *exactly*: for the plane
    ``Z = 0`` the world→image map is ``K[r₁ r₂ t]``, i.e. two rotation columns and the
    translation, which is precisely what a plane transform acts on. ``K`` never enters, so this
    is the same right-multiply as :func:`adjusted_calibration` and cannot disagree with it.

    The SVD snap only removes float noise here: for a similarity the columns come out orthogonal
    already, scaled by ``σ``, which is what dividing by ``‖m₀‖`` takes out.
    """
    if camera is None:
        return None
    frames = np.asarray(camera.frames, dtype=int)
    per_frame = [plane_adjustment(corrections, int(f)) for f in frames]
    if all(np.array_equal(b, np.eye(3)) for b in per_frame):
        return camera

    quat = np.asarray(camera.rotation_quat, dtype=float)
    transl = np.asarray(camera.translation, dtype=float)
    rots = np.zeros((len(per_frame), 3, 3))
    out_t = np.zeros_like(transl)
    for i, b in enumerate(per_frame):
        r = quat_to_rotation_matrix(quat[i])
        m = np.column_stack([r[:, 0], r[:, 1], transl[i]]) @ b
        scale = np.linalg.norm(m[:, 0])
        r1, r2, out_t[i] = m[:, 0] / scale, m[:, 1] / scale, m[:, 2] / scale
        u, _, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
        rots[i] = u @ vt
        if np.linalg.det(rots[i]) < 0:
            rots[i] = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return replace(camera, rotation_quat=matrix_to_quat(rots), translation=out_t)


def adjusted_calibration(calibration, corrections):
    """``calibration`` with the user's layout drags folded in — or itself, if there are none.

    Returns a copy: the stored solve stays untouched, so disabling the corrections restores it
    exactly. ``H_i2w`` is the inverse direction, hence ``B⁻¹`` on the left.
    """
    frames = np.asarray(calibration.frames, dtype=int)
    per_frame = [plane_adjustment(corrections, int(f)) for f in frames]
    if all(np.array_equal(b, np.eye(3)) for b in per_frame):
        return calibration
    h = np.asarray(calibration.homographies, dtype=float)
    moved = np.stack([np.linalg.inv(b) @ h[i] for i, b in enumerate(per_frame)])
    return replace(calibration, homographies=moved)
