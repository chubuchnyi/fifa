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
    flipped = bool(-R[1, 2] < 0)
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
