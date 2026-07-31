"""Recover a real :class:`CameraTrack` from a solved :class:`FieldCalibration` — or refuse to.

A scene carries two descriptions of one camera: a ``CameraTrack`` (``K [R | t]``) and a
per-frame image↔world-plane homography. The pipeline only ever solved the second, so
``AppController`` filled the first with a *synthetic* broadcast pose (#107) and every export
carried a camera that had never seen the clip. This module closes that gap where it can be
closed honestly.

**A plane homography determines the camera once the focal is known.** For the pitch plane
``Z = 0`` the world→image map is ``H ≃ K [r₁ r₂ t]``, so ``K⁻¹H`` hands back the first two
columns of the rotation and the translation, up to one scale fixed by ``‖r₁‖ = 1`` and one sign
fixed by putting the camera in front of the pitch.

**And the focal comes from the same homographies, for free.** ``r₁`` and ``r₂`` are columns of a
rotation, so they must be unit and orthogonal — Zhang's constraint. That is two equations per
frame on one unknown, which is why :func:`camera_from_calibration` needs nothing but the
calibration it is handed. On a calibration that really is a camera it is exact: the #119 fit's
own homographies return 4169 px, the focal that produced them, to five figures.

**The refusal is the point.** Sixty *free* per-frame homographies are not a camera and cannot be
made into one — no focal makes ``r₁·r₂`` vanish, because nothing ever asked them to come from a
shared ``K``. Measured on ``out/carry_off/export/scene.json``: best-case ``r₁·r₂ = 0.185`` and the
rebuilt camera misses its own homography by 5048 px on the pitch. So this returns a
:class:`PlaneCameraFit` whose ``camera`` is ``None`` rather than a plausible-looking wrong camera,
and the caller keeps its labelled synthetic fallback. #107 was never a coding oversight — for a
free-homography calibration there was no measured camera to render.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..correction.rotations import matrix_to_quat
from .camera import CameraIntrinsics, CameraTrack

#: Bracket for the focal search, in pixels of the calibration's own image space. Wide on purpose:
#: a broadcast long lens and a phone wide angle both have to fall inside it.
FOCAL_BOUNDS = (300.0, 20000.0)

#: How far the rebuilt camera may miss the homography it came from, on the pitch plane, before
#: the calibration is declared not-a-camera. A real camera lands at ~1e-4 px (float noise); the
#: shipped free-homography solve lands at 5048. Nothing observed sits near this line.
REALIZABLE_PX = 1.0


@dataclass(frozen=True)
class PlaneCameraFit:
    """The outcome of asking a calibration for its camera.

    Attributes:
        camera: The recovered track, or ``None`` when the calibration is not camera-realizable.
        focal_px: The focal used — measured from the homographies unless one was supplied.
        reprojection_px: Worst distance, over all frames and probe points, between the rebuilt
            camera and the homography it was built from. This is the evidence for ``camera``
            being real, and it is reported even when the fit is refused.
        realizable: ``reprojection_px <= REALIZABLE_PX``.
    """

    camera: CameraTrack | None
    focal_px: float
    reprojection_px: float
    realizable: bool


def _decompose(h_world_to_image: np.ndarray, k_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World→camera ``(R, t)`` from a plane homography, given ``K⁻¹``.

    Completes ``r₃ = r₁ × r₂`` and snaps to the nearest rotation. The snap is what makes a wrong
    focal visible: at the right one the columns are already orthonormal and it does nothing.
    """
    m = k_inv @ h_world_to_image
    scale = 1.0 / np.linalg.norm(m[:, 0])
    r1, r2, t = m[:, 0] * scale, m[:, 1] * scale, m[:, 2] * scale
    if t[2] < 0:  # the two-fold planar sign ambiguity: keep the pitch in front of the camera
        r1, r2, t = -r1, -r2, -t
    u, _, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        rot = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return rot, t


def _orthonormality(h_world_to_image: np.ndarray, k_inv: np.ndarray) -> float:
    """Zhang's residual for one frame: how far ``K⁻¹H``'s first two columns are from a rotation's."""
    m = k_inv @ h_world_to_image
    r1, r2 = m[:, 0], m[:, 1]
    n1 = np.linalg.norm(r1)
    if n1 < 1e-12:
        return np.inf
    r1, r2 = r1 / n1, r2 / n1
    return float((r1 @ r2) ** 2 + (np.linalg.norm(r2) - 1.0) ** 2)


def _k_inv(focal: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[1.0 / focal, 0.0, -cx / focal], [0.0, 1.0 / focal, -cy / focal],
                     [0.0, 0.0, 1.0]])


def _measure_focal(h_w2i: np.ndarray, cx: float, cy: float) -> float:
    """The focal that best makes every frame's homography come from one rotation.

    Coarse log grid then golden-section refine — the residual is smooth in ``f`` but not convex
    over three octaves, so a pure local search from a fixed start can settle in the wrong basin.
    """

    def cost(f: float) -> float:
        ki = _k_inv(f, cx, cy)
        return float(sum(_orthonormality(h, ki) for h in h_w2i))

    grid = np.geomspace(*FOCAL_BOUNDS, 96)
    best = int(np.argmin([cost(float(f)) for f in grid]))
    lo = float(grid[max(best - 1, 0)])
    hi = float(grid[min(best + 1, len(grid) - 1)])

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = cost(c), cost(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = cost(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = cost(d)
    return float((a + b) / 2.0)


def _probe_points(h_i2w: np.ndarray, width: float, height: float) -> np.ndarray:
    """World-plane points spanning what this frame actually sees.

    Deliberately not a fixed pitch grid: two cameras that disagree in focal still agree near the
    principal point, so the probe has to reach the image corners to have any power.
    """
    u = np.array([[0.0, 0.0], [width, 0.0], [0.0, height], [width, height],
                  [width / 2.0, height / 2.0]])
    p = np.column_stack([u, np.ones(len(u))]) @ h_i2w.T
    return p[:, :2] / p[:, 2, None]


def camera_from_calibration(
    calibration,
    *,
    width: int,
    height: int,
    focal_px: float | None = None,
) -> PlaneCameraFit:
    """Recover the camera a :class:`FieldCalibration` came from, or report that there is none.

    Args:
        calibration: Anything with ``homographies`` (``(T, 3, 3)`` image→world) and ``frames``.
        width, height: Pixel size of the image space the homographies are in. The principal point
            is taken at its centre — the same assumption the focal is measured under.
        focal_px: Skip the measurement and use this focal (the project's auto→manual override
            rule). The rotations and translations are still the calibration's own; a supplied
            focal that the homographies do not support simply shows up in ``reprojection_px``.

    Returns:
        A :class:`PlaneCameraFit`. ``camera`` is ``None`` unless the rebuilt track reproduces the
        input homographies to within :data:`REALIZABLE_PX` on the pitch plane.
    """
    h_i2w = np.asarray(calibration.homographies, dtype=float)
    frames = np.asarray(calibration.frames, dtype=int)
    h_w2i = np.linalg.inv(h_i2w)
    cx, cy = width / 2.0, height / 2.0

    focal = float(focal_px) if focal_px is not None else _measure_focal(h_w2i, cx, cy)
    k_inv = _k_inv(focal, cx, cy)
    k = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])

    rots = np.zeros((len(h_i2w), 3, 3))
    transl = np.zeros((len(h_i2w), 3))
    worst = 0.0
    for i, h in enumerate(h_w2i):
        rots[i], transl[i] = _decompose(h, k_inv)
        world = _probe_points(h_i2w[i], width, height)
        through_cam = (np.column_stack([world, np.zeros(len(world))]) @ rots[i].T + transl[i]) @ k.T
        through_hom = np.column_stack([world, np.ones(len(world))]) @ h.T
        got = through_cam[:, :2] / through_cam[:, 2, None]
        want = through_hom[:, :2] / through_hom[:, 2, None]
        worst = max(worst, float(np.abs(got - want).max()))

    realizable = worst <= REALIZABLE_PX
    camera = None
    if realizable:
        camera = CameraTrack(
            intrinsics=CameraIntrinsics(
                fx=focal, fy=focal, cx=cx, cy=cy, width=int(width), height=int(height)
            ),
            frames=frames,
            rotation_quat=matrix_to_quat(rots),
            translation=transl,
            estimated=True,
            raw_frame_aligned=True,
        )
    return PlaneCameraFit(
        camera=camera, focal_px=focal, reprojection_px=worst, realizable=realizable
    )
