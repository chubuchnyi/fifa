"""M2-6 fast low-q preview lever (UX-9) — the quality→resolution policy + camera rescale.

"Fast low-q preview before the expensive final" is implemented honestly as a *resolution*
downscale: :attr:`RenderQuality.scale` is the single policy (preview = half-res, final = full),
and :meth:`CameraIntrinsics.scaled` / :meth:`CameraTrack.scaled` apply it by shrinking only the
intrinsics — extrinsics, frame indices and frame *count* are preserved, so the same content is
rasterised at ~¼ the pixels. ``scale(1.0)`` is a no-op that keeps camera identity, so a FINAL
render reuses the caller's camera object. These pin the mechanism; the two render paths that
consume it are pinned in ``test_avatar_splat.py`` (splat) and ``test_render_orbit.py`` (seam A).
"""

from __future__ import annotations

import numpy as np

from pitch3d.core.ports.render import RenderQuality
from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack


def _intr(w: int = 64, h: int = 48) -> CameraIntrinsics:
    return CameraIntrinsics(fx=50.0, fy=50.0, cx=w / 2.0, cy=h / 2.0, width=w, height=h)


# --- the policy ---------------------------------------------------------------
def test_quality_scale_preview_is_half_final_is_full():
    assert RenderQuality.PREVIEW.scale == 0.5
    assert RenderQuality.FINAL.scale == 1.0


# --- CameraIntrinsics.scaled --------------------------------------------------
def test_intrinsics_scaled_halves_every_pixel_quantity():
    out = _intr(64, 48).scaled(0.5)
    assert (out.fx, out.fy) == (25.0, 25.0)
    assert (out.cx, out.cy) == (16.0, 12.0)
    assert (out.width, out.height) == (32, 24)


def test_intrinsics_scaled_one_is_identity_noop():
    intr = _intr()
    assert intr.scaled(1.0) is intr  # FINAL must not allocate a new camera (identity preserved)


def test_intrinsics_scaled_preserves_distortion():
    dist = np.array([0.1, -0.2, 0.0, 0.0])
    intr = CameraIntrinsics(
        fx=50.0, fy=50.0, cx=32.0, cy=24.0, width=64, height=48, distortion=dist
    )
    assert intr.scaled(0.5).distortion is dist  # normalised coeffs ride through a pixel rescale


def test_intrinsics_scaled_floors_at_one_pixel():
    # A 1 px image can't halve to 0 px — the floor keeps a renderable (degenerate but valid) frame.
    out = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.5, cy=0.5, width=1, height=1).scaled(0.5)
    assert (out.width, out.height) == (1, 1)


# --- CameraTrack.scaled -------------------------------------------------------
def test_track_scaled_shrinks_intrinsics_but_preserves_extrinsics_and_frames():
    cam = CameraTrack.identity(_intr(64, 48), 8)
    cam.frames = np.arange(10, 18)  # non-trivial frame indices to prove they survive untouched
    out = cam.scaled(0.5)
    assert (out.intrinsics.width, out.intrinsics.height) == (32, 24)
    assert out.n_frames == cam.n_frames == 8
    np.testing.assert_array_equal(out.frames, cam.frames)
    np.testing.assert_array_equal(out.rotation_quat, cam.rotation_quat)
    np.testing.assert_array_equal(out.translation, cam.translation)
    assert out.estimated == cam.estimated


def test_track_scaled_one_is_identity_noop():
    cam = CameraTrack.identity(_intr(), 4)
    assert cam.scaled(1.0) is cam
