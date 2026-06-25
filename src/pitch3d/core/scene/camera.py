"""Camera model: pinhole intrinsics + per-frame extrinsics.

Used for (a) the estimated broadcast camera of a scene and (b) the *synthesized*
camera trajectories that a :class:`ViewSynthesizer` produces (stored on
:class:`~pitch3d.core.scene.assets.SynthViewRef`).

Extrinsics convention: world→camera. A world point ``X_w`` maps to camera space by
``X_c = R @ X_w + t`` where ``R`` is stored as a quaternion ``(w, x, y, z)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics in pixels.

    Attributes:
        fx, fy: Focal lengths (px).
        cx, cy: Principal point (px).
        width, height: Image size (px).
        distortion: Optional radial/tangential coefficients (k1,k2,p1,p2,k3,...).
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: np.ndarray | None = None

    def matrix(self) -> np.ndarray:
        """Return the 3x3 intrinsics matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    def scaled(self, factor: float) -> CameraIntrinsics:
        """Uniformly rescale to a lower pixel resolution (the M2-6 fast-preview lever, UX-9).

        Multiplies the focal lengths and principal point by ``factor`` and rounds the image size,
        so a downstream raster touches ~``factor**2`` of the pixels. ``factor == 1.0`` is a no-op
        (returns ``self``); ``distortion`` is in normalised coords and rides through unchanged. The
        image stays at least 1 px on each side.
        """
        if factor == 1.0:
            return self
        return replace(
            self,
            fx=self.fx * factor,
            fy=self.fy * factor,
            cx=self.cx * factor,
            cy=self.cy * factor,
            width=max(1, int(round(self.width * factor))),
            height=max(1, int(round(self.height * factor))),
        )


@dataclass
class CameraTrack:
    """A camera over a frame range: shared intrinsics + per-frame pose (world→camera).

    Attributes:
        intrinsics: Shared pinhole intrinsics (mono broadcast: one zoom track is a
            future refinement; per-frame intrinsics can be added without touching core).
        frames: Integer frame indices, shape ``(T,)``.
        rotation_quat: World→camera rotation per frame, shape ``(T, 4)`` as (w,x,y,z).
        translation: World→camera translation per frame, shape ``(T, 3)`` in meters.
        estimated: True if poses are estimated (broadcast cam); False if prescribed
            (a synthesized ViewSynthesizer trajectory).
    """

    intrinsics: CameraIntrinsics
    frames: np.ndarray
    rotation_quat: np.ndarray
    translation: np.ndarray
    estimated: bool = True

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=int)
        self.rotation_quat = np.asarray(self.rotation_quat, dtype=float).reshape(-1, 4)
        self.translation = np.asarray(self.translation, dtype=float).reshape(-1, 3)

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @classmethod
    def identity(cls, intrinsics: CameraIntrinsics, n_frames: int) -> CameraTrack:
        """A static camera at the origin looking along world axes (test/default)."""
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, 1))
        return cls(
            intrinsics=intrinsics,
            frames=np.arange(n_frames),
            rotation_quat=quat,
            translation=np.zeros((n_frames, 3)),
        )

    def scaled(self, factor: float) -> CameraTrack:
        """A copy at ``factor`` of this camera's pixel resolution; extrinsics unchanged (UX-9).

        Only the intrinsics shrink (the M2-6 fast-preview lever) — the per-frame pose and frame
        indices are identical, so ``n_frames`` and the projected *content* are preserved, just at
        fewer pixels. ``factor == 1.0`` returns ``self`` so a FINAL render keeps camera identity.
        """
        if factor == 1.0:
            return self
        return replace(self, intrinsics=self.intrinsics.scaled(factor))
