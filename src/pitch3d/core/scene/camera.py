"""Camera model: pinhole intrinsics + per-frame extrinsics.

Used for (a) the estimated broadcast camera of a scene and (b) the *synthesized*
camera trajectories that a :class:`ViewSynthesizer` produces (stored on
:class:`~pitch3d.core.scene.assets.SynthViewRef`).

Extrinsics convention: world→camera. A world point ``X_w`` maps to camera space by
``X_c = R @ X_w + t`` where ``R`` is stored as a quaternion ``(w, x, y, z)``.
"""

from __future__ import annotations

from dataclasses import dataclass

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
