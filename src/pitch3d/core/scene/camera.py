"""Camera model: pinhole intrinsics + per-frame extrinsics.

Used for (a) the estimated broadcast camera of a scene and (b) the *synthesized*
camera trajectories that a :class:`ViewSynthesizer` produces (stored on
:class:`~pitch3d.core.scene.assets.SynthViewRef`).

Extrinsics convention: world→camera. A world point ``X_w`` maps to camera space by
``X_c = R @ X_w + t`` where ``R`` is stored as a quaternion ``(w, x, y, z)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np


class CameraSource(str, Enum):
    """How a :class:`CameraTrack` was obtained. **R-6 applied to the camera itself (#140).**

    The controller ends with ``scene.camera = _measured_camera(...) or _static_camera(...)``. The
    refusal on the left is correct and deliberate (#61), but the substitution on the right used to
    be **silent**: on disk a synthetic stand-in and a real solve were byte-indistinguishable, and
    the only record — ``PlaneCameraFit`` — lived in memory and was never serialized.

    Measured 2026-08-08: **nine of nine scenes on disk** carried the synthetic fallback, including
    the reference scene the #135 eye labels were made on. Every judgement that compared one of
    those scenes to the source pixels was reading a camera at 772 px against a clip whose real
    focal is ~4200. "Mark, never hide" is the rule we apply to phantom players; it was not being
    applied here.
    """

    PLANE_FIT = "plane_fit"              # reduced from the field calibration and accepted
    STATIC_FALLBACK = "static_fallback"  # the plane fit refused; this is NOT the clip's camera
    PRESCRIBED = "prescribed"            # a synthesized trajectory (ViewSynthesizer, virtual op)


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
        raw_frame_aligned: True if the extrinsics already project onto the video's
            native (as-decoded) frame — i.e. the camera was rebuilt from the
            raw-frame homography (scripts/recalibrate_camera.py). Consumers that
            carry a legacy 180°-roll workaround (poseannot.camera.frame_projector)
            must skip it when this is set. Legacy PnLCalib solves leave it False.
    """

    intrinsics: CameraIntrinsics
    frames: np.ndarray
    rotation_quat: np.ndarray
    translation: np.ndarray
    estimated: bool = True
    raw_frame_aligned: bool = False
    #: Per-frame focal length in px, shape ``(T,)`` — the operator's zoom. ``None`` means the
    #: track has one focal for its whole span, which is what every solve here produced until
    #: camlab's schema 2 arrived.
    #:
    #: **This is the "future refinement" the class docstring promised, and it is not cosmetic.**
    #: Collapsing a zooming clip to one focal is measured, by camlab, against the paint:
    #: `fan` zooms 1.59x and goes 1.65 px -> **4.56 px**, dropping 2 of 12 frames out of the 20 px
    #: band; `broadcast` zooms 1.03x and goes 2.29 -> 4.16. So one focal is honest on a clip that
    #: does not zoom and misreports by 3x on one that does.
    #:
    #: Position needed nothing: ``translation`` was already per frame.
    focal_px: np.ndarray | None = None
    #: How this camera was obtained (#140). Defaults to ``PLANE_FIT`` so an old scene decodes,
    #: but the controller always sets it explicitly.
    source: CameraSource = CameraSource.PLANE_FIT
    #: The plane fit's reprojection in px, whether it was accepted or refused. ``None`` when no
    #: fit was attempted. This is the number that decides ``realizable``.
    fit_reprojection_px: float | None = None
    #: The focal the plane fit recovered, in px — recorded even when the fit was refused, because
    #: a sane focal with a large reprojection is a different diagnosis from a wild one.
    fit_focal_px: float | None = None

    @property
    def is_measured(self) -> bool:
        """False when this camera is a stand-in and must not be compared to the source pixels."""
        return self.source is CameraSource.PLANE_FIT

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
        scale = self.intrinsics.scaled(factor)
        focal = None if self.focal_px is None else np.asarray(self.focal_px, float) * factor
        return replace(self, intrinsics=scale, focal_px=focal)

    def intrinsics_at(self, frame_index: int) -> CameraIntrinsics:
        """The intrinsics **at one frame** — per-frame focal when the track carries the zoom.

        Every consumer that projects a point must go through this rather than reading
        ``.intrinsics`` directly, or a zooming clip is rendered through whichever single focal the
        track happened to be built with. Falls back to the shared intrinsics when ``focal_px`` is
        ``None``, so a track that genuinely has one focal costs nothing and every caller is safe.
        """
        if self.focal_px is None:
            return self.intrinsics
        row = int(np.argmin(np.abs(np.asarray(self.frames) - int(frame_index))))
        f = float(np.asarray(self.focal_px)[row])
        return replace(self.intrinsics, fx=f, fy=f)

    def zoom_ratio(self) -> float:
        """Largest focal over smallest, or 1.0 when the track has a single focal.

        Reported rather than assumed: camlab's schema 2 carries it so that "does this clip zoom"
        needs no arithmetic, and it is the number that says whether one focal would have been
        honest here.
        """
        if self.focal_px is None:
            return 1.0
        f = np.asarray(self.focal_px, float)
        lo = float(f.min())
        return float(f.max() / lo) if lo > 0 else float("inf")
