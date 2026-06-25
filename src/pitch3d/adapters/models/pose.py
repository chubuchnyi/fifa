"""GVHMR SMPL-X pose estimator — fourth real adapter, the central one (M1, FR-8, FR-22c).

Self-hosted default for `PoseEstimator`: a gravity-view HMR network (GVHMR / WHAM / TRAM),
behind the optional ``hmr`` extra. Split so the *model-independent* logic is testable with
**no torch, no GPU**:

* :class:`GVHMRPoseEstimator` — the **pure** half. The network only knows *articulation*
  (per-frame ``global_orient`` + ``body_pose`` + shape ``betas``); turning that into a scene
  subject means **anchoring the world root on the pitch** — exactly FR-8's "root from
  homography". This half projects each tracklet's foot point to world metres through the field
  calibration, stacks the metric root, assembles the canonical :class:`SubjectMotion`, and
  applies geometric :meth:`refit` constraints (root nudges/locks/floor) deterministically.
  Numpy only; unit-tested via an injected backend.
* :class:`GVHMRBackend` — the **heavy** half: *not wired yet* (roadmap M1/P2.4) and GPU-bound.
  GVHMR is a research repo, not a pip package, so the ``hmr`` extra ships only its substrate
  (torch/smplx/chumpy) — not the network or its weights; :meth:`GVHMRBackend.estimate_bodies`
  raises an actionable ``NotImplementedError`` pointing at ``--pose fake`` or an injected backend.
  The pure half above is complete and tested, so the pose path runs end to end on the fake today.

Swap it in via ``default_ports(pose="gvhmr")`` (wiring) — one fake replaced at a time,
satisfying the very same ``PoseEstimator`` port test the fake passes (roadmap M1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ...core.ports.io import ClipRef
from ...core.ports.perception import Tracklet, Tracks
from ...core.ports.pose import PoseEstimator
from ...core.scene.field import FieldCalibration
from ...core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from ...core.scene.provenance import Backend, ModelInfo

#: Nominal SMPL-X pelvis height above the ground plane (m) — the mono Z anchor (R-4).
_DEFAULT_PELVIS_HEIGHT_M = 0.92


@dataclass
class RawBodyMotion:
    """Backend output for one subject: camera-space SMPL-X articulation + shape, no world root.

    The world root is intentionally absent — placing it on the pitch is the field-homography
    job done in the pure half. This is the canned hand-off the grounding logic consumes.

    Attributes:
        track_id: The tracklet this articulation belongs to.
        frames: ``(T,)`` source frame indices.
        global_orient: ``(T, 3)`` root orientation, axis-angle.
        body_pose: ``(T, J, 3)`` body-joint rotations, axis-angle.
        betas: ``(n_betas,)`` per-subject shape coefficients (frame-invariant).
        pelvis_above_foot: ``(T,)`` per-frame vertical foot→pelvis offset (metres) from the
            backend's SMPL-X forward kinematics, or ``None``. When present it drives the world
            root Z in :meth:`GVHMRPoseEstimator._ground_root` (the T2 foot-plane anchor — pelvis
            height that varies with crouch/run/stride); when absent the estimator falls back to a
            fixed nominal pelvis height. Backends that do not compute FK leave it ``None``.
    """

    track_id: int
    frames: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray
    betas: np.ndarray
    pelvis_above_foot: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=int).reshape(-1)
        t = self.frames.shape[0]
        self.global_orient = np.asarray(self.global_orient, dtype=float).reshape(-1, 3)
        body_pose = np.asarray(self.body_pose, dtype=float)
        self.body_pose = body_pose.reshape(body_pose.shape[0], -1, 3)  # (rows, J, 3)
        self.betas = np.asarray(self.betas, dtype=float).reshape(-1)
        if not (self.global_orient.shape[0] == self.body_pose.shape[0] == t):
            raise ValueError(
                f"ragged raw body motion {self.track_id}: {t} frames, "
                f"{self.global_orient.shape[0]} orient, {self.body_pose.shape[0]} body rows"
            )
        if self.pelvis_above_foot is not None:
            self.pelvis_above_foot = np.asarray(self.pelvis_above_foot, dtype=float).reshape(-1)
            if self.pelvis_above_foot.shape[0] != t:
                raise ValueError(
                    f"pelvis_above_foot for {self.track_id} has "
                    f"{self.pelvis_above_foot.shape[0]} rows, expected {t}"
                )


@runtime_checkable
class HMRBackend(Protocol):
    """The heavy half: run the HMR network per tracklet → camera-space articulation + shape.

    Kept behind this protocol so :class:`GVHMRPoseEstimator`'s grounding/assembly/refit logic
    can be tested with a stub returning canned :class:`RawBodyMotion` — no GPU required.
    """

    def estimate_bodies(self, clip: ClipRef, tracks: Tracks) -> dict[int, RawBodyMotion]:
        """Return SMPL-X articulation per subject, keyed by ``track_id``."""
        ...


def _smooth_path(values: np.ndarray, window: int) -> np.ndarray:
    """Box-average a ``(T, D)`` path over a centred frame window (anti-jitter, edge-clamped)."""
    t = values.shape[0]
    if window <= 1 or t <= 2:
        return values
    half = (window if window % 2 else window + 1) // 2
    out = np.empty_like(values)
    for i in range(t):
        out[i] = values[max(0, i - half):min(t, i + half + 1)].mean(axis=0)
    return out


@dataclass
class GVHMRPoseEstimator(PoseEstimator):
    """SMPL-X HMR (FR-8) — pure root-grounding + assembly over an injected backend.

    Attributes:
        backend: The HMR network backend. If ``None``, a real :class:`GVHMRBackend` is built
            lazily on first use (needs the ``hmr`` extra + weights + GPU).
        pelvis_height_m: World Z assigned to the grounded root (mono height anchor, R-4).
        smooth_window: Centred window (frames) for smoothing the grounded root path
            (anti-foot-sliding from box jitter); 1 disables it.
        n_betas: Shape-coefficient count expected from the backend (provenance only).
        device: Inference device for the default backend.
    """

    backend: HMRBackend | None = None
    pelvis_height_m: float = _DEFAULT_PELVIS_HEIGHT_M
    smooth_window: int = 1
    n_betas: int = 10
    device: str = "cuda"

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="GVHMR",
            backend=Backend.LOCAL,
            license="see upstream (non-commercial SMPL-X)",
            params={"pelvis_height_m": self.pelvis_height_m, "device": self.device},
        )

    def estimate(
        self, clip: ClipRef, tracks: Tracks, calibration: FieldCalibration
    ) -> dict[int, SubjectMotion]:
        bodies = self._backend().estimate_bodies(clip, tracks)
        out: dict[int, SubjectMotion] = {}
        for tl in tracks.tracklets:
            if tl.cls == "ball":
                continue  # the ball has its own BallTracker; HMR is for people only
            raw = bodies.get(tl.track_id)
            if raw is None:
                continue
            rows = _align_rows(raw.frames, tl.frames)
            height = raw.pelvis_above_foot[rows] if raw.pelvis_above_foot is not None else None
            transl = _smooth_path(
                self._ground_root(tl, calibration, height), self.smooth_window
            )
            pose = PoseSequence(
                frames=tl.frames,
                global_orient=raw.global_orient[rows],
                body_pose=raw.body_pose[rows],
                transl=transl,
            )
            out[tl.track_id] = SubjectMotion(shape=SmplxShape(betas=raw.betas), pose=pose)
        return out

    def refit(
        self, clip: ClipRef, motion: SubjectMotion, constraints: dict, frames: np.ndarray,
    ) -> SubjectMotion:
        """Apply operator constraints on ``frames`` and return a NEW motion (non-destructive).

        Geometric constraints are honoured purely here (no network): ``root_z_nudge`` (float, m),
        ``root_xy`` ((2,) world lock), ``foot_floor`` (clamp root Z ≥ floor), ``relax_to_rest``
        (scale body pose toward rest in ``[0, 1]``). The correction engine wraps the result as a
        REFIT correction, so this stays a pure function of its inputs.
        """
        refined = motion.copy()
        rows = np.isin(refined.pose.frames, np.asarray(frames, dtype=int).reshape(-1))

        relax = constraints.get("relax_to_rest")
        if relax is not None:
            refined.pose.body_pose[rows] *= float(relax)
        nudge = float(constraints.get("root_z_nudge", 0.0))
        if nudge:
            refined.pose.transl[rows, 2] += nudge
        xy = constraints.get("root_xy")
        if xy is not None:
            refined.pose.transl[rows, :2] = np.asarray(xy, dtype=float).reshape(2)
        floor = constraints.get("foot_floor")
        if floor is not None:
            refined.pose.transl[rows, 2] = np.maximum(refined.pose.transl[rows, 2], float(floor))
        return refined

    def _ground_root(
        self,
        tl: Tracklet,
        calibration: FieldCalibration,
        pelvis_above_foot: np.ndarray | None = None,
    ) -> np.ndarray:
        """Place the root on the pitch: bbox foot point → world metres via the homography.

        The root Z is the pelvis height above the foot plane (z=0): the per-frame
        ``pelvis_above_foot`` the backend computed from SMPL-X FK (the T2 foot-plane anchor,
        varying with crouch/run/stride) when present, else the fixed nominal
        :attr:`pelvis_height_m`. Foot XY comes from the bbox bottom-centre through the homography.
        """
        foot_uv = np.column_stack(
            [(tl.bboxes_xyxy[:, 0] + tl.bboxes_xyxy[:, 2]) / 2.0, tl.bboxes_xyxy[:, 3]]
        )
        world_xy = np.stack(
            [calibration.image_to_world(int(f), foot_uv[i])[0] for i, f in enumerate(tl.frames)]
        )
        if pelvis_above_foot is None:
            z = np.full(tl.frames.shape[0], self.pelvis_height_m)
        else:
            z = np.asarray(pelvis_above_foot, dtype=float).reshape(-1)
        return np.column_stack([world_xy, z])

    def _backend(self) -> HMRBackend:
        return self.backend or GVHMRBackend(device=self.device)


def _align_rows(raw_frames: np.ndarray, want_frames: np.ndarray) -> np.ndarray:
    """Row indices selecting ``want_frames`` out of (sorted) ``raw_frames``; raises if uncovered."""
    idx = np.clip(np.searchsorted(raw_frames, want_frames), 0, raw_frames.shape[0] - 1)
    if not np.array_equal(raw_frames[idx], want_frames):
        raise ValueError("HMR backend frames do not cover the tracklet frames")
    return idx


@dataclass
class GVHMRBackend:
    """Real GVHMR inference: lazy torch/HMR imports, no import cost.

    Imports the heavy stack only when :meth:`estimate_bodies` is first called, so this module
    stays import-safe without the ``hmr`` extra installed.
    """

    weights: str | None = None
    device: str = "cuda"
    _model: object = None

    def estimate_bodies(  # pragma: no cover - heavy path
        self, clip: ClipRef, tracks: Tracks
    ) -> dict[int, RawBodyMotion]:
        self._load()
        raise NotImplementedError(
            "GVHMR inference is not wired yet (roadmap M1/P2.4) and is GPU-bound. Unlike the "
            "rfdetr/bytetrack reals, GVHMR is a research repo (not a pip package), so the `hmr` "
            "extra ships only its substrate (torch/smplx/chumpy, non-commercial SMPL-X) — not the "
            "network or its weights — and it is the one stage that may not be CPU-viable even for "
            "a single clip (ADR-0009). The pure half (root grounding + assembly + refit) is "
            "complete and tested; inject an HMRBackend that yields per-tracklet SMPL-X params, or "
            "keep the fake (`--pose fake`) until GVHMR is wired."
        )

    def _load(self) -> object:  # pragma: no cover - exercised only without the extra
        if self._model is None:
            try:
                import torch  # noqa: F401  (stand-in for the GVHMR/SMPL-X stack)
            except ImportError as exc:
                raise RuntimeError(
                    "GVHMR is not installed. Install the HMR extra: "
                    "`pip install 'pitch3d[hmr]'`, or inject an HMRBackend."
                ) from exc
            self._model = object()
        return self._model
