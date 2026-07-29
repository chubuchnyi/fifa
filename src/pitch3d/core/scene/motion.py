"""Motion data: SMPL-X body parameters and the ball trajectory.

This is the **editable source of truth** (the "edit" half of the dual representation,
ADR-0002). Everything the operator edits and everything the correction engine touches
lives here; the render representation is derived from it and is never edited directly.

Pose rotations are stored as **axis-angle** 3-vectors, exactly as SMPL/SMPL-X expects.
Root translation, ball position and any curve are plain metric 3-vectors in the world
frame (Z-up, meters).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

N_SMPLX_BODY_JOINTS = 21
"""SMPL-X body pose joints (excludes global orient, hands, jaw, eyes)."""


class Provenance(str, Enum):
    """Where a per-frame state value came from — R-6 ("mark, don't erase") as a type.

    Before this existed the pipeline encoded the same thing as a magic ``subject_frame_conf``
    value (1.0 measured, 0.3 bridged, 0.2 coasted, 0.15 teleport-interpolated), so a 0.2 could
    equally mean "we did measure this and the detector was unsure". Those are different facts
    and a photoreal renderer must be able to tell them apart.
    """

    MEASURED = "measured"
    INTERPOLATED = "interpolated"  # bridged between two measured anchors on either side
    IMPUTED = "imputed"            # no anchor on one side: coasted, held, or otherwise inferred


class BallMode(str, Enum):
    """Ball height regime. Replaces a bare ``on_ground`` bool, which conflated two facts.

    ``not on_ground`` used to mean *both* "we fitted a parabola through it" and "we have no
    idea" — the lead/trail frames of `ball_lift` carry no bracketing contact and cannot pin a
    parabola at all. :attr:`UNMEASURED` says so.
    """

    ON_GROUND = "on_ground"    # Z pinned to the pitch plane by a measured contact
    BALLISTIC = "ballistic"    # Z from a gravity parabola between two contacts
    UNMEASURED = "unmeasured"  # height unknown — never present this as a measurement


_PROVENANCE_DTYPE = "<U12"
_BALL_MODE_DTYPE = "<U10"


def _label_array(
    values: object, n: int, enum_cls: type[Enum], dtype: str, default: Enum
) -> np.ndarray:
    """Normalise ``values`` to an ``(n,)`` array of ``enum_cls`` values; ``None`` → ``default``."""
    if values is None:
        return np.full(n, default.value, dtype=dtype)
    arr = np.asarray(values, dtype=dtype).reshape(n)
    allowed = {m.value for m in enum_cls}
    bad = sorted(set(arr.tolist()) - allowed)
    if bad:
        raise ValueError(f"not valid {enum_cls.__name__} values: {bad}")
    return arr


class BodyModel(str, Enum):
    """Which parametric body is in use. Pose dimensions follow from this."""

    SMPL = "SMPL"       # 23 body joints, no hands/face
    SMPL_H = "SMPL-H"   # body + hands
    SMPL_X = "SMPL-X"   # body + hands + face (default; matches Meshcapade addon)


@dataclass
class SmplxShape:
    """Per-subject shape — shared across all frames (β is identity, not motion).

    Attributes:
        betas: Shape coefficients, shape ``(n_betas,)`` (typically 10 or 16).
        body_model: Which parametric model these params target.
    """

    betas: np.ndarray
    body_model: BodyModel = BodyModel.SMPL_X

    def __post_init__(self) -> None:
        self.betas = np.asarray(self.betas, dtype=float).reshape(-1)


@dataclass
class PoseSequence:
    """Per-frame SMPL-X pose + root for one subject.

    Attributes:
        frames: Frame indices, shape ``(T,)``.
        global_orient: Root orientation as axis-angle, shape ``(T, 3)``.
        body_pose: Body joint rotations as axis-angle, shape ``(T, J, 3)``.
        transl: Root translation in world meters, shape ``(T, 3)`` (anchored by
            the field homography; FR-8).
        left_hand_pose, right_hand_pose, jaw_pose: Optional SMPL-X extras,
            shape ``(T, K, 3)`` / ``(T, 3)``; ``None`` for SMPL / SMPL-H.
        provenance: Per-frame :class:`Provenance`, shape ``(T,)``. Defaults to all
            ``MEASURED`` — every construction site *except* the row-fabricating gates
            (`coherence`, `kinematics`, `pose_motion_sync`) is copying detector output, and a
            third "unknown" state would only push the ambiguity downstream. The gates that
            invent rows must call :meth:`mark`.
    """

    frames: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray
    transl: np.ndarray
    left_hand_pose: np.ndarray | None = None
    right_hand_pose: np.ndarray | None = None
    jaw_pose: np.ndarray | None = None
    provenance: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=int).reshape(-1)
        t = self.frames.shape[0]
        self.global_orient = np.asarray(self.global_orient, dtype=float).reshape(t, 3)
        self.body_pose = np.asarray(self.body_pose, dtype=float).reshape(t, -1, 3)
        self.transl = np.asarray(self.transl, dtype=float).reshape(t, 3)
        self.provenance = _label_array(
            self.provenance, t, Provenance, _PROVENANCE_DTYPE, Provenance.MEASURED
        )

    def mark(self, rows: object, value: Provenance) -> None:
        """Stamp ``rows`` (indices or a bool mask) with ``value``. Mutates in place."""
        idx = np.asarray(rows)
        if idx.size:
            self.provenance[idx] = value.value

    @property
    def measured_mask(self) -> np.ndarray:
        """``(T,)`` bool — rows a detector actually produced, as opposed to rows we invented."""
        return self.provenance == Provenance.MEASURED.value

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def n_joints(self) -> int:
        return int(self.body_pose.shape[1])

    def frame_pos(self, frame_index: int) -> int:
        """Return the array row for a given frame index (raises if absent)."""
        hits = np.nonzero(self.frames == frame_index)[0]
        if hits.size == 0:
            raise KeyError(f"frame {frame_index} not in pose sequence")
        return int(hits[0])

    def copy(self) -> PoseSequence:
        return PoseSequence(
            frames=self.frames.copy(),
            global_orient=self.global_orient.copy(),
            body_pose=self.body_pose.copy(),
            transl=self.transl.copy(),
            left_hand_pose=None if self.left_hand_pose is None else self.left_hand_pose.copy(),
            right_hand_pose=None if self.right_hand_pose is None else self.right_hand_pose.copy(),
            jaw_pose=None if self.jaw_pose is None else self.jaw_pose.copy(),
            provenance=self.provenance.copy(),
        )

    @classmethod
    def rest(cls, frames: np.ndarray, n_joints: int = N_SMPLX_BODY_JOINTS) -> PoseSequence:
        """A rest-pose sequence (all zeros) over the given frames — handy for tests/fakes."""
        frames = np.asarray(frames, dtype=int).reshape(-1)
        t = frames.shape[0]
        return cls(
            frames=frames,
            global_orient=np.zeros((t, 3)),
            body_pose=np.zeros((t, n_joints, 3)),
            transl=np.zeros((t, 3)),
        )


@dataclass
class SubjectMotion:
    """Shape + pose for one subject. The unit the correction engine resolves."""

    shape: SmplxShape
    pose: PoseSequence

    def copy(self) -> SubjectMotion:
        return SubjectMotion(
            shape=SmplxShape(self.shape.betas.copy(), self.shape.body_model),
            pose=self.pose.copy(),
        )


@dataclass
class BallTrack:
    """Ball trajectory in 3D with explicit height confidence (FR-9, R-4).

    Mono height is recovered by ballistics and is genuinely uncertain, so
    ``height_confidence`` is a first-class, per-frame field rather than an afterthought.
    ``mode`` says *how* the height was arrived at, which confidence alone cannot.

    Attributes:
        frames: Frame indices, shape ``(T,)``.
        positions_3d: World positions (meters), shape ``(T, 3)``.
        height_confidence: Per-frame confidence in the Z component, ``[0, 1]``, shape ``(T,)``.
        track_2d: Optional image-space track (px), shape ``(T, 2)``.
        mode: Per-frame :class:`BallMode`, shape ``(T,)``. Defaults to all ``UNMEASURED``:
            a bare ``BallTrack`` has no contact segmentation behind it, and claiming ground
            contact we never established is exactly the fabrication R-6 forbids.
    """

    frames: np.ndarray
    positions_3d: np.ndarray
    height_confidence: np.ndarray
    track_2d: np.ndarray | None = None
    mode: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=int).reshape(-1)
        t = self.frames.shape[0]
        self.positions_3d = np.asarray(self.positions_3d, dtype=float).reshape(t, 3)
        self.height_confidence = np.asarray(self.height_confidence, dtype=float).reshape(t)
        if self.track_2d is not None:
            self.track_2d = np.asarray(self.track_2d, dtype=float).reshape(t, 2)
        self.mode = _label_array(
            self.mode, t, BallMode, _BALL_MODE_DTYPE, BallMode.UNMEASURED
        )

    @property
    def on_ground(self) -> np.ndarray:
        """``(T,)`` bool ground-contact mask, derived — :attr:`mode` is the source of truth."""
        return self.mode == BallMode.ON_GROUND.value

    def copy(self) -> BallTrack:
        return BallTrack(
            frames=self.frames.copy(),
            positions_3d=self.positions_3d.copy(),
            height_confidence=self.height_confidence.copy(),
            track_2d=None if self.track_2d is None else self.track_2d.copy(),
            mode=self.mode.copy(),
        )


@dataclass
class Ball2DTrack:
    """Raw 2D ball track from a :class:`BallTracker` adapter, before the core 3D lift."""

    frames: np.ndarray
    positions_2d: np.ndarray  # (T, 2) image px
    confidence: np.ndarray    # (T,) detection confidence

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=int).reshape(-1)
        t = self.frames.shape[0]
        self.positions_2d = np.asarray(self.positions_2d, dtype=float).reshape(t, 2)
        self.confidence = np.asarray(self.confidence, dtype=float).reshape(t)


@dataclass
class VectorCurve:
    """A generic dense per-frame vector curve (used by editors/tests for root paths).

    Not all trajectories need this — poses use :class:`PoseSequence`, the ball uses
    :class:`BallTrack`. This exists so corrections can be expressed uniformly over any
    ``(T, D)`` quantity when convenient.
    """

    frames: np.ndarray
    values: np.ndarray  # (T, D)
    label: str = "curve"

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=int).reshape(-1)
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)

    @property
    def dim(self) -> int:
        return int(self.values.shape[1])
