"""The three-layer, non-destructive edit model and confidence metadata.

Layers (FR-21, UX-5):
    ``proposal``    raw model output, stored on each :class:`Subject` / :class:`BallTrack`.
    ``corrections`` a list of :class:`Correction` deltas — the only thing edits create.
    ``resolved``    ``proposal ⊕ corrections``, computed by ``core.correction`` on demand.

A correction is a *typed delta*: a target (which quantity), a frame range, a mode (one of
the four propagation modes, FR-22) and a mode-specific payload. Corrections are toggleable
(``enabled``) so the operator can compare/reset/disable without losing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Layer(str, Enum):
    PROPOSAL = "proposal"
    CORRECTIONS = "corrections"
    RESOLVED = "resolved"


class TargetKind(str, Enum):
    """What a correction acts on."""

    POSE_BODY_JOINT = "pose_body_joint"     # one SMPL-X body joint rotation
    ROOT_ORIENTATION = "root_orientation"   # global_orient rotation
    ROOT_TRANSLATION = "root_translation"   # root transl (meters)
    SHAPE_BETA = "shape_beta"               # β shape coefficients
    BALL_POSITION = "ball_position"         # ball 3D position (meters)
    FIELD_CALIBRATION = "field_calibration"  # where the pitch model sits on its own plane


class CorrectionMode(str, Enum):
    """The four propagation modes (FR-22 a–d)."""

    CONSTANT_OFFSET = "constant_offset"        # (a)
    KEYFRAME_INTERP = "keyframe_interp"        # (b)
    REFIT = "refit"                            # (c) constraint-guided HMR (injected port)
    TEMPORAL_SMOOTHING = "temporal_smoothing"  # (d)


@dataclass
class FrameRange:
    """Inclusive ``[start, end]`` frame range."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"FrameRange end {self.end} < start {self.start}")

    def frames(self) -> np.ndarray:
        return np.arange(self.start, self.end + 1, dtype=int)

    def __contains__(self, frame: int) -> bool:
        return self.start <= frame <= self.end


@dataclass
class CorrectionTarget:
    """Addresses the exact quantity a correction edits."""

    kind: TargetKind
    subject_track_id: int | None = None  # None for ball / global quantities
    joint_index: int | None = None       # required iff kind == POSE_BODY_JOINT


# --- Mode-specific payloads ----------------------------------------------------
# Each carries exactly what its propagation mode needs; all are plain serializable data.


@dataclass
class OffsetPayload:
    """CONSTANT_OFFSET: a fixed delta applied across the range.

    For vector targets (translation/ball) this is added. For rotation targets it is an
    **axis-angle offset rotation** composed onto the proposal (left-multiply).
    """

    delta: np.ndarray  # (3,) vector or axis-angle; (n_betas,) for SHAPE_BETA

    def __post_init__(self) -> None:
        self.delta = np.asarray(self.delta, dtype=float).reshape(-1)


@dataclass
class PlaneTransformPayload:
    """CONSTANT_OFFSET for FIELD_CALIBRATION: a 3x3 similarity of the pitch plane.

    Composed on the **world** side — ``H'_world→image = H_world→image @ matrix`` — which is
    what makes it safe. A world-plane similarity leaves ``K`` and the camera's rotation basis
    intact (``K[r₁ r₂ t] @ B`` is again ``K[r₁' r₂' t']`` with ``r₁' ⟂ r₂'``), so re-registering
    the pitch by hand cannot turn a camera-realizable calibration into an unrealizable one — the
    #107 check keeps reading "one camera". An image-side ``A @ H`` would offer no such guarantee.

    Restricting to a similarity (rotate, uniform scale, translate — 4 DOF) is the same argument
    from the other end: a general projective nudge would leave the pitch a non-rectangle, i.e. no
    longer the object whose dimensions we know.
    """

    matrix: np.ndarray  # (3, 3), acts on homogeneous pitch-plane points [x, y, 1]

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float).reshape(3, 3)


@dataclass
class KeyframePayload:
    """KEYFRAME_INTERP: operator keyframes; fill the range between them.

    Vectors interpolate linearly; rotations via slerp. Values are **absolute** target
    values at the key frames (not deltas).
    """

    key_frames: np.ndarray   # (K,) frame indices
    key_values: np.ndarray   # (K, D) absolute values
    interp: str = "linear"   # "linear" | "slerp" (auto-selected by target kind)

    def __post_init__(self) -> None:
        self.key_frames = np.asarray(self.key_frames, dtype=int).reshape(-1)
        self.key_values = np.asarray(self.key_values, dtype=float)
        if self.key_values.ndim == 1:
            self.key_values = self.key_values.reshape(-1, 1)


@dataclass
class RefitPayload:
    """REFIT: re-run constraint-guided HMR on the range via the injected port.

    ``constraints`` is adapter-defined (e.g. 2D keypoint hints, foot-contact locks);
    the core treats it as opaque and passes it to :meth:`PoseEstimator.refit`.
    """

    constraints: dict = field(default_factory=dict)


@dataclass
class SmoothingPayload:
    """TEMPORAL_SMOOTHING: windowed smoothing over the range."""

    window: int = 5                 # odd window length in frames
    method: str = "moving_average"  # "moving_average" | "gaussian"
    sigma: float = 1.0              # for gaussian


@dataclass
class Correction:
    """A single non-destructive edit delta.

    Attributes:
        id: Stable identifier (for undo/compare/toggle).
        target: What this edits.
        frame_range: Where it applies (inclusive).
        mode: Which of the four propagation modes produced/defines it.
        payload: Mode-specific data (one of the *Payload dataclasses above).
        enabled: If False the resolver skips it (compare / reset without deletion).
        note: Optional operator note.
        created_at: Optional ISO timestamp string.
    """

    id: str
    target: CorrectionTarget
    frame_range: FrameRange
    mode: CorrectionMode
    payload: object
    enabled: bool = True
    note: str | None = None
    created_at: str | None = None


@dataclass
class ConfidenceMap:
    """Per-subject / per-field confidence driving the "needs attention" list (FR-17, UX-4).

    All arrays are keyed by ``track_id``. Reprojection error is in pixels (FR-16).
    """

    subject_frame_conf: dict = field(default_factory=dict)      # track_id -> (T,)
    subject_joint_conf: dict = field(default_factory=dict)      # track_id -> (T, J)
    reprojection_error_px: dict = field(default_factory=dict)   # track_id -> (T,)
    field_homography_conf: np.ndarray | None = None             # (T,)
