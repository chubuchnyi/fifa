"""The correction engine: ``resolved = proposal ⊕ corrections`` (FR-21, FR-22).

Pure functions, no ports except the optional :class:`PoseEstimator` injected for the
REFIT mode. The engine never mutates the proposal — :func:`resolve_subject_motion` and
:func:`resolve_ball` work on a deep copy, which is what makes the edit model
non-destructive and gives FR-23 (preview) for free.

The four propagation modes (FR-22) are implemented as small pure ops so they can be unit
tested in isolation:

* ``CONSTANT_OFFSET``     add a vector / compose a rotation across the range.
* ``KEYFRAME_INTERP``     fill the range between operator keyframes (lerp / slerp).
* ``TEMPORAL_SMOOTHING``  windowed moving-average / gaussian (quaternion-aware).
* ``REFIT``               re-run constraint-guided HMR via the injected port and splice.

Rotation targets are handled honestly in rotation space (compose / slerp / quaternion
average) — never by adding axis-angle vectors componentwise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..scene.layers import (
    Correction,
    CorrectionMode,
    CorrectionTarget,
    FrameRange,
    KeyframePayload,
    OffsetPayload,
    RefitPayload,
    SmoothingPayload,
    TargetKind,
)
from ..scene.motion import BallTrack, SubjectMotion
from .rotations import (
    average_quats,
    axis_angle_to_quat,
    compose_axis_angle,
    quat_to_axis_angle,
    slerp_quat,
)

if TYPE_CHECKING:
    from ..ports.io import ClipRef
    from ..ports.pose import PoseEstimator

_ROTATION_KINDS = (TargetKind.POSE_BODY_JOINT, TargetKind.ROOT_ORIENTATION)


# --- low-level pure ops (one per propagation mode) ----------------------------------


def apply_offset_vector(values: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Add a constant ``delta`` to every row of ``values`` (``(M, D)`` + ``(D,)``)."""
    return np.asarray(values, dtype=float) + np.asarray(delta, dtype=float).reshape(1, -1)


def apply_offset_rotation(aa_values: np.ndarray, offset_aa: np.ndarray) -> np.ndarray:
    """Left-compose ``offset_aa`` onto each axis-angle row (``(M, 3)``)."""
    a = np.asarray(aa_values, dtype=float).reshape(-1, 3)
    off = np.tile(np.asarray(offset_aa, dtype=float).reshape(1, 3), (a.shape[0], 1))
    return compose_axis_angle(off, a)


def interp_vector(
    target_frames: np.ndarray, key_frames: np.ndarray, key_values: np.ndarray
) -> np.ndarray:
    """Piecewise-linear interpolation per dimension, clamped at the ends."""
    tf = np.asarray(target_frames, dtype=float).reshape(-1)
    order = np.argsort(np.asarray(key_frames, dtype=float))
    kf = np.asarray(key_frames, dtype=float)[order]
    kv = np.asarray(key_values, dtype=float)[order]
    out = np.empty((tf.shape[0], kv.shape[1]))
    for d in range(kv.shape[1]):
        out[:, d] = np.interp(tf, kf, kv[:, d])  # np.interp clamps outside [kf0, kf-1]
    return out


def interp_rotation(
    target_frames: np.ndarray, key_frames: np.ndarray, key_values_aa: np.ndarray
) -> np.ndarray:
    """Piecewise spherical interpolation (slerp) between axis-angle keyframes."""
    tf = np.asarray(target_frames, dtype=float).reshape(-1)
    order = np.argsort(np.asarray(key_frames, dtype=float))
    kf = np.asarray(key_frames, dtype=float)[order]
    quats = axis_angle_to_quat(np.asarray(key_values_aa, dtype=float)[order].reshape(-1, 3))
    quats = quats.reshape(-1, 4)
    out = np.empty((tf.shape[0], 3))
    for i, f in enumerate(tf):
        if f <= kf[0]:
            out[i] = quat_to_axis_angle(quats[0])
        elif f >= kf[-1]:
            out[i] = quat_to_axis_angle(quats[-1])
        else:
            j = int(np.searchsorted(kf, f, side="right") - 1)  # segment [j, j+1]
            span = kf[j + 1] - kf[j]
            t = 0.0 if span == 0 else float((f - kf[j]) / span)
            q = slerp_quat(quats[j], quats[j + 1], np.atleast_1d(t))
            out[i] = quat_to_axis_angle(q[0])
    return out


def _smoothing_kernel(window: int, method: str, sigma: float) -> np.ndarray:
    half = max(int(window) // 2, 0)
    offsets = np.arange(-half, half + 1, dtype=float)
    if method == "gaussian":
        w = np.exp(-0.5 * (offsets / max(sigma, _TINY)) ** 2)
    else:  # moving_average
        w = np.ones(offsets.shape[0])
    return w / w.sum()


_TINY = 1e-12


def smooth_vector(
    values: np.ndarray, window: int = 5, method: str = "moving_average", sigma: float = 1.0
) -> np.ndarray:
    """Windowed smoothing of ``(M, D)`` along time (axis 0), edges clamped."""
    v = np.asarray(values, dtype=float)
    m = v.shape[0]
    if m == 0 or window <= 1:
        return v.copy()
    w = _smoothing_kernel(window, method, sigma)
    half = w.shape[0] // 2
    offsets = np.arange(-half, half + 1)
    out = np.empty_like(v)
    for i in range(m):
        idx = np.clip(i + offsets, 0, m - 1)
        out[i] = (w[:, None] * v[idx]).sum(axis=0)
    return out


def smooth_rotation(
    aa_values: np.ndarray, window: int = 5, method: str = "moving_average", sigma: float = 1.0
) -> np.ndarray:
    """Windowed smoothing of axis-angle rotations via quaternion averaging, edges clamped."""
    a = np.asarray(aa_values, dtype=float).reshape(-1, 3)
    m = a.shape[0]
    if m == 0 or window <= 1:
        return a.copy()
    quats = axis_angle_to_quat(a).reshape(-1, 4)
    w = _smoothing_kernel(window, method, sigma)
    half = w.shape[0] // 2
    offsets = np.arange(-half, half + 1)
    out = np.empty((m, 3))
    for i in range(m):
        idx = np.clip(i + offsets, 0, m - 1)
        out[i] = quat_to_axis_angle(average_quats(quats[idx], w))
    return out


# --- dispatch helpers ---------------------------------------------------------------


def _range_rows(frames_arr: np.ndarray, frame_range: FrameRange) -> np.ndarray:
    f = np.asarray(frames_arr)
    return np.nonzero((f >= frame_range.start) & (f <= frame_range.end))[0]


def _apply_inplace(
    arr: np.ndarray, frames_arr: np.ndarray, corr: Correction, *, is_rotation: bool
) -> None:
    """Apply one non-REFIT correction to the rows of ``arr`` that fall in its range."""
    rows = _range_rows(frames_arr, corr.frame_range)
    if rows.size == 0:
        return
    mode = corr.mode
    if mode == CorrectionMode.CONSTANT_OFFSET:
        delta = corr.payload.delta
        arr[rows] = (
            apply_offset_rotation(arr[rows], delta)
            if is_rotation
            else apply_offset_vector(arr[rows], delta)
        )
    elif mode == CorrectionMode.KEYFRAME_INTERP:
        tf = np.asarray(frames_arr)[rows]
        p = corr.payload
        arr[rows] = (
            interp_rotation(tf, p.key_frames, p.key_values)
            if is_rotation
            else interp_vector(tf, p.key_frames, p.key_values)
        )
    elif mode == CorrectionMode.TEMPORAL_SMOOTHING:
        p = corr.payload
        arr[rows] = (
            smooth_rotation(arr[rows], p.window, p.method, p.sigma)
            if is_rotation
            else smooth_vector(arr[rows], p.window, p.method, p.sigma)
        )
    else:  # REFIT is resolved at the motion level, not per-array
        raise ValueError(f"{mode} is not an in-place array mode")


def _splice_refit(
    resolved: SubjectMotion,
    corr: Correction,
    refit_port: PoseEstimator | None,
    clip: ClipRef | None,
) -> None:
    if refit_port is None or clip is None:
        raise ValueError(
            "REFIT correction requires a refit_port and clip; "
            "pass refit_port=<PoseEstimator> and clip=<ClipRef> to resolve."
        )
    rows = _range_rows(resolved.pose.frames, corr.frame_range)
    if rows.size == 0:
        return
    frames = resolved.pose.frames[rows]
    refit = refit_port.refit(clip, resolved, corr.payload.constraints, frames)
    for f in frames:
        src, dst = refit.pose.frame_pos(int(f)), resolved.pose.frame_pos(int(f))
        resolved.pose.global_orient[dst] = refit.pose.global_orient[src]
        resolved.pose.body_pose[dst] = refit.pose.body_pose[src]
        resolved.pose.transl[dst] = refit.pose.transl[src]


# --- public resolve / preview -------------------------------------------------------


def resolve_subject_motion(
    proposal: SubjectMotion,
    corrections: Iterable[Correction],
    *,
    refit_port: PoseEstimator | None = None,
    clip: ClipRef | None = None,
) -> SubjectMotion:
    """Return ``proposal ⊕ corrections`` for one subject; ``proposal`` is never mutated.

    Corrections apply in list order (later ones see earlier results). Ball corrections are
    ignored here (see :func:`resolve_ball`). Pass only one subject's corrections, e.g. via
    ``Scene.corrections_for(track_id)``.
    """
    resolved = proposal.copy()
    for corr in corrections:
        if not corr.enabled:
            continue
        if corr.mode == CorrectionMode.REFIT:
            _splice_refit(resolved, corr, refit_port, clip)
            continue
        kind = corr.target.kind
        if kind == TargetKind.POSE_BODY_JOINT:
            j = corr.target.joint_index
            if j is None:
                raise ValueError("POSE_BODY_JOINT correction requires target.joint_index")
            _apply_inplace(
                resolved.pose.body_pose[:, j, :], resolved.pose.frames, corr, is_rotation=True
            )
        elif kind == TargetKind.ROOT_ORIENTATION:
            _apply_inplace(
                resolved.pose.global_orient, resolved.pose.frames, corr, is_rotation=True
            )
        elif kind == TargetKind.ROOT_TRANSLATION:
            _apply_inplace(resolved.pose.transl, resolved.pose.frames, corr, is_rotation=False)
        elif kind == TargetKind.SHAPE_BETA:
            if corr.mode != CorrectionMode.CONSTANT_OFFSET:
                raise ValueError("SHAPE_BETA supports CONSTANT_OFFSET only (β is frame-invariant)")
            b, delta = resolved.shape.betas, corr.payload.delta
            n = min(b.shape[0], delta.shape[0])
            b[:n] = b[:n] + delta[:n]
        elif kind == TargetKind.BALL_POSITION:
            continue  # resolved by resolve_ball
        else:
            raise ValueError(f"unknown target kind {kind}")
    return resolved


def resolve_ball(proposal: BallTrack, corrections: Iterable[Correction]) -> BallTrack:
    """Return the ball trajectory with its BALL_POSITION corrections applied (copy-safe)."""
    resolved = proposal.copy()
    for corr in corrections:
        if not corr.enabled or corr.target.kind != TargetKind.BALL_POSITION:
            continue
        if corr.mode == CorrectionMode.REFIT:
            raise ValueError("REFIT is not defined for the ball trajectory")
        _apply_inplace(resolved.positions_3d, resolved.frames, corr, is_rotation=False)
    return resolved


def preview_subject_motion(
    proposal: SubjectMotion,
    corrections: Sequence[Correction],
    candidate: Correction,
    *,
    refit_port: PoseEstimator | None = None,
    clip: ClipRef | None = None,
) -> SubjectMotion:
    """FR-23: resolve *as if* ``candidate`` were added, without storing it or mutating state."""
    return resolve_subject_motion(
        proposal, [*corrections, candidate], refit_port=refit_port, clip=clip
    )


# --- correction constructors (typed, serializable deltas) ---------------------------


def _as_range(frame_range: FrameRange | tuple[int, int]) -> FrameRange:
    if isinstance(frame_range, FrameRange):
        return frame_range
    start, end = frame_range
    return FrameRange(int(start), int(end))


def make_offset(
    id: str,
    target: CorrectionTarget,
    frame_range: FrameRange | tuple[int, int],
    delta: np.ndarray,
    *,
    note: str | None = None,
    created_at: str | None = None,
) -> Correction:
    """Build a CONSTANT_OFFSET correction (vector add or axis-angle compose)."""
    return Correction(
        id=id,
        target=target,
        frame_range=_as_range(frame_range),
        mode=CorrectionMode.CONSTANT_OFFSET,
        payload=OffsetPayload(delta=np.asarray(delta, dtype=float)),
        note=note,
        created_at=created_at,
    )


def make_keyframes(
    id: str,
    target: CorrectionTarget,
    frame_range: FrameRange | tuple[int, int],
    key_frames: np.ndarray,
    key_values: np.ndarray,
    *,
    interp: str = "linear",
    note: str | None = None,
    created_at: str | None = None,
) -> Correction:
    """Build a KEYFRAME_INTERP correction (lerp for vectors, slerp for rotations)."""
    return Correction(
        id=id,
        target=target,
        frame_range=_as_range(frame_range),
        mode=CorrectionMode.KEYFRAME_INTERP,
        payload=KeyframePayload(key_frames=key_frames, key_values=key_values, interp=interp),
        note=note,
        created_at=created_at,
    )


def make_refit(
    id: str,
    target: CorrectionTarget,
    frame_range: FrameRange | tuple[int, int],
    constraints: dict | None = None,
    *,
    note: str | None = None,
    created_at: str | None = None,
) -> Correction:
    """Build a REFIT correction; ``constraints`` is opaque and passed to the port."""
    return Correction(
        id=id,
        target=target,
        frame_range=_as_range(frame_range),
        mode=CorrectionMode.REFIT,
        payload=RefitPayload(constraints=dict(constraints or {})),
        note=note,
        created_at=created_at,
    )


def make_smoothing(
    id: str,
    target: CorrectionTarget,
    frame_range: FrameRange | tuple[int, int],
    *,
    window: int = 5,
    method: str = "moving_average",
    sigma: float = 1.0,
    note: str | None = None,
    created_at: str | None = None,
) -> Correction:
    """Build a TEMPORAL_SMOOTHING correction."""
    return Correction(
        id=id,
        target=target,
        frame_range=_as_range(frame_range),
        mode=CorrectionMode.TEMPORAL_SMOOTHING,
        payload=SmoothingPayload(window=window, method=method, sigma=sigma),
        note=note,
        created_at=created_at,
    )
