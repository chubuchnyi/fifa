"""Rotation math for SMPL-X poses — axis-angle, quaternion, Rodrigues, slerp.

Pose joints are stored as **axis-angle** 3-vectors (SMPL-X convention). Correcting a
rotation must happen *in rotation space* (compose rotations, slerp between them) — never
by adding axis-angle vectors componentwise, which is wrong for anything but tiny angles.
These pure-numpy helpers give the correction engine that honest math.

Quaternions are ``(w, x, y, z)``. Functions accept a single vector ``(3,)``/``(4,)`` or a
batch ``(N, 3)``/``(N, 4)``.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _atleast_2d(x: np.ndarray, dim: int) -> tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(1, dim), True
    return x.reshape(-1, dim), False


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle → unit quaternion (w, x, y, z)."""
    a, squeeze = _atleast_2d(aa, 3)
    theta = np.linalg.norm(a, axis=1, keepdims=True)
    small = theta < _EPS
    half = theta / 2.0
    # sin(half)/theta, stable as theta->0 (→ 1/2)
    k = np.where(small, 0.5, np.sin(half) / np.where(small, 1.0, theta))
    w = np.cos(half)
    xyz = a * k
    q = np.concatenate([w, xyz], axis=1)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return q[0] if squeeze else q


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) → axis-angle."""
    a, squeeze = _atleast_2d(q, 4)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    w = np.clip(a[:, 0:1], -1.0, 1.0)
    xyz = a[:, 1:4]
    s = np.sqrt(np.clip(1.0 - w * w, 0.0, 1.0))
    theta = 2.0 * np.arccos(w)
    small = s < _EPS
    axis = np.where(small, 0.0, xyz / np.where(small, 1.0, s))
    aa = axis * theta
    return aa[0] if squeeze else aa


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (w, x, y, z)."""
    aa, sa = _atleast_2d(a, 4)
    bb, sb = _atleast_2d(b, 4)
    aw, ax, ay, az = aa[:, 0], aa[:, 1], aa[:, 2], aa[:, 3]
    bw, bx, by, bz = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
    out = np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )
    return out[0] if (sa and sb) else out


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Axis-angle → 3x3 rotation matrix (Rodrigues), batched."""
    a, squeeze = _atleast_2d(aa, 3)
    theta = np.linalg.norm(a, axis=1)
    out = np.tile(np.eye(3), (a.shape[0], 1, 1))
    nz = theta > _EPS
    if np.any(nz):
        axis = a[nz] / theta[nz, None]
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        zeros = np.zeros_like(x)
        k = np.stack(
            [zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=1
        ).reshape(-1, 3, 3)
        st = np.sin(theta[nz])[:, None, None]
        ct = np.cos(theta[nz])[:, None, None]
        eye = np.eye(3)[None]
        out[nz] = eye + st * k + (1.0 - ct) * (k @ k)
    return out[0] if squeeze else out


def matrix_to_axis_angle(r: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → axis-angle (via quaternion, numerically stable)."""
    m = np.asarray(r, dtype=float)
    squeeze = m.ndim == 2
    m = m.reshape(-1, 3, 3)
    trace = np.clip((m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(trace)
    q = np.zeros((m.shape[0], 4))
    q[:, 0] = np.cos(theta / 2.0)
    vec = np.stack(
        [m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]],
        axis=1,
    )
    n = np.linalg.norm(vec, axis=1, keepdims=True)
    sin_half = np.sin(theta / 2.0)[:, None]
    axis = np.where(n < _EPS, 0.0, vec / np.where(n < _EPS, 1.0, n))
    q[:, 1:4] = axis * sin_half
    aa = quat_to_axis_angle(q)
    aa = aa.reshape(-1, 3)
    return aa[0] if squeeze else aa


def compose_axis_angle(offset_aa: np.ndarray, base_aa: np.ndarray) -> np.ndarray:
    """Left-compose: result rotation = ``R(offset) · R(base)``, returned as axis-angle.

    This is the correct way to apply a rotational offset correction.
    """
    q = quat_mul(axis_angle_to_quat(offset_aa), axis_angle_to_quat(base_aa))
    return quat_to_axis_angle(q)


def slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions."""
    a = np.asarray(q0, dtype=float).reshape(4)
    b = np.asarray(q1, dtype=float).reshape(4)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:  # take the shorter arc
        b = -b
        dot = -dot
    t = np.asarray(t, dtype=float)
    if dot > 0.9995:  # nearly parallel → normalized lerp
        out = a[None, :] + t[:, None] * (b - a)[None, :]
        out = out / np.linalg.norm(out, axis=1, keepdims=True)
        return out
    theta0 = np.arccos(dot)
    sin0 = np.sin(theta0)
    s0 = np.sin((1.0 - t) * theta0) / sin0
    s1 = np.sin(t * theta0) / sin0
    return s0[:, None] * a[None, :] + s1[:, None] * b[None, :]


def slerp_axis_angle(aa0: np.ndarray, aa1: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Slerp between two axis-angle rotations; ``t`` may be scalar or ``(K,)``."""
    q = slerp_quat(axis_angle_to_quat(aa0), axis_angle_to_quat(aa1), np.atleast_1d(t))
    return quat_to_axis_angle(q)


def average_quats(quats: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted average of quaternions (Markley eigenvector method), sign-robust.

    Used by rotational temporal smoothing. ``quats`` is ``(K, 4)``.
    """
    q = np.asarray(quats, dtype=float).reshape(-1, 4)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    if weights is None:
        weights = np.ones(q.shape[0])
    w = np.asarray(weights, dtype=float).reshape(-1)
    m = (q * w[:, None]).T @ q  # 4x4 accumulation
    eigvals, eigvecs = np.linalg.eigh(m)
    out = eigvecs[:, int(np.argmax(eigvals))]
    if out[0] < 0:
        out = -out
    return out / np.linalg.norm(out)
