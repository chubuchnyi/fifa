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


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → unit quaternion (w, x, y, z), Shepperd's method.

    Robust for *all* rotations, including the ``theta = π`` case where the naive
    antisymmetric-part axis extraction degenerates (the difference terms vanish). We pick the
    largest of the four denominators per matrix for numerical stability.
    """
    m = np.asarray(r, dtype=float)
    squeeze = m.ndim == 2
    m = m.reshape(-1, 3, 3)
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    m21, m12 = m[:, 2, 1], m[:, 1, 2]
    m02, m20 = m[:, 0, 2], m[:, 2, 0]
    m10, m01 = m[:, 1, 0], m[:, 0, 1]
    t = m00 + m11 + m22
    q = np.zeros((m.shape[0], 4))

    c0 = t > 0.0
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    c2 = (~c0) & (~c1) & (m11 >= m22)
    c3 = ~(c0 | c1 | c2)

    def _s(diag: np.ndarray, mask: np.ndarray) -> np.ndarray:
        s = np.sqrt(np.maximum(diag[mask], 0.0)) * 2.0
        return np.where(s < _EPS, 1.0, s)

    if np.any(c0):  # trace > 0: largest term is w
        s = _s(t + 1.0, c0)
        q[c0, 0] = 0.25 * s
        q[c0, 1] = (m21[c0] - m12[c0]) / s
        q[c0, 2] = (m02[c0] - m20[c0]) / s
        q[c0, 3] = (m10[c0] - m01[c0]) / s
    if np.any(c1):  # m00 dominant: largest term is x
        s = _s(1.0 + m00 - m11 - m22, c1)
        q[c1, 0] = (m21[c1] - m12[c1]) / s
        q[c1, 1] = 0.25 * s
        q[c1, 2] = (m01[c1] + m10[c1]) / s
        q[c1, 3] = (m02[c1] + m20[c1]) / s
    if np.any(c2):  # m11 dominant: largest term is y
        s = _s(1.0 + m11 - m00 - m22, c2)
        q[c2, 0] = (m02[c2] - m20[c2]) / s
        q[c2, 1] = (m01[c2] + m10[c2]) / s
        q[c2, 2] = 0.25 * s
        q[c2, 3] = (m12[c2] + m21[c2]) / s
    if np.any(c3):  # m22 dominant: largest term is z
        s = _s(1.0 + m22 - m00 - m11, c3)
        q[c3, 0] = (m10[c3] - m01[c3]) / s
        q[c3, 1] = (m02[c3] + m20[c3]) / s
        q[c3, 2] = (m12[c3] + m21[c3]) / s
        q[c3, 3] = 0.25 * s

    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(q[:, 0:1] < 0, -q, q)  # canonical hemisphere (w >= 0)
    return q[0] if squeeze else q


def matrix_to_axis_angle(r: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → axis-angle (via the robust quaternion extraction)."""
    return quat_to_axis_angle(matrix_to_quat(r))


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
