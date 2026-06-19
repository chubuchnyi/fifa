"""Rotation math — round-trips, the θ=π Shepperd branch, compose/slerp/average."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.correction.rotations import (
    average_quats,
    axis_angle_to_matrix,
    axis_angle_to_quat,
    compose_axis_angle,
    matrix_to_quat,
    quat_mul,
    quat_to_axis_angle,
    slerp_quat,
)


def _R(aa) -> np.ndarray:
    return axis_angle_to_matrix(np.asarray(aa, dtype=float))


def test_axis_angle_quat_roundtrip_via_matrix():
    rng = np.random.default_rng(0)
    for _ in range(25):
        v = rng.normal(size=3)
        aa = v / (np.linalg.norm(v) + 1e-9) * rng.uniform(0.01, 3.0)
        q = axis_angle_to_quat(aa)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-9)
        np.testing.assert_allclose(_R(aa), _R(quat_to_axis_angle(q)), atol=1e-7)


@pytest.mark.parametrize("axis", list(np.eye(3)))
def test_matrix_to_quat_handles_pi(axis):
    aa = axis * np.pi
    m = axis_angle_to_matrix(aa)
    assert np.trace(m) < 0.0  # degenerate branch (naive axis extraction would fail here)
    q = matrix_to_quat(m)  # Shepperd's method must stay finite and reconstruct the matrix
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-9)
    np.testing.assert_allclose(_R(quat_to_axis_angle(q)), m, atol=1e-6)


def test_compose_equals_matrix_product():
    a = np.array([0.3, -0.4, 0.1])
    b = np.array([-0.2, 0.5, 0.25])
    np.testing.assert_allclose(_R(compose_axis_angle(a, b)), _R(a) @ _R(b), atol=1e-7)


def test_slerp_endpoints_and_midpoint():
    q0 = axis_angle_to_quat(np.zeros(3))
    q1 = axis_angle_to_quat(np.array([0.0, 0.0, np.pi / 2]))
    out = slerp_quat(q0, q1, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(_R(quat_to_axis_angle(out[0])), np.eye(3), atol=1e-7)
    np.testing.assert_allclose(quat_to_axis_angle(out[1]), [0.0, 0.0, np.pi / 4], atol=1e-6)
    np.testing.assert_allclose(_R(quat_to_axis_angle(out[2])), _R([0, 0, np.pi / 2]), atol=1e-7)


def test_average_of_identical_quats():
    q = axis_angle_to_quat(np.array([0.0, 0.0, 0.3]))
    avg = average_quats(np.stack([q, q, q]))
    np.testing.assert_allclose(_R(quat_to_axis_angle(avg)), _R([0, 0, 0.3]), atol=1e-7)


def test_quat_mul_identity():
    q = axis_angle_to_quat(np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(quat_mul(np.array([1.0, 0, 0, 0]), q), q, atol=1e-9)
