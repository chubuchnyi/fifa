"""M3-2: homography-anchor validation + hardened anchor-locked re-fit + gated occlusion seam.

The *pure* halves are real and exercised with no torch/GPU: the measured-over-generative anchor
validator (``core.correction.anchor``) and the ``foot_anchor`` re-fit lock shared by the real and
fake :class:`PoseEstimator`s. The generative cluster-occlusion completer (Diffusion-VAS + SAM-3)
is gated — it raises until wired (R-8) — but its seam is verified: protocol, injection, and that
re-fit engages it before applying the anchor lock on top.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.adapters.fakes import FakePoseEstimator
from pitch3d.adapters.models.pose import (
    DiffusionVasOcclusionBackend,
    GVHMRPoseEstimator,
    OcclusionBackend,
)
from pitch3d.app.wiring import default_ports
from pitch3d.core.correction.anchor import (
    anchor_residuals,
    blend_to_anchor,
    validate_against_anchor,
)
from pitch3d.core.ports.io import ClipRef
from pitch3d.core.scene.motion import (
    N_SMPLX_BODY_JOINTS,
    PoseSequence,
    SmplxShape,
    SubjectMotion,
)

_OCCL_SPEC = "pitch3d.adapters.models.pose:DiffusionVasOcclusionBackend"


def _clip(n: int = 3) -> ClipRef:
    return ClipRef(source_id="s", uri="x", frames=np.arange(n), width=1280, height=720, fps=25.0)


def _motion(frames, transl) -> SubjectMotion:
    frames = np.asarray(frames, dtype=int)
    t = frames.shape[0]
    return SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames,
            global_orient=np.zeros((t, 3)),
            body_pose=np.ones((t, N_SMPLX_BODY_JOINTS, 3)),
            transl=np.asarray(transl, dtype=float),
        ),
    )


# --- pure anchor math ---------------------------------------------------------------
def test_anchor_residuals_per_frame_and_broadcast():
    transl = np.array([[0, 0, 0.9], [3, 4, 0.9], [1, 0, 0.9]], dtype=float)
    np.testing.assert_allclose(anchor_residuals(transl, np.zeros((3, 2))), [0, 5, 1])
    np.testing.assert_allclose(anchor_residuals(transl, [0, 0]), [0, 5, 1])  # (2,) broadcast


def test_blend_to_anchor_endpoints_and_midpoint():
    cur = np.array([[0, 0], [10, 0]], dtype=float)
    target = np.array([[2, 0], [2, 0]], dtype=float)
    np.testing.assert_allclose(blend_to_anchor(cur, target, 1.0), target)        # hard lock
    np.testing.assert_allclose(blend_to_anchor(cur, target, 0.0), cur)           # no-op
    np.testing.assert_allclose(blend_to_anchor(cur, target, 0.5), [[1, 0], [6, 0]])
    np.testing.assert_allclose(blend_to_anchor(cur, [2, 0], 1.0), target)        # (2,) broadcast


def test_validate_against_anchor_flags_off_anchor():
    frames = np.array([0, 1, 2])
    transl = np.array([[0, 0, 0.9], [3, 0, 0.9], [0.1, 0, 0.9]], dtype=float)
    rep = validate_against_anchor(frames, transl, np.zeros((3, 2)), max_residual_m=0.5)
    assert (rep.n_frames, rep.n_valid, rep.n_off_anchor) == (3, 2, 1)
    np.testing.assert_array_equal(rep.off_anchor_frames, [1])  # only the 3.0 m drift fails
    assert rep.worst_residual_m == pytest.approx(3.0)
    assert rep.mean_residual_m == pytest.approx((0 + 3 + 0.1) / 3)


def test_anchor_residuals_rejects_ragged_per_frame_anchor():
    with pytest.raises(ValueError, match="expected 3"):
        anchor_residuals(np.zeros((3, 3)), np.zeros((2, 2)))


# --- the hardened re-fit anchor lock (both adapters share the contract) --------------
@pytest.mark.parametrize("make", [FakePoseEstimator, GVHMRPoseEstimator], ids=["fake", "gvhmr"])
def test_refit_foot_anchor_locks_root_xy_only(make):
    pse = make()
    motion = _motion([0, 1, 2], [[5, 5, 0.9], [6, 6, 0.9], [7, 7, 0.9]])
    refined = pse.refit(_clip(), motion, {"foot_anchor": [1.0, 2.0]}, np.array([0, 1, 2]))
    np.testing.assert_allclose(refined.pose.transl[:, :2], [[1, 2]] * 3)   # XY locked
    np.testing.assert_allclose(refined.pose.transl[:, 2], 0.9)             # Z untouched (R-4)
    np.testing.assert_allclose(motion.pose.transl[:, :2], [[5, 5], [6, 6], [7, 7]])  # input intact


def test_refit_foot_anchor_per_frame_on_selected_rows():
    pse = GVHMRPoseEstimator()
    motion = _motion([0, 1, 2], [[5, 5, 0.9], [6, 6, 0.9], [7, 7, 0.9]])
    refined = pse.refit(
        _clip(), motion, {"foot_anchor": np.array([[0, 0], [9, 9]])}, np.array([0, 2])
    )
    np.testing.assert_allclose(refined.pose.transl[0, :2], [0, 0])  # frame 0 → first anchor row
    np.testing.assert_allclose(refined.pose.transl[2, :2], [9, 9])  # frame 2 → second anchor row
    np.testing.assert_allclose(refined.pose.transl[1, :2], [6, 6])  # frame 1 not selected


def test_refit_anchor_blend_pulls_partway():
    pse = GVHMRPoseEstimator()
    motion = _motion([0], [[10, 0, 0.9]])
    refined = pse.refit(
        _clip(), motion, {"foot_anchor": [0, 0], "anchor_blend": 0.25}, np.array([0])
    )
    np.testing.assert_allclose(refined.pose.transl[0, :2], [7.5, 0.0])


def test_refit_to_anchor_then_validate_is_all_on_anchor():
    pse = GVHMRPoseEstimator()
    frames = np.array([0, 1, 2])
    anchor = np.array([[1, 1], [2, 2], [3, 3]], dtype=float)
    motion = _motion(frames, [[5, 5, 0.9], [6, 6, 0.9], [7, 7, 0.9]])
    refined = pse.refit(_clip(), motion, {"foot_anchor": anchor}, frames)
    rep = validate_against_anchor(refined.pose.frames, refined.pose.transl, anchor)
    assert rep.n_off_anchor == 0 and rep.n_valid == 3


# --- gated cluster-occlusion completion (R-8) ---------------------------------------
def test_occlusion_backend_satisfies_protocol_and_is_gated():
    be = DiffusionVasOcclusionBackend()
    assert isinstance(be, OcclusionBackend)  # structural: has complete_occlusions
    with pytest.raises(NotImplementedError, match="occlusion"):
        be.complete_occlusions(_clip(), _motion([0], [[0, 0, 0.9]]), np.array([0]))


def test_refit_complete_occlusions_without_backend_is_actionable():
    pse = GVHMRPoseEstimator()  # no occlusion backend injected
    with pytest.raises(NotImplementedError, match="OcclusionBackend"):
        pse.refit(
            _clip(), _motion([0], [[0, 0, 0.9]]), {"complete_occlusions": True}, np.array([0])
        )


class _StubOcclusion:
    """A non-gated stand-in: marks the motion so we can prove re-fit engaged it (ADR-0006 BYO)."""

    def __init__(self) -> None:
        self.called = False

    def complete_occlusions(
        self, clip: ClipRef, motion: SubjectMotion, frames: np.ndarray
    ) -> SubjectMotion:
        self.called = True
        out = motion.copy()
        out.pose.transl[:, 2] = 7.0  # sentinel height proving this output flows downstream
        return out


def test_refit_runs_injected_occlusion_then_applies_anchor_on_top():
    stub = _StubOcclusion()
    assert isinstance(stub, OcclusionBackend)
    pse = GVHMRPoseEstimator(occlusion_backend=stub)
    motion = _motion([0, 1], [[5, 5, 0.9], [6, 6, 0.9]])
    refined = pse.refit(
        _clip(), motion,
        {"complete_occlusions": True, "foot_anchor": [1, 2]},
        np.array([0, 1]),
    )
    assert stub.called
    np.testing.assert_allclose(refined.pose.transl[:, 2], [7.0, 7.0])      # completion output used
    np.testing.assert_allclose(refined.pose.transl[:, :2], [[1, 2], [1, 2]])  # anchor lock on top


# --- wiring (ADR-0006 dotted-path injection, gated to --pose gvhmr) ------------------
def test_wiring_injects_occlusion_backend_into_gvhmr():
    ports = default_ports(pose="gvhmr", occlusion_backend=_OCCL_SPEC)
    assert isinstance(ports.pose.occlusion_backend, OcclusionBackend)


def test_wiring_occlusion_backend_requires_gvhmr():
    with pytest.raises(ValueError, match="requires --pose gvhmr"):
        default_ports(occlusion_backend=_OCCL_SPEC)  # pose defaults to fake
