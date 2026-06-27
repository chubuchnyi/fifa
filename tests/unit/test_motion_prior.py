"""M3-8: learned motion-prior smoothing — the seam, the fake-real denoiser, the gated model.

``method="learned"`` routes a ``TEMPORAL_SMOOTHING`` correction through an injected ``MotionPrior``
port (mirroring how REFIT routes through ``refit_port``), so a learned denoiser stays a normal,
inspectable correction (ADR-0002). The :class:`FakeMotionPrior` is a real, GPU-free gaussian
denoiser (the seam runs and is tested now); :class:`LearnedMotionPrior` (HTD-Refine/StableMotion)
is the gated swap-in (R-8). All exercised with no torch/GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.adapters.fakes import FakeMotionPrior
from pitch3d.adapters.models import LearnedMotionPrior
from pitch3d.app.wiring import default_ports
from pitch3d.core.correction.engine import (
    make_smoothing,
    preview_subject_motion,
    resolve_subject_motion,
)
from pitch3d.core.ports.motion_prior import MotionPrior
from pitch3d.core.scene.layers import CorrectionTarget, TargetKind
from pitch3d.core.scene.motion import (
    N_SMPLX_BODY_JOINTS,
    PoseSequence,
    SmplxShape,
    SubjectMotion,
)
from pitch3d.core.scene.provenance import Backend, ModelInfo

_MP_SPEC = "pitch3d.adapters.fakes.motion_prior:FakeMotionPrior"


def _step_transl(n: int = 7) -> np.ndarray:
    t = np.zeros((n, 3))
    t[n // 2:, 2] = 1.0  # a step in Z — the discontinuity a denoiser visibly softens
    return t


def _motion(transl=None, global_orient=None, n: int = 7) -> SubjectMotion:
    frames = np.arange(n)
    transl = _step_transl(n) if transl is None else np.asarray(transl, float)
    go = np.zeros((n, 3)) if global_orient is None else np.asarray(global_orient, float)
    return SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=go,
            body_pose=np.zeros((n, N_SMPLX_BODY_JOINTS, 3)), transl=transl,
        ),
    )


def _root_t(tid: int = 1) -> CorrectionTarget:
    return CorrectionTarget(kind=TargetKind.ROOT_TRANSLATION, subject_track_id=tid)


def _root_o(tid: int = 1) -> CorrectionTarget:
    return CorrectionTarget(kind=TargetKind.ROOT_ORIENTATION, subject_track_id=tid)


class _RecordingPrior(MotionPrior):
    """Identity denoiser that records how the engine called it (proves is_rotation routing)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def info(self) -> ModelInfo:
        return ModelInfo(name="RecordingPrior", backend=Backend.FAKE)

    def denoise(self, values, frames, *, is_rotation):
        self.calls.append((np.asarray(values).shape, is_rotation))
        return np.asarray(values, dtype=float)


# --- the fake-real denoiser ---------------------------------------------------------
def test_fake_prior_is_a_motion_prior_and_smooths_a_step():
    mp = FakeMotionPrior()
    assert isinstance(mp, MotionPrior)
    step = _step_transl()
    out = mp.denoise(step, np.arange(step.shape[0]), is_rotation=False)
    tv_in = float(np.abs(np.diff(step[:, 2])).sum())
    tv_out = float(np.abs(np.diff(out[:, 2])).sum())
    assert tv_out < tv_in                            # zero-phase gaussian softens the jump
    np.testing.assert_allclose(step, _step_transl())  # input not mutated


def test_fake_prior_deterministic_and_rotation_aware():
    mp = FakeMotionPrior()
    aa = np.zeros((5, 3))
    aa[:, 2] = np.linspace(0.0, 1.0, 5)
    a = mp.denoise(aa, np.arange(5), is_rotation=True)
    b = mp.denoise(aa, np.arange(5), is_rotation=True)
    np.testing.assert_allclose(a, b)  # deterministic
    assert a.shape == (5, 3)


# --- engine routing: method="learned" through the injected port ---------------------
def test_resolve_learned_smoothing_denoises_root_translation():
    proposal = _motion()
    mp = FakeMotionPrior()
    resolved = resolve_subject_motion(
        proposal, [make_smoothing("c", _root_t(), (0, 6), method="learned")], motion_prior=mp
    )
    expected = mp.denoise(proposal.pose.transl, proposal.pose.frames, is_rotation=False)
    np.testing.assert_allclose(resolved.pose.transl, expected)
    assert not np.allclose(resolved.pose.transl, proposal.pose.transl)  # actually changed
    np.testing.assert_allclose(proposal.pose.transl, _step_transl())     # proposal intact


def test_resolve_learned_smoothing_without_prior_is_actionable():
    proposal = _motion()
    with pytest.raises(ValueError, match="MotionPrior"):
        resolve_subject_motion(proposal, [make_smoothing("c", _root_t(), (0, 6), method="learned")])


def test_learned_routing_passes_is_rotation_per_target():
    proposal = _motion(global_orient=_step_transl())
    rec = _RecordingPrior()
    resolve_subject_motion(
        proposal, [make_smoothing("t", _root_t(), (0, 6), method="learned")], motion_prior=rec
    )
    resolve_subject_motion(
        proposal, [make_smoothing("o", _root_o(), (0, 6), method="learned")], motion_prior=rec
    )
    assert [c[1] for c in rec.calls] == [False, True]  # translation euclidean, orientation rotation


def test_pure_smoothing_still_needs_no_prior():
    proposal = _motion()
    resolved = resolve_subject_motion(
        proposal, [make_smoothing("c", _root_t(), (0, 6), method="gaussian", window=5)]
    )
    assert not np.allclose(resolved.pose.transl, proposal.pose.transl)  # pure path unaffected


def test_learned_smoothing_only_touches_its_range():
    proposal = _motion(n=7)
    resolved = resolve_subject_motion(
        proposal, [make_smoothing("c", _root_t(), (2, 4), method="learned")],
        motion_prior=FakeMotionPrior(),
    )
    np.testing.assert_allclose(resolved.pose.transl[:2], proposal.pose.transl[:2])   # before intact
    np.testing.assert_allclose(resolved.pose.transl[5:], proposal.pose.transl[5:])   # after intact


def test_preview_subject_motion_threads_the_prior():
    proposal = _motion()
    cand = make_smoothing("c", _root_t(), (0, 6), method="learned")
    out = preview_subject_motion(proposal, [], cand, motion_prior=FakeMotionPrior())
    assert not np.allclose(out.pose.transl, proposal.pose.transl)


# --- the gated learned model (R-8) --------------------------------------------------
def test_learned_motion_prior_is_gated():
    mp = LearnedMotionPrior()
    assert isinstance(mp, MotionPrior)
    assert mp.info().name == "LearnedMotionPrior"
    with pytest.raises(NotImplementedError, match="motion"):
        mp.denoise(np.zeros((4, 3)), np.arange(4), is_rotation=False)


# --- wiring (default fake-real, named gated, dotted-path BYO) ------------------------
def test_wiring_defaults_to_fake_motion_prior():
    assert isinstance(default_ports().motion_prior, FakeMotionPrior)


def test_wiring_selects_gated_learned_motion_prior():
    mp = default_ports(motion_prior="learned").motion_prior
    assert isinstance(mp, LearnedMotionPrior)
    with pytest.raises(NotImplementedError):
        mp.denoise(np.zeros((3, 3)), np.arange(3), is_rotation=False)


def test_wiring_accepts_dotted_path_byo_motion_prior():
    assert isinstance(default_ports(motion_prior=_MP_SPEC).motion_prior, FakeMotionPrior)


def test_wiring_bad_motion_prior_spec_is_actionable():
    with pytest.raises(ValueError):
        default_ports(motion_prior="not_a_real.module:Nope")
