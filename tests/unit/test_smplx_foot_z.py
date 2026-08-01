"""SMPL-X FK-based pelvis target provider — gated on real SMPL-X model presence."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.adapters.models.smplx_foot_z import (
    SmplxFootZConfig,
    make_smplx_foot_z_provider,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Subject

smplx = pytest.importorskip("pitch3d.adapters.models.smplx_lbs")
locate_smplx_model = smplx.locate_smplx_model


def _subject_for_fk(track_id: int, T: int = 5) -> Subject:
    frames = np.arange(T, dtype=int)
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=frames, global_orient=np.zeros((T, 3)),
            body_pose=np.zeros((T, 21, 3)), transl=np.zeros((T, 3)),
        ),
    )
    return Subject(track_id=track_id, proposal=motion)


def test_make_provider_returns_none_when_model_missing(tmp_path):
    """No model path + no env → None so foot_plant falls back to cfg.target."""
    provider = make_smplx_foot_z_provider(model_path=str(tmp_path / "does_not_exist.npz"))
    assert provider is None


@pytest.mark.skipif(locate_smplx_model() is None,
                    reason="requires SMPL-X model .npz (PITCH3D_SMPLX_MODEL/MODELS)")
def test_provider_returns_realistic_offsets():
    """For a neutral-shape rest pose the pelvis-above-foot lands in the
    normal adult range (0.7-1.5m). The wide bound covers different SMPL-X
    versions / betas — the exact value is the WHOLE POINT: it's measured,
    not the 0.92 constant we used to bake in."""
    provider = make_smplx_foot_z_provider()
    assert provider is not None
    subject = _subject_for_fk(1)
    offsets = provider(subject)
    assert offsets is not None
    assert offsets.shape[0] >= 1
    assert 0.7 < float(np.median(offsets)) < 1.5


@pytest.mark.skipif(locate_smplx_model() is None,
                    reason="requires SMPL-X model .npz (PITCH3D_SMPLX_MODEL/MODELS)")
def test_provider_respects_max_frames_sampled():
    """cfg caps the number of FK evaluations per subject."""
    provider = make_smplx_foot_z_provider(cfg=SmplxFootZConfig(max_frames_sampled=3))
    assert provider is not None
    subject = _subject_for_fk(1, T=100)
    offsets = provider(subject)
    assert offsets is not None
    assert offsets.shape[0] <= 3


@pytest.mark.skipif(locate_smplx_model() is None,
                    reason="requires SMPL-X model .npz")
def test_provider_different_betas_yield_different_offsets():
    """Two subjects with different shape betas get DIFFERENT measured
    pelvis-above-foot offsets — the whole point of the T6a v2 upgrade."""
    provider = make_smplx_foot_z_provider()
    assert provider is not None
    subj_short = _subject_for_fk(1)
    subj_tall = _subject_for_fk(2)
    # override betas on the second subject to something noticeably different
    subj_tall.proposal.shape.betas = np.array([2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    off_short = provider(subj_short)
    off_tall = provider(subj_tall)
    assert off_short is not None and off_tall is not None
    # Non-zero beta[0] measurably changes standing offset (either direction)
    assert abs(float(np.median(off_tall)) - float(np.median(off_short))) > 0.01
