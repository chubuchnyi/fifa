"""SMPL-X world foot-position provider — gated on real SMPL-X model presence.

The regression these pin: the provider used to pick ``argmin`` over *native* ``y`` and remap with
an improper ``[x, z, y]`` matrix (det -1). On real camera-frame HMR output that selects the top
of the head, mirrors the forward axis, and carries the model-origin bias — measured at 167 mm of
sink below the pitch across six real subjects, versus 4 mm after the fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.adapters.models.smplx_foot_pos import (
    SmplxFootPosConfig,
    make_smplx_foot_position_provider,
)
from pitch3d.core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from pitch3d.core.scene.scene import Subject

smplx = pytest.importorskip("pitch3d.adapters.models.smplx_lbs")
locate_smplx_model = smplx.locate_smplx_model

needs_model = pytest.mark.skipif(
    locate_smplx_model() is None,
    reason="requires SMPL-X model .npz (PITCH3D_SMPLX_MODEL/MODELS)",
)

PELVIS_Z = 0.92
#: Rotate the canonical body head-down, which is how real camera-frame HMR output arrives.
CAMERA_ORIENT = np.array([np.pi, 0.0, 0.0])


def _subject(orient: np.ndarray, T: int = 4) -> Subject:
    transl = np.tile([3.0, -2.0, PELVIS_Z], (T, 1))
    motion = SubjectMotion(
        shape=SmplxShape(betas=np.zeros(10)),
        pose=PoseSequence(
            frames=np.arange(T, dtype=int),
            global_orient=np.tile(orient, (T, 1)),
            body_pose=np.zeros((T, 21, 3)),
            transl=transl,
        ),
    )
    return Subject(track_id=1, proposal=motion)


def test_make_provider_returns_none_when_model_missing(tmp_path):
    assert make_smplx_foot_position_provider(model_path=str(tmp_path / "nope.npz")) is None


@needs_model
@pytest.mark.parametrize("orient", [CAMERA_ORIENT, np.zeros(3)])
def test_standing_player_puts_feet_on_the_pitch(orient):
    """Both source frames must land the foot ~a leg-length below the pelvis, i.e. near z=0."""
    out = make_smplx_foot_position_provider()(_subject(orient))
    assert out.shape == (4, 3)
    drop = PELVIS_Z - out[:, 2]
    assert np.all((drop > 0.80) & (drop < 1.05)), f"foot drop {drop} is not a leg length"
    assert np.all(np.abs(out[:, 2]) < 0.10), f"feet off the pitch plane: {out[:, 2]}"


@needs_model
def test_foot_xy_tracks_the_root():
    out = make_smplx_foot_position_provider()(_subject(CAMERA_ORIENT))
    assert np.allclose(out[:, 0], 3.0, atol=0.35) and np.allclose(out[:, 1], -2.0, atol=0.35)


@needs_model
def test_source_frame_override_is_honoured():
    """Forcing the wrong frame must visibly break — proof the knob reaches the maths."""
    cfg = SmplxFootPosConfig(source_frame="canonical")
    wrong = make_smplx_foot_position_provider(cfg=cfg)(_subject(CAMERA_ORIENT))
    right = make_smplx_foot_position_provider()(_subject(CAMERA_ORIENT))
    assert abs(wrong[0][2]) > abs(right[0][2]) + 0.1, "override had no effect"


@needs_model
def test_returns_none_without_a_proposal():
    assert make_smplx_foot_position_provider()(Subject(track_id=1, proposal=None)) is None
