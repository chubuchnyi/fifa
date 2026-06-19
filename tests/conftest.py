"""Shared fixtures — factory helpers + a fakes-backed Application (no GPU/Blender/models)."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.app import build_app
from pitch3d.core.ports.io import ClipRef
from pitch3d.core.scene.layers import ConfidenceMap, Correction
from pitch3d.core.scene.motion import (
    N_SMPLX_BODY_JOINTS,
    BallTrack,
    PoseSequence,
    SmplxShape,
    SubjectMotion,
)
from pitch3d.core.scene.scene import Scene
from pitch3d.core.scene.subject import Subject


@pytest.fixture
def make_motion():
    """Factory: a rest-pose :class:`SubjectMotion` over ``frames`` (zeros, optional root Z)."""

    def _make(
        frames, *, n_joints: int = N_SMPLX_BODY_JOINTS, transl_z: float = 0.0, n_betas: int = 10
    ) -> SubjectMotion:
        frames = np.asarray(frames, dtype=int).reshape(-1)
        t = frames.shape[0]
        pose = PoseSequence(
            frames=frames,
            global_orient=np.zeros((t, 3)),
            body_pose=np.zeros((t, n_joints, 3)),
            transl=np.column_stack([np.zeros(t), np.zeros(t), np.full(t, transl_z)]),
        )
        return SubjectMotion(shape=SmplxShape(betas=np.zeros(n_betas)), pose=pose)

    return _make


@pytest.fixture
def make_scene():
    """Factory: a minimal :class:`Scene` (subjects/corrections/ball/confidence optional)."""

    def _make(
        *,
        subjects: list[Subject] | None = None,
        corrections: list[Correction] | None = None,
        ball: BallTrack | None = None,
        confidence: ConfidenceMap | None = None,
    ) -> Scene:
        return Scene(
            id="s-test",
            episode_id="ep-test",
            source_id="src-test",
            subjects=list(subjects or []),
            corrections=list(corrections or []),
            ball=ball,
            confidence=confidence,
        )

    return _make


@pytest.fixture
def clip() -> ClipRef:
    return ClipRef(
        source_id="t", uri="mem://t.mp4", frames=np.arange(8), width=640, height=360, fps=25.0
    )


@pytest.fixture
def app(tmp_path):
    """A fakes-backed Application writing artifacts under a temp dir."""
    return build_app(out_dir=tmp_path / "out")


@pytest.fixture
def reconstructed(app, clip):
    """``(app, scene_id)`` for a freshly reconstructed scene (grounded ball)."""
    episode = app.register_clip(clip, name="t")
    scene_id = app.run_reconstruction(episode.id)
    return app, scene_id
