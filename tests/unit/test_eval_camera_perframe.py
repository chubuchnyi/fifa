"""SyntheticScene world↔camera maps handle a per-frame (moving) camera, not just a static one.

Real accessible pose datasets (3DPW, EMDB) are shot with a *moving* camera: one extrinsic
``(R_t, t_t)`` per frame. :class:`SyntheticScene` originally assumed a single static broadcast
camera; these tests pin the generalised behaviour the 3DPW loader relies on:

* per-frame ``world_to_camera`` / ``camera_to_world`` round-trip for every harness shape;
* a per-frame camera built by *broadcasting* a static one gives byte-identical results to the
  static path (so the generalisation is a strict superset — synthetic scenes are unaffected);
* ``field_calibration`` refuses a per-frame camera (no fixed Z=0 pitch plane).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pitch3d.eval.synthetic import generate_scene


def _orthonormal(rng, n):
    return np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(n)])


def test_perframe_round_trip_all_harness_shapes():
    rng = np.random.default_rng(0)
    base = generate_scene(seed=3)
    t_frames = base.n_frames
    moving = dataclasses.replace(
        base,
        rotation=_orthonormal(rng, t_frames),
        translation=rng.normal(size=(t_frames, 3)),
    )
    # (T,3) root, (T,J,3) one subject, (T,N,J,3) all subjects — every shape the harness passes.
    for shape in [(t_frames, 3), (t_frames, 16, 3), (t_frames, base.n_subjects, 16, 3)]:
        x = rng.normal(size=shape)
        back = moving.camera_to_world(moving.world_to_camera(x))
        assert np.allclose(back, x, atol=1e-9)


def test_perframe_broadcast_matches_static():
    base = generate_scene(seed=4)
    t_frames = base.n_frames
    moving = dataclasses.replace(
        base,
        rotation=np.broadcast_to(base.rotation, (t_frames, 3, 3)).copy(),
        translation=np.broadcast_to(base.translation, (t_frames, 3)).copy(),
    )
    x = np.random.default_rng(5).normal(size=(t_frames, base.n_subjects, 16, 3))
    assert np.allclose(moving.world_to_camera(x), base.world_to_camera(x), atol=1e-12)
    assert np.allclose(moving.project(x), base.project(x), atol=1e-9)


def test_field_calibration_rejects_perframe_camera():
    base = generate_scene(seed=6)
    moving = dataclasses.replace(
        base,
        rotation=np.broadcast_to(base.rotation, (base.n_frames, 3, 3)).copy(),
        translation=np.broadcast_to(base.translation, (base.n_frames, 3)).copy(),
    )
    with pytest.raises(ValueError, match="static camera"):
        moving.field_calibration()
