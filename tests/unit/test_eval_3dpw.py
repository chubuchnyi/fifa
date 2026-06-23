"""3DPW loader: parsing + projection geometry, validated against a synthetic 3DPW-shaped pickle.

We cannot ship real 3DPW data (licence) or run a real backend here, so these tests pin what is
verifiable *without* the dataset: that :func:`load_3dpw_sequence` selects the right canonical
joints, builds a per-frame (moving) camera, reprojects GT consistently, and **loudly rejects** an
inverted ``cam_poses`` convention — the one field-semantics mistake a real run is most likely to
hit. The remaining unknown (whether a *real* pickle's ``cam_poses`` is world→camera) surfaces via
:func:`diagnose_3dpw_scene` on first use.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from pitch3d.eval.dataset import PoseEvalScene
from pitch3d.eval.datasets_3dpw import (
    SMPL24_TO_CANONICAL,
    diagnose_3dpw_scene,
    load_3dpw_sequence,
)

_T, _N = 4, 2
_K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _fake_3dpw(tmp_path, *, invert_cam=False, seed=0):
    """Write a synthetic 3DPW-shaped pickle; return (path, world joints per actor, cams, K).

    The camera sits *behind* the subjects (translation +8 in z) so the correct world→camera
    extrinsic gives positive depth while its inverse gives negative — letting us test the guard.
    """
    rng = np.random.default_rng(seed)
    jp = [rng.normal(scale=0.25, size=(_T, 24, 3)) + np.array([0.0, 0.0, 5.0]) for _ in range(_N)]
    cams = []
    for f in range(_T):
        m = np.eye(4)
        m[:3, :3] = _rot_z(0.03 * f)               # small per-frame rotation → moving camera
        m[:3, 3] = np.array([0.05 * f, 0.0, 8.0])  # camera behind the subjects
        cams.append(np.linalg.inv(m) if invert_cam else m)
    cams = np.stack(cams)
    seq = {
        "sequence": "fakeSeq",
        "jointPositions": [a.reshape(_T, 72) for a in jp],
        "cam_poses": cams,
        "cam_intrinsics": _K,
        "campose_valid": [np.ones(_T, dtype=bool) for _ in range(_N)],
    }
    path = tmp_path / "fakeSeq.pkl"
    with open(path, "wb") as fh:
        pickle.dump(seq, fh, protocol=2)  # 3DPW pickles are Python-2 protocol
    return path, jp, cams, _K


def test_load_selects_canonical_joints_and_reprojects(tmp_path):
    path, jp, cams, k = _fake_3dpw(tmp_path)
    scene = load_3dpw_sequence(path, images_dir=tmp_path)

    assert isinstance(scene, PoseEvalScene)
    assert scene.rotation.shape == (_T, 3, 3)          # per-frame moving camera
    assert scene.joints_world.shape == (_T, _N, 16, 3)

    sel = list(SMPL24_TO_CANONICAL)
    expected_world = np.stack([jp[a][:, sel, :] for a in range(_N)], axis=1)
    assert np.allclose(scene.joints_world, expected_world)

    rot, transl = cams[:, :3, :3], cams[:, :3, 3]
    cam = np.einsum("tij,tnkj->tnki", rot, expected_world) + transl[:, None, None, :]
    img = cam @ k.T
    assert np.allclose(scene.joints_image, img[..., :2] / img[..., 2:3], atol=1e-6)

    assert scene.clip_uri.startswith("file://")
    diag = diagnose_3dpw_scene(scene)
    assert diag["depth_positive_fraction"] == 1.0
    assert diag["in_frame_fraction"] == 1.0


def test_boxes_enclose_joints_and_clip_to_frame(tmp_path):
    path, *_ = _fake_3dpw(tmp_path)
    scene = load_3dpw_sequence(path, images_dir=tmp_path)
    u, v = scene.joints_image[..., 0], scene.joints_image[..., 1]
    x0, y0, x1, y1 = (scene.boxes_xyxy[..., i] for i in range(4))
    assert np.all(x0 <= u.min(-1)) and np.all(x1 >= u.max(-1))
    assert np.all(y0 <= v.min(-1)) and np.all(y1 >= v.max(-1))
    assert scene.boxes_xyxy[..., 0::2].min() >= 0.0
    assert scene.boxes_xyxy[..., 0::2].max() <= scene.intrinsics.width


def test_inverted_camera_convention_is_rejected(tmp_path):
    path, *_ = _fake_3dpw(tmp_path, invert_cam=True)
    with pytest.raises(ValueError, match="behind the camera"):
        load_3dpw_sequence(path)


def test_stride_subsamples_frames(tmp_path):
    path, *_ = _fake_3dpw(tmp_path)
    scene = load_3dpw_sequence(path, stride=2)
    assert scene.n_frames == 2
    assert list(scene.frames) == [0, 2]
