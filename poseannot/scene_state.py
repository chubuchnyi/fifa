"""In-memory scene state — loaded once at start, cached FK per subject.

Reading scene.json + running SMPL-X FK for every joint request would kill
UI latency. We front-load: on first access, run FK for every subject over
every frame and cache the resulting joint positions (T, J, 3) + verts
(T, V, 3). Subsequent /api/joints/... calls are dict lookups.

If the user edits a pose (v1+), we recompute FK for the affected subject
only and update the cache — a single edit is < 1s round-trip.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import smplx
import torch

from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.scene.scene import Scene
from pitch3d.core.scene.serialization import load_scene as _pitch3d_load_scene

from .config import PoseAnnotConfig, load as load_config

# SMPLest-X → z-up world remap (see scripts/render_smplx_mesh.py comment).
R_SMPLX_TO_OURS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)

#: SMPL-X body joint index → human name.  Body has 22 joints (pelvis + 21).
BODY_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]


@dataclass
class SubjectCache:
    track_id: int
    frames: np.ndarray            # (T,)
    verts: np.ndarray             # (T, V, 3)  z-up world coords
    joints: np.ndarray            # (T, J, 3)  z-up world coords
    faces: np.ndarray             # (F, 3)     shared
    transl: np.ndarray            # (T, 3)     for camera framing
    body_pose: np.ndarray         # (T, 21, 3) editable
    betas: np.ndarray             # (num_betas,)
    global_orient: np.ndarray     # (T, 3)     editable


@dataclass
class SceneState:
    scene: Scene
    subjects: dict[int, SubjectCache] = field(default_factory=dict)
    n_frames: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


@lru_cache(maxsize=1)
def _smplx_model_cache(models_dir: str, num_betas: int) -> smplx.SMPLX:
    return smplx.create(
        models_dir, model_type="smplx", gender="neutral",
        num_betas=num_betas, use_pca=False, flat_hand_mean=True, batch_size=1,
    )


def _fk_forward(model, betas, global_orient, body_pose, transl):
    with torch.no_grad():
        out = model(
            betas=torch.tensor(betas[None], dtype=torch.float32),
            global_orient=torch.tensor(global_orient[None], dtype=torch.float32),
            body_pose=torch.tensor(body_pose.reshape(1, -1), dtype=torch.float32),
        )
    verts = out.vertices[0].numpy() @ R_SMPLX_TO_OURS.T + transl.astype(np.float32)
    joints = out.joints[0].numpy() @ R_SMPLX_TO_OURS.T + transl.astype(np.float32)
    return verts.astype(np.float32), joints.astype(np.float32)


def _build_subject_cache(subject, corrections, models_dir: str) -> SubjectCache:
    resolved = resolve_subject_motion(subject.proposal, corrections)
    frames = np.asarray(resolved.pose.frames, dtype=int)
    global_orient = np.asarray(resolved.pose.global_orient, dtype=float)
    body_pose = np.asarray(resolved.pose.body_pose, dtype=float)
    transl = np.asarray(resolved.pose.transl, dtype=float)
    betas = np.asarray(resolved.shape.betas, dtype=float)
    T = frames.shape[0]

    model = _smplx_model_cache(models_dir, int(betas.shape[0]))
    verts = np.zeros((T, 10475, 3), dtype=np.float32)
    joints = np.zeros((T, 127, 3), dtype=np.float32)
    for k in range(T):
        v, j = _fk_forward(model, betas, global_orient[k], body_pose[k], transl[k])
        verts[k] = v
        joints[k] = j

    return SubjectCache(
        track_id=int(subject.track_id),
        frames=frames,
        verts=verts,
        joints=joints[:, :22],   # keep only body joints (pelvis + 21)
        faces=model.faces.astype(np.int32),
        transl=transl.astype(np.float32),
        body_pose=body_pose.astype(np.float32),
        betas=betas.astype(np.float32),
        global_orient=global_orient.astype(np.float32),
    )


def build_scene_state(cfg: PoseAnnotConfig | None = None) -> SceneState:
    cfg = cfg or load_config()
    scene = _pitch3d_load_scene(str(cfg.scene_json))
    st = SceneState(scene=scene)
    n = 0
    for s in scene.subjects:
        corrs = scene.corrections_for(s.track_id)
        cache = _build_subject_cache(s, corrs, str(cfg.smplx_models))
        st.subjects[cache.track_id] = cache
        n = max(n, cache.frames.shape[0])
    st.n_frames = n
    return st


# ─── module-level lazy singleton (rebuilt on demand) ────────────────────────
_STATE: SceneState | None = None
_STATE_LOCK = threading.Lock()


def get_state(force_reload: bool = False) -> SceneState:
    global _STATE
    with _STATE_LOCK:
        if _STATE is None or force_reload:
            _STATE = build_scene_state()
    return _STATE
