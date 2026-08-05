"""Placeholder forward-kinematics — the FK seam the bake-off shares with the product.

A pose backend yields SMPL-X *articulation* per subject (axis-angle ``global_orient`` +
``body_pose`` + shape ``betas``) — exactly :class:`~pitch3d.adapters.models.pose.RawBodyMotion`.
To score MPJPE we need joint *positions*, so the harness runs forward kinematics on those
params. :class:`JointModel` is that seam:

* :class:`PlaceholderJointModel` — pure/numpy FK on a fixed 16-joint skeleton (no kinematic
  chain: every joint is a direct child of the pelvis). It validates the *geometry / metric*
  path with no box, no SMPL-X mesh, no GPU. ``betas`` is accepted (real contract) but ignored
  (the placeholder has fixed shape).
* On the box, the real SMPL-X body model implements the same :class:`JointModel` and drops in —
  the harness, the synthetic GT generator and the product all consume one FK contract.

Conventions match the core scene model: right-handed, **Z-up, metres**; the canonical skeleton
is pelvis-at-origin with feet at ``z ≈ -0.90`` so a pelvis grounded at ~0.92 m sits on the plane.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..core.scene.frames import R_SMPLX_CANONICAL_TO_WORLD

JOINT_NAMES: tuple[str, ...] = (
    "pelvis", "spine", "neck", "head",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
)

#: Canonical body joints in a local frame (pelvis at origin; x-right, y-forward, z-up), metres.
CANONICAL_SKELETON: np.ndarray = np.array(
    [
        [0.00, 0.0, 0.00],   # pelvis
        [0.00, 0.0, 0.25],   # spine
        [0.00, 0.0, 0.50],   # neck
        [0.00, 0.0, 0.62],   # head
        [0.17, 0.0, 0.45],   # l_shoulder
        [-0.17, 0.0, 0.45],  # r_shoulder
        [0.21, 0.0, 0.22],   # l_elbow
        [-0.21, 0.0, 0.22],  # r_elbow
        [0.23, 0.0, -0.02],  # l_wrist
        [-0.23, 0.0, -0.02],  # r_wrist
        [0.09, 0.0, -0.06],  # l_hip
        [-0.09, 0.0, -0.06],  # r_hip
        [0.10, 0.0, -0.50],  # l_knee
        [-0.10, 0.0, -0.50],  # r_knee
        [0.11, 0.0, -0.90],  # l_ankle
        [-0.11, 0.0, -0.90],  # r_ankle
    ],
    dtype=float,
)


def _rodrigues(aa: np.ndarray) -> np.ndarray:
    """Batched axis-angle ``(M, 3)`` → rotation matrices ``(M, 3, 3)`` (Rodrigues)."""
    aa = np.asarray(aa, dtype=float).reshape(-1, 3)
    theta = np.linalg.norm(aa, axis=1, keepdims=True)            # (M, 1)
    k = np.divide(aa, theta, out=np.zeros_like(aa), where=theta > 0)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    cross = np.zeros((aa.shape[0], 3, 3))
    cross[:, 0, 1], cross[:, 0, 2] = -kz, ky
    cross[:, 1, 0], cross[:, 1, 2] = kz, -kx
    cross[:, 2, 0], cross[:, 2, 1] = -ky, kx
    sin = np.sin(theta)[..., None]                               # (M, 1, 1)
    cos = np.cos(theta)[..., None]
    return np.eye(3)[None] + sin * cross + (1.0 - cos) * (cross @ cross)


def _rotmat_to_aa(rot: np.ndarray) -> np.ndarray:
    """Batched rotation matrices ``(M, 3, 3)`` → axis-angle ``(M, 3)``.

    Via quaternion (Shepperd's method, branch on the largest diagonal term) so it is robust at
    all angles including ``θ ≈ π`` — the inverse of :func:`_rodrigues`. Used by the synthetic
    generator to express ``R_camera @ R_z(yaw)`` (a camera-space root orientation) as the
    axis-angle ``global_orient`` a real HMR backend would emit.
    """
    rot = np.asarray(rot, dtype=float).reshape(-1, 3, 3)
    m = rot.shape[0]
    quat = np.zeros((m, 4))  # (w, x, y, z)
    for i in range(m):
        r = rot[i]
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            quat[i] = (0.25 * s,
                       (r[2, 1] - r[1, 2]) / s,
                       (r[0, 2] - r[2, 0]) / s,
                       (r[1, 0] - r[0, 1]) / s)
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            quat[i] = ((r[2, 1] - r[1, 2]) / s, 0.25 * s,
                       (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            quat[i] = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s,
                       0.25 * s, (r[1, 2] + r[2, 1]) / s)
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            quat[i] = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
                       (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    w = np.clip(quat[:, 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)                                  # [0, 2π)
    sin_half = np.sqrt(1.0 - w * w)
    axis = np.divide(quat[:, 1:], sin_half[:, None],
                     out=np.zeros((m, 3)), where=sin_half[:, None] > 1e-8)
    flip = angle > np.pi                                        # wrap to [0, π], flip axis
    angle = np.where(flip, 2.0 * np.pi - angle, angle)
    axis = np.where(flip[:, None], -axis, axis)
    return axis * angle[:, None]


@runtime_checkable
class JointModel(Protocol):
    """Forward kinematics: SMPL-X articulation → joint positions (root-relative, metres).

    The seam the synthetic GT, the bake-off harness, and the box-side SMPL-X share. ``betas`` is
    part of the contract (real shape parameters) even where an implementation ignores it.

    Note the two joint counts: ``body_pose`` carries :attr:`n_pose_joints` *input* articulations
    (16 placeholder, 21 SMPL-X body), but :meth:`joints` always returns the 16 *canonical*
    :data:`JOINT_NAMES` — backends with a denser kinematic tree select down to the shared set.
    """

    @property
    def n_pose_joints(self) -> int:
        """Joints carried by the ``body_pose`` axis (16 placeholder, 21 SMPL-X body)."""
        ...

    def joints(
        self, global_orient: np.ndarray, body_pose: np.ndarray, betas: np.ndarray
    ) -> np.ndarray:
        """``global_orient (T,3)`` + ``body_pose (T,P,3)`` + ``betas (B,)`` → joints ``(T,16,3)``.

        ``P`` is :attr:`n_pose_joints`; the output is always the 16 canonical joints.
        """
        ...


@dataclass
class PlaceholderJointModel:
    """Fixed-skeleton FK (no kinematic chain): each joint rotates about the pelvis.

    ``joints = R(global_orient) @ R(body_pose_j) @ skeleton_j`` — root-relative (pelvis at the
    origin). Pure numpy; ``betas`` is ignored (fixed shape). Stand-in for SMPL-X FK until the box.
    """

    skeleton: np.ndarray = field(default_factory=lambda: CANONICAL_SKELETON.copy())

    @property
    def n_pose_joints(self) -> int:
        return int(self.skeleton.shape[0])

    def joints(
        self, global_orient: np.ndarray, body_pose: np.ndarray, betas: np.ndarray
    ) -> np.ndarray:
        go = np.asarray(global_orient, dtype=float).reshape(-1, 3)        # (T, 3)
        bp = np.asarray(body_pose, dtype=float)
        bp = bp.reshape(bp.shape[0], -1, 3)                              # (T, J, 3)
        rot_global = _rodrigues(go)                                      # (T, 3, 3)
        rot_joint = _rodrigues(bp.reshape(-1, 3)).reshape(*bp.shape[:2], 3, 3)  # (T, J, 3, 3)
        local = np.einsum("tjac,jc->tja", rot_joint, self.skeleton)      # (T, J, 3)
        return np.einsum("tac,tjc->tja", rot_global, local)             # (T, J, 3)


_SMPLX_ENV_PATH = "PITCH3D_SMPLX_MODEL_PATH"
_SMPLX_DEFAULT_DIR = "models/smplx"

#: SMPL-X joint index for each of the 16 canonical :data:`JOINT_NAMES`, in order (rest-pose probe).
SMPLX_TO_CANONICAL: tuple[int, ...] = (0, 6, 12, 15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8)

#: This module runs its own ``global_orient = 0`` forward pass, so its input is always the
#: SMPL-X *canonical* frame; external orientation is applied afterwards. See
#: :mod:`pitch3d.core.scene.frames` for why the camera-frame remap is a different matrix.
_R_SMPLX_TO_OURS: np.ndarray = R_SMPLX_CANONICAL_TO_WORLD


def _build_smplx(model_path: str | None, gender: str, num_betas: int) -> Any:
    """Construct an ``smplx`` SMPL-X model, resolving the asset dir from arg → env → default.

    Raises a clear error if ``smplx`` is missing or the ``.npz`` is absent, so the no-asset path
    stays on :class:`PlaceholderJointModel` rather than hitting a cryptic ImportError later.
    """
    try:
        import smplx
    except ImportError as exc:
        raise RuntimeError(
            "SmplxJointModel needs the 'smplx' package (pip install smplx — the 'hmr' extra). "
            "Use PlaceholderJointModel for the asset-free / CPU-only path."
        ) from exc
    root = model_path or os.environ.get(_SMPLX_ENV_PATH) or _SMPLX_DEFAULT_DIR
    model_file = Path(root) / "smplx" / f"SMPLX_{gender.upper()}.npz"
    if not model_file.exists():
        raise FileNotFoundError(
            f"SMPL-X model not found at {model_file}. Download the SMPL-X .npz and set "
            f"${_SMPLX_ENV_PATH} (or pass model_path=) to its parent 'models' dir."
        )
    return smplx.create(
        root,
        model_type="smplx",
        gender=gender,
        use_pca=False,
        flat_hand_mean=True,
        num_betas=num_betas,
        batch_size=1,
    )


@dataclass
class SmplxJointModel:
    """Real SMPL-X forward kinematics behind the :class:`JointModel` seam (CPU, torch).

    Runs the SMPL-X kinematic tree with the supplied ``body_pose`` (21 body joints) and **zero**
    internal ``global_orient``, selects the 16 canonical :data:`JOINT_NAMES`, roots them at the
    pelvis, maps native→our frame (:data:`_R_SMPLX_TO_OURS`), then applies the external
    ``global_orient`` via Rodrigues — mirroring :class:`PlaceholderJointModel` exactly so the
    synthetic oracle round-trips. The torch/smplx import is lazy and the model is built on first
    use, so importing this module stays pure (no torch at load time).

    Asset: ``smplx`` package + ``SMPLX_{GENDER}.npz``, resolved from ``model_path`` →
    ``$PITCH3D_SMPLX_MODEL_PATH`` → ``models/smplx``.
    """

    model_path: str | None = None
    gender: str = "neutral"
    num_betas: int = 10
    _model: Any = field(default=None, init=False, repr=False, compare=False)

    @property
    def n_pose_joints(self) -> int:
        return 21

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = _build_smplx(self.model_path, self.gender, self.num_betas)
        return self._model

    def joints(
        self, global_orient: np.ndarray, body_pose: np.ndarray, betas: np.ndarray
    ) -> np.ndarray:
        import torch

        model = self._ensure_model()
        go = np.asarray(global_orient, dtype=float).reshape(-1, 3)           # (T, 3)
        bp = np.asarray(body_pose, dtype=float)
        bp = bp.reshape(bp.shape[0], -1, 3)                                  # (T, 21, 3)
        t = go.shape[0]

        b = np.zeros(self.num_betas)
        src = np.asarray(betas, dtype=float).reshape(-1)
        b[: min(src.shape[0], self.num_betas)] = src[: self.num_betas]
        betas_t = torch.as_tensor(
            np.broadcast_to(b, (t, self.num_betas)).copy(), dtype=torch.float32
        )
        zeros3 = torch.zeros(t, 3)
        with torch.no_grad():
            out = model(
                betas=betas_t,
                global_orient=zeros3,
                body_pose=torch.as_tensor(bp.reshape(t, -1), dtype=torch.float32),
                left_hand_pose=torch.zeros(t, 45),
                right_hand_pose=torch.zeros(t, 45),
                jaw_pose=zeros3,
                leye_pose=zeros3,
                reye_pose=zeros3,
                expression=torch.zeros(t, 10),
            )
        canon = out.joints.detach().cpu().numpy()[:, list(SMPLX_TO_CANONICAL), :]  # (T,16,3) native
        canon = canon - canon[:, :1, :]                                     # root-relative
        canon = canon @ _R_SMPLX_TO_OURS.T                                  # native → ours
        rot_global = _rodrigues(go)                                         # (T, 3, 3)
        return np.einsum("tac,tjc->tja", rot_global, canon)                 # (T, 16, 3)
