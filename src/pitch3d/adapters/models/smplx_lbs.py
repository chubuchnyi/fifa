"""Pure-numpy SMPL-X linear-blend-skinning forward (M2-8a) — posing without torch/GPU.

Turning a subject's resolved per-frame SMPL-X parameters into *posed world vertices* is the
geometry half of M2-8 ("geometry follows the edited pose, incl. limbs"). The forward kinematics +
LBS is plain linear algebra, so it runs in the repo venv on the **CPU with no torch at runtime**
(:class:`SmplxModel` reads the model ``.npz`` directly) — the same "model-independent maths stays
dependency-free" split the rest of the avatar stack uses (see :mod:`.avatar`). Correctness is not
taken on faith: a gated unit test cross-checks this against the reference ``smplx`` package.

Only the **body** is articulated here — root (``global_orient``) + the 21 body joints — which is
exactly what the pipeline measures and stores (:class:`~pitch3d.core.scene.motion.PoseSequence`);
hands / jaw / eyes are left at rest (identity), so their pose-corrective blendshape and LBS
contribution vanish. Hands may be passed explicitly for completeness but are unused by the soccer
path (broadcast soccer GT is body-only — see the pose-backend notes).

The model is located like every gated asset (mirroring ``locate_blender``): an explicit path →
``$PITCH3D_SMPLX_MODEL`` (a direct ``.npz``) → ``$PITCH3D_SMPLX_MODELS`` (the models dir, the
``smplx/SMPLX_<gender>.npz`` convention) → a repo-local ``models/smplx`` fallback → ``None``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...core.correction.rotations import axis_angle_to_matrix
from ...core.scene.motion import N_SMPLX_BODY_JOINTS

N_SMPLX_VERTS = 10475
N_SMPLX_JOINTS = 55  # 1 root + 21 body + jaw + 2 eyes + 2*15 hand joints
_JAW_JOINT = 22
_LEFT_HAND_START = 25
_RIGHT_HAND_START = 40


def locate_smplx_model(explicit: str | None = None, gender: str = "NEUTRAL") -> str | None:
    """Find a SMPL-X model ``.npz``: explicit → ``$PITCH3D_SMPLX_MODEL`` → ``$PITCH3D_SMPLX_MODELS``
    → repo-local ``models/smplx`` → ``None`` (mirrors :func:`...adapters.blender.locate_blender`).
    """
    name = f"SMPLX_{gender.upper()}.npz"
    if explicit and os.path.isfile(explicit):
        return explicit
    direct = os.environ.get("PITCH3D_SMPLX_MODEL")
    if direct and os.path.isfile(direct):
        return direct
    roots = [os.environ.get("PITCH3D_SMPLX_MODELS"), "models/smplx"]
    for root in roots:
        if not root:
            continue
        cand = Path(root) / "smplx" / name
        if cand.is_file():
            return str(cand)
    return None


@dataclass
class SmplxModel:
    """A loaded SMPL-X body model — the buffers needed for shape + pose-blend + FK + LBS.

    Pure arrays (no torch). :meth:`shaped` applies the shape blendshape (the rest/canonical mesh
    for a subject's ``betas``); :meth:`pose` additionally applies pose-corrective blendshapes,
    forward kinematics and linear blend skinning to produce **posed world vertices**.
    """

    v_template: np.ndarray  # (V, 3) rest template
    shapedirs: np.ndarray   # (V, 3, n_shape) shape blendshape basis
    posedirs: np.ndarray    # (V, 3, 9*(J-1)) pose-corrective blendshape basis
    j_regressor: np.ndarray  # (J, V) joints from vertices
    parents: np.ndarray     # (J,) int kinematic parents; parents[0] = -1 (root)
    weights: np.ndarray     # (V, J) LBS skinning weights
    faces: np.ndarray       # (F, 3) int triangle topology

    @classmethod
    def load(cls, path: str | Path) -> SmplxModel:
        """Read a SMPL-X ``.npz`` (e.g. ``SMPLX_NEUTRAL.npz``) into pure numpy buffers."""
        d = np.load(str(path), allow_pickle=True)
        parents = d["kintree_table"][0].astype(np.int64).copy()
        parents[0] = -1  # the npz stores the uint32 -1 sentinel for the root
        return cls(
            v_template=np.asarray(d["v_template"], dtype=float),
            shapedirs=np.asarray(d["shapedirs"], dtype=float),
            posedirs=np.asarray(d["posedirs"], dtype=float),
            j_regressor=np.asarray(d["J_regressor"], dtype=float),
            parents=parents,
            weights=np.asarray(d["weights"], dtype=float),
            faces=np.asarray(d["f"], dtype=np.int64),
        )

    @property
    def n_verts(self) -> int:
        return int(self.v_template.shape[0])

    @property
    def n_joints(self) -> int:
        return int(self.weights.shape[1])

    def shaped(self, betas: np.ndarray) -> np.ndarray:
        """Rest-pose vertices ``(V, 3)`` for ``betas`` — the canonical mesh (no pose applied)."""
        b = np.asarray(betas, dtype=float).reshape(-1)
        nb = min(b.shape[0], self.shapedirs.shape[2])
        if nb == 0:
            return self.v_template.copy()
        return self.v_template + self.shapedirs[:, :, :nb] @ b[:nb]

    def pose(
        self,
        betas: np.ndarray,
        global_orient: np.ndarray,
        body_pose: np.ndarray,
        transl: np.ndarray | None = None,
        *,
        left_hand_pose: np.ndarray | None = None,
        right_hand_pose: np.ndarray | None = None,
        jaw_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        """LBS-posed **world** vertices ``(V, 3)`` for one frame's SMPL-X parameters.

        ``global_orient`` (3,) and ``body_pose`` (21, 3) are axis-angle; ``transl`` (3,) is added
        last in world metres. Hands / jaw default to rest (identity). The pipeline stores exactly
        these (root + 21 body joints), so a ``POSE_BODY_JOINT`` edit flows straight into the limbs.
        """
        v_shaped = self.shaped(betas)
        joints = self.j_regressor @ v_shaped  # (J, 3)
        full_pose = self._full_pose(
            global_orient, body_pose, left_hand_pose, right_hand_pose, jaw_pose
        )
        rot = axis_angle_to_matrix(full_pose)  # (J, 3, 3)
        # Pose-corrective blendshape: flattened (R - I) over the non-root joints.
        pose_feature = (rot[1:] - np.eye(3)).reshape(-1)
        v_posed = v_shaped + (
            self.posedirs.reshape(-1, pose_feature.shape[0]) @ pose_feature
        ).reshape(-1, 3)
        rel = self._relative_transforms(rot, joints)  # (J, 4, 4)
        skin = np.einsum("vj,jab->vab", self.weights, rel)  # (V, 4, 4)
        v_h = np.concatenate([v_posed, np.ones((v_posed.shape[0], 1))], axis=1)
        verts = np.einsum("vab,vb->va", skin, v_h)[:, :3]
        if transl is not None:
            verts = verts + np.asarray(transl, dtype=float).reshape(3)
        return verts

    def pose_sequence(
        self,
        betas: np.ndarray,
        global_orient: np.ndarray,
        body_pose: np.ndarray,
        transl: np.ndarray,
    ) -> np.ndarray:
        """Per-frame posed world vertices ``(T, V, 3)`` for a whole :class:`PoseSequence`."""
        go = np.asarray(global_orient, dtype=float).reshape(-1, 3)
        bp = np.asarray(body_pose, dtype=float).reshape(go.shape[0], -1, 3)
        tr = np.asarray(transl, dtype=float).reshape(-1, 3)
        out = np.empty((go.shape[0], self.n_verts, 3))
        for i in range(go.shape[0]):
            out[i] = self.pose(betas, go[i], bp[i], tr[i])
        return out

    def _full_pose(
        self,
        global_orient: np.ndarray,
        body_pose: np.ndarray,
        left_hand_pose: np.ndarray | None,
        right_hand_pose: np.ndarray | None,
        jaw_pose: np.ndarray | None,
    ) -> np.ndarray:
        """Assemble the full ``(J, 3)`` axis-angle pose; unset joints stay at rest (zeros)."""
        full = np.zeros((self.n_joints, 3))
        full[0] = np.asarray(global_orient, dtype=float).reshape(3)
        bp = np.asarray(body_pose, dtype=float).reshape(-1, 3)
        nb = min(bp.shape[0], N_SMPLX_BODY_JOINTS)
        full[1 : 1 + nb] = bp[:nb]
        if jaw_pose is not None:
            full[_JAW_JOINT] = np.asarray(jaw_pose, dtype=float).reshape(3)
        hands = ((_LEFT_HAND_START, left_hand_pose), (_RIGHT_HAND_START, right_hand_pose))
        for start, hand in hands:
            if hand is not None:
                h = np.asarray(hand, dtype=float).reshape(-1, 3)
                k = min(h.shape[0], 15)
                full[start : start + k] = h[:k]
        return full

    def _relative_transforms(self, rot: np.ndarray, joints: np.ndarray) -> np.ndarray:
        """Per-joint LBS transforms ``(J, 4, 4)``: FK down the tree, rest pose removed.

        Builds each joint's global transform from its parent (root-relative joint offsets), then
        subtracts the rest-pose joint so a zero pose maps every vertex to itself (the standard
        SMPL skinning matrices ``A``).
        """
        j = joints.shape[0]
        rel_j = joints.copy()
        rel_j[1:] -= joints[self.parents[1:]]
        local = np.tile(np.eye(4), (j, 1, 1))
        local[:, :3, :3] = rot
        local[:, :3, 3] = rel_j
        glob = np.empty_like(local)
        glob[0] = local[0]
        for i in range(1, j):
            glob[i] = glob[self.parents[i]] @ local[i]
        joints_h = np.concatenate([joints, np.zeros((j, 1))], axis=1)  # (J, 4)
        offset = np.einsum("jab,jb->ja", glob, joints_h)  # G @ [J; 0]
        rel = glob.copy()
        rel[:, :, 3] -= offset
        return rel


__all__ = ["N_SMPLX_JOINTS", "N_SMPLX_VERTS", "SmplxModel", "locate_smplx_model"]
