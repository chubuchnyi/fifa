"""Synthetic broadcast-soccer ground truth — a pure, deterministic eval fixture.

Why this exists: the pose bake-off (``docs/pose-bakeoff-runbook.md``) is blocked on
WorldPose *frames* (we hold the Light annotations, not the video). This module
generates a scene we fully control — a virtual broadcast camera, ``N`` articulated
subjects on the pitch, and **perfect ground truth** (GT camera ``K, R, t``; per-frame
per-subject world 3D joints in metres; their 2D projections; bboxes; the GT image→world
homography; and the GT SMPL-X-style articulation that produced the joints). It is
numpy-only — no Blender, no GPU, no asset — so it lets us build and unit-test the whole
bake-off harness *before* any real footage arrives.

Crucially, the GT joints are generated *through the FK seam* (:class:`PlaceholderJointModel`):
the subjects' world joints are ``camera_to_world(root_cam + FK(global_orient, body_pose))``.
So a "perfect" pose backend — one that returns :attr:`SyntheticScene.gt_global_orient` /
``gt_body_pose`` / ``gt_betas`` — reconstructs the GT exactly (MPJPE → 0), and the GT camera
does real work placing the camera-space prediction into the world (condition A).

What it is NOT: photoreal pixels, nor a SMPL-X mesh (the articulated body is the fixed
placeholder skeleton in :mod:`pitch3d.eval.bodymodel`). It validates the *geometry / grounding
/ metric* path, not a pose network's robustness to real pixels. Two seams swap cleanly later:
the placeholder FK is replaced by SMPL-X FK on the box, and RGB rendering bolts onto the same
camera + world joints via Blender (M2). Conventions match the core scene model: right-handed,
**Z-up, metres**, pitch plane ``Z = 0``, camera extrinsics world→camera (``X_c = R @ X_w + t``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.scene.camera import CameraIntrinsics
from ..core.scene.field import FieldCalibration
from .bodymodel import (
    CANONICAL_SKELETON,
    JOINT_NAMES,
    JointModel,
    PlaceholderJointModel,
    _rotmat_to_aa,
)

#: Per-joint swing amplitude (rad) about the local x-axis, modulated by a per-frame sine.
#: Arms and the contralateral legs swing out of phase so Local MPJPE is exercised.
_BODY_SWING: np.ndarray = np.zeros(CANONICAL_SKELETON.shape[0])
_BODY_SWING[[6, 8]] = (0.15, 0.25)      # l_elbow, l_wrist  (forward)
_BODY_SWING[[7, 9]] = (-0.15, -0.25)    # r_elbow, r_wrist  (back)
_BODY_SWING[[12, 14]] = (-0.15, -0.25)  # l_knee,  l_ankle  (contralateral to the left arm)
_BODY_SWING[[13, 15]] = (0.15, 0.25)    # r_knee,  r_ankle


def _look_at(center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World→camera ``(R, t)`` for a pinhole looking from ``center`` at ``target`` (Z-up world).

    Camera axes (OpenCV): x right, y down, z forward. ``X_c = R @ X_w + t``.
    """
    center = np.asarray(center, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rot = np.stack([right, down, forward], axis=0)  # rows = camera axes in world
    return rot, -rot @ center


def _rot_z(yaw: np.ndarray) -> np.ndarray:
    """Stack of rotations about the world up-axis (Z), shape ``(N, 3, 3)``."""
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.zeros((yaw.shape[0], 3, 3))
    out[:, 0, 0] = c
    out[:, 0, 1] = -s
    out[:, 1, 0] = s
    out[:, 1, 1] = c
    out[:, 2, 2] = 1.0
    return out


@dataclass
class SyntheticScene:
    """A fully-known synthetic broadcast scene: GT camera + world/image joints + GT articulation.

    Attributes:
        intrinsics: Shared pinhole intrinsics.
        frames: Frame indices, shape ``(T,)``.
        rotation: World→camera rotation ``R``, shape ``(3, 3)`` (static camera).
        translation: World→camera translation ``t``, shape ``(3,)``.
        joints_world: GT joints in world metres, shape ``(T, N, J, 3)``.
        joints_image: GT joint projections in pixels, shape ``(T, N, J, 2)``.
        boxes_xyxy: GT per-subject bounding boxes (px, clipped to frame), shape ``(T, N, 4)``.
        pelvis_height_m: World Z of the grounded pelvis (joint 0).
        gt_global_orient: GT camera-space root orientation (axis-angle), shape ``(T, N, 3)`` —
            what a perfect HMR backend would emit.
        gt_body_pose: GT per-joint articulation (axis-angle), shape ``(T, N, J, 3)``.
        gt_betas: GT per-subject shape coefficients, shape ``(N, B)`` (placeholder: zeros).
        joint_names: Names aligned with the ``J`` axis.
        joint_model: The FK that produced the joints (the harness must use the same one).
    """

    intrinsics: CameraIntrinsics
    frames: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    joints_world: np.ndarray
    joints_image: np.ndarray
    boxes_xyxy: np.ndarray
    pelvis_height_m: float
    gt_global_orient: np.ndarray
    gt_body_pose: np.ndarray
    gt_betas: np.ndarray
    joint_names: tuple[str, ...] = JOINT_NAMES
    joint_model: JointModel = field(default_factory=PlaceholderJointModel)

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def n_subjects(self) -> int:
        return int(self.joints_world.shape[1])

    @property
    def root_world(self) -> np.ndarray:
        """GT pelvis (root) position per frame/subject, shape ``(T, N, 3)``."""
        return self.joints_world[:, :, 0, :]

    def world_to_camera(self, pts_world: np.ndarray) -> np.ndarray:
        """Map world points ``(..., 3)`` into camera space (``X_c = R @ X_w + t``)."""
        return np.asarray(pts_world, dtype=float) @ self.rotation.T + self.translation

    def camera_to_world(self, pts_cam: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`world_to_camera`: camera points ``(..., 3)`` → world metres."""
        return (np.asarray(pts_cam, dtype=float) - self.translation) @ self.rotation

    def project(self, pts_world: np.ndarray) -> np.ndarray:
        """Project world points ``(..., 3)`` to pixels ``(..., 2)`` through the GT camera."""
        cam = self.world_to_camera(pts_world)
        img = cam @ self.intrinsics.matrix().T
        return img[..., :2] / img[..., 2:3]

    def field_calibration(self) -> FieldCalibration:
        """The GT image→world(plane Z=0) homography track — a perfect-calibration baseline."""
        k = self.intrinsics.matrix()
        plane = np.column_stack([self.rotation[:, 0], self.rotation[:, 1], self.translation])
        h_iw = np.linalg.inv(k @ plane)
        homographies = np.broadcast_to(h_iw, (self.n_frames, 3, 3)).copy()
        return FieldCalibration(
            homographies=homographies,
            frames=self.frames,
            confidence=np.ones(self.n_frames),
        )


def generate_scene(
    n_subjects: int = 3,
    n_frames: int = 8,
    image_size: tuple[int, int] = (1280, 720),
    seed: int = 0,
    pelvis_height_m: float = 0.92,
) -> SyntheticScene:
    """Generate a deterministic synthetic broadcast-soccer scene with perfect GT.

    The camera sits behind the ``-Y`` sideline, elevated, looking at the pitch centre.
    Subjects start at random central pitch positions, walk in a straight line, and swing
    their limbs sinusoidally (so root-relative articulation is non-trivial). The GT joints
    are produced *through the FK seam* so a perfect backend scores zero. All outputs are
    pure functions of ``seed``.
    """
    width, height = image_size
    rng = np.random.default_rng(seed)
    frames = np.arange(n_frames)

    intrinsics = CameraIntrinsics(
        fx=1400.0, fy=1400.0, cx=width / 2.0, cy=height / 2.0, width=width, height=height
    )
    rotation, translation = _look_at(np.array([0.0, -50.0, 15.0]), np.zeros(3))

    start_xy = rng.uniform([-20.0, -12.0], [20.0, 12.0], size=(n_subjects, 2))
    velocity = rng.uniform(-0.15, 0.15, size=(n_subjects, 2))
    yaw = rng.uniform(0.0, 2.0 * np.pi, size=n_subjects)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_subjects)

    joint_model = PlaceholderJointModel()
    n_joints = joint_model.skeleton.shape[0]

    # World root path: pelvis walks in a straight line at a constant grounded height.
    root_xy = start_xy[None] + velocity[None] * frames[:, None, None]      # (T, N, 2)
    root_world = np.concatenate(
        [root_xy, np.full((n_frames, n_subjects, 1), pelvis_height_m)], axis=-1
    )

    # GT articulation: camera-space root orientation R_cam @ R_z(yaw) → axis-angle, + limb swing.
    rot_cam_yaw = np.einsum("ij,njk->nik", rotation, _rot_z(yaw))          # (N, 3, 3)
    global_orient_n = _rotmat_to_aa(rot_cam_yaw)                           # (N, 3)
    gt_global_orient = np.broadcast_to(
        global_orient_n[None], (n_frames, n_subjects, 3)
    ).copy()
    period = max(n_frames, 2)
    sway = np.sin(2.0 * np.pi * frames[:, None] / period + phase[None, :])  # (T, N)
    gt_body_pose = np.zeros((n_frames, n_subjects, n_joints, 3))
    gt_body_pose[..., 0] = _BODY_SWING[None, None, :] * sway[:, :, None]    # swing about local x
    gt_betas = np.zeros((n_subjects, 10))

    # FK → camera-space root-relative joints, place at the GT root through the GT camera.
    flat = n_frames * n_subjects
    fk_cam = joint_model.joints(
        gt_global_orient.reshape(flat, 3),
        gt_body_pose.reshape(flat, n_joints, 3),
        gt_betas[0],
    ).reshape(n_frames, n_subjects, n_joints, 3)
    root_cam = root_world @ rotation.T + translation                      # world_to_camera(root)
    joints_cam = root_cam[:, :, None, :] + fk_cam
    joints_world = (joints_cam - translation) @ rotation                  # camera_to_world

    scene = SyntheticScene(
        intrinsics=intrinsics,
        frames=frames,
        rotation=rotation,
        translation=translation,
        joints_world=joints_world,
        joints_image=np.zeros(joints_world.shape[:-1] + (2,)),
        boxes_xyxy=np.zeros((n_frames, n_subjects, 4)),
        pelvis_height_m=pelvis_height_m,
        gt_global_orient=gt_global_orient,
        gt_body_pose=gt_body_pose,
        gt_betas=gt_betas,
        joint_model=joint_model,
    )
    scene.joints_image = scene.project(joints_world)
    u, v = scene.joints_image[..., 0], scene.joints_image[..., 1]
    scene.boxes_xyxy = np.stack(
        [
            np.clip(u.min(-1), 0, width), np.clip(v.min(-1), 0, height),
            np.clip(u.max(-1), 0, width), np.clip(v.max(-1), 0, height),
        ],
        axis=-1,
    )
    return scene
