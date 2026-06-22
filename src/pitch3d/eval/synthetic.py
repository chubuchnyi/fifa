"""Synthetic broadcast-soccer ground truth — a pure, deterministic eval fixture.

Why this exists: the pose bake-off (``docs/pose-bakeoff-runbook.md``) is blocked on
WorldPose *frames* (we hold the Light annotations, not the video). This module
generates a scene we fully control — a virtual broadcast camera, ``N`` articulated
subjects on the pitch, and **perfect ground truth** (GT camera ``K, R, t``; per-frame
per-subject world 3D joints in metres; their 2D projections; bboxes; and the GT
image→world homography). It is numpy-only — no Blender, no GPU, no asset — so it lets
us build and unit-test the whole bake-off harness *before* any real footage arrives.

What it is NOT: photoreal pixels. The articulated body here is a fixed placeholder
skeleton (see :data:`CANONICAL_SKELETON`), not a SMPL-X mesh, so this validates the
*geometry / grounding / metric* path, not a pose network's robustness to real pixels.
Two seams make it swap cleanly later: the placeholder forward-kinematics is replaced by
SMPL-X FK on the box, and RGB rendering bolts onto the same camera + world joints via
Blender (M2). Conventions match the core scene model: right-handed, **Z-up, metres**,
pitch plane ``Z = 0``, camera extrinsics world→camera (``X_c = R @ X_w + t``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.scene.camera import CameraIntrinsics
from ..core.scene.field import FieldCalibration

#: Placeholder body joints in a local frame (pelvis at origin; x-right, y-forward, z-up),
#: metres. Feet sit at ``z = -0.90`` so a pelvis grounded at ~0.92 m puts them on the plane.
JOINT_NAMES: tuple[str, ...] = (
    "pelvis", "spine", "neck", "head",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
)

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

#: Per-joint articulation axis (local +Y swing): arms and legs swing out of phase so
#: Local MPJPE is exercised. Amplitude in metres, modulated by a per-frame sine.
_SWAY = np.zeros_like(CANONICAL_SKELETON)
_SWAY[[6, 8], 1] = (0.06, 0.10)     # l_elbow, l_wrist forward
_SWAY[[7, 9], 1] = (-0.06, -0.10)   # r_elbow, r_wrist back
_SWAY[[12, 14], 1] = (-0.06, -0.10)  # l_knee, l_ankle back
_SWAY[[13, 15], 1] = (0.06, 0.10)   # r_knee, r_ankle forward


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
    """A fully-known synthetic broadcast scene: GT camera + world/image joints + bboxes.

    Attributes:
        intrinsics: Shared pinhole intrinsics.
        frames: Frame indices, shape ``(T,)``.
        rotation: World→camera rotation ``R``, shape ``(3, 3)`` (static camera).
        translation: World→camera translation ``t``, shape ``(3,)``.
        joints_world: GT joints in world metres, shape ``(T, N, J, 3)``.
        joints_image: GT joint projections in pixels, shape ``(T, N, J, 2)``.
        boxes_xyxy: GT per-subject bounding boxes (px, clipped to frame), shape ``(T, N, 4)``.
        pelvis_height_m: World Z of the grounded pelvis (joint 0).
        joint_names: Names aligned with the ``J`` axis.
    """

    intrinsics: CameraIntrinsics
    frames: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    joints_world: np.ndarray
    joints_image: np.ndarray
    boxes_xyxy: np.ndarray
    pelvis_height_m: float
    joint_names: tuple[str, ...] = JOINT_NAMES

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
    their limbs sinusoidally (so root-relative articulation is non-trivial). All outputs
    are pure functions of ``seed``.
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

    period = max(n_frames, 2)
    sway = np.sin(2.0 * np.pi * frames[:, None] / period + phase[None, :])  # (T, N)
    local = CANONICAL_SKELETON[None, None] + _SWAY[None, None] * sway[..., None, None]
    rotated = np.einsum("nij,tnkj->tnki", _rot_z(yaw), local)  # (T, N, J, 3)

    root_xy = start_xy[None] + velocity[None] * frames[:, None, None]  # (T, N, 2)
    root = np.concatenate(
        [root_xy, np.full((n_frames, n_subjects, 1), pelvis_height_m)], axis=-1
    )
    joints_world = root[:, :, None, :] + rotated

    scene = SyntheticScene(
        intrinsics=intrinsics,
        frames=frames,
        rotation=rotation,
        translation=translation,
        joints_world=joints_world,
        joints_image=np.zeros(joints_world.shape[:-1] + (2,)),
        boxes_xyxy=np.zeros((n_frames, n_subjects, 4)),
        pelvis_height_m=pelvis_height_m,
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
