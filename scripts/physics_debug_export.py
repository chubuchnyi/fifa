"""Export per-frame SMPL-X vertex sequences for the physics-only Blender preview.

Loads a scene.json, resolves each subject's motion (proposal ⊕ corrections),
forwards SMPL-X per frame, and dumps a single npz that
``blender_physics_debug.py`` reads to animate the mesh. The pipeline venv has
torch/smplx; Blender's bundled Python does not — so the FK must live here.

Output format (single npz):
    subjects: object array — one dict per subject with
        track_id  int
        verts     (T, V, 3)  float32, z-up world coords
        faces     (F, 3)     int64
        color     (3,)       float32, per-team tint
        transl    (T, 3)     float32  (for camera framing)
    pitch     dict with x_size, y_size (meters)
    fps       float
    n_frames  int

Env:
    PITCH3D_SMPLX_MODELS  dir containing smplx/SMPLX_NEUTRAL.npz
    PITCH3D_SCENE_JSON    input scene.json path
    PITCH3D_DEBUG_OUT     output npz path

Run:
    .venv/bin/python scripts/physics_debug_export.py \
        --scene out/anim_full_realism/scene.json \
        --out out/physics_debug/frames.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import smplx
import torch

from pitch3d.core.correction.engine import resolve_subject_motion
from pitch3d.core.scene.frames import R_SMPLX_CAMERA_TO_WORLD
from pitch3d.core.scene.serialization import load_scene
from pitch3d.env import load_env

load_env()

# SMPLest-X emits camera-frame verts with y pointing down (head at -y). Rotate
# into our z-up world (new = [x, z, -y]) before adding transl.
R_SMPLX_TO_OURS = R_SMPLX_CAMERA_TO_WORLD.astype(np.float32)  # real HMR output


def _team_color(track_id: int, palette: np.ndarray) -> np.ndarray:
    return palette[track_id % len(palette)].astype(np.float32)


def _forward_frame(
    model: smplx.SMPLX,
    betas: np.ndarray,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        out = model(
            betas=torch.tensor(betas[None], dtype=torch.float32),
            global_orient=torch.tensor(global_orient[None], dtype=torch.float32),
            body_pose=torch.tensor(body_pose.reshape(1, -1), dtype=torch.float32),
        )
    return out.vertices[0].numpy().astype(np.float32) @ R_SMPLX_TO_OURS.T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default=os.environ.get(
        "PITCH3D_SCENE_JSON", "out/anim_full_realism/scene.json"))
    ap.add_argument("--out", default=os.environ.get(
        "PITCH3D_DEBUG_OUT", "out/physics_debug/frames.npz"))
    ap.add_argument("--models", default=os.environ.get(
        "PITCH3D_SMPLX_MODELS", "SMPL-X/models"))
    ap.add_argument("--fps", type=float, default=29.97)
    args = ap.parse_args()

    scene = load_scene(args.scene)
    n_subjects = len(scene.subjects)
    if n_subjects == 0:
        raise SystemExit(f"scene has no subjects: {args.scene}")

    palette = matplotlib.colormaps["tab10"](np.linspace(0, 1, 10))[:, :3]

    n_frames = max(
        np.asarray(s.proposal.pose.frames).shape[0] for s in scene.subjects
    )
    print(f"scene: {n_subjects} subjects, {n_frames} frames, out → {args.out}")

    subjects_out = []
    faces_ref: np.ndarray | None = None
    for i, s in enumerate(scene.subjects):
        resolved = resolve_subject_motion(
            s.proposal, scene.corrections_for(s.track_id),
        )
        frames = np.asarray(resolved.pose.frames, dtype=int)
        betas = np.asarray(resolved.shape.betas, dtype=float)
        global_orient = np.asarray(resolved.pose.global_orient, dtype=float)
        body_pose = np.asarray(resolved.pose.body_pose, dtype=float)
        transl = np.asarray(resolved.pose.transl, dtype=float)
        T = frames.shape[0]

        model = smplx.create(
            args.models,
            model_type="smplx",
            gender="neutral",
            num_betas=int(betas.shape[0]),
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
        )
        if faces_ref is None:
            faces_ref = model.faces.astype(np.int64)

        verts_seq = np.zeros((T, 10475, 3), dtype=np.float32)
        for k in range(T):
            v = _forward_frame(model, betas, global_orient[k], body_pose[k])
            verts_seq[k] = v + transl[k].astype(np.float32)

        subjects_out.append({
            "track_id": int(s.track_id),
            "verts": verts_seq,
            "faces": faces_ref,
            "color": _team_color(int(s.track_id), palette),
            "transl": transl.astype(np.float32),
        })
        print(f"  subj {s.track_id:3d}: T={T}  transl_range="
              f"[{transl[:, 0].min():.1f},{transl[:, 0].max():.1f}]"
              f"×[{transl[:, 1].min():.1f},{transl[:, 1].max():.1f}]m")

    pitch_x = 105.0
    pitch_y = 68.0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        subjects=np.array(subjects_out, dtype=object),
        pitch=np.array({"x_size": pitch_x, "y_size": pitch_y}, dtype=object),
        fps=np.float32(args.fps),
        n_frames=np.int32(n_frames),
    )
    print(f"DEBUG_EXPORT_OK ({n_subjects} subjects × {n_frames} frames) → {args.out}")


if __name__ == "__main__":
    main()
