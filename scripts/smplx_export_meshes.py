"""Forward exported subject_*.npz through SMPL-X and dump Blender-ready geometry.

Blender's bundled python has numpy but NOT torch/smplx, so the SMPL-X forward must happen
here (in the repo venv). Writes one mesh_<name>.npz {verts (V,3) world z-up, faces (F,3),
color (3,)} per subject; scripts/blender_render_meshes.py then loads these and renders a lit,
shadowed scene with the real Blender engine.

Env (machine paths come from the repo-root .env; see .env.example):
  PITCH3D_SMPLX_MODELS  dir containing smplx/SMPLX_NEUTRAL.npz
  PITCH3D_NPZ_DIR       dir of subject_*.npz (pipeline --format smplx_npz export)
  PITCH3D_MESH_OUT      output dir for mesh_*.npz
  PITCH3D_MESH_FRAME    frame index to pose (default 0)

Run:  .venv/bin/python scripts/smplx_export_meshes.py
"""

import glob
import os

import matplotlib
import numpy as np
import smplx
import torch

from pitch3d.env import load_env

load_env()  # PITCH3D_SMPLX_MODELS and friends come from the repo-root .env, never hard-coded

MODELS = os.environ.get("PITCH3D_SMPLX_MODELS", "SMPL-X/models")
NPZ_DIR = os.environ.get("PITCH3D_NPZ_DIR", "out/cuda/export/scene.smplx_npz")
OUT = os.environ.get("PITCH3D_MESH_OUT", "out/cuda/mesh")
FRAME = int(os.environ.get("PITCH3D_MESH_FRAME", "0"))

# Real SMPLest-X output lives in an image/camera frame whose vertical axis points DOWN, so
# a posed standing body has its head at -y; map camera(y-down) -> z-up world: new = [x, z, -y].
# Degenerate/fake-pose exports instead carry a *canonical* SMPL-X body (global_orient=0,
# y-up, head at +y); for those set PITCH3D_CANONICAL_UP=1 to use the plain y-up -> z-up map
# (new = [x, -z, y]) so the body still stands. Then add the z-up world transl (pelvis pos).
if os.environ.get("PITCH3D_CANONICAL_UP", "0") == "1":
    R_SMPLX_TO_OURS = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
else:
    R_SMPLX_TO_OURS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)

os.makedirs(OUT, exist_ok=True)
paths = sorted(glob.glob(os.path.join(NPZ_DIR, "subject_*.npz")))
assert paths, f"no subject_*.npz in {NPZ_DIR}"
colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, 10))[:, :3]

for i, p in enumerate(paths):
    d = np.load(p, allow_pickle=True)
    fr = min(FRAME, d["frames"].shape[0] - 1)
    model = smplx.create(
        MODELS,
        model_type="smplx",
        gender="neutral",
        num_betas=int(d["betas"].shape[0]),
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    )
    with torch.no_grad():
        out = model(
            betas=torch.tensor(d["betas"][None], dtype=torch.float32),
            global_orient=torch.tensor(d["global_orient"][fr][None], dtype=torch.float32),
            body_pose=torch.tensor(d["body_pose"][fr].reshape(1, -1), dtype=torch.float32),
        )
    verts = out.vertices[0].numpy() @ R_SMPLX_TO_OURS.T + d["transl"][fr].astype(np.float32)
    faces = model.faces.astype(np.int64)
    name = os.path.basename(p).replace(".npz", "")
    dst = os.path.join(OUT, f"mesh_{name}.npz")
    np.savez(dst, verts=verts.astype(np.float32), faces=faces, color=colors[i % 10].astype(np.float32))
    print(
        f"{name}: betas[:4]={np.round(d['betas'][:4], 3).tolist()} "
        f"transl={np.round(d['transl'][fr], 2).tolist()} -> {os.path.basename(dst)}"
    )
print(f"EXPORT_MESHES_OK ({len(paths)} subjects -> {OUT})")
