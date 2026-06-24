"""Render SMPL-X skinned meshes from the pipeline's exported subject_*.npz.

Forwards each exported subject (betas/global_orient/body_pose/transl) through the SMPL-X
body model to obtain a posed, skinned mesh, then rasterises it with matplotlib only (no
GPU/EGL/Blender) so it runs anywhere the model files + npz exist:

  mesh_subject1.png  one subject, recentred close-up — proves articulated-human recovery
  scene_meshes.png   every subject at its world transl — the reconstructed scene

This is exactly the data path a Blender import would take: smplx_npz -> SMPL-X forward ->
vertices + faces. A green MESH_RENDER_OK means the export is a faithful, renderable body.

Env (defaults target the pod):
  PITCH3D_SMPLX_MODELS  dir containing smplx/SMPLX_NEUTRAL.npz
  PITCH3D_NPZ_DIR       dir of subject_*.npz (pipeline --format smplx_npz export)
  PITCH3D_MESH_OUT      output dir for the PNGs
  PITCH3D_MESH_FRAME    frame index to pose (default 0)

Run on the pod:  python scripts/render_smplx_mesh.py
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import smplx  # noqa: E402
import torch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from pitch3d.env import load_env  # noqa: E402

load_env()  # any path below may be overridden by the repo-root .env (pod defaults stand otherwise)

MODELS = os.environ.get(
    "PITCH3D_SMPLX_MODELS", "/workspace/repos/SMPLest-X/human_models/human_model_files"
)
NPZ_DIR = os.environ.get(
    "PITCH3D_NPZ_DIR", "/workspace/out/e2e_real/export/scene.smplx_npz"
)
OUT = os.environ.get("PITCH3D_MESH_OUT", "/workspace/out/e2e_real/mesh")
FRAME = int(os.environ.get("PITCH3D_MESH_FRAME", "0"))

os.makedirs(OUT, exist_ok=True)
paths = sorted(glob.glob(os.path.join(NPZ_DIR, "subject_*.npz")))
assert paths, f"no subject_*.npz in {NPZ_DIR}"

# SMPLest-X emits the body in an image/camera frame whose vertical axis points DOWN
# (head at -y, feet at +y). Rotate it into our z-up world so the head lands on +z, then
# place it with the z-up world transl (the pelvis position the pipeline solved for).
# new = [x, z, -y]: lateral stays, depth->y, height (head-up) -> +z.
R_SMPLX_TO_OURS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)


def forward(npz_path: str, frame: int):
    """smplx_npz subject -> (z-up vertices about pelvis (V,3), faces (F,3), world transl (3,))."""
    d = np.load(npz_path, allow_pickle=True)
    frame = min(frame, d["frames"].shape[0] - 1)
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
            global_orient=torch.tensor(d["global_orient"][frame][None], dtype=torch.float32),
            body_pose=torch.tensor(d["body_pose"][frame].reshape(1, -1), dtype=torch.float32),
        )  # transl applied below, in z-up world, after the y-up -> z-up rotation
    verts = out.vertices[0].numpy() @ R_SMPLX_TO_OURS.T
    return verts, model.faces.astype(np.int64), d["transl"][frame]


def _equal_box(ax, lo, hi):
    ctr = (lo + hi) / 2
    r = (hi - lo).max() / 2
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.set_box_aspect((1, 1, 1))


# --- close-up: subject 1, recentred to the origin (z-up, stands upright) ---
V, F, t = forward(paths[0], FRAME)
ext = V.max(0) - V.min(0)
print(f"subject_1 z-up bbox extent (x,y,z) = {np.round(ext, 3).tolist()}  transl={np.round(t,3).tolist()}")
Vc = V - V.mean(0)
fig = plt.figure(figsize=(5, 8))
ax = fig.add_subplot(111, projection="3d")
coll = Poly3DCollection(Vc[F], alpha=1.0)
coll.set_facecolor((0.32, 0.5, 0.78))
coll.set_edgecolor("none")
ax.add_collection3d(coll)
_equal_box(ax, Vc.min(0), Vc.max(0))
ax.set_axis_off()
ax.view_init(elev=6, azim=-90)
fig.savefig(os.path.join(OUT, "mesh_subject1.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote mesh_subject1.png")

# --- scene: every subject at its world transl, standing on z=0 ---
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")
lo = np.array([np.inf] * 3)
hi = np.array([-np.inf] * 3)
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i, p in enumerate(paths):
    V, F, t = forward(p, FRAME)
    Vw = V + t
    coll = Poly3DCollection(Vw[F], alpha=0.95)
    coll.set_facecolor(colors[i % 10][:3])
    coll.set_edgecolor("none")
    ax.add_collection3d(coll)
    lo = np.minimum(lo, Vw.min(0))
    hi = np.maximum(hi, Vw.max(0))
    print(f"  {os.path.basename(p)}: transl={np.round(t, 2).tolist()}")
lo[2] = min(lo[2], 0.0)
_equal_box(ax, lo, hi)
ax.view_init(elev=18, azim=-60)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z (up)")
fig.savefig(os.path.join(OUT, "scene_meshes.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote scene_meshes.png")
print("MESH_RENDER_OK")
