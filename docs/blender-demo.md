# Blender skinned-mesh demo (from pipeline output)

Turn a pipeline **`smplx_npz` export** into a lit, shadowed **skinned SMPL-X render** — the honest
"what a Blender import sees": `smplx_npz → SMPL-X forward → vertices+faces → Blender Cycles`. This is
distinct from `--observer blender`, which renders cheap **box proxies** (cubes) for live editing
feedback; this path renders the **actual body mesh**.

Two scripts, because Blender's bundled Python has `numpy` but **not** `torch`/`smplx`, so the SMPL-X
forward must run in the repo venv and hand geometry to Blender via files:

1. [`scripts/smplx_export_meshes.py`](../scripts/smplx_export_meshes.py) — venv: each `subject_*.npz`
   → `mesh_<name>.npz` `{verts (V,3) world z-up, faces (F,3), color (3,)}`.
2. [`scripts/blender_render_meshes.py`](../scripts/blender_render_meshes.py) — `bpy`: loads those,
   builds meshes on a grass ground plane with a sun + sky, frames them, and renders with **Cycles CPU**
   (works headless, no GPU/display needed). Writes `blender_scene.png` (all subjects) +
   `blender_scene_hero.png` (close-up of the first body).

For a **live, interactive** look instead of a still, [`scripts/blender_view_meshes.py`](../scripts/blender_view_meshes.py)
builds the *same* scene but leaves Blender **open** with a realtime material-preview viewport framed
through the camera, so you can orbit the reconstructed crowd. It needs a display, so do **not** pass
`--background` (see "Live, interactive GUI" below).

A matplotlib-only variant, [`scripts/render_smplx_mesh.py`](../scripts/render_smplx_mesh.py), renders
the same mesh **without Blender** (quick look / pod sanity check).

## Prerequisites (verified present locally, 2026-06-23)

| Need | Local path |
|---|---|
| Blender binary | `/home/chubuchnyi/Downloads/blender-5.1.2-linux-x64/blender` (5.1.2) |
| SMPL-X body model (gated) | `/home/chubuchnyi/AVATAR/SMPL-X/models/` (`smplx/SMPLX_NEUTRAL.npz`) |
| `smplx` python pkg | repo venv `/home/chubuchnyi/AVATAR/.venv` |
| A `smplx_npz` export | e.g. `out/cuda/export/scene.smplx_npz/subject_*.npz` |

## Run

```bash
# 1. forward the export to Blender-ready geometry (repo venv)
.venv/bin/python scripts/smplx_export_meshes.py            # writes out/cuda/mesh/mesh_subject_*.npz

# 2. render with Blender (headless, Cycles CPU) — $PITCH3D_BLENDER comes from .env
"$PITCH3D_BLENDER" --background \
    --python scripts/blender_render_meshes.py -- \
    --in out/cuda/mesh --out out/cuda/mesh/blender_scene.png
# -> out/cuda/mesh/blender_scene.png  +  blender_scene_hero.png
```

> The fastest path is [`scripts/demo.sh`](../scripts/demo.sh) (see [`demo.md`](demo.md)) — it runs
> both steps for you (and the GPU half) with `.env`-driven paths.

## Live, interactive GUI

Same scene, but an **open** Blender window you can orbit (realtime material preview), instead of a
headless still. Needs a display — on a local GNOME/Wayland box pass the session env through and **omit**
`--background`:

```bash
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 DISPLAY=:0 \
  "$PITCH3D_BLENDER" \
  --python scripts/blender_view_meshes.py -- --in out/live_real/mesh
# log prints `BLENDER_VIEW_READY -> N bodies ...`; the window opens framed through the camera.
```

### Full live chain (real models on a GPU pod → local GUI)

The end-to-end "real video → real SMPL-X → live Blender" demo. The heavy half runs on the GPU box
(real RF-DETR + ByteTrack + SMPLest-X); the visual half runs locally on CPU:

```bash
# 1. ON THE POD: real perception → real SMPL-X pose → smplx_npz export (see docs/cloud-dev.md §5).
#    The committed runner injects the wired SMPLest-X backend by dotted path (ADR-0006):
OUT=out/live_demo FRAMES=6 bash scripts/pod_real_e2e.sh
#    (equivalently, the explicit command the runner wraps:)
#    PYTHONPATH=src python -m pitch3d --clip clip.mp4 --frames 6 \
#      --detector rfdetr --tracker bytetrack --device cuda \
#      --pose gvhmr --pose-backend pitch3d.adapters.models.smplestx_backend:make \
#      --render overlay --export gltf --format smplx_npz --out-dir out/live_demo

# 2. LOCAL: pull the real export down (scp from the pod), into out/live_real/export/...
#    then forward it through SMPL-X (real-pose → DEFAULT orientation, NOT canonical):
PITCH3D_NPZ_DIR=out/live_real/export/scene.smplx_npz \
  PITCH3D_MESH_OUT=out/live_real/mesh \
  .venv/bin/python scripts/smplx_export_meshes.py

# 3. LOCAL: open the live GUI on the reconstructed crowd (command above), --in out/live_real/mesh
```

## Env / flags

`smplx_export_meshes.py` reads:

- `PITCH3D_SMPLX_MODELS` — dir containing `smplx/SMPLX_NEUTRAL.npz` (default local path above).
- `PITCH3D_NPZ_DIR` — dir of `subject_*.npz` (the `--format smplx_npz` export).
- `PITCH3D_MESH_OUT` — output dir for `mesh_*.npz`.
- `PITCH3D_MESH_FRAME` — which frame to pose (default `0`).
- `PITCH3D_CANONICAL_UP` — orientation, see below.

`blender_render_meshes.py` takes `--in <mesh_dir>` and `--out <png>`.

## Orientation gotcha (important)

SMPL-X is y-up; our world is z-up. **Real SMPLest-X** output lives in an image/camera frame whose
vertical axis points **down** (a standing body has its head at −y), so the default rotation maps
`new = [x, z, −y]` and the body stands up. But a **fake/degenerate** export (the no-GPU `--pose fake`
path writes `global_orient = 0`, a *canonical* y-up body) renders **upside-down** under that map —
for those set `PITCH3D_CANONICAL_UP=1` (uses `new = [x, −z, y]`).

Rule of thumb: real-pose export → default; fake-pose / canonical export → `PITCH3D_CANONICAL_UP=1`.

> The local `out/cuda` export is a **fake-pose** run (`betas=0`, `body_pose=0`) → default-shape bodies
> in the SMPL-X rest pose. **Dynamic, person-specific** meshes come from a **real-pose** export (e.g. a
> GPU run's `out/e2e_real/export/scene.smplx_npz`), rendered the same way with the default orientation.
