"""Forward a canonical scene JSON through SMPL-X for ALL frames → Blender-ready animation.

The still-image cousin (scripts/smplx_export_meshes.py) poses ONE frame from the bodies-only
`smplx_npz` export. This one reads the **full canonical scene** (`--format json`, which alone
carries the ball) and forwards every subject across *all* frames, so the result is an animation
with the ball — exactly what scripts/blender_animate.py needs.

Per subject it writes `anim_subject_<track_id>.npz` {verts (T,V,3) world z-up, faces (F,3),
color (3,), frames (T,)}; the resolved ball (if any) goes to `ball.npz`
{frames (T,), positions_3d (T,3), height_confidence (T,)}. Edits (the correction stack) are
applied here via the same resolve_* path the exporters use, so a corrected scene animates corrected.

Env (machine paths come from the repo-root .env; see .env.example):
  PITCH3D_SMPLX_MODELS  dir containing smplx/SMPLX_NEUTRAL.npz
  PITCH3D_SCENE_JSON    canonical scene JSON (pipeline --format json export)
  PITCH3D_ANIM_OUT      output dir for anim_subject_*.npz + ball.npz
  PITCH3D_CANONICAL_UP  1 for a fake/canonical export (see smplx_export_meshes.py)

Run:  .venv/bin/python scripts/anim_export.py
"""

import glob
import os

import matplotlib
import numpy as np
import smplx
import torch

from pitch3d.adapters.render.overlay import appearance_alpha
from pitch3d.core.correction.engine import resolve_ball, resolve_subject_motion
from pitch3d.core.scene.serialization import load_scene
from pitch3d.env import load_env

load_env()  # PITCH3D_SMPLX_MODELS and friends come from the repo-root .env, never hard-coded

MODELS = os.environ.get("PITCH3D_SMPLX_MODELS", "SMPL-X/models")
SCENE_JSON = os.environ.get("PITCH3D_SCENE_JSON", "out/anim/export/scene.json")
OUT = os.environ.get("PITCH3D_ANIM_OUT", "out/anim/mesh")
# Entry/exit opacity fade (#98/#100): bake a per-frame alpha into each subject npz so the Blender
# render can ramp a body in/out at GENUINE entries/exits instead of popping it. 0 disables (opaque).
FADE_FRAMES = int(os.environ.get("PITCH3D_FADE_FRAMES", "4"))

# Same orientation gotcha as smplx_export_meshes.py: real SMPLest-X output is camera-frame
# (y-down) → map to z-up world with new = [x, z, -y]; a fake/canonical export needs the plain
# y-up → z-up map (PITCH3D_CANONICAL_UP=1). Then add the z-up world transl (pelvis position).
if os.environ.get("PITCH3D_CANONICAL_UP", "0") == "1":
    R_SMPLX_TO_OURS = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
else:
    R_SMPLX_TO_OURS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)

os.makedirs(OUT, exist_ok=True)
# The pod-side OUT dir lives on the persistent volume and is reused across runs, so a track
# that existed only in a PRIOR run (e.g. anim_subject_22.npz) would otherwise linger and be
# globbed by blender_animate.py — rendering a phantom body this scene never had. Purge the
# per-subject + ball artifacts up front so the mesh dir reflects EXACTLY this scene.
for _stale in glob.glob(os.path.join(OUT, "anim_subject_*.npz")) + [os.path.join(OUT, "ball.npz")]:
    if os.path.exists(_stale):
        os.remove(_stale)

scene = load_scene(SCENE_JSON)
assert scene.subjects, f"no subjects in {SCENE_JSON}"

# Rendered clip span = union of every present frame (subjects + ball) — the same range
# blender_animate.py iterates. A subject touching this span's edge was clipped by the window, not a
# genuine entry/exit, so it is NOT faded. Frames are resolve-invariant (the engine never inserts
# rows; coherence gap-fill is already baked into the proposal), so proposal frames are exact here.
_present = [np.asarray(s.proposal.pose.frames, dtype=int) for s in scene.subjects]
if scene.ball is not None and np.asarray(scene.ball.frames).size:
    _present.append(np.asarray(scene.ball.frames, dtype=int))
clip_first = int(min(int(f[0]) for f in _present if f.size))
clip_last = int(max(int(f[-1]) for f in _present if f.size))

# Team colours when the tracker classified them (team A vs B reads clearly in the render);
# otherwise fall back to a distinct per-subject palette.
team_color = {t.id: t.color_rgb for t in scene.teams if t.color_rgb is not None}
palette = matplotlib.colormaps["tab10"](np.linspace(0, 1, 10))[:, :3]

for i, subj in enumerate(scene.subjects):
    motion = resolve_subject_motion(subj.proposal, scene.corrections_for(subj.track_id))
    betas = np.asarray(motion.shape.betas, dtype=np.float32)
    frames = np.asarray(motion.pose.frames)
    n_frames = int(frames.shape[0])
    model = smplx.create(
        MODELS,
        model_type="smplx",
        gender="neutral",
        num_betas=int(betas.shape[0]),
        use_pca=False,
        flat_hand_mean=True,
        batch_size=n_frames,
    )
    with torch.no_grad():
        out = model(
            betas=torch.tensor(np.tile(betas[None], (n_frames, 1)), dtype=torch.float32),
            global_orient=torch.tensor(motion.pose.global_orient, dtype=torch.float32),
            body_pose=torch.tensor(
                np.asarray(motion.pose.body_pose).reshape(n_frames, -1), dtype=torch.float32
            ),
        )
    transl = np.asarray(motion.pose.transl, dtype=np.float32)  # (T,3) z-up world
    verts = out.vertices.numpy() @ R_SMPLX_TO_OURS.T + transl[:, None, :]  # (T,V,3)
    color = np.asarray(team_color.get(subj.team_id, palette[i % 10]), dtype=np.float32)
    alpha = appearance_alpha(frames, clip_first, clip_last, FADE_FRAMES)  # (T,) in [0,1]
    dst = os.path.join(OUT, f"anim_subject_{subj.track_id}.npz")
    np.savez(
        dst,
        verts=verts.astype(np.float32),
        faces=model.faces.astype(np.int32),
        color=color,
        frames=frames.astype(np.int64),
        alpha=alpha.astype(np.float32),
    )
    span = float(np.linalg.norm(transl.max(0) - transl.min(0)))
    print(
        f"subject_{subj.track_id}: team={subj.team_id} frames={n_frames} "
        f"transl_span={span:.2f}m -> {os.path.basename(dst)}"
    )

if scene.ball is not None:
    ball = resolve_ball(scene.ball, scene.corrections_for(None))
    bdst = os.path.join(OUT, "ball.npz")
    np.savez(
        bdst,
        frames=np.asarray(ball.frames, dtype=np.int64),
        positions_3d=np.asarray(ball.positions_3d, dtype=np.float32),
        height_confidence=np.asarray(ball.height_confidence, dtype=np.float32),
    )
    print(f"ball: {int(np.asarray(ball.frames).shape[0])} frames -> {os.path.basename(bdst)}")
else:
    print("ball: none in scene (skipping ball.npz)")

print(f"ANIM_EXPORT_OK ({len(scene.subjects)} subjects -> {OUT})")
