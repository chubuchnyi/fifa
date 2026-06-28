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

from pitch3d.adapters.io.frames import resolve_source_path
from pitch3d.adapters.models.avatar import measured_texture_from_clip
from pitch3d.adapters.render.overlay import appearance_alpha
from pitch3d.core.correction.engine import resolve_ball, resolve_subject_motion
from pitch3d.core.scene.pitch import goal_frame_geometry, pitch_line_ribbons
from pitch3d.core.scene.serialization import load_scene
from pitch3d.core.scene.stadium import (
    bowl_tile_loop_uvs,
    fill_holes_by_copy,
    stadium_bowl_geometry,
)
from pitch3d.env import load_env

load_env()  # PITCH3D_SMPLX_MODELS and friends come from the repo-root .env, never hard-coded

MODELS = os.environ.get("PITCH3D_SMPLX_MODELS", "SMPL-X/models")
SCENE_JSON = os.environ.get("PITCH3D_SCENE_JSON", "out/anim/export/scene.json")
OUT = os.environ.get("PITCH3D_ANIM_OUT", "out/anim/mesh")
# Entry/exit opacity fade (#98/#100): bake a per-frame alpha into each subject npz so the Blender
# render can ramp a body in/out at GENUINE entries/exits instead of popping it. 0 disables (opaque).
FADE_FRAMES = int(os.environ.get("PITCH3D_FADE_FRAMES", "4"))
# The broadcast source clip drives every MEASURED appearance: the per-vertex body texture (in the
# subject loop) and the stadium crowd (further down). PITCH3D_STADIUM_VIDEO names it for historical
# reasons (it first fed only the stadium); bodies sample the SAME clip through the solved camera.
# Unset/missing -> bodies keep their flat kit colour (R-6: never fabricate pixels we can't measure).
SOURCE_VIDEO = os.environ.get("PITCH3D_STADIUM_VIDEO", "")
SOURCE_OK = bool(SOURCE_VIDEO) and os.path.exists(resolve_source_path(SOURCE_VIDEO))

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
for _stale in glob.glob(os.path.join(OUT, "anim_subject_*.npz")) + [
    os.path.join(OUT, "ball.npz"), os.path.join(OUT, "pitch.npz"),
    os.path.join(OUT, "stadium.npz")
]:
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

    # Measured per-vertex body texture (M2-8b carried into the video path): sample each player's
    # real broadcast pixels onto its posed mesh through the solved camera, averaged over an even
    # spread of the frames it appears in. Vertices never seen front-facing (the far/occluded
    # surface) fall back to the flat kit colour; the measured flag, also written, is the honest R-6
    # channel. No clip -> vcolor stays None and the body keeps its flat kit colour as before.
    vcolor = None
    measured = None
    if SOURCE_OK:
        vcolor, measured = measured_texture_from_clip(
            verts, model.faces, scene.camera, frames, SOURCE_VIDEO
        )
        vcolor[~measured] = color

    # Shirt number plate (#numbers, v1): when the subject carries a jersey number, bake a per-frame
    # upper-back anchor + outward "back" direction so the renderer can place a number on the shirt
    # without any SMPL-X knowledge. Anchor = spine3↔neck blend pushed out along the back normal; the
    # back normal is the *posterior* horizontal direction, derived as -(facing), facing = eyes−head
    # (SMPL-X joints 23/24 are the eyeballs, in front of the head joint 15).
    num_extra: dict[str, np.ndarray] = {}
    if subj.jersey_number is not None:
        joints = out.joints.numpy() @ R_SMPLX_TO_OURS.T + transl[:, None, :]  # (T, J, 3) z-up world
        spine3, neck, head = joints[:, 9], joints[:, 12], joints[:, 15]
        facing = 0.5 * (joints[:, 23] + joints[:, 24]) - head
        facing[:, 2] = 0.0
        fn = np.linalg.norm(facing, axis=1, keepdims=True)
        facing = np.divide(facing, fn, out=np.zeros_like(facing), where=fn > 1e-6)
        back = -facing  # (T,3) unit posterior horizontal; zero rows where facing was degenerate
        # Mid-upper-back height (spine3-weighted so it sits between the shoulder blades, not on the
        # neck) pushed ~0.19 m out along the posterior normal. The spine joints sit at the torso
        # centre, so a smaller offset leaves the plate buried in the mesh (only fragments poke
        # through); 0.19 floats it a few cm proud of the curved back skin so the digits read clean.
        anchor = 0.62 * spine3 + 0.38 * neck + 0.19 * back
        lum = float(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
        num_rgb = np.array([0.04, 0.04, 0.07] if lum > 0.5 else [0.97, 0.97, 0.97], np.float32)
        num_extra = dict(
            jersey_number=np.int64(int(subj.jersey_number)),
            back_anchor=anchor.astype(np.float32),
            back_dir=back.astype(np.float32),
            number_rgb=num_rgb,
        )

    tex_extra: dict[str, np.ndarray] = {}
    if vcolor is not None and measured is not None:
        tex_extra = dict(vcolor=vcolor.astype(np.float32), measured=measured.astype(np.uint8))
    dst = os.path.join(OUT, f"anim_subject_{subj.track_id}.npz")
    np.savez(
        dst,
        verts=verts.astype(np.float32),
        faces=model.faces.astype(np.int32),
        color=color,
        frames=frames.astype(np.int64),
        alpha=alpha.astype(np.float32),
        **num_extra,
        **tex_extra,
    )
    span = float(np.linalg.norm(transl.max(0) - transl.min(0)))
    num_msg = ""
    if num_extra:
        bz = float(np.abs(num_extra["back_dir"][:, 2]).mean())  # ~0 ⇒ horizontal (sane)
        num_msg = f" number={int(num_extra['jersey_number'])} back_dir|z|~{bz:.2f}"
    tex_msg = f" tex={measured.mean() * 100:.0f}%" if measured is not None else ""
    print(
        f"subject_{subj.track_id}: team={subj.team_id} frames={n_frames} "
        f"transl_span={span:.2f}m{num_msg}{tex_msg} -> {os.path.basename(dst)}"
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

# Measured pitch markings + goal frames as world geometry for the renderer (#205). The dims are the
# field's own (the homography anchors bodies to THIS template), so the lines/goals line up with the
# subjects once placement is correct. blender_animate.py loads pitch.npz exactly like ball.npz.
_dims = scene.field.dimensions
_pv, _pf = pitch_line_ribbons(_dims)
_gv, _gf = goal_frame_geometry(_dims)
pdst = os.path.join(OUT, "pitch.npz")
np.savez(
    pdst,
    pitch_verts=_pv.astype(np.float32),
    pitch_faces=_pf.astype(np.int32),
    goal_verts=_gv.astype(np.float32),
    goal_faces=_gf.astype(np.int32),
)
print(
    f"pitch: {_pf.shape[0]} line-tris + {_gf.shape[0]} goal-tris "
    f"({_dims.length:g}x{_dims.width:g} m) -> {os.path.basename(pdst)}"
)

# Hybrid stadium backdrop (M2 stadium): a procedural seating bowl around the pitch given REAL
# appearance from THIS clip — a *tinted mosaic*. A clean crowd patch cut from the broadcast tiles
# over the bowl (so spectators stay crisp instead of one stretched pixel per vertex), modulated by
# the per-vertex median colour the camera measured (so each stand keeps its real tint and the near
# side it never saw is copy-filled from its mirror). Gated on PITCH3D_STADIUM_VIDEO (the source
# clip): with no clip we cannot measure crowd colour, so the renderer omits the bowl, not invent it.
STADIUM_REPEAT_AROUND = 40.0  # crowd-tile copies laid around the loop (mirror-tiled in Blender)
STADIUM_REPEAT_UP = 4.0       # copies up the rake; broadcast crowd band is short, so tile upward
if SOURCE_OK:
    from pitch3d.adapters.render.stadium_backdrop import bake_backdrop_colors, extract_crowd_tile

    _sv, _sf, _sp = stadium_bowl_geometry(_dims)
    _scolors, _scov = bake_backdrop_colors(scene.camera, _sv, SOURCE_VIDEO)
    _sfilled, _ = fill_holes_by_copy(_sv, _scolors, _scov)
    _stile = extract_crowd_tile(scene.camera, _sv, _sp, _scov, SOURCE_VIDEO)
    _suv = bowl_tile_loop_uvs(
        _sf, _sp, repeat_around=STADIUM_REPEAT_AROUND, repeat_up=STADIUM_REPEAT_UP
    )
    sdst = os.path.join(OUT, "stadium.npz")
    np.savez(
        sdst,
        verts=_sv.astype(np.float32),
        faces=_sf.astype(np.int32),
        colors=_sfilled.astype(np.float32),
        uv=_suv.astype(np.float32),
        tile=_stile.astype(np.float32),
    )
    print(
        f"stadium: {_sv.shape[0]} verts {_sf.shape[0]} tris; covered "
        f"{int(_scov.sum())}/{_scov.size} ({_scov.mean() * 100:.0f}%); "
        f"tile {_stile.shape[1]}x{_stile.shape[0]} -> {os.path.basename(sdst)}"
    )
else:
    print("stadium: PITCH3D_STADIUM_VIDEO unset/missing (skipping stadium.npz)")

print(f"ANIM_EXPORT_OK ({len(scene.subjects)} subjects -> {OUT})")
