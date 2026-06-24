"""Render a multi-camera ANIMATION of the reconstructed bodies + ball with Blender Cycles.

Input is scripts/anim_export.py's output: `anim_subject_*.npz` {verts (T,V,3) z-up, faces, color}
+ optional `ball.npz` {frames, positions_3d (T,3), height_confidence}. For every frame we re-pose
each body (`foreach_set("co", ...)`) and move the ball, then render the SAME instant from N fixed
cameras (broadcast / sideline / top-down / goal-end) — so each camera yields a PNG sequence that
scripts/pod_make_video.sh stitches into one mp4 per angle.

Runs either as a Blender-binary script (`blender --background --python ... -- --in DIR`) or as the
`bpy` pip module (`python scripts/blender_animate.py --in DIR`) — argv parsing handles both. Cycles
on CPU is the reliable headless default; `--device gpu` tries OPTIX/CUDA and falls back to CPU.

Flags (after `--` when run via the binary):
  --in DIR        dir of anim_subject_*.npz (+ ball.npz)         [default out/anim/mesh]
  --out DIR       root for <camera>/frame_*.png sequences        [default <in>/frames]
  --device cpu|gpu                                               [default cpu]
  --res-x N --res-y N --samples N --fps N --frame-step N --cameras a,b,c
"""

import glob
import os
import sys

import numpy as np

import bpy
import mathutils

_argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]


def _arg(name, default):
    return _argv[_argv.index(name) + 1] if name in _argv else default


IN = _arg("--in", "out/anim/mesh")
OUT = _arg("--out", os.path.join(IN, "frames"))
DEVICE = _arg("--device", "cpu").lower()
RES_X = int(_arg("--res-x", "1280"))
RES_Y = int(_arg("--res-y", "720"))
SAMPLES = int(_arg("--samples", "32"))
FPS = int(_arg("--fps", "25"))
STEP = int(_arg("--frame-step", "1"))
WANT_CAMS = _arg("--cameras", "broadcast,sideline,top,goal").split(",")

BALL_RADIUS = 0.11  # FIFA size-5 ball ≈ 0.11 m radius


def _look_at(cam, target):
    cam.rotation_euler = (
        mathutils.Vector(target) - mathutils.Vector(cam.location)
    ).to_track_quat("-Z", "Y").to_euler()


bpy.ops.wm.read_factory_settings(use_empty=True)

# ── load the animation export ────────────────────────────────────────────────
mesh_files = sorted(glob.glob(os.path.join(IN, "anim_subject_*.npz")))
assert mesh_files, f"no anim_subject_*.npz in {IN}"

# Each subject covers its OWN (possibly partial) frame range — real tracks come and go (a player
# tracked only on frames 24-25 must not truncate the whole clip). So we key everything by GLOBAL
# frame index and toggle each body's visibility per frame.
bodies = []          # (object, mesh, verts (Ti,V,3), {global_frame: row}, bsdf, alpha (Ti,))
all_frames = set()
lo = np.array([np.inf, np.inf, np.inf])
hi = -lo
for mp in mesh_files:
    d = np.load(mp)
    verts, faces, col, frames = d["verts"], d["faces"], d["color"], d["frames"]
    # Per-frame opacity baked by anim_export.py (#98/#100); absent in older exports → opaque.
    alpha = d["alpha"] if "alpha" in d.files else np.ones(len(frames), dtype=np.float32)
    me = bpy.data.meshes.new(os.path.basename(mp))
    me.from_pydata(verts[0].tolist(), [], faces.tolist())
    me.update()
    for poly in me.polygons:
        poly.use_smooth = True
    ob = bpy.data.objects.new(os.path.basename(mp), me)
    bpy.context.collection.objects.link(ob)
    mat = bpy.data.materials.new("body")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (float(col[0]), float(col[1]), float(col[2]), 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    me.materials.append(mat)
    frame_row = {int(f): i for i, f in enumerate(frames)}
    bodies.append((ob, me, verts, frame_row, bsdf, alpha))
    all_frames.update(frame_row)
    lo = np.minimum(lo, verts.reshape(-1, 3).min(0))
    hi = np.maximum(hi, verts.reshape(-1, 3).max(0))

ctr = (lo + hi) / 2.0
span = float(max((hi - lo)[0], (hi - lo)[1], 1.0))  # horizontal extent of all motion

# ── ball (optional) ──────────────────────────────────────────────────────────
ball_path = os.path.join(IN, "ball.npz")
ball_ob = None
ball_pos = None
ball_row = {}
if os.path.exists(ball_path):
    bd = np.load(ball_path)
    ball_pos = np.asarray(bd["positions_3d"], dtype=np.float32)
    ball_row = {int(f): i for i, f in enumerate(bd["frames"])}
    all_frames.update(ball_row)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=BALL_RADIUS, segments=24, ring_count=16)
    ball_ob = bpy.context.active_object
    for poly in ball_ob.data.polygons:
        poly.use_smooth = True
    bmat = bpy.data.materials.new("ball")
    bmat.use_nodes = True
    bbsdf = bmat.node_tree.nodes.get("Principled BSDF")
    bbsdf.inputs["Base Color"].default_value = (0.95, 0.95, 0.95, 1.0)
    bbsdf.inputs["Roughness"].default_value = 0.4
    ball_ob.data.materials.append(bmat)

# ── pitch, light, sky ────────────────────────────────────────────────────────
bpy.ops.mesh.primitive_plane_add(size=max(120.0, span * 3), location=(ctr[0], ctr[1], 0.0))
gmat = bpy.data.materials.new("grass")
gmat.use_nodes = True
gmat.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (0.06, 0.3, 0.09, 1)
bpy.context.active_object.data.materials.append(gmat)

bpy.ops.object.light_add(type="SUN", location=(ctr[0] + 8, ctr[1] - 8, 30))
bpy.context.active_object.data.energy = 4.0
bpy.context.active_object.data.angle = 0.1

world = bpy.data.worlds.new("w")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.55, 0.72, 0.92, 1.0)
bg.inputs[1].default_value = 1.0

# ── fixed cameras (static; the players + ball move within frame) ─────────────
look = (ctr[0], ctr[1], ctr[2] + 0.6)
cam_specs = {
    "broadcast": (ctr[0] + span * 0.35, ctr[1] - span * 1.5 - 8.0, ctr[2] + span * 0.55 + 6.0),
    "sideline":  (ctr[0], ctr[1] - span * 1.2 - 6.0, ctr[2] + 2.0),
    "top":       (ctr[0], ctr[1], ctr[2] + span * 1.8 + 20.0),
    "goal":      (ctr[0] + span * 1.4 + 10.0, ctr[1], ctr[2] + span * 0.4 + 4.0),
}
cameras = []
for name in WANT_CAMS:
    loc = cam_specs.get(name)
    if loc is None:
        continue
    cam_data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = loc
    _look_at(cam, (look[0], look[1], 0.6) if name == "top" else look)
    cameras.append((name, cam))
assert cameras, f"no known cameras among {WANT_CAMS} (try broadcast,sideline,top,goal)"

# ── render settings ──────────────────────────────────────────────────────────
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = True
sc.render.resolution_x = RES_X
sc.render.resolution_y = RES_Y
sc.render.image_settings.file_format = "PNG"
sc.cycles.device = "CPU"
if DEVICE == "gpu":
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        chosen = ""
        for backend in ("OPTIX", "CUDA"):
            prefs.compute_device_type = backend
            prefs.get_devices()
            if any(getattr(dev, "type", "") == backend for dev in prefs.devices):
                chosen = backend
                break
        if chosen:
            for dev in prefs.devices:
                dev.use = getattr(dev, "type", "") in (chosen, "CPU")
            sc.cycles.device = "GPU"
            print(f"BLENDER_ANIM_GPU {chosen}")
        else:
            print("BLENDER_ANIM_GPU none-found -> CPU")
    except Exception as exc:  # noqa: BLE001 - best-effort; CPU always works headless
        print(f"BLENDER_ANIM_GPU failed ({exc}) -> CPU")

for name, _ in cameras:
    cam_out = os.path.join(OUT, name)
    os.makedirs(cam_out, exist_ok=True)
    # Same reused-dir hazard as the npz export: a prior run with MORE frames would leave
    # frame_NNNN.png files that ffmpeg's glob would splice into this clip. Clear them so each
    # camera's PNG sequence is exactly this run's frames.
    for _stale in glob.glob(os.path.join(cam_out, "frame_*.png")):
        os.remove(_stale)

# ── animate over the GLOBAL frame range; per frame, show only bodies present then ─
gframes = sorted(all_frames)[::STEP]
assert gframes, "no frames to render (empty export?)"
rendered = 0
for gf in gframes:
    visible = 0
    for ob, me, verts, frame_row, bsdf, alpha in bodies:
        row = frame_row.get(gf)
        if row is None:
            ob.hide_render = True
            continue
        ob.hide_render = False
        visible += 1
        # Ramp opacity at genuine entries/exits (Cycles honours the Principled BSDF Alpha input
        # directly); a body present across the whole clip stays at alpha 1.0 → opaque as before.
        bsdf.inputs["Alpha"].default_value = float(alpha[row])
        me.vertices.foreach_set("co", np.ascontiguousarray(verts[row], dtype=np.float32).ravel())
        me.update()
    if ball_ob is not None:
        brow = ball_row.get(gf)
        ball_ob.hide_render = brow is None
        if brow is not None:
            ball_ob.location = tuple(float(x) for x in ball_pos[brow])
    for name, cam in cameras:
        sc.camera = cam
        sc.render.filepath = os.path.join(OUT, name, f"frame_{gf:04d}.png")
        bpy.ops.render.render(write_still=True)
    rendered += 1
    print(f"BLENDER_ANIM_FRAME {rendered}/{len(gframes)} (global={gf}, {visible} bodies, {len(cameras)} cams)")

print(
    f"BLENDER_ANIM_OK frames={rendered} cams={[c for c, _ in cameras]} "
    f"res={RES_X}x{RES_Y} samples={SAMPLES} fps={FPS} -> {OUT}"
)
