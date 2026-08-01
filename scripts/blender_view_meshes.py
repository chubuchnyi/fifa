"""Open mesh_*.npz (from smplx_export_meshes.py) in a LIVE, interactive Blender GUI.

Unlike blender_render_meshes.py (headless Cycles stills that quit when done), this builds
the SAME scene — SMPL-X bodies in team colors on a grass plane under a sun + sky — but
leaves Blender OPEN with a realtime (MATERIAL-preview) viewport framed through the camera,
so you can orbit the actual reconstructed crowd live. Needs a display; do NOT pass --background.

Run ($PITCH3D_BLENDER points at the Blender 5.x binary; see .env):
  DISPLAY=:0 "$PITCH3D_BLENDER" \
      --python scripts/blender_view_meshes.py -- --in out/live_real/mesh
"""

import glob
import os
import sys

import bpy
import mathutils
import numpy as np

_argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def _arg(name, default):
    return _argv[_argv.index(name) + 1] if name in _argv else default


IN = _arg("--in", "out/live_real/mesh")

bpy.ops.wm.read_factory_settings(use_empty=True)

mesh_files = sorted(glob.glob(os.path.join(IN, "mesh_*.npz")))
assert mesh_files, f"no mesh_*.npz in {IN}"

all_v = []
for mp in mesh_files:
    d = np.load(mp)
    verts, faces, col = d["verts"], d["faces"], d["color"]
    all_v.append(verts)
    me = bpy.data.meshes.new(os.path.basename(mp))
    me.from_pydata(verts.tolist(), [], faces.tolist())
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
    ob.data.materials.append(mat)

all_v = np.concatenate(all_v, 0)
ctr = all_v.mean(0)
lo, hi = all_v.min(0), all_v.max(0)
span = float((hi - lo).max())

# grass ground at z=0 (data API — bpy.context.active_object is None in GUI startup)
ghalf = max(60.0, span * 4) / 2.0
gx, gy = float(ctr[0]), float(ctr[1])
gme = bpy.data.meshes.new("grass")
gme.from_pydata(
    [
        (gx - ghalf, gy - ghalf, 0.0),
        (gx + ghalf, gy - ghalf, 0.0),
        (gx + ghalf, gy + ghalf, 0.0),
        (gx - ghalf, gy + ghalf, 0.0),
    ],
    [],
    [(0, 1, 2, 3)],
)
gme.update()
gmat = bpy.data.materials.new("grass")
gmat.use_nodes = True
gmat.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (0.06, 0.3, 0.09, 1)
gme.materials.append(gmat)
bpy.context.collection.objects.link(bpy.data.objects.new("grass", gme))

# sun (data API)
sun = bpy.data.lights.new("sun", type="SUN")
sun.energy = 4.0
sun.angle = 0.1
sob = bpy.data.objects.new("sun", sun)
sob.location = (ctr[0] + 8, ctr[1] - 8, 20)
bpy.context.collection.objects.link(sob)

# sky-ish world
world = bpy.data.worlds.new("w")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.55, 0.72, 0.92, 1.0)
bg.inputs[1].default_value = 1.0

# camera, framing the whole crowd from a broadcast-ish 3/4 angle
cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
cam.location = (ctr[0] + span * 0.4, ctr[1] - span * 1.7 - 6.0, ctr[2] + span * 0.5 + 4.0)
look_at = mathutils.Vector((ctr[0], ctr[1], ctr[2] + 0.6))
cam.rotation_euler = (look_at - mathutils.Vector(cam.location)).to_track_quat("-Z", "Y").to_euler()
bpy.context.scene.camera = cam


def _setup_viewport():
    # run once, after the GUI is up: every 3D viewport -> camera view, MATERIAL preview
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type != "VIEW_3D":
                    continue
                space.shading.type = "MATERIAL"
                if space.region_3d is not None:
                    space.region_3d.view_perspective = "CAMERA"
            area.tag_redraw()
    return None  # one-shot timer


bpy.app.timers.register(_setup_viewport, first_interval=0.5)
print(f"BLENDER_VIEW_READY -> {len(mesh_files)} bodies from {IN}")
