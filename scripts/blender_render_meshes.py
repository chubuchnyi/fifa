"""Render Blender-ready mesh_*.npz (from smplx_export_meshes.py) as a lit, shadowed scene.

This is the real Blender skinned-mesh demo: it builds each SMPL-X body as a Blender mesh,
drops them onto a grass ground plane with a sun + sky, frames them with a camera, and
renders with Cycles (CPU, so it works headless without a GPU/display).

Run headless ($PITCH3D_BLENDER points at the Blender 5.x binary; see .env):
  "$PITCH3D_BLENDER" --background \
      --python scripts/blender_render_meshes.py -- \
      --in out/cuda/mesh --out out/cuda/mesh/blender_scene.png
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


IN = _arg("--in", "out/cuda/mesh")
OUT = _arg("--out", "out/cuda/mesh/blender_scene.png")

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

# grass ground at z=0
bpy.ops.mesh.primitive_plane_add(size=max(60.0, span * 4), location=(ctr[0], ctr[1], 0.0))
gmat = bpy.data.materials.new("grass")
gmat.use_nodes = True
gmat.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (0.06, 0.3, 0.09, 1)
bpy.context.active_object.data.materials.append(gmat)

# sun
bpy.ops.object.light_add(type="SUN", location=(ctr[0] + 8, ctr[1] - 8, 20))
bpy.context.active_object.data.energy = 4.0
bpy.context.active_object.data.angle = 0.1

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

sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = 48
sc.cycles.use_denoising = True
sc.render.resolution_x = 1280
sc.render.resolution_y = 720
sc.render.filepath = OUT
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
bpy.ops.render.render(write_still=True)
print(f"BLENDER_RENDER_OK -> {OUT}")

# hero close-up of the first body (fills the frame so the skinned mesh is clearly visible)
v0 = np.load(mesh_files[0])["verts"]
c0 = v0.mean(0)
h = float(v0[:, 2].max() - v0[:, 2].min())
cam.location = (c0[0] + 0.25 * h, c0[1] - 2.4 * h, c0[2] + 0.35 * h)
cam.rotation_euler = (
    mathutils.Vector((float(c0[0]), float(c0[1]), float(c0[2])))
    - mathutils.Vector(cam.location)
).to_track_quat("-Z", "Y").to_euler()
hero = OUT.replace(".png", "_hero.png")
sc.render.filepath = hero
bpy.ops.render.render(write_still=True)
print(f"BLENDER_HERO_OK -> {hero}")
