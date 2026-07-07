"""Physics-only Blender preview: SMPL-X meshes animate on a flat pitch.

Reads the per-frame vertex sequences dumped by ``physics_debug_export.py`` and
renders an EEVEE animation with:

* one grey mesh per subject (per-team tint, no textures, no crowd, no photoreal)
* a flat 105 × 68 m green pitch with a white boundary rectangle
* one sun overhead
* a broadcast-angle camera framing all subjects

Fast iteration is the point — no v2v, no shadows-catcher, no auto-target
lighting. If a physics fix makes the run cycle look right here, it will look
right in the photoreal path too.

Env:
    PITCH3D_BLENDER — Blender ≥ 5.x binary (from .env)

Run headless:
    "$PITCH3D_BLENDER" --background \\
        --python scripts/blender_physics_debug.py -- \\
        --in out/physics_debug/frames.npz \\
        --out out/physics_debug/render/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

import bpy
import mathutils

_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _arg(name: str, default: str) -> str:
    return _argv[_argv.index(name) + 1] if name in _argv else default


IN = _arg("--in", "out/physics_debug/frames.npz")
OUT = _arg("--out", "out/physics_debug/render/")
RES_X = int(_arg("--res-x", "960"))
RES_Y = int(_arg("--res-y", "540"))
SAMPLES = int(_arg("--samples", "16"))

Path(OUT).mkdir(parents=True, exist_ok=True)

# ─── load export ─────────────────────────────────────────────────────────────
data = np.load(IN, allow_pickle=True)
subjects_arr = data["subjects"]
pitch = data["pitch"].item()
fps = float(data["fps"])
n_frames = int(data["n_frames"])

subjects = [s for s in subjects_arr]
print(f"loaded {len(subjects)} subjects × {n_frames} frames @ {fps} fps")

# ─── empty scene, world, sun ─────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# World: overcast grey — floodlit night reads flat-neutral in eye
world = bpy.data.worlds.new("physdbg_world")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.20, 0.22, 0.25, 1.0)
bg.inputs[1].default_value = 0.6

# Sun overhead
bpy.ops.object.light_add(type="SUN", location=(0, 0, 30))
sun = bpy.context.active_object
sun.data.energy = 3.0
sun.data.angle = 0.05
sun.rotation_euler = (np.radians(60), 0, np.radians(30))

# ─── flat pitch ──────────────────────────────────────────────────────────────
px = float(pitch["x_size"])
py = float(pitch["y_size"])
bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
plane = bpy.context.active_object
plane.scale = (px / 2 + 4.0, py / 2 + 4.0, 1.0)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
pitch_mat = bpy.data.materials.new("pitch")
pitch_mat.use_nodes = True
bsdf = pitch_mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = (0.08, 0.28, 0.10, 1.0)
bsdf.inputs["Roughness"].default_value = 0.9
plane.data.materials.append(pitch_mat)

# White boundary rectangle (thin flat strips just above the pitch)
def _line(a, b, thickness=0.15):
    ax, ay = a; bx, by = b
    cx, cy = (ax + bx) / 2, (ay + by) / 2
    length = float(np.hypot(bx - ax, by - ay))
    yaw = float(np.arctan2(by - ay, bx - ax))
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(cx, cy, 0.01))
    line = bpy.context.active_object
    line.scale = (length / 2, thickness / 2, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    line.rotation_euler = (0, 0, yaw)
    mat = bpy.data.materials.new(f"line_{cx:.1f}_{cy:.1f}")
    mat.use_nodes = True
    b = mat.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
    b.inputs["Roughness"].default_value = 0.6
    line.data.materials.append(mat)


hx, hy = px / 2, py / 2
_line((-hx, -hy), (hx, -hy))
_line((hx, -hy), (hx, hy))
_line((hx, hy), (-hx, hy))
_line((-hx, hy), (-hx, -hy))
_line((0, -hy), (0, hy))  # halfway line

# ─── subject meshes ──────────────────────────────────────────────────────────
subject_meshes: list[tuple[bpy.types.Object, np.ndarray]] = []
for s in subjects:
    verts_seq = s["verts"]   # (T, V, 3)
    faces = s["faces"]       # (F, 3)
    color = s["color"]       # (3,)
    tid = int(s["track_id"])

    T = verts_seq.shape[0]
    me = bpy.data.meshes.new(f"subj_{tid}_mesh")
    me.from_pydata(verts_seq[0].tolist(), [], faces.tolist())
    me.update()
    for poly in me.polygons:
        poly.use_smooth = True

    ob = bpy.data.objects.new(f"subj_{tid}", me)
    bpy.context.collection.objects.link(ob)

    mat = bpy.data.materials.new(f"body_{tid}")
    mat.use_nodes = True
    b = mat.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (
        float(color[0]), float(color[1]), float(color[2]), 1.0,
    )
    b.inputs["Roughness"].default_value = 0.65
    me.materials.append(mat)

    subject_meshes.append((ob, verts_seq))

# ─── per-frame vertex morph handler ──────────────────────────────────────────
def _update_verts_for_frame(scene: bpy.types.Scene) -> None:
    f = scene.frame_current - 1   # blender frames 1-based; our data 0-based
    for ob, verts_seq in subject_meshes:
        T = verts_seq.shape[0]
        if T == 0:
            continue
        k = max(0, min(T - 1, f))
        me = ob.data
        me.vertices.foreach_set("co", verts_seq[k].reshape(-1))
        me.update()


bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(_update_verts_for_frame)

# ─── camera (broadcast-ish, framing the pitch) ───────────────────────────────
cam_data = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
cam.location = (0.0, -py * 1.2, 20.0)
look_at = mathutils.Vector((0.0, 0.0, 1.0))
cam.rotation_euler = (
    look_at - mathutils.Vector(cam.location)
).to_track_quat("-Z", "Y").to_euler()
cam_data.lens = 35.0
scene.camera = cam

# ─── render settings (EEVEE, fast) ───────────────────────────────────────────
# Blender 5.1: engine name is BLENDER_EEVEE (BLENDER_EEVEE_NEXT ships in 5.2+).
scene.render.engine = "BLENDER_EEVEE"
try:
    scene.eevee.taa_render_samples = SAMPLES
except AttributeError:
    pass
scene.render.fps = int(round(fps))
scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.image_settings.file_format = "PNG"
scene.frame_start = 1
scene.frame_end = int(n_frames)
scene.render.filepath = str(Path(OUT) / "frame_")

print(f"rendering {n_frames} frames @ {RES_X}x{RES_Y}, EEVEE samples={SAMPLES}")
bpy.ops.render.render(animation=True)
print(f"PHYSICS_DEBUG_OK -> {OUT}")
