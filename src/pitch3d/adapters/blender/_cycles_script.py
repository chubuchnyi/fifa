"""In-Blender Cycles render script — runs *inside* Blender's Python, never imported by us (M2-7).

:func:`~pitch3d.adapters.blender.runner.run_cycles_render` invokes this via
``blender --background --factory-startup --python _cycles_script.py -- --plan plan.json
--mesh-dir <dir> --render-dir <dir>``. It reads the JSON
:class:`~pitch3d.adapters.blender.cycles_plan.CyclesPlan`, builds the *resolved* scene as real
geometry — each avatar mesh from its NPZ (vertices + triangles + per-vertex colour), a neutral
ground plane, a sun + sky world — places a physical camera from the plan's lens/shift/sensor, and
renders every plan frame with **Cycles** to ``frame_{index:05d}.png``. The camera ``matrix_world``
and each avatar's placement come straight from the plan (all the OpenCV→Blender maths happened on
the pure side), so this file only *applies* transforms.

Self-contained on purpose: only stdlib + ``bpy``/``mathutils`` + ``numpy`` (all Blender-bundled), so
it carries no dependency on the ``pitch3d`` package, and ``bpy`` is imported only when run as
Blender's ``__main__`` — importing this file in a normal interpreter never pulls in ``bpy``.

Honest scope (M2-7): avatars are placed by their root **rigid** transform only (no LBS — M2-8); the
environment is a single neutral matte ground plane (measured grass/line material is M2-9).
"""

from __future__ import annotations

import json
import math
import os
import sys

_SUCCESS = "PITCH3D_BLENDER_OK"
_SKY_RGB = (0.55, 0.72, 0.92, 1.0)        # daytime world background
_GROUND_RGB = (0.15, 0.18, 0.12, 1.0)     # neutral matte pitch plane (NOT measured grass — M2-9)


def _parse_args(argv):
    """Parse post-``--`` flags ``--plan`` / ``--mesh-dir`` / ``--render-dir`` (pure + testable)."""
    after = argv[argv.index("--") + 1:] if "--" in argv else []
    out = {"plan": None, "mesh_dir": None, "render_dir": None}
    flags = {"--plan": "plan", "--mesh-dir": "mesh_dir", "--render-dir": "render_dir"}
    i = 0
    while i < len(after):
        key = flags.get(after[i])
        if key is not None and i + 1 < len(after):
            out[key] = after[i + 1]
            i += 2
        else:
            i += 1
    return out


def _configure_cycles(bpy, scene, plan):  # pragma: no cover - needs bpy
    scene.render.engine = "CYCLES"
    scene.cycles.device = plan.get("device", "CPU")
    scene.cycles.samples = int(plan.get("samples", 48))
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = int(plan["width"])
    scene.render.resolution_y = int(plan["height"])
    scene.render.resolution_percentage = 100


def _build_world(bpy, scene):  # pragma: no cover - needs bpy
    world = bpy.data.worlds.new("sky")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = _SKY_RGB
        bg.inputs[1].default_value = 1.0
    scene.world = world


def _add_sun(bpy):  # pragma: no cover - needs bpy
    light = bpy.data.lights.new("sun", type="SUN")
    light.energy = 4.0
    light.angle = 0.1
    obj = bpy.data.objects.new("sun", light)
    obj.rotation_euler = (math.radians(50.0), 0.0, math.radians(30.0))
    bpy.context.collection.objects.link(obj)


def _matte_material(bpy, name, rgba, roughness):  # pragma: no cover - needs bpy
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = roughness
    return mat


def _add_ground(bpy, plan):  # pragma: no cover - needs bpy
    bpy.ops.mesh.primitive_plane_add(
        size=float(plan.get("ground_size", 140.0)),
        location=(0.0, 0.0, float(plan.get("ground_z", 0.0))),
    )
    ground = bpy.context.active_object
    ground.name = "ground"
    ground.data.materials.append(_matte_material(bpy, "ground_mat", _GROUND_RGB, 0.9))
    return ground


def _build_avatar(bpy, mesh_dir, spec):  # pragma: no cover - needs bpy
    import numpy as np

    data = np.load(os.path.join(mesh_dir, spec["npz"]))
    verts, faces, rgb = data["verts"], data["faces"], data["rgb"]
    me = bpy.data.meshes.new(spec["name"])
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update()
    for poly in me.polygons:
        poly.use_smooth = True
    # Per-vertex colour as a BYTE_COLOR attribute (sRGB) read by a Vertex Color shader node, so the
    # measured texture — and the R-6 unmeasured tint baked into ``rgb`` upstream — render faithful.
    attr = me.color_attributes.new(name="Col", type="BYTE_COLOR", domain="POINT")
    rgba = np.concatenate([rgb, np.ones((rgb.shape[0], 1), dtype=rgb.dtype)], axis=1)
    attr.data.foreach_set("color", rgba.reshape(-1).tolist())
    mat = bpy.data.materials.new(f"{spec['name']}_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    vcol = nt.nodes.new("ShaderNodeVertexColor")
    vcol.layer_name = "Col"
    nt.links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.7
    me.materials.append(mat)
    obj = bpy.data.objects.new(spec["name"], me)
    bpy.context.collection.objects.link(obj)
    return obj


def _make_camera(bpy, plan):  # pragma: no cover - needs bpy
    ci = plan["camera_intrinsics"]
    cam_data = bpy.data.cameras.new("cam")
    cam_data.sensor_fit = ci["sensor_fit"]
    cam_data.sensor_width = float(ci["sensor_width_mm"])
    cam_data.lens = float(ci["lens_mm"])
    cam_data.shift_x = float(ci["shift_x"])
    cam_data.shift_y = float(ci["shift_y"])
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.pixel_aspect_x = float(ci["pixel_aspect_x"])
    scene.render.pixel_aspect_y = float(ci["pixel_aspect_y"])
    return cam


def _render_frames(bpy, Matrix, plan, cam, avatars, render_dir):  # pragma: no cover - needs bpy
    os.makedirs(render_dir, exist_ok=True)
    scene = bpy.context.scene
    paths = []
    for fr in plan["frames"]:
        cam.matrix_world = Matrix(fr["camera_matrix_world"])
        for pl in fr["placements"]:
            obj = avatars.get(pl["name"])
            if obj is None:
                continue
            obj.hide_render = not pl["visible"]
            obj.matrix_world = Matrix(pl["matrix_world"])
        out = os.path.join(render_dir, f"frame_{int(fr['index']):05d}.png")
        scene.render.filepath = out
        bpy.ops.render.render(write_still=True)
        paths.append(out)
    return paths


def main():  # pragma: no cover - runs only as Blender's __main__
    import bpy
    from mathutils import Matrix

    args = _parse_args(sys.argv)
    with open(args["plan"], encoding="utf-8") as fh:
        plan = json.load(fh)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    _configure_cycles(bpy, scene, plan)
    _build_world(bpy, scene)
    _add_sun(bpy)
    _add_ground(bpy, plan)

    avatars = {spec["name"]: _build_avatar(bpy, args["mesh_dir"], spec) for spec in plan["meshes"]}
    cam = _make_camera(bpy, plan)
    _render_frames(bpy, Matrix, plan, cam, avatars, args["render_dir"])

    print(_SUCCESS)


if __name__ == "__main__":
    main()
