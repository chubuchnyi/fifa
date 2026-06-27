"""In-Blender Cycles render script — runs *inside* Blender's Python, never imported by us (M2-7).

:func:`~pitch3d.adapters.blender.runner.run_cycles_render` invokes this via
``blender --background --factory-startup --python _cycles_script.py -- --plan plan.json
--mesh-dir <dir> --render-dir <dir>``. It reads the JSON
:class:`~pitch3d.adapters.blender.cycles_plan.CyclesPlan`, builds the *resolved* scene as real
geometry — each avatar mesh from its NPZ (vertices + triangles + per-vertex colour), a grass-PBR
ground plane carrying the measured pitch markings, a sun + physical-sky world — places a physical
camera from the plan's lens/shift/sensor, and renders every plan frame with **Cycles** to
``frame_{index:05d}.png``. The camera ``matrix_world``
and each avatar's placement come straight from the plan (all the OpenCV→Blender maths happened on
the pure side), so this file only *applies* transforms.

Self-contained on purpose: only stdlib + ``bpy``/``mathutils`` + ``numpy`` (all Blender-bundled), so
it carries no dependency on the ``pitch3d`` package, and ``bpy`` is imported only when run as
Blender's ``__main__`` — importing this file in a normal interpreter never pulls in ``bpy``.

Avatars render either **rigid** (one canonical ``(V, 3)`` mesh placed by its root ``matrix_world``,
M2-7) or **posed** (per-frame LBS vertices ``(T, V, 3)`` in the NPZ; the script swaps row
``vert_index`` into the mesh each frame so the limbs follow the resolved pose, M2-8). The
environment is a procedural grass pitch carrying the measured line markings (the ``pitch_npz``
ribbons) under a physically-based (multiple-scattering) sky (M2-9).
"""

from __future__ import annotations

import json
import math
import os
import sys

_SUCCESS = "PITCH3D_BLENDER_OK"
_LINE_RGB = (0.9, 0.9, 0.9, 1.0)          # measured pitch markings (painted white on grass, M2-9)
_SUN_ELEVATION_DEG = 45.0                 # sky + key-light sun share one elevation/azimuth (M2-9)
_SUN_AZIMUTH_DEG = 30.0


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
    # Physically-based outdoor lighting: a multiple-scattering atmosphere drives the world
    # background, so the sky dome (soft fill) is measured-real, not a flat tint. The sun disc is
    # off here — the crisp key comes from a matched SUN lamp, which converges cleaner at 48 spp.
    world = bpy.data.worlds.new("sky")
    nt = world.node_tree  # worlds carry a node tree by default on Blender 5.x (use_nodes is gone)
    bg = nt.nodes.get("Background") or nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.get("World Output") or nt.nodes.new("ShaderNodeOutputWorld")
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"  # Blender 5.x successor to the Nishita physical sky
    sky.sun_disc = False
    sky.sun_elevation = math.radians(_SUN_ELEVATION_DEG)
    sky.sun_rotation = math.radians(_SUN_AZIMUTH_DEG)
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Strength"].default_value = 1.0
    scene.world = world


def _add_sun(bpy):  # pragma: no cover - needs bpy
    # Key light, aimed to match the Nishita sun: elevation tips the lamp up from the horizon (a SUN
    # points down its local -Z at euler 0), azimuth swings it round. angle≈the real solar disc, so
    # shadow edges are softly penumbral rather than razor-sharp.
    light = bpy.data.lights.new("sun", type="SUN")
    light.energy = 3.0
    light.angle = math.radians(0.53)
    obj = bpy.data.objects.new("sun", light)
    obj.rotation_euler = (
        math.radians(90.0 - _SUN_ELEVATION_DEG),
        0.0,
        math.radians(_SUN_AZIMUTH_DEG),
    )
    bpy.context.collection.objects.link(obj)


def _matte_material(bpy, name, rgba, roughness):  # pragma: no cover - needs bpy
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = roughness
    return mat


def _grass_material(bpy):  # pragma: no cover - needs bpy
    # Procedural grass for the measured pitch plane (M2-9): banded mowing stripes (the iconic pitch
    # look) drive the base colour, a fine noise drives a subtle bump for blade micro-texture, and a
    # high roughness keeps it matte. Built purely from generator nodes — no image texture to ship.
    mat = bpy.data.materials.new("grass")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    # Mowing stripes: a banded wave quantised to two greens by a constant-interpolation ramp.
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = 6.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    dark, light = ramp.color_ramp.elements
    dark.position, dark.color = 0.0, (0.045, 0.20, 0.035, 1.0)
    light.position, light.color = 0.5, (0.075, 0.28, 0.055, 1.0)
    nt.links.new(coord.outputs["Object"], wave.inputs["Vector"])
    nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    # Grass micro-texture as a bump only (no extra colour mix → fewer sockets to get wrong).
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 120.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.15
    nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.9
    return mat


def _add_ground(bpy, plan):  # pragma: no cover - needs bpy
    bpy.ops.mesh.primitive_plane_add(
        size=float(plan.get("ground_size", 140.0)),
        location=(0.0, 0.0, float(plan.get("ground_z", 0.0))),
    )
    ground = bpy.context.active_object
    ground.name = "ground"
    ground.data.materials.append(_grass_material(bpy))
    return ground


def _build_pitch(bpy, mesh_dir, plan):  # pragma: no cover - needs bpy
    """Build the measured line-marking ribbons (``pitch_npz``) as a flat matte-white mesh, or skip.

    The ribbon geometry is the *measured* pitch template the calibration anchors to (M2-9), already
    given thickness on the pure side (:func:`~pitch3d.core.scene.pitch.pitch_line_ribbons`); here it
    just becomes a white-painted mesh sitting a hair above the grass. ``None`` when the plan carries
    no markings (``draw_pitch=False``) — an honest empty pitch, never a fabricated one.
    """
    import numpy as np

    name = plan.get("pitch_npz")
    if not name:
        return None
    data = np.load(os.path.join(mesh_dir, name))
    verts, faces = data["verts"], data["faces"]
    me = bpy.data.meshes.new("pitch_lines")
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update()
    me.materials.append(_matte_material(bpy, "pitch_lines_mat", _LINE_RGB, 0.6))
    obj = bpy.data.objects.new("pitch_lines", me)
    bpy.context.collection.objects.link(obj)
    return obj


def _build_avatar(bpy, mesh_dir, spec):  # pragma: no cover - needs bpy
    import numpy as np

    data = np.load(os.path.join(mesh_dir, spec["npz"]))
    verts, faces, rgb = data["verts"], data["faces"], data["rgb"]
    # Posed avatars carry per-frame LBS vertices (T, V, 3); build topology from frame 0 and keep the
    # stack so _render_frames can swap in each frame's row. Rigid avatars are a single (V, 3) mesh.
    posed = verts if verts.ndim == 3 else None
    base = verts[0] if posed is not None else verts
    me = bpy.data.meshes.new(spec["name"])
    me.from_pydata(base.tolist(), [], faces.tolist())
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
    return obj, posed


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
            entry = avatars.get(pl["name"])
            if entry is None:
                continue
            obj, posed = entry
            obj.hide_render = not pl["visible"]
            obj.matrix_world = Matrix(pl["matrix_world"])
            idx = int(pl.get("vert_index", -1))
            if posed is not None and idx >= 0:
                # Swap this frame's LBS vertices into the shared mesh (identity placement, so the
                # baked world coords land 1:1). update() marks it dirty for the depsgraph re-eval.
                obj.data.vertices.foreach_set("co", posed[idx].reshape(-1))
                obj.data.update()
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
    _build_pitch(bpy, args["mesh_dir"], plan)

    avatars = {spec["name"]: _build_avatar(bpy, args["mesh_dir"], spec) for spec in plan["meshes"]}
    cam = _make_camera(bpy, plan)
    _render_frames(bpy, Matrix, plan, cam, avatars, args["render_dir"])

    print(_SUCCESS)


if __name__ == "__main__":
    main()
