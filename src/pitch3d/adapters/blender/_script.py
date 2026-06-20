"""In-Blender build/render script — runs *inside* Blender's Python, never imported by us.

:mod:`pitch3d.adapters.blender.runner` invokes this via
``blender --background --factory-startup --python _script.py -- --plan plan.json [--out-blend …]
[--render-dir …]``. It reads the JSON :class:`~pitch3d.adapters.blender.proxy.ProxyPlan` and
builds the editable proxy: one root **Empty** controller per subject carrying location +
axis-angle **F-curves** (the "ball/root as F-curves" editing surface, M1 step 10), β as a custom
property and the per-joint body pose as keyframed channels, each with a team-coloured marker mesh
child so the proxy is visible; the ball is an animated sphere. With ``--out-blend`` it saves a
``.blend`` a human opens to edit; with ``--render-dir`` it renders each plan view (Workbench, CPU)
to a PNG — the proxy ``SCENE_3D`` feedback (A-7).

It is intentionally self-contained (only stdlib + ``bpy``/``mathutils``, both Blender-bundled) so
it has no dependency on the ``pitch3d`` package, and ``bpy`` is imported only when run as Blender's
``__main__`` — importing this file in a normal interpreter never pulls in ``bpy``.
"""

from __future__ import annotations

import json
import math
import os
import sys


def _parse_args(argv):  # pragma: no cover - exercised only inside Blender
    after = argv[argv.index("--") + 1:] if "--" in argv else []
    out = {"plan": None, "out_blend": None, "render_dir": None}
    flags = {"--plan": "plan", "--out-blend": "out_blend", "--render-dir": "render_dir"}
    i = 0
    while i < len(after):
        key = flags.get(after[i])
        if key is not None and i + 1 < len(after):
            out[key] = after[i + 1]
            i += 2
        else:
            i += 1
    return out


def _clear_scene(bpy):  # pragma: no cover - needs bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _frame_bounds(plan):  # pragma: no cover - needs bpy
    frames = [int(f) for obj in plan["objects"] for f in obj["frames"]]
    return (min(frames), max(frames)) if frames else (0, 0)


def _marker_mesh(bpy, kind, name, rgb):  # pragma: no cover - needs bpy
    if kind == "ball":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.11, segments=12, ring_count=8)
        obj = bpy.context.active_object
    else:  # a standing 0.4 x 0.4 x 1.8 m player marker resting on the root (feet at z=0)
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        obj.scale = (0.2, 0.2, 0.9)
        obj.location = (0.0, 0.0, 0.9)
    obj.name = f"{name}_marker"
    color = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
    obj.color = color  # Workbench OBJECT colour
    mat = bpy.data.materials.new(f"{name}_mat")
    mat.diffuse_color = color
    obj.data.materials.append(mat)
    return obj


def _build_object(bpy, Vector, obj_spec):  # pragma: no cover - needs bpy
    name, frames, loc = obj_spec["name"], obj_spec["frames"], obj_spec["location"]
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.5
    bpy.context.collection.objects.link(empty)

    rot = obj_spec.get("rotation_aa")
    if rot is not None:
        empty.rotation_mode = "AXIS_ANGLE"
    for i, f in enumerate(frames):
        empty.location = Vector(loc[i])
        empty.keyframe_insert("location", frame=int(f))
        if rot is not None:
            ang = math.sqrt(sum(c * c for c in rot[i]))
            axis = [c / ang for c in rot[i]] if ang > 1e-12 else [0.0, 0.0, 1.0]
            empty.rotation_axis_angle = (ang, axis[0], axis[1], axis[2])
            empty.keyframe_insert("rotation_axis_angle", frame=int(f))

    if obj_spec.get("betas") is not None:
        empty["betas"] = list(obj_spec["betas"])
    body_pose = obj_spec.get("body_pose")
    if body_pose is not None:
        try:
            for i, f in enumerate(frames):
                empty["body_pose"] = [c for joint in body_pose[i] for c in joint]
                empty.keyframe_insert('["body_pose"]', frame=int(f))
        except (RuntimeError, TypeError):
            empty["body_pose"] = [c for joint in body_pose[0] for c in joint]  # static fallback

    marker = _marker_mesh(bpy, obj_spec["kind"], name, obj_spec["color_rgb"])
    marker.parent = empty
    marker.matrix_parent_inverse.identity()  # keep the marker's clean local offset
    return empty


def _render_views(bpy, Vector, plan, render_dir):  # pragma: no cover - needs bpy
    os.makedirs(render_dir, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.color_type = "OBJECT"
    scene.render.image_settings.file_format = "PNG"
    paths = []
    for i, v in enumerate(plan["views"]):
        cam_data = bpy.data.cameras.new(f"cam_{i}")
        cam_data.lens = float(v["lens_mm"])
        cam_data.sensor_width = 36.0
        cam = bpy.data.objects.new(f"cam_{i}", cam_data)
        bpy.context.collection.objects.link(cam)
        eye, target = Vector(v["eye"]), Vector(v["target"])
        cam.location = eye
        cam.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()
        scene.camera = cam
        scene.render.resolution_x = int(v["resolution"][0])
        scene.render.resolution_y = int(v["resolution"][1])
        scene.render.resolution_percentage = 100
        scene.frame_set(int(v.get("frame", 0)))
        out = os.path.join(render_dir, f"{i:02d}_{v['viewpoint']}.png")
        scene.render.filepath = out
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam)
        paths.append(out)
    return paths


def main():  # pragma: no cover - runs only as Blender's __main__
    import bpy
    from mathutils import Vector

    args = _parse_args(sys.argv)
    with open(args["plan"], encoding="utf-8") as fh:
        plan = json.load(fh)

    _clear_scene(bpy)
    scene = bpy.context.scene
    scene.render.fps = max(1, int(round(plan.get("fps", 25.0))))
    start, end = _frame_bounds(plan)
    scene.frame_start, scene.frame_end = start, end

    for obj_spec in plan["objects"]:
        _build_object(bpy, Vector, obj_spec)

    if args["out_blend"]:
        os.makedirs(os.path.dirname(args["out_blend"]) or ".", exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=args["out_blend"])

    if args["render_dir"] and plan.get("views"):
        _render_views(bpy, Vector, plan, args["render_dir"])

    print("PITCH3D_BLENDER_OK")


if __name__ == "__main__":
    main()
