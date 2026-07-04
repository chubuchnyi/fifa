"""Render a multi-camera ANIMATION of the reconstructed bodies + ball with Blender Cycles.

Input is an anim export directory (pitch3d.app.anim_export), VALIDATED first against its
`manifest.json` — the versioned contract in `anim_contract.py` — so a stale/partial/drifted
export fails loudly here instead of by eye in a finished render. For every frame we re-pose
each body (`foreach_set("co", ...)`) and move the ball, then render the SAME instant from the
VIRTUAL OPERATOR's cameras (`cameras.npz`: fixed mounts inside the stadium bowl whose look-at
and fov pan/zoom with the action per frame — broadcast / sideline / top / goal). Each camera
yields a PNG sequence that scripts/pod_make_video.sh stitches into one mp4 per angle. Exports
predating the operator fall back to the old bbox-derived static cameras.

Runs either as a Blender-binary script (`blender --background --python ... -- --in DIR`) or as the
`bpy` pip module (`python scripts/blender_animate.py --in DIR`) — argv parsing handles both. Cycles
on CPU is the reliable headless default; `--device gpu` tries OPTIX/CUDA and falls back to CPU.

Flags (after `--` when run via the binary):
  --in DIR        dir of anim_subject_*.npz (+ ball.npz)         [default out/anim/mesh]
  --out DIR       root for <camera>/frame_*.png sequences        [default <in>/frames]
  --device cpu|gpu                                               [default cpu]
  --res-x N --res-y N --samples N --fps N --frame-step N --cameras a,b,c
  --light-rgb r,g,b --light-energy N --sky-strength N           manual lighting overrides; each
  --sun-count N --sun-elevation N --sun-angle N                 wins over the measured lighting.npz
"""

import glob
import math
import os
import sys

import numpy as np

import bpy
import mathutils

# Shared pitch3d-free Blender modules (scene node-graphs + the export contract) — imported by
# file so this script stays self-contained (--factory-startup, no pitch3d install needed).
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "pitch3d", "adapters", "blender",
))
import anim_contract  # noqa: E402
import scene_builders  # noqa: E402

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
# Team-mask AOV pass: same cameras/frames/bodies, but every body renders a flat UNLIT team code
# (A=red, B=green, untracked=blue) on black — no lights, no stadium, no plates, 1 sample. The
# generative finisher (v2v) is structure-locked, so these render-space masks say WHERE each kit
# is in ITS output too; scripts/hue_pin.py uses them to undo the v2v/upscaler kit-hue drift.
TEAM_MASK = _arg("--team-mask", "0") == "1"

BALL_RADIUS = 0.11  # FIFA size-5 ball ≈ 0.11 m radius

# The export↔render contract (both sides versioned): refuse to build anything from a directory
# that is not a complete, current anim export. ContractError names exactly what is missing or
# stale — the drift this catches used to surface only by eye in a finished render.
anim_contract.load_manifest(IN)


def _look_at(cam, target):
    cam.rotation_euler = (
        mathutils.Vector(target) - mathutils.Vector(cam.location)
    ).to_track_quat("-Z", "Y").to_euler()


def _make_number_plate(num, rgb):
    """A centred FONT object carrying the shirt number; slightly emissive so it stays legible in
    shadow. Placed/oriented per frame from the body's baked back-anchor (#numbers, v1)."""
    tc = bpy.data.curves.new(f"num{num}", type="FONT")
    tc.body = str(int(num))
    tc.align_x = "CENTER"
    tc.align_y = "CENTER"
    tc.size = 0.30        # ~0.3 m cap height — a real shirt number
    tc.extrude = 0.004    # tiny depth so it is not a zero-thickness sliver
    ob = bpy.data.objects.new(f"plate{num}", tc)
    bpy.context.collection.objects.link(ob)
    mat = bpy.data.materials.new(f"num{num}")
    mat.use_nodes = True
    b = mat.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
    b.inputs["Roughness"].default_value = 0.5
    if "Emission Color" in b.inputs:  # Blender 4+/5 naming
        b.inputs["Emission Color"].default_value = (
            float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0
        )
        b.inputs["Emission Strength"].default_value = 0.55
    tc.materials.append(mat)
    return ob


def _orient_plate(ob, anchor, back):
    """Face the plate's front (+Z) along the body's outward back normal, text-up = world up. Hidden
    when the back direction is degenerate (facing straight up/down ⇒ no horizontal posterior)."""
    n = mathutils.Vector((float(back[0]), float(back[1]), float(back[2])))
    if n.length < 1e-4:
        ob.hide_render = True
        return
    n.normalize()
    up = mathutils.Vector((0.0, 0.0, 1.0))
    # reading direction (left→right as seen by a viewer behind the player). MUST be up×n, not n×up:
    # with up×n the text-up axis y=n×x resolves to +world-Z (upright, unmirrored); the reversed
    # cross rolls the plate 180° → digits render upside-down AND mirrored.
    x = up.cross(n)
    if x.length < 1e-4:
        x = mathutils.Vector((1.0, 0.0, 0.0))
    x.normalize()
    y = n.cross(x)  # text-up, orthonormal; = +world-Z when n is horizontal
    rot = mathutils.Matrix((x, y, n)).transposed().to_4x4()  # columns = local X,Y,Z in world
    ob.matrix_world = mathutils.Matrix.Translation(
        mathutils.Vector((float(anchor[0]), float(anchor[1]), float(anchor[2])))
    ) @ rot
    ob.hide_render = False


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
    # Measured per-vertex body texture (lever 1): real broadcast pixels projected onto the posed
    # SMPL-X mesh by anim_export.py. Absent in older exports / no source clip → flat kit colour.
    vcolor = d["vcolor"] if "vcolor" in d.files else None
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
    if TEAM_MASK:
        team = str(d["team"]) if "team" in d.files else ""
        code = {"A": (1.0, 0.0, 0.0, 1.0), "B": (0.0, 1.0, 0.0, 1.0)}.get(team, (0.0, 0.0, 1.0, 1.0))
        bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bsdf.inputs["Emission Color"].default_value = code
        bsdf.inputs["Emission Strength"].default_value = 1.0
    elif vcolor is not None:
        # BYTE_COLOR (sRGB) → Base Color, lit by the scene sun. Unmeasured verts were filled
        # with the flat kit colour upstream (R-6), so the body is opaque-coloured, never black.
        rgb = np.asarray(vcolor, dtype=np.float32).reshape(-1, 3)
        rgba = np.concatenate([rgb, np.ones((rgb.shape[0], 1), dtype=np.float32)], axis=1)
        attr = me.color_attributes.new(name="Col", type="BYTE_COLOR", domain="POINT")
        attr.data.foreach_set("color", rgba.reshape(-1).tolist())
        vcol = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        vcol.layer_name = "Col"
        mat.node_tree.links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (float(col[0]), float(col[1]), float(col[2]), 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    me.materials.append(mat)
    frame_row = {int(f): i for i, f in enumerate(frames)}
    # Shirt-number plate baked by anim_export.py (#numbers): a back-anchored FONT object oriented
    # per frame. Absent for subjects with no read number (older exports never carry these keys).
    plate = None
    if "jersey_number" in d.files and not TEAM_MASK:
        plate = {
            "ob": _make_number_plate(int(d["jersey_number"]), d["number_rgb"]),
            "anchor": np.asarray(d["back_anchor"], dtype=np.float32),
            "back": np.asarray(d["back_dir"], dtype=np.float32),
        }
    bodies.append((ob, me, verts, frame_row, bsdf, alpha, plate))
    all_frames.update(frame_row)
    lo = np.minimum(lo, verts.reshape(-1, 3).min(0))
    hi = np.maximum(hi, verts.reshape(-1, 3).max(0))

# Bodies-only bbox, captured BEFORE the pitch expands lo/hi below — drives an "action" camera that
# frames just the players (not the whole 105×68 m field), so shirt numbers are big enough to read.
body_lo, body_hi = lo.copy(), hi.copy()

# Measured pitch markings + goal frames (anim_export.py's pitch.npz, #205). Loaded BEFORE the
# camera framing so lo/hi — hence ctr/span — span the whole pitch, not just the bodies: that keeps
# the full field (and its lines/goals) in frame even while world placement is still being fixed.
pitch_path = os.path.join(IN, "pitch.npz")
pitch_npz = np.load(pitch_path) if os.path.exists(pitch_path) else None
if pitch_npz is not None:
    for _key in ("pitch_verts", "goal_verts"):
        _pv = np.asarray(pitch_npz[_key], dtype=float).reshape(-1, 3)
        if _pv.size:
            lo = np.minimum(lo, _pv.min(0))
            hi = np.maximum(hi, _pv.max(0))

ctr = (lo + hi) / 2.0
span = float(max((hi - lo)[0], (hi - lo)[1], 1.0))  # horizontal extent of pitch + all motion

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
# Grass PBR (v2 lever 2): the procedural mowing-stripe + bump material shared with the formal Cycles
# path via scene_builders, so the deliverable pitch is no longer a flat green plane.
bpy.ops.mesh.primitive_plane_add(size=max(120.0, span * 3), location=(ctr[0], ctr[1], 0.0))


def _env_rgb(name, default):
    raw = os.environ.get(name, "")
    return tuple(float(v) for v in raw.split(",")) if raw else default


bpy.context.active_object.data.materials.append(
    scene_builders.build_grass_material(
        bpy,
        dark_rgb=_env_rgb("PITCH3D_GRASS_DARK", scene_builders.GRASS_DARK_RGB),
        light_rgb=_env_rgb("PITCH3D_GRASS_LIGHT", scene_builders.GRASS_LIGHT_RGB),
    )
)


def _add_static_mesh(name, verts, faces, rgb, roughness):
    """Build a flat-shaded static mesh (pitch lines / goals) from world verts+faces."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(
        np.asarray(verts, dtype=float).tolist(), [], np.asarray(faces, dtype=int).tolist()
    )
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    b = mat.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    b.inputs["Roughness"].default_value = roughness
    me.materials.append(mat)
    return ob


def _add_stadium_mesh(name, verts, faces, tint, uv=None, tile=None, *, tile_ext="MIRROR",
                      emission_strength=1.0):
    """Build the stadium bowl as a *tinted mosaic*: a crowd image (``tile``) tiled over the bowl by
    per-loop ``uv`` and modulated by the per-vertex measured colour (``tint``). Their product drives
    *emission* (base colour black) so the crowd renders at its clip brightness, not lit by the
    novel-view sun — a far backdrop shouldn't pick up our one key light's direction.

    The tile is normalised to unit mean, turning it into a pure detail map, so ``tint`` sets each
    stand's real colour (its regional yellow/red) while the tile only carries crowd texture; that
    keeps the multiply from double-darkening or distorting hue. ``tile_ext`` is the sampler edge
    mode the exporter chose: MIRROR for the legacy small-tile repeat (reflection hides its seams),
    REPEAT for the non-repeating 0-1 quilt (the wrap-seam faces sample u slightly past 1). Falls
    back to flat vertex colour if no tile/uv is given (older exports) or the Blender build predates
    the emission input.
    """
    me = bpy.data.meshes.new(name)
    me.from_pydata(
        np.asarray(verts, dtype=float).tolist(), [], np.asarray(faces, dtype=int).tolist()
    )
    me.update()
    for poly in me.polygons:
        poly.use_smooth = True
    tcol = np.asarray(tint, dtype=np.float32).reshape(-1, 3)
    rgba = np.concatenate([tcol, np.ones((tcol.shape[0], 1), np.float32)], axis=1)
    attr = me.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
    attr.data.foreach_set("color", rgba.ravel())

    have_tex = uv is not None and tile is not None
    if have_tex:
        uvw = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
        me.uv_layers.new(name="UVMap").data.foreach_set("uv", uvw.ravel())

    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 1.0
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "Col"
    emissive = "Emission Color" in bsdf.inputs
    if emissive:
        bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission_strength

    if have_tex and emissive:
        til = np.asarray(tile)
        til = til.astype(np.float32) / 255.0 if til.dtype == np.uint8 else til.astype(np.float32)
        mean = np.clip(til.reshape(-1, 3).mean(axis=0), 1e-3, None)
        norm = (til / mean)[::-1]  # unit-mean detail map; flip to Blender's bottom-left origin
        # Clip crowds read as a WARM mass varying in luma, not hue confetti: pull the tile's
        # per-pixel colour toward its luma so the measured tint owns the hue (0 = pure luma).
        chroma = float(os.environ.get("PITCH3D_CROWD_CHROMA", "0.15"))
        if chroma < 1.0:
            luma = norm @ np.float32([0.2126, 0.7152, 0.0722])
            norm = luma[..., None] + (norm - luma[..., None]) * chroma
        hh, ww = norm.shape[:2]
        img = bpy.data.images.new(name + "_tile", width=ww, height=hh, float_buffer=True)
        img.colorspace_settings.name = "Non-Color"  # raw values, matching the linear vertex tint
        alpha = np.ones((hh, ww, 1), np.float32)
        img.pixels.foreach_set(np.concatenate([norm, alpha], axis=2).ravel())
        img.pack()
        uvmap = nt.nodes.new("ShaderNodeUVMap")
        uvmap.uv_map = "UVMap"
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.extension = tile_ext
        nt.links.new(uvmap.outputs["UV"], tex.inputs["Vector"])
        mul = nt.nodes.new("ShaderNodeVectorMath")
        mul.operation = "MULTIPLY"
        nt.links.new(tex.outputs["Color"], mul.inputs[0])
        nt.links.new(vc.outputs["Color"], mul.inputs[1])
        nt.links.new(mul.outputs["Vector"], bsdf.inputs["Emission Color"])
    elif emissive:
        nt.links.new(vc.outputs["Color"], bsdf.inputs["Emission Color"])
    else:
        nt.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    me.materials.append(mat)
    return ob


# Measured pitch lines (white) + goal frames (off-white) — the geometric reference that makes
# placement judgeable by eye (#205). Nothing drawn if anim_export wrote no pitch.npz (older runs).
if pitch_npz is not None:
    if np.asarray(pitch_npz["pitch_faces"]).size:
        _add_static_mesh("pitch_lines", pitch_npz["pitch_verts"], pitch_npz["pitch_faces"],
                         (0.90, 0.90, 0.90), 0.5)
    if np.asarray(pitch_npz["goal_faces"]).size:
        _add_static_mesh("goals", pitch_npz["goal_verts"], pitch_npz["goal_faces"],
                         (0.95, 0.95, 0.95), 0.35)

# Hybrid stadium bowl (M2 stadium): a procedural seating bowl wearing a tinted crowd mosaic measured
# from the clip (anim_export.py's stadium.npz). Deliberately NOT folded into the lo/hi framing:
# it rings the pitch, so driving ctr/span off it would zoom every camera out until players are dots.
stadium_path = os.path.join(IN, "stadium.npz")
if os.path.exists(stadium_path) and not TEAM_MASK:  # emissive crowd would pollute the mask
    sd = np.load(stadium_path)
    # Crowd emission boost: the tint is measured from the RAW clip (already a dark night
    # broadcast), then grade3 darkens the render AGAIN — the same double-grade trap as the
    # grass albedo. Boost so the post-grade crowd V matches the clip's. The tint saturation
    # is pre-compensated too: AgX desaturates bright emission toward gray, and grade3 then
    # paints dark gray with its blue shadow cast — the measured warmth needs headroom.
    # Defaults measured 2026-07-04 (5 render→grade iterations, pure-crowd ROI): E 3.6 +
    # chroma 0.15 + tint_sat 1.35 → post-grade V .184 H 69 S .58 vs clip V .188 H 48 S .42
    # (V exact, hue = the tint's own yellow — reads right by eye, the fans ARE in yellow).
    _crowd_tint = np.asarray(sd["colors"], dtype=np.float32).reshape(-1, 3)
    _tsat = float(os.environ.get("PITCH3D_CROWD_TINT_SAT", "1.35"))
    if _tsat != 1.0:
        _lum = _crowd_tint @ np.float32([0.2126, 0.7152, 0.0722])
        _crowd_tint = np.clip(_lum[:, None] + (_crowd_tint - _lum[:, None]) * _tsat, 0.0, None)
    # tile_gain: the exporter's structured quilt (walkway/aisles/top-fade) lowered the tile
    # mean; the unit-mean norm below is scale-invariant, so without this the seated rows would
    # render brighter than the tuned emission (measured 2026-07-04: x1.41 -> oversaturated
    # amber panels after the night grade).
    _tgain = float(sd["tile_gain"]) if "tile_gain" in sd.files else 1.0
    _add_stadium_mesh(
        "stadium", sd["verts"], sd["faces"], _crowd_tint,
        uv=sd["uv"] if "uv" in sd.files else None,
        tile=sd["tile"] if "tile" in sd.files else None,
        tile_ext=str(sd["tile_ext"]) if "tile_ext" in sd.files else "MIRROR",
        emission_strength=float(os.environ.get("PITCH3D_CROWD_EMISSION", "3.6")) * _tgain,
    )

# LED ad-board ring + dark walkway band (anim_export's boards.npz, 2026-07-03): flat
# vertex-colour emission. Boards saturate the PNG (strength >1) so the night grade still
# leaves them the brightest element — matching the clip, where the LED strip glows.
boards_path = os.path.join(IN, "boards.npz")
if os.path.exists(boards_path) and not TEAM_MASK:
    bdz = np.load(boards_path)
    _add_stadium_mesh(
        "adboards", bdz["verts"], bdz["faces"], bdz["colors"],
        emission_strength=float(os.environ.get("PITCH3D_BOARD_EMISSION", "4.0")),
    )

# Floodlit-NIGHT lighting (v2 lever 3) with BOTH auto-detect and manual override. AUTO baseline:
# anim_export.py measured the floodlight colour (+ night-model defaults) into lighting.npz. MANUAL:
# any --light-* flag wins over that. Fallback when neither exists: the measured night defaults baked
# into scene_builders. The model itself — a dark world + a ring of soft suns — is the floodlit-night
# finding (no daytime sun); the formal Cycles path keeps its own separate daytime sky.
light_kwargs: dict = {}
_lp = os.path.join(IN, "lighting.npz")
if os.path.exists(_lp):
    _ld = np.load(_lp)
    light_kwargs["light_rgb"] = tuple(float(x) for x in _ld["light_rgb"])
    for _k in ("key_energy", "sky_strength", "elevation_deg", "sun_angle_deg"):
        if _k in _ld.files:
            light_kwargs[_k] = float(_ld[_k])
    if "sun_count" in _ld.files:
        light_kwargs["sun_count"] = int(_ld["sun_count"])
_rgb_arg = _arg("--light-rgb", "")
if _rgb_arg:
    light_kwargs["light_rgb"] = tuple(float(x) for x in _rgb_arg.split(","))
for _flag, _key, _cast in (
    ("--light-energy", "key_energy", float),
    ("--sky-strength", "sky_strength", float),
    ("--sun-count", "sun_count", int),
    ("--sun-elevation", "elevation_deg", float),
    ("--sun-angle", "sun_angle_deg", float),
):
    _v = _arg(_flag, "")
    if _v != "":
        light_kwargs[_key] = _cast(_v)
if TEAM_MASK:
    # No lights + a zero-strength world: only the emissive team codes reach the sensor; every
    # diffuse surface (grass, lines, goals, ball) stays black yet still occludes bodies exactly
    # as in the beauty pass.
    _mask_world = bpy.data.worlds.new("mask_world")
    _mask_world.use_nodes = True
    _mask_world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    bpy.context.scene.world = _mask_world
else:
    scene_builders.build_stadium_lighting(bpy, **light_kwargs)

# ── cameras: the virtual operator (cameras.npz), static bbox specs as legacy fallback ─────
# anim_export plans FIXED mounts inside the stadium bowl with a per-frame look-at + horizontal
# fov that pan/zoom with the action (core/scene/cameras.py). The old static cameras below were
# derived from the bbox of everything loaded — with the 105x68 m pitch folded in, every camera
# framed the whole bowl from outside and the players became specks. They remain only for
# exports that predate the operator.
cam_path = os.path.join(IN, "cameras.npz")
virtual_cams = {}
cam_row = {}
if os.path.exists(cam_path):
    cd = np.load(cam_path)
    cam_row = {int(f): i for i, f in enumerate(cd["frames"])}
    for _name in (str(n) for n in cd["names"]):
        virtual_cams[_name] = {
            "pos": np.asarray(cd[f"{_name}_pos"], dtype=float),
            "look": np.asarray(cd[f"{_name}_look"], dtype=float),
            "fov": np.asarray(cd[f"{_name}_fov_deg"], dtype=float),
        }

look = (ctr[0], ctr[1], ctr[2] + 0.6)
# Players-only centre/extent for the tight "action" framing (numbers must be legible).
bctr = (body_lo + body_hi) / 2.0
bspan = float(max((body_hi - body_lo)[0], (body_hi - body_lo)[1], 6.0))
look_action = (bctr[0], bctr[1], bctr[2] + 0.6)
cam_specs = {
    "broadcast": (ctr[0] + span * 0.35, ctr[1] - span * 1.5 - 8.0, ctr[2] + span * 0.55 + 6.0),
    "sideline":  (ctr[0], ctr[1] - span * 1.2 - 6.0, ctr[2] + 2.0),
    "top":       (ctr[0], ctr[1], ctr[2] + span * 1.8 + 20.0),
    "goal":      (ctr[0] + span * 1.4 + 10.0, ctr[1], ctr[2] + span * 0.4 + 4.0),
    "action":    (bctr[0] + bspan * 0.45, bctr[1] - bspan * 1.3 - 6.0, bctr[2] + bspan * 0.8 + 5.0),
}
cameras = []
for name in WANT_CAMS:
    vc = virtual_cams.get(name)
    if vc is None and name not in cam_specs:
        continue
    cam_data = bpy.data.cameras.new(name)
    cam_data.clip_end = 2000.0  # bowl+pitch reach ~150 m from a mount; the 100 m default clips them
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam)
    if vc is not None:
        cam_data.sensor_fit = "HORIZONTAL"  # planned fov_x_deg IS the horizontal angle
        cam.location = tuple(float(x) for x in vc["pos"])
        _look_at(cam, vc["look"][0])
        cam_data.angle = math.radians(float(vc["fov"][0]))
    else:
        cam.location = cam_specs[name]
        if name == "action":
            target = look_action
        elif name == "top":
            target = (look[0], look[1], 0.6)
        else:
            target = look
        _look_at(cam, target)
    cameras.append((name, cam))
assert cameras, f"no known cameras among {WANT_CAMS} (try broadcast,sideline,top,goal)"
print(f"BLENDER_ANIM_CAMS {'virtual-operator' if virtual_cams else 'static-legacy'} "
      f"{[c for c, _ in cameras]}")

# ── render settings ──────────────────────────────────────────────────────────
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = 1 if TEAM_MASK else SAMPLES  # flat emission needs one sample, no denoise
sc.cycles.use_denoising = not TEAM_MASK
# Keep the built scene resident on the device between renders. Without this, every
# bpy.ops.render.render() call below tears down and re-uploads the whole Cycles scene (BVH for the
# ~20 deforming bodies, geometry, denoiser), so the GPU idled (~0–25%) waiting on CPU re-sync for
# each camera; persistent data reuses it across the 4 cameras and rebuilds only changed geometry.
sc.render.use_persistent_data = True
# Broadcast-faithful tone-map: the target clip is standard Rec.709, so render through the "Standard"
# view transform rather than Blender's default AgX, which desaturated the floodlit grass to grey
# and lifted the night sky. Keeps the measured colours (kit, grass, floodlight tint) reading as the
# clip does (v2 lever 3).
sc.view_settings.view_transform = "Standard"
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
                # GPU-only (exclude CPU): enabling CPU too makes Cycles render in hybrid mode, pegging
                # all CPU cores path-tracing alongside the GPU — the high-CPU / low-GPU-util symptom.
                dev.use = getattr(dev, "type", "") == chosen
            sc.cycles.device = "GPU"
            if chosen == "OPTIX":
                sc.cycles.denoiser = "OPTIX"  # denoise on the GPU, not the CPU OpenImageDenoiser
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
    for ob, me, verts, frame_row, bsdf, alpha, plate in bodies:
        row = frame_row.get(gf)
        if row is None:
            ob.hide_render = True
            if plate is not None:
                plate["ob"].hide_render = True
            continue
        ob.hide_render = False
        visible += 1
        # Ramp opacity at genuine entries/exits (Cycles honours the Principled BSDF Alpha input
        # directly); a body present across the whole clip stays at alpha 1.0 → opaque as before.
        bsdf.inputs["Alpha"].default_value = float(alpha[row])
        me.vertices.foreach_set("co", np.ascontiguousarray(verts[row], dtype=np.float32).ravel())
        me.update()
        if plate is not None:
            _orient_plate(plate["ob"], plate["anchor"][row], plate["back"][row])
    if ball_ob is not None:
        brow = ball_row.get(gf)
        ball_ob.hide_render = brow is None
        if brow is not None:
            ball_ob.location = tuple(float(x) for x in ball_pos[brow])
    # Persistent data reuses device geometry between renders, so force a depsgraph re-eval here to
    # push this frame's re-posed meshes to Cycles (else it could reuse the previous frame's pose).
    bpy.context.view_layer.update()
    crow = cam_row.get(gf)
    for name, cam in cameras:
        vc = virtual_cams.get(name)
        if vc is not None and crow is not None:
            # The operator pans (aim at the smoothed action) and zooms (fov fits the action).
            _look_at(cam, vc["look"][crow])
            cam.data.angle = math.radians(float(vc["fov"][crow]))
        sc.camera = cam
        sc.render.filepath = os.path.join(OUT, name, f"frame_{gf:04d}.png")
        bpy.ops.render.render(write_still=True)
    rendered += 1
    print(f"BLENDER_ANIM_FRAME {rendered}/{len(gframes)} (global={gf}, {visible} bodies, {len(cameras)} cams)")

print(
    f"BLENDER_ANIM_OK frames={rendered} cams={[c for c, _ in cameras]} "
    f"res={RES_X}x{RES_Y} samples={SAMPLES} fps={FPS} -> {OUT}"
)
