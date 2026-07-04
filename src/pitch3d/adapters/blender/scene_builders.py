"""Shared, pitch3d-free Blender node-graph builders — the "B" data/contract layer.

Both render scripts build the *same* procedural environment but stay deliberately self-contained
(only ``bpy`` + stdlib, run under ``--factory-startup``): the formal Cycles path
(:mod:`pitch3d.adapters.blender._cycles_script`) and the deliverable video path
(``scripts/blender_animate.py``). Rather than import ``pitch3d`` (which neither may), each imports
THIS module by file via a tiny ``sys.path`` shim, so the procedural pieces they share — grass PBR
now, sky + sun next — have ONE definition and the two render paths cannot silently drift.

Self-contained on purpose: no ``pitch3d`` import, and ``bpy`` is only ever received as an argument
(never imported at module load), so importing this file in a plain interpreter is safe and cheap.
"""

from __future__ import annotations

import math


GRASS_DARK_RGB = (0.311, 0.446, 0.109)
GRASS_LIGHT_RGB = (0.327, 0.469, 0.114)


def build_grass_material(
    bpy, name="grass", stripe_scale=0.1, dark_rgb=GRASS_DARK_RGB, light_rgb=GRASS_LIGHT_RGB
):  # pragma: no cover - needs bpy
    """Procedural grass for the pitch plane: banded mowing stripes (the iconic look) drive the base
    colour, a fine noise drives a subtle bump for blade micro-texture, and a high roughness keeps it
    matte. Built purely from generator nodes — no image texture to ship.

    ``stripe_scale`` is the wave Scale on the plane's Object coords (metres): ~0.1 gives ≈5 m mowing
    bands that read at broadcast distance — the original M2-9 Scale 6 averages to flat green on a
    full 105 m pitch.

    ``dark_rgb``/``light_rgb`` are the two stripe albedos (linear). Defaults are matched to the
    target clip (2026-07-04, 4 measured render→grade iterations): the old emerald pair (hue ≈115°)
    rendered acid-green through the night grade; the clip's night-graded grass is yellow-green —
    post-grade medians H 81.9° S 0.663 match the clip exactly, V within 6%. Stripe contrast is
    measured end-to-end (p90/p10 of the detrended grass-luma profile): the clip's bands are nearly
    flat at 1.015 while albedo ratio 1.18 rendered 1.089 post-grade (~6×) and 1.03 vanished into
    the denoiser floor (1.003) — ratio 1.05 around the same per-channel mean lands at 1.019."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    # Mowing stripes: a banded wave quantised to two greens by a constant-interpolation ramp.
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = float(stripe_scale)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    dark, light = ramp.color_ramp.elements
    dark.position, dark.color = 0.0, (*dark_rgb, 1.0)
    light.position, light.color = 0.5, (*light_rgb, 1.0)
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


def build_stadium_lighting(
    bpy,
    *,
    light_rgb=(0.96, 0.96, 1.0),
    key_energy=1.6,
    sky_strength=0.03,
    sun_count=4,
    elevation_deg=65.0,
    sun_angle_deg=9.0,
):  # pragma: no cover - needs bpy
    """Floodlit-NIGHT stadium lighting, measured from the target clip (the deliverable is a night
    match, not a sunny day): a DARK world faintly tinted by the floodlight colour — no blue Nishita
    sky — plus a RING of soft, high SUN lamps tinted the same. Together they give an even,
    low-contrast fill that casts faint *multi-directional* shadows: the look of many roof
    floodlights, not one hard sun. The formal Cycles path keeps its own daytime sky — this
    builds only the deliverable's night look.

    ``light_rgb`` is the measured floodlight colour (a white-patch illuminant estimate; ~neutral,
    faintly cool). The world background is ``light_rgb × sky_strength`` so the night sky is dark but
    not pure black. ``sun_count`` lamps are spread evenly in azimuth at ``elevation_deg`` above the
    horizon, each a wide ``sun_angle_deg`` (soft shadow edges) at ``key_energy`` strength. Returns
    the created sun objects.
    """
    r, g, b = float(light_rgb[0]), float(light_rgb[1]), float(light_rgb[2])
    world = bpy.data.worlds.new("stadium_night")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (r * sky_strength, g * sky_strength, b * sky_strength, 1.0)
    bg.inputs[1].default_value = 1.0
    # A SUN points down its local −Z; rotate it tilt° off vertical (0 = straight down, 90 = horizon)
    # then spin around Z so the ring's rays arrive from evenly-spaced compass directions → the faint
    # multi-directional shadows. High elevation (small tilt) keeps those shadows short, as overhead.
    tilt = math.radians(90.0 - float(elevation_deg))
    n = max(1, int(sun_count))
    suns = []
    for i in range(n):
        az = 2.0 * math.pi * i / n
        light = bpy.data.lights.new(name=f"floodlight_{i}", type="SUN")
        light.energy = float(key_energy)
        light.angle = math.radians(float(sun_angle_deg))
        light.color = (r, g, b)
        obj = bpy.data.objects.new(name=f"floodlight_{i}", object_data=light)
        obj.rotation_euler = (tilt, 0.0, az)
        bpy.context.collection.objects.link(obj)
        suns.append(obj)
    return suns
