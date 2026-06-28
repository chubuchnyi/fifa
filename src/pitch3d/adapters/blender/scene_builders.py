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


def build_grass_material(bpy, name="grass", stripe_scale=0.1):  # pragma: no cover - needs bpy
    """Procedural grass for the pitch plane: banded mowing stripes (the iconic look) drive the base
    colour, a fine noise drives a subtle bump for blade micro-texture, and a high roughness keeps it
    matte. Built purely from generator nodes — no image texture to ship.

    ``stripe_scale`` is the wave Scale on the plane's Object coords (metres): ~0.1 gives ≈5 m mowing
    bands that read at broadcast distance — the original M2-9 Scale 6 averages to flat green on a
    full 105 m pitch."""
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
