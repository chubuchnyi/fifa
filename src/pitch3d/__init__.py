"""pitch3d — editable photorealistic 3D football reconstruction (architecture scaffold).

Hexagonal layout:
    * ``pitch3d.core``     — pure core (numpy only): scene model, correction math,
      orchestration contracts, port ABCs. Imports no ``bpy`` and no ML/render library.
    * ``pitch3d.adapters`` — infrastructure behind ports (models, viewsynth, blender,
      render, export) plus deterministic ``fakes`` for testing.
    * ``pitch3d.app``      — composition root and the CLI dry-run.
"""

__version__ = "0.0.1"
