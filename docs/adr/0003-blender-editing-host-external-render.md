# ADR-0003 — Blender is the editing host; photoreal render is external and imported back

- **Status:** Accepted
- **Date:** 2026-06-19
- **Deciders:** architecture
- **Related:** ADR-0001 (hexagonal core), ADR-0002 (source of truth), ADR-0005 (canonical model), ADR-0007 (ViewSynthesizer seams); C2, C6, R-2, R-9; roadmap M1/M2

## Context

Two distinct questions sit behind "where does the work happen":

1. **Where does the operator edit?** Manual pose/trajectory correction needs a mature 3D UI —
   bone gizmos, F-curves, a timeline, a viewport, a 2D radar overlay. Building that from scratch is
   a multi-year effort orthogonal to the product's value.
2. **Where do photoreal pixels come from?** Photoreal output is 3D Gaussian Splatting, per-subject
   avatars, and (for moderate moves) video-diffusion. These are specialised, heavy, GPU-bound
   renderers that evolve fast and are not Blender-native.

These must not collapse into "do everything inside Blender's Cycles/Eevee", which cannot render
splats or run video-diffusion, nor into "build our own editor", which is wasteful.

## Decision

1. **Blender is the editing platform.** The operator works on a **proxy** SMPL-X mesh + 2D radar +
   reprojection overlay inside Blender. A gizmo/F-curve bridge in `adapters/blender` translates
   operator gestures into `Correction` records (ADR-0002) — it never writes resolved geometry.
2. **The world frame is chosen to match Blender:** right-handed, **Z-up**, **meters** (`core/scene/units.py`).
   Exporters convert to Y-up where a target format (glTF/USD) requires it; that conversion lives in
   `adapters/export`, never in core.
3. **Photoreal rendering is external, behind ports.** `RenderPass` (splat/avatar) and the
   `ViewSynthesizer` seam-A orbit video (ADR-0007) run in their own renderers; their outputs come
   back as `RenderAssetRef`/`SynthViewRef` **pointers** and are imported for viewing, **not re-edited**.
4. **`bpy` is quarantined in `adapters/blender`.** The core never imports `bpy`; the Blender
   adapter is kept thin so the untestable surface is small.

## Consequences

**Positive**
- We reuse a battle-tested 3D editor instead of building one; gizmos/F-curves map cleanly to the
  four propagation modes (FR-22).
- The renderer is swappable (splat today, something else tomorrow) without touching the editor or core.
- Z-up/meters removes a whole class of scale/axis bugs at the edit boundary (matches Blender + sports intuition).

**Negative / costs**
- The editing adapter depends on Blender; `bpy` is awkward to unit-test, so it stays a thin,
  mostly-integration-tested bridge.
- Photoreal results round-trip through an import step (R-9 splat-render integration) rather than
  living natively in the editor.
- Per-subject avatar cost scales with ×24 people (R-2) — mitigated by building avatars on demand as
  cached `RenderAssetRef`s with a textured-SMPL-X default.

## Alternatives considered

- **Render everything inside Blender (Cycles/Eevee only)** — rejected: cannot do 3DGS or
  video-diffusion; external-renderer + import keeps the photoreal path open and swappable.
- **Build a custom WebGL/three.js editor for the MVP** — rejected: enormous build cost; three.js is
  kept as an *export/viewer* target (FR-27), not the editing host.
- **Edit directly on the photoreal render** — rejected: violates the single source of truth (ADR-0002)
  and the non-editable nature of ViewSynthesizer output (R-15).
