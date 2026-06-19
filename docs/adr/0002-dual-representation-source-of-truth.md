# ADR-0002 — Dual representation, single source of truth (`proposal ⊕ corrections → resolved`)

- **Status:** Accepted
- **Date:** 2026-06-19
- **Deciders:** architecture
- **Related:** ADR-0001 (hexagonal core), ADR-0005 (canonical model), ADR-0008 (LLM loop); C2, FR-21, FR-22, FR-23, FR-24, UX-5, R-3

## Context

The operator edits a *parametric* body (SMPL-X θ/β + root) and curves; the audience sees
*photoreal* pixels (splats, avatars, ball, or a ViewSynthesizer video). These are two
representations of one thing. If both are independently editable they desynchronise — the classic
edit↔render drift (R-3): you fix a pose in the render and the geometry no longer matches, or vice
versa. Edits must also be **non-destructive**: comparable, toggleable, reversible (FR-21).

## Decision

1. **The edit space is the single source of truth:** SMPL-X (θ, β, root) per subject + the ball
   curve. The render representation is *derived* and is **never edited**.
2. **Three layers** (`core/scene/layers.py`):
   - `proposal` — raw model output, stored on each `Subject`/`BallTrack`.
   - `corrections` — a list of typed `Correction` deltas; **the only thing an edit creates**.
   - `resolved = proposal ⊕ corrections`, computed on demand by `core/correction`, never hand-stored.
3. **A `Correction` is a typed delta**: a `target` (which quantity), a `frame_range`, a `mode`
   (one of the four propagation modes, FR-22) and a mode-specific payload. Corrections are
   `enabled`-toggleable so the operator can compare/reset/disable without losing them (UX-5).
4. **Resolution bakes the stack empty.** `resolve_scene` returns a fully-resolved copy whose
   `corrections` list is empty and whose geometry already includes the deltas — render, observe
   and export consume *only* this resolved scene. There is no back-channel from pixels to geometry.
5. **Preview = resolve-without-commit** (FR-23); **batch = one op over many subjects/ranges** (FR-24).

## Consequences

**Positive**
- No edit↔render desync (R-3): every render representation is re-driven from `resolved`.
- Edits are inherently non-destructive, comparable and reversible (FR-21, UX-5).
- The LLM agent edits through the *same* `Correction` unit a human does — safe by construction (ADR-0008).
- Preview is free: it is just `resolve` on a candidate stack without storing it.

**Negative / costs**
- `resolve` runs on every observe/render/export; cost is real (mitigated by the cache, ADR-0004).
- The store keeps corrections, not baked geometry — exporters must resolve first (they do).
- Operators/agents must express *every* change as a `Correction`; there is deliberately no
  "just nudge the mesh" escape hatch.

## Alternatives considered

- **Bake edits straight into geometry** — rejected: destructive, and re-introduces R-3 desync.
- **Store `resolved` as the truth and diff backwards** — rejected: loses the clean,
  toggleable provenance of each edit and makes compare/reset lossy.
- **Let the renderer own an editable copy** — rejected: two editable copies *is* the desync risk.
