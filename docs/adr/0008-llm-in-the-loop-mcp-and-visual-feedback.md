# ADR-0008 — LLM-in-the-loop editing via MCP + multi-view visual feedback

- **Status:** Accepted
- **Date:** 2026-06-19
- **Deciders:** architecture
- **Related:** ADR-0001 (hexagonal core), ADR-0002 (dual representation / single source of truth), ADR-0004 (queue + cache); FR-21 (non-destructive), FR-22 (propagation), FR-23 (preview), UX-4 (attention list)

## Context

A product goal is to **automate editor work with an LLM**: an agent should be able to open an
episode, find the wrong poses/trajectories, fix them, and check the result — the same loop a
human operator runs. Two things are needed that we did not yet have contracts for:

1. **A control surface** the LLM can call (open episode, inspect, edit, preview, render, export).
2. **Visual feedback.** An LLM cannot judge 3D from numbers alone. Limb crossings, ball height,
   foot-skate, and whether the rig lines up with the broadcast pixels are all *visual* facts. The
   agent must *see* the consequences of its edits — from **several viewpoints**, plus the source
   frame with the reprojection overlay, plus (optionally) the editor UI.

We must add this without breaking the hexagonal core (ADR-0001) or the single source of truth
(ADR-0002): the agent must not edit render output or poke SMPL arrays directly.

## Decision

1. **MCP server is a *driving* adapter**, parallel to the CLI (`adapters/mcp/`). Its **tool
   catalog** is the application's use-cases expressed as pure data
   (`adapters/mcp/tools.py::tool_catalog`): `list_episodes`, `run_reconstruction`, `observe`,
   `get_attention`, `apply_offset|keyframes|smoothing|refit`, `set_correction_enabled`,
   `preview`, `render`, `export`. The catalog is import-free (no SDK), so the agreed surface is
   testable today; the live server is gated behind the optional `mcp` extra.

2. **`SceneObserver` is a *driven* port** (`core/ports/observation.py`) producing snapshots in
   three kinds — `SCENE_3D` (the resolved scene rendered from N viewpoints), `FRAME_OVERLAY`
   (source frame + reprojection), `UI` (editor screenshot) — bundled into an `Observation`
   (images + textual `summary`). It composes the existing `RenderPass`; producing real pixels is
   an adapter, so the core ships the contract + a stdlib `FakeSceneObserver`.

3. **Viewpoint generation is pure core math** (`core/agent/`): `look_at`, `standard_viewpoints`
   (front/left/top/broadcast + orbit ring around the action centroid), and `scene_summary`, which
   turns the `attention_list` (UX-4) into the text half of the feedback.

4. **The agent loop is: `observe → reason(images + summary) → mutate via a correction tool →
   observe`.** Every agent edit is a `Correction` (FR-21): toggleable, previewable (FR-23),
   reversible. The agent never edits render output and never writes resolved state — `resolve()`
   stays the only path from proposal+corrections to geometry. This gives the LLM exactly the
   guardrails a human operator has.

5. **Feedback images return as MCP image content blocks** built from `ObservationImage` URIs, so
   the model literally sees what it changed.

## Consequences

**Positive**
- LLM automation is a first-class, vendor-neutral capability that *reuses* the correction,
  render, and export contracts — zero new coupling in `core`.
- Feedback is honest: snapshots come from the same `RenderPass` a human sees, from canonical
  viewpoints, with the attention list as guidance.
- Agent edits inherit the non-destructive guarantees (compare/disable/reset, preview-before-commit).
- The whole loop runs in tests and the dry-run with `FakeSceneObserver` (no GPU/renderer).

**Negative / costs**
- Rendering several viewpoints per `observe` is expensive; mitigated by `PREVIEW` quality, the
  content-addressable cache (ADR-0004), and letting the agent request only the viewpoints it needs.
- The live server needs the application controller (Task 7) before it can serve; until then the
  catalog is inspectable but `serve()` is an honest `NotImplementedError`.

## Alternatives considered

- **Custom REST/WebSocket API instead of MCP** — rejected: MCP is the emerging standard for LLM
  tool use and plugs directly into agent hosts (Claude Desktop/CLI), so we get transport + schema
  for free and stay vendor-neutral.
- **Text-only feedback (no images)** — rejected: the errors operators fix are visual; numbers miss
  limb crossing, occlusion, and foot-skate.
- **Let the LLM write SMPL/θ arrays directly** — rejected: violates the single source of truth and
  the non-destructive model; the safe, reviewable unit of change is a `Correction`.
