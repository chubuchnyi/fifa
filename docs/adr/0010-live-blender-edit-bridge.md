# ADR-0010 — Live Blender edits become Corrections over a socket; the host owns the scene

- **Status:** Accepted
- **Date:** 2026-06-20
- **Deciders:** architecture
- **Related:** ADR-0001 (hexagonal core), ADR-0002 (dual representation / Corrections are the only edit), ADR-0003 (Blender editing host), ADR-0008 (LLM-in-the-loop, one code path for human+agent), ADR-0009 (the un-runnable-real honesty pattern); FR-14, FR-21, FR-22a, UX-3; roadmap M1 step 10 (P1)

## Context

The Blender adapter was **batch-only**: `runner.run_blender` drives `blender --background` to write
the editable `.blend` and render proxy `SCENE_3D` views, and the top-down radar
(`render_radar` / `observe(include_radar=…)`) was **read-only**. The product needs the inverse: a
human dragging a player — a dot on the tactical radar, or the root Empty in the 3D viewport — should
move that subject in the canonical scene. Per ADR-0002 the *only* way anything enters the scene is a
`Correction`, and per ADR-0008 the human and the LLM should drive the **same** use-cases.

A live, GUI Blender session (kept alive, not `--background`) cannot run in our CI/dev box — there is
no display, and Blender is not pip-installable. This is the same shape of limitation ADR-0009 hit
with the GVHMR/TrackNet networks: the honest move is to build and **unit-test the parts that don't
need the GUI**, and be explicit about the one piece that does.

## Decision

1. **The host process owns the `Scene`; Blender is a thin transform reporter.** The wire protocol
   carries a subject root's **new world location** at a frame, not a pre-computed delta. The host
   diffs the drop against the **resolved** root (`Application.resolved`, which already bakes
   `proposal ⊕ corrections`, REFIT included) and mints the offset. Because the new offset stacks on
   the resolved base, the subject lands *exactly* where it was dropped, regardless of prior edits.
2. **A drag is committed through the same `apply_offset` use-case the MCP agent calls** (ADR-0008).
   `apply_drag` builds a `ROOT_TRANSLATION` / `CONSTANT_OFFSET` correction — no new mode, no new
   path. Human placement and LLM tool calls converge on `Application.apply_offset`.
3. **Drag semantics:** a 2-vector location (a radar drag) moves only XY and **keeps the resolved
   pelvis height** (no accidental lift off the pitch); a 3-vector (a 3D-viewport drag) moves all
   axes. The offset applies over the subject's **whole track** by default (the drag nudges the entire
   trajectory — the most predictable gesture); an explicit `frame_range` scopes it.
4. **The adapter keeps the pure/heavy split (ADR-0001/0006/0009).** The host side — `radar_to_world`
   (exact inverse of `world_to_radar`), the `subject_<id>` id↔name contract
   (`subject_object_name`/`parse_subject_name`, the single source of truth shared by the proxy
   builder and the bridge), `apply_drag`, and `serve_edits` (a newline-JSON socket loop) — imports no
   `bpy` and is unit-tested over a real `socket.socketpair`. The in-Blender client (`_live.py`,
   `bpy`-only, a `depsgraph_update_post` watcher) is self-contained and never imported by us, exactly
   like the batch `_script.py`.
5. **The live GUI session is the one un-runnable-in-CI piece, and we say so (ADR-0009 pattern).**
   `launch_live_session` spawns GUI Blender (no `--background`) and serves edits until the human
   closes it; it is `# pragma: no cover` because it needs a display + a Blender binary. Everything it
   composes is tested in isolation, so the gap is the *session*, not the logic.
6. **A wobbly editor never corrupts the scene.** `serve_edits` replies with a typed `error` and
   skips any malformed/unknown message or out-of-range frame, and terminates cleanly on `bye` **or**
   socket EOF (the human quitting Blender mid-session).

## Consequences

**Positive**
- One code path for human and LLM edits (ADR-0008): a GUI drag and an `apply_offset` tool call are
  the same correction, so they undo/compare/serialize identically.
- The whole drag→Correction loop is verified headlessly — the inverse math, the id↔name contract,
  the socket framing, error-skip, and EOF/`bye` termination — without a display.
- Host-owns-scene keeps Blender dumb: it reports world positions; it never decides what a correction
  is. The proxy can drift (we don't push state back yet) without risking the canonical data.
- The radar is no longer read-only: `radar_to_world` closes the loop a dragged dot needs.

**Negative / costs**
- The live GUI session and the in-Blender depsgraph watcher are not exercised in CI (no display);
  their correctness is best-effort + documented until validated on a workstation with Blender.
- The drag heuristic in `_live.py` distinguishes a *frame scrub* (refresh caches, emit nothing) from
  a *move* (emit) by frame equality; grabbing an Empty exactly on a keyframe edit is an accepted
  ambiguity for the MVP (the host stays correct regardless).
- We do **not** push resolved state back into the live proxy after a commit yet, so the Empty stays
  where the human left it rather than re-snapping to the resolved value (they already coincide for a
  single drag). A round-trip refresh is future work.

## Alternatives considered

- **Blender computes and sends the delta** — rejected: it would duplicate the resolve logic and the
  source of truth in the GUI, against ADR-0002. The host already knows the resolved base.
- **Embed the pitch3d package in Blender's Python and edit the scene in-process** — rejected:
  Blender ships its own interpreter (ADR-0003); importing our package there is fragile and couples
  the GUI to core internals. A thin socket keeps the boundary clean and unit-testable.
- **A new `LIVE_DRAG` correction mode** — rejected: a drag is exactly a `CONSTANT_OFFSET` on
  `ROOT_TRANSLATION`; a new mode would be dead weight (no half-finished implementations).
- **Write edits to a watched file instead of a socket** — rejected: a localhost socket gives clean
  framing, backpressure and EOF semantics, and is trivially testable via `socketpair`.
