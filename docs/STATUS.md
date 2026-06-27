# pitch3d — STATUS (single source of truth)

> **This file is the durable project tracker.** Claude Code sessions and its task list are NOT
> reliable (sessions break without recovery; context compresses with data loss), and chat history is
> not durable either. Everything project-relevant — goal, plan, current focus, defect status, progress
> — lives **here, in git**. Update this file and **commit at every step**.

**Last updated:** 2026-06-27

---

## Goal (confirmed 2026-06-27)

From a source broadcast clip → a **realistic novel-view video of the SAME episode** (a different
camera angle), as faithful as possible. Players look like the originals (same kit + shirt numbers);
the stadium is realistic and the same as the source. **Judged by eye.**

- **Target clip:** `samples/video/Colombia-1-0-Congo-DR1080p.mp4`
- **Approximations OK** for exact numbers / exact stadium when unrecoverable from one clip — backstopped
  by **manual Blender editing** + **generative prompt-editing** (ADR-0008 LLM-over-MCP).

## Staged bar (do in order)

- [ ] **v0 — correct GEOMETRY (CURRENT FOCUS).** Stable ~22 players, correct world placement/scale,
  correct poses, virtual cameras that frame the action, pitch with lines. Output: a clean *geometric*
  novel-view video. First "good" result; everything builds on it.
- [ ] **v1 — recognizability.** Team kit colors; numbers (OCR where readable, else roster); simple
  stadium backdrop.
- [ ] **v2 — photoreal.** Textured/Gaussian avatars + photoreal stadium + view-synth (the gated
  `avatars`/`viewsynth` heavy halves). The full stated goal; a long research stage.

---

## v0 punch-list (the work right now)

Detailed analysis + exact code root-causes: [`v0-geometry-defects.md`](v0-geometry-defects.md).
Found by eye in the 300-frame render of the real clip (`out/anim/video/*`, real CUDA models).

| ID | Defect | Status | Root cause (located in code) | Next step |
|----|--------|--------|------------------------------|-----------|
| #202 | Too many bodies (track-ID fragmentation); swarm grows over clip | TODO | `min_track_frames`=1; fragment-stitch pass wired but OFF by default (`continuity.py`/`pipeline.py`) | Enable stitch by default + raise `min_track_frames` (~5–10). Local, no GPU. **Cheapest first win.** |
| #203 | Depth collapse / wrong world scale (players not spread across pitch) | TODO | homography `H` collapses to identity/degenerate fallback → `image_to_world` piles feet at one spot (`field.py`/`pose.py`/`calibration.py`) | Verify/repair `H` on the real clip (log `image_to_world`). **Deepest root** — also fixes #204. Likely a pod re-run. |
| #204 | Virtual cameras don't frame the action (players tiny at horizon) | TODO | `action_centroid` averages collapsed roots; render uses one static camera frozen at frame 0 (`viewpoints.py`/`controller.py`) | Mostly resolves with #203; then optional per-frame centroid-tracking camera. |
| #205 | Bare pitch (no lines / no goals) | TODO | pitch-line code exists but only via Cycles pass; 300f video used a non-Cycles path; goal mesh genuinely absent (`_cycles_script.py`/`cycles.py`/`pitch.py`) | Render via Cycles (lines appear) + add `_build_goals()` mesh. |

**Validation:** after fixes, one GPU re-run on the pod renders v0 end-to-end; it **must save
`scene.json`** so #202's body count becomes a measured number, and we look at the output by eye.

---

## Progress log (newest first)

- **2026-06-27** — Strategic reset: results over process (stop milestone-ticking; make ONE real clip
  good, judged by eye). Defined the goal + v0→v1→v2 ladder. Looked at the first real 300-frame render
  (`out/anim/video`, real Colombia clip, real CUDA models) → found 4 v0 geometry defects (#202–#205)
  and located their root causes in code. Created this tracker + `v0-geometry-defects.md`; added a
  results-first reset banner to `roadmap.md`.

---

## Key references

- **v0 defects (detail + code root-causes):** [`v0-geometry-defects.md`](v0-geometry-defects.md)
- **Historical build log (M0–M4 = platform plumbing, NOT result quality):** [`roadmap.md`](roadmap.md)
- **M1 live state:** [`m1-status-and-plan.md`](m1-status-and-plan.md)
