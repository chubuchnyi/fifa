# pitch3d — STATUS (single source of truth)

<!--
  LLM COLD-START DOC. This file is the durable project tracker and the primary
  context-reload surface for a Claude Code session that starts with no memory.
  Claude Code sessions break without recovery and the context window compresses
  with data loss, so DO NOT rely on the CC task list (#...) or chat history.
  Everything project-relevant lives HERE, in git. Update this file and COMMIT at
  every meaningful step (decision, defect, status change, validation result).
  Keep it dense, structured, path/command-explicit — optimised for an LLM to
  rehydrate fast, not for prose.
-->

**Last updated:** 2026-06-27 · **Branch:** main · **Repo:** /home/chubuchnyi/AVATAR

---

## 0. TL;DR for a cold-start LLM

- **Goal:** from ONE broadcast clip → a realistic novel-view video of the *same* episode (different camera angle). Players look like originals (kit + shirt numbers), same realistic stadium. **Judged by eye.**
- **Mode:** results over process. Do NOT tick milestones / wire seams / pass tests on fake adapters. Only do work that makes the real-clip output visibly better.
- **Current focus:** **v0 = correct GEOMETRY.** Working the v0 punch-list (#202→#205) in ID order.
- **NEXT ACTION:** #202 LOCAL fix landed (stitch on by default + singleton drop; 557 tests pass). Next local-doable work is **#205** (add `_build_goals()` mesh; ensure render uses the Cycles path). #203/#204 are GPU-bound (diagnosis needs a pod). Plan: finish local work, then ONE batched pod run to diagnose #203/#204 + validate #202/#205 by eye (must save `scene.json`).
- **Target clip:** `samples/video/Colombia-1-0-Congo-DR1080p.mp4`

---

## 1. Goal (confirmed 2026-06-27)

From a source broadcast clip → a **realistic novel-view video of the SAME episode** (a different
camera angle), as faithful as possible. Players look like the originals (same kit + shirt numbers);
the stadium is realistic and the same as the source. **Judged by eye.**

- **Approximations OK** for exact numbers / exact stadium when unrecoverable from one clip — backstopped
  by **manual Blender editing** + **generative prompt-editing** (ADR-0008 LLM-over-MCP).

## 2. Staged bar (do in order; gate each on eye-judgement)

- [ ] **v0 — correct GEOMETRY (CURRENT FOCUS).** Stable ~22 players, correct world placement/scale,
  correct poses, virtual cameras that frame the action, pitch with lines. Output: a clean *geometric*
  novel-view video. First "good" result; everything builds on it.
- [ ] **v1 — recognizability.** Team kit colors; numbers (OCR where readable, else roster); simple
  stadium backdrop.
- [ ] **v2 — photoreal.** Textured/Gaussian avatars + photoreal stadium + view-synth (the gated
  `avatars`/`viewsynth` heavy halves). The full stated goal; a long research stage.

---

## 3. v0 punch-list — the work right now

Detail + exact code root-causes: [`v0-geometry-defects.md`](v0-geometry-defects.md).
Found by eye in the 300-frame render of the real clip (`out/anim/video/*`, real CUDA models, 4 virtual
cameras, 25 fps / 12 s / 1280×720). That run saved **no `scene.json`** → body count is visual-only.

| ID | Defect | Status | GPU? | Root cause (file:line) | Next step |
|----|--------|--------|------|------------------------|-----------|
| #202 | Too many bodies (track-ID fragmentation); swarm grows over clip | **LOCAL FIX LANDED — validate on pod** | done(local) / pod(validate) | was: stitch OFF unless flag; tracker `min_track_frames=1` keeps 1-frame blips | DONE locally: stitch now ON by default everywhere (CLI `--no-stitch` opt-out; `pod_real_e2e.sh`/`pod_make_video.sh` default on; `wiring.py` real ByteTrack `min_track_frames=2` drops un-stitchable singletons). **KEY:** stitch is PIXEL-space → independent of #203; `demo_video.sh` ALREADY defaulted stitch on, so the swarm persisted DESPITE stitch → the body-count fix needs MEASUREMENT, not blind tuning. Validate: pod re-run logs `len(scene.subjects)` from `scene.json`; only THEN tune stitch gates / add ByteTrack re-id if still high. |
| #203 | Depth collapse / wrong world scale (players not spread across pitch) | TODO | yes | homography `H` degenerate/identity fallback → `image_to_world` piles feet at ~origin (`core/scene/field.py`, `adapters/models/pose.py` `_ground_root`, `adapters/models/calibration.py`) | Verify/repair `H` on the real clip (log `image_to_world` of known foot pt). **Deepest root** — also fixes #204. Pod re-run. |
| #204 | Virtual cameras don't frame the action (players tiny at horizon) | TODO | partial | cameras placed on 63 m radius around `action_centroid` (mean of collapsed roots); render uses one static cam frozen at frame 0 (`core/agent/viewpoints.py`, `app/controller.py`) | Mostly resolves with #203; then optional per-frame centroid-tracking camera. |
| #205 | Bare pitch (no lines / no goals) | TODO | yes | pitch lines drawn only via Cycles pass; 300f video likely used non-Cycles path; goal mesh genuinely absent (`adapters/blender/_cycles_script.py`, `adapters/render/cycles.py`, `core/scene/pitch.py`) | Render via Cycles (lines appear) + add `_build_goals()` mesh. |

**Validation:** after the local fixes (#202, #205-code), ONE GPU pod re-run renders v0 end-to-end; it
**must save `scene.json`** (so #202's body count becomes a measured number) and we judge by eye. Batch
GPU-needing work (#203 diagnosis, #204, #205 render) into that single pod session to save cost.

---

## 4. Conventions & commands (rehydration cheat-sheet)

**Run from repo root** `/home/chubuchnyi/AVATAR`. Local Python env is `.venv` (CPU/core work needs no GPU).

```bash
# tests / lint / types (pure-core work is fully testable locally)
.venv/bin/python -m pytest                 # full suite (~557 tests baseline)
.venv/bin/python -m pytest tests/<path>    # focused
.venv/bin/ruff check <files>               # lint
.venv/bin/mypy <files>                     # types

# end-to-end CLI (real pipeline entrypoint)
.venv/bin/python -m pitch3d.app.cli ...    # see app/cli.py for args (e.g. --stitch, --real-calib)
```

**Lint policy:** repo has ~46 pre-existing ruff violations. Lint **only changed lines** — baseline-diff
with `git show HEAD:<file> | .venv/bin/ruff check --stdin-filename <file> -`. Don't "fix" unrelated
pre-existing violations.

**Git:** commit at every checkpoint (durability). Push uses a dedicated SSH key:
```bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/<fifa_key>' git push   # remote: git@github.com:chubuchnyi/fifa.git
```
(see memory `reference_github_push` for the exact key path.)

**GPU pod (RunPod):** ~$0.74/hr — **STOP it whenever not actively rendering**; the network volume
persists and restart is free. The pod repo `/workspace/fifa` is a **stale mirror**: it shows
already-pushed work as "uncommitted". Reconcile by byte-comparing vs pushed HEAD, then
checkout+rm+pull — never blind-commit on the pod. (memory `reference_pod_git_state`, `feedback_pod_cost`.)

**Architecture (ADR-0001 hexagonal):** pure `core/` (numpy/CPU, unit-tested) + `adapters/` (heavy ML
behind extras, lazy-imported via dotted-path injection, ADR-0006). Corrections are the sole edit path
(ADR-0002). LLM-over-MCP editing, human≡LLM (ADR-0008/0010). R-6 honesty: mark/interpolate, never
fabricate or silently hide.

**Working rules (from user feedback memory):**
- Results over process — only result-bearing work; no milestone flips / seam-wiring / fake-adapter tests.
- Verify the WHOLE decode→…→export path every iteration, not just the changed unit.
- Track everything here in `docs/STATUS.md` + commit; CC task list is at most a transient mirror.
- Be decisive when scope is known; don't over-poll or wait for obvious confirmations.

---

## 5. Code map (where the v0 work lives)

- **Tracking / fragmentation (#202):** `src/pitch3d/adapters/models/tracking.py` (`ByteTrackTracker`,
  `min_track_frames`, `ByteTrackBackend.associate`), `src/pitch3d/core/orchestration/pipeline.py`
  (`stitch_cfg` gate), `src/pitch3d/core/orchestration/continuity.py` (`StitchConfig`,
  `stitch_tracks_with_report`), `src/pitch3d/core/orchestration/assemble.py` (one Subject per track_id),
  `src/pitch3d/app/cli.py` (`--stitch`, `run_dry_run`).
- **Calibration / world scale (#203):** `src/pitch3d/core/scene/field.py` (`image_to_world`),
  `src/pitch3d/adapters/models/pose.py` (`_ground_root`, foot=bbox bottom),
  `src/pitch3d/adapters/models/calibration.py` (identity fallback; `CameraModuleFieldCalibrator`),
  `src/pitch3d/core/scene/units.py` (`FieldDimensions` 105×68 m).
- **Cameras (#204):** `src/pitch3d/core/agent/viewpoints.py` (`standard_viewpoints` 63 m radius,
  `action_centroid`), `src/pitch3d/app/controller.py` (`_static_camera`, frozen frame-0).
- **Render / pitch / goals (#205):** `src/pitch3d/adapters/blender/_cycles_script.py` (`_add_ground`,
  `_build_pitch`), `src/pitch3d/adapters/render/cycles.py` (`draw_pitch`), `src/pitch3d/core/scene/pitch.py`
  (markings). Goals: add a `_build_goals()` mesh (absent today).

---

## 6. Progress log (newest first)

- **2026-06-27** — **#202 local fix landed.** Made stitch the uniform default across ALL
  reconstruction entrypoints (CLI flag flipped `--stitch`→`--no-stitch`, default ON;
  `pod_real_e2e.sh` + `pod_make_video.sh` default stitch on) and set the real ByteTrack path's
  `min_track_frames=2` in `wiring.py` to drop un-stitchable 1-frame singletons. Discovered while
  diagnosing: stitch runs in PIXEL space (so #202 is *independent* of #203's homography), the tracker
  `min_track_frames` filter runs *before* stitch (so raising it too high starves stitch), and
  `demo_video.sh` already defaulted stitch on — i.e. the 300f swarm appeared *with* stitch on, so the
  body-count fix must be MEASURED on a pod re-run (read `len(scene.subjects)`), not blind-tuned.
  Full suite green (557 passed, 12 skipped); changed-line ruff clean; mypy clean. Body-count
  validation deferred to the batched pod run.
- **2026-06-27** — Reformatted this file into an LLM-friendly cold-start doc (dense sections +
  conventions/commands + code map). Pushed docs reset (7af66e9). Starting v0 punch-list #202.
- **2026-06-27** — Strategic reset: results over process. Defined the goal + v0→v1→v2 ladder. Inspected
  the first real 300-frame render (`out/anim/video`, real Colombia clip, real CUDA models) → found 4 v0
  geometry defects (#202–#205) and located their root causes in code. Created this tracker +
  `v0-geometry-defects.md`; added a results-first reset banner to `roadmap.md`.

---

## 7. Key references

- **v0 defects (detail + code root-causes):** [`v0-geometry-defects.md`](v0-geometry-defects.md)
- **Historical build log (M0–M4 = platform plumbing, NOT result quality):** [`roadmap.md`](roadmap.md)
- **M1 live state:** [`m1-status-and-plan.md`](m1-status-and-plan.md)
- **Memory (outside repo):** `feedback_results_over_process`, `project_goal_definition`,
  `feedback_durable_tracking`, `feedback_pod_cost`, `reference_github_push`, `reference_pod_git_state`.
