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
- **Current focus:** **v0 = correct GEOMETRY is DONE (validated 2026-06-27).** Next bar = **v1 (recognizability)**: kit colours, shirt numbers, simple stadium.
- **NEXT ACTION:** **v0 = correct GEOMETRY is ACHIEVED & VALIDATED** (pod `zueopp6nzozxb7`, 48f real run, now STOPPED). All four defects closed: #202 body count = **20** (was a swarm); #203 root spread = **34×40 m** real-scale (fake-calib 22×7 m collapse gone, real PnLCalib default); #204 broadcast/top cameras frame the action; #205 pitch lines + goal frames render. Eye-judged from `out/val/video/{broadcast,top}.mp4` (local copies `/tmp/val_video/`, scene `/tmp/val_scene.json`). **NEXT BAR → v1 (recognizability):** team kit colours (teams already split A/B in the render), shirt numbers (OCR where readable, else roster), simple stadium backdrop. Pick the first v1 sub-task and scope it.
- **Target clip:** `samples/video/Colombia-1-0-Congo-DR1080p.mp4`

---

## 1. Goal (confirmed 2026-06-27)

From a source broadcast clip → a **realistic novel-view video of the SAME episode** (a different
camera angle), as faithful as possible. Players look like the originals (same kit + shirt numbers);
the stadium is realistic and the same as the source. **Judged by eye.**

- **Approximations OK** for exact numbers / exact stadium when unrecoverable from one clip — backstopped
  by **manual Blender editing** + **generative prompt-editing** (ADR-0008 LLM-over-MCP).

## 2. Staged bar (do in order; gate each on eye-judgement)

- [x] **v0 — correct GEOMETRY (ACHIEVED 2026-06-27).** Stable ~22 players (measured **20**), correct
  world placement/scale (root spread **34×40 m**), poses, virtual cameras that frame the action, pitch
  with lines + goals. Validated end-to-end on the pod (`out/val/`); see §6. First "good" result.
- [ ] **v1 — recognizability (CURRENT FOCUS).** Team kit colors (teams already split A/B in the render);
  numbers (OCR where readable, else roster); simple stadium backdrop.
- [ ] **v2 — photoreal.** Textured/Gaussian avatars + photoreal stadium + view-synth (the gated
  `avatars`/`viewsynth` heavy halves). The full stated goal; a long research stage.

---

## 3. v0 punch-list — the work right now

Detail + exact code root-causes: [`v0-geometry-defects.md`](v0-geometry-defects.md).
Found by eye in the 300-frame render of the real clip (`out/anim/video/*`, real CUDA models, 4 virtual
cameras, 25 fps / 12 s / 1280×720). That run saved **no `scene.json`** → body count is visual-only.

| ID | Defect | Status | GPU? | Root cause (file:line) | Next step |
|----|--------|--------|------|------------------------|-----------|
| #202 | Too many bodies (track-ID fragmentation); swarm grows over clip | **VALIDATED on pod ✓ (20 subjects)** | done | was: stitch OFF unless flag; tracker `min_track_frames=1` keeps 1-frame blips | DONE + VALIDATED 2026-06-27: stitch ON by default everywhere + `wiring.py` real ByteTrack `min_track_frames=2`. **Pod 48f real run (`out/val/export/scene.json`) → `len(scene.subjects) = 20`** (vs the old swarm of dozens; target ~22). The "swarm grows over the clip" defect is gone. **KEY:** stitch is PIXEL-space → independent of #203. No further tuning needed for v0. |
| #203 | Depth collapse / wrong world scale (players not spread across pitch) | **VALIDATED on pod ✓ (34×40 m spread)** | done | NOT the identity fallback: the default render used `--calibrator fake` = `FakeFieldCalibrator` (`adapters/fakes/perception.py:125`), a 30 m **top-down orthographic toy** `H` with NO perspective. Numeric proof: whole frame → 30×17 m world box; realistic feet → 22.7×7.3 m; the ~7 m y-axis is the "thin horizon" blob. Real PnLCalib was wired (`adapters/models/pnlcalib_backend.py`) but opt-in & OFF. | DONE + VALIDATED 2026-06-27: real PnLCalib is now the **default** render calib (`pod_real_e2e.sh` defaults `PNLCALIB_REPO=/workspace/repos/PnLCalib`; run echoed `calibration: REAL PnLCalib`). **Pod 48f scene.json root spread: X (length) 34.1 m, Y (width) 40.0 m, Z (pelvis) 0.69–1.01 m** — i.e. real-scale spread within the touchlines, NOT the fake-calib 22×7 m collapse; the ~7 m "thin horizon" is gone. Players cluster in one half (localized broadcast action), as expected. **Deepest root — also fixes #204.** |
| #204 | Virtual cameras don't frame the action (players tiny at horizon) | **VALIDATED on pod ✓ (broadcast frames the pitch)** | done | the VIDEO render (`blender_animate.py`) derives its OWN cameras from `ctr`/`span` of the loaded geometry — NOT `viewpoints.py`/`controller.py` (those drive the *in-pipeline* render, a different path). With #203's collapsed 22×7 m blob + a bare plane, the cams framed empty grass. | DONE + VALIDATED 2026-06-27: #205 folds the FULL pitch bounds into `ctr`/`span` + #203 puts bodies on the field. **Pod render eye-check (`out/val/video/{broadcast,top}.mp4`): broadcast camera frames the whole pitch at a realistic oblique angle (players fill the action third, not a horizon speck); top camera frames the full 105×68 m.** No per-frame tracking camera needed for v0. |
| #205 | Bare pitch (no lines / no goals) | **VALIDATED on pod ✓ (lines + goals render)** | done | the ACTUAL video render is `scripts/blender_animate.py` (builds its scene from scratch — only drew a bare grass plane); pitch lines existed only in the *other* (in-pipeline Cycles) path, and the goal mesh was genuinely absent | DONE + VALIDATED 2026-06-27: measured `goal_frame_geometry()` + `pitch_line_ribbons()` in pure core; `anim_export.py` writes `pitch.npz`, `blender_animate.py` builds `pitch_lines` + `goals`. **Pod 48f run: `anim_export` logged `pitch: 2848 line-tris + 72 goal-tris (105x68 m)`; render eye-check confirms full markings (boxes, circle, arcs, halfway) + goal frames on both views.** |

**Validation — DONE 2026-06-27 (pod `zueopp6nzozxb7`, 48f real run, `out/val/`):** all four v0 defects
validated end-to-end. #202 body count = **20** (scene.json); #203 root spread = **34×40 m** real-scale;
#204 broadcast/top cameras frame the action; #205 pitch lines + goals render. Numbers from `scene.json`,
eye-judged from `out/val/video/{broadcast,top}.mp4`. Pod STOPPED. **v0 = correct GEOMETRY: ACHIEVED.**

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

- **2026-06-27** — **v0 = correct GEOMETRY: ACHIEVED & VALIDATED end-to-end on the pod.** ONE batched
  48-frame real run (pod `zueopp6nzozxb7`: RF-DETR · ByteTrack · real PnLCalib · SMPLest-X · WASB →
  `out/val/export/scene.json`, then a cheap top/broadcast re-render via `REUSE_SCENE=1`). Measured: **#202
  body count = 20** (`len(scene.subjects)`; was a swarm of dozens) — CLOSED. **#203 root spread = 34 m
  (length) × 40 m (width)**, pelvis 0.69–1.01 m, run echoed `calibration: REAL PnLCalib` (the fake-calib
  22×7 m collapse is gone) — CLOSED. **#204** broadcast + top cameras frame the action (no horizon speck)
  — CLOSED. **#205** `anim_export` logged `pitch: 2848 line-tris + 72 goal-tris (105x68 m)`; render shows
  full markings + goal frames — CLOSED. Eye-judged from `out/val/video/{broadcast,top}.mp4` (local:
  `/tmp/val_video/`, scene `/tmp/val_scene.json`). Pod STOPPED ($0.012/hr storage only). All four v0
  tasks #202–#205 complete. **NEXT BAR → v1 (recognizability):** kit colours (teams already A/B), shirt
  numbers, simple stadium.
- **2026-06-27** — **#203 root found locally + fix landed (no pod needed for diagnosis).** Traced the
  depth collapse to the *calibrator*, not a degenerate homography: the default render path uses
  `--calibrator fake` (`cli.py` default) = `FakeFieldCalibrator`, a **top-down orthographic toy** with
  `_FAKE_PITCH_SPAN_M = 30.0` — it maps the WHOLE frame into a 30×17 m world box (the 105×68 m pitch is
  unrepresentable) and has no perspective term, so oblique broadcast depth folds onto a ~7 m band.
  Proven numerically on this machine (realistic feet → 22.7×7.3 m world span). Real PnLCalib was wired
  (`pnlcalib_backend.py`) but opt-in & OFF. **Fix (mirrors #202):** made real PnLCalib the DEFAULT —
  `pod_real_e2e.sh` defaults `PNLCALIB_REPO=/workspace/repos/PnLCalib` (proxy only if the staged repo is
  truly absent; force with `PNLCALIB_REPO=`), `demo_video.sh` flips `REAL_CALIB`→1 (`--no-real-calib` to
  opt out). This is the deepest root and also addresses #204 (the video cameras already frame the full
  pitch via #205's bounds fold; #203 puts bodies on it). Both scripts `bash -n` clean; default-resolution
  logic unit-checked. Still must be VALIDATED on the pod (confirm spread in `scene.json`).
- **2026-06-27** — **#205 code done + validated locally.** The real video render is
  `scripts/blender_animate.py` (builds its scene from scratch — it only drew a bare grass plane), so
  the fix lives there, NOT in the in-pipeline Cycles adapter the original root-cause note guessed.
  Added measured `goal_frame_geometry()` (2 posts + crossbar at Laws dims: 7.32 m mouth, 2.44 m high,
  0.12 m square section) to pure core `core/scene/pitch.py`, beside the existing `pitch_line_ribbons()`;
  `anim_export.py` now writes `pitch.npz` (line ribbons + goal frames, world metres) and purges it with
  the other artifacts; `blender_animate.py` loads `pitch.npz`, folds the pitch bounds into the camera
  framing (so the whole field stays in frame), and builds `pitch_lines` + `goals` meshes (bare-plane
  fallback if absent). 3 new geometry tests (now 12 in `test_pitch_geometry.py`). Validated by rendering
  the fixtures locally with the Blender binary (top view = full markings; goal close-up = correct
  posts+crossbar). Full tree green (560 passed, 12 skipped); changed-line ruff clean; mypy clean.
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
