# Pipeline Studio — a glass-box workbench for the pitch3d pipeline

> **Goal.** Turn the pipeline from a black box that emits one `scene.json` into a **glass box**:
> every stage's **input** and **output** is inspectable **per frame**, editable **manually or by an
> AI/model**, its **parameters** are tunable, and any stage can be **re-run from here** with a
> **diff** against the previous result. This serves two jobs at once — **debug** the algorithms
> (see exactly where a reconstruction goes wrong) and **experiment** with them (branch, tune,
> compare) with full control instead of a single opaque answer.
>
> **Host.** This extends the existing `poseannot` app (`docs/poseannot-architecture.md`); it does not
> replace it. It **runs on the pod** (GPU + models + network volume), in-process with the pipeline.
>
> **Two parts.** Part I is the **interface design**. Part II is the **implementation plan**.
> Every claim about existing code is `file:symbol` for navigation and verified as of 2026-07-09.

---

## The one principle: edits are layered overrides, never destruction (R-6, pipeline-wide)

The pipeline already has one honest edit model — the **`Correction` layer** (ADR-0002 / R-6): physics
and coherence never rewrite a pose, they *append* `Correction` objects resolved at export
(`core/correction/engine.py:resolve_subject_motion`, `app/controller.py:resolved`). poseannot already
reuses it: a human joint edit is one `Correction` row in `edits.json`
(`poseannot/edits.py:build_body_pose_edit`).

**Pipeline Studio generalizes exactly this pattern to every stage.** An edit to *any* stage's input,
output, or parameters — whether typed by a human or proposed by a model — is a **layered override**
keyed by `(stage, target, frame_range)`, applied at that stage's boundary, and **resolved** into the
re-run. The immutable pipeline output and the source video are never mutated. This gives us undo,
provenance, branching, and honest confidence stamping *for free*, at every stage, the same way we
already get it for pose corrections today.

---

# PART I — INTERFACE DESIGN

## 1. Honest leverage: ~70% of this already exists

The design is cheap because the pipeline was built for it. What we exploit:

| We need… | It already exists as… | Where |
|---|---|---|
| Per-stage artifacts, content-addressed | `run_cached` + `content_key(stage, input_hash, params, model_version)` — **process-independent** ("a cache written by one run is found by another") | `core/orchestration/stages.py:78`, `core/ports/cache.py:content_key` |
| A stage list / DAG | `Stage` enum + `RECON_ORDER` | `stages.py:24,42` |
| Stage-run metadata (key, cache-hit) | `StageRun` | `stages.py:69` |
| Async, non-blocking heavy re-runs | `JobQueue` / `JobState` port (submit/state/result/cancel) | `core/ports/jobs.py` |
| An LLM visual-feedback loop **as a first-class stage** | `Stage.OBSERVE` — *"multi-view snapshots for the LLM feedback loop (ADR-0008)"* | `stages.py:38` |
| A layered, toggleable, resolvable edit model | `Correction` layer + `resolve_subject_motion` + `resolve_scene` | `core/correction/*`, `controller.py:resolved` |
| Per-frame frame server + JPEG | `/api/frame/{n}` | `poseannot/app.py`, `video.py` |
| World→pixel projection (calibration + 180° roll) | `camera.frame_projector` / `project_points` | `poseannot/camera.py` |
| Client-side overlay renderer (SVG bones/points, zoom/pan) | `projectOverlay` / `rebuildOverlay` | `poseannot/static/index.html` |
| 3D SMPL-X editor with a rotation gizmo | Three.js `updateThreePose` + TransformControls | `poseannot/static/index.html` |
| Per-frame status timeline | timeline strip + `editedFrames` colouring | `poseannot/static/index.html`, `/api/edits` |
| Runtime clip/bundle switch + upload | `clips.py`, `/api/clips*` | `poseannot/clips.py` |
| Persisted, resolvable human edits | `/api/edit`, `edits.json` | `poseannot/edits.py`, `scene_state.py` |

**What is genuinely new** (Part II builds these): a **disk cache adapter** that materialises every
stage's I/O to a browsable run bundle (the port is designed for it — *"concrete stores (memory, disk)
are adapters"* — but only the memory fake exists today, `adapters/fakes/cache.py`); a **real job
queue** for GPU re-runs (only the in-process fake exists, `adapters/fakes/jobs.py`); **per-stage
visualizers/editors** beyond pose; the **override-resolution glue** for input/param edits; and the
**stage-rail UI**. The core seams are all present.

## 2. The pipeline has four kinds of "stage" — the Studio unifies them behind one view

Be honest about the substrate: today's stages do **not** all flow through `run_cached`. Four families:

1. **`run_cached` recon stages** — `DETECT, TRACK, CALIBRATE, POSE, BALL, ASSEMBLE` (+ `AMPLIFY, ENV,
   AVATAR, RENDER, EXPORT, OBSERVE`). Content-addressed, cached, job-queued. *Easiest to expose.*
2. **Post-assembly correction stages** — `COHERENCE`, the **M3-9 kinematic gate**, and the 14 opt-in
   physics gates. Run in `controller.run_reconstruction:203-292`; they **emit `Correction` layers**
   (already inspectable, disableable, resolvable). *This is the gold edit model — copy it outward.*
3. **Pre-pose continuity** — `STITCH`, `IDENTITY`. Run in `pipeline.run` around the cached stages
   (`core/orchestration/continuity.py`); not cached, not in the `Stage` enum yet.
4. **Export/render tail** — resolve → `anim_export.py` → `blender_animate.py` → ffmpeg (scripts).

The Studio presents all four uniformly through a single **`StageView`** descriptor (§4). Part II
proposes pulling (2)/(3)/(4) under the same `Stage`/cache/override model so the UI code is uniform,
but the UI abstraction does not *require* that on day one — a `StageView` can wrap a non-cached stage.

## 3. The four core abstractions

### 3.1 `StageView` — the per-stage UI contract
One descriptor per stage, so the frontend is generic and stages are data, not special cases:

```
StageView = {
  id:            Stage,                 # "pose"
  label:         str,                   # "7 · Pose estimation"
  temporal:      "per_frame" | "per_track" | "whole_clip",
  input:         DataSpec,              # type + how to visualise + how to edit
  output:        DataSpec,
  params:        ParamSchema,           # tunable knobs, with defaults + current + provenance
  rerun_cost:    "instant" | "seconds" | "gpu_minutes",  # drives UI (auto vs explicit re-run)
  edit_actions:  [ActionSpec],          # the buttons == the MCP tools (§8)
  depends_on:    [Stage],               # DAG edges → invalidation
}
```

`DataSpec` names a **visualizer** (a layer renderer: boxes / trails / homography-grid / smplx-3d /
ball-marker / correction-ribbon / rendered-frame) and an **editor** (the manual affordances). Because
input and output are *both* "a frame + an overlay layer", they reuse the existing
`projectOverlay`/`rebuildOverlay` pipeline — we only add new layer renderers.

### 3.2 Run bundle (session) — the materialised DAG on disk
Formalise the disk cache into a browsable bundle. The key already exists (`content_key`), so this is
mostly a `DiskCache` adapter + a manifest:

```
runs/<run_id>/
  manifest.json         # DAG, per-stage {key, input_hash, params, model_version, timing, cache_hit}, parent branch
  cache/<stage>-<hash>  # the StageRun.result artifact (content-addressed; == cache.put target)
  overrides/<stage>.json# layered input/output/param edits for this branch (Correction-style rows)
  observe/<frame>.jpg   # OBSERVE snapshots for the AI loop (ADR-0008)
```

Two properties fall out of `content_key` being deterministic and process-independent:
- **Re-run = cache miss only where something changed.** Editing `CALIBRATE` params changes its key →
  miss → re-run; everything upstream is a hit; everything downstream is a miss (its `input_hash`
  changed). This *is* DAG invalidation, for free.
- **Branch = a new overrides file + a shared cache.** Two branches that differ only at `POSE` share
  every upstream artifact by hash. The A/B pose bake-off becomes "branch at `POSE`", generalised to
  any stage.

### 3.3 Override layer — one edit model for input, output, and params
Every edit is a row (the `Correction` pattern, widened). Targets now address any stage, not just poses:

```json
{ "stage": "detect", "side": "output", "frame_range": [12, 12],
  "op": "add_box", "value": {"cls": "goalkeeper", "xyxy": [910,540,970,700], "score": 1.0},
  "source": "manual:admin", "confidence": 1.0, "note": "keeper missed by RF-DETR",
  "ts": "2026-07-09T14:02Z" }
```

- `side: "input"` overrides feed the stage's *input* (e.g. a drawn detection box that re-tracking must
  honour). `side: "output"` overrides post-edit the stage's *result* (e.g. a nudged joint, today's
  behaviour). `side: "params"` set a knob.
- Resolution mirrors `resolve_subject_motion`: `stage_input = resolve(upstream_output ⊕ input_overrides)`,
  run the stage, then `stage_output = resolve(raw_output ⊕ output_overrides)`.
- **Provenance + confidence** are mandatory (`source: manual:<user>` | `ai:<model>` | `stage`), so an
  AI's guess never masquerades as a measurement — the same R-6 discipline the pipeline already stamps
  (`docs/pipeline-en.md §12`). Undo = pop the last row. Audit = read the file.

### 3.4 Action API — one set of operations, two drivers (human + LLM)
The heart of the "edit via AI" requirement and ADR-0008. There is **one** verb set; the UI buttons and
the MCP tools call the **same** endpoints. "Auto + manual override" at the UI layer.

```
list_stages() · get_io(stage, frame) · get_params(stage) · set_param(stage, key, value)
edit_input(stage, frame_range, op, value) · edit_output(stage, frame_range, op, value)
rerun(from_stage, subjects?) → job · job_status(job) · diff(stage, frame, branch_a, branch_b)
observe(frames, cameras) → images · branch(from_stage, name) · resolve_export()
```

A human clicks "draw box → re-run"; an LLM calls `edit_input` then `rerun` then `observe` then reads
the image and iterates. **Identical surface.** The `OBSERVE` stage (`stages.py:38`) is the visual
feedback channel that closes the LLM loop.

## 4. Temporal shapes — how "input/output on every frame" works for each stage

Stages differ in *time shape*; the inspector adapts (this is the single most important UI subtlety):

- **`per_frame`** (`DETECT, CALIBRATE, BALL-2D`): scrubbing the timeline shows a genuinely different
  I/O each frame. Timeline row = per-frame status of *this stage*.
- **`per_track`** (`TRACK, STITCH, POSE, COHERENCE, PHYSICS`): the object is a track over time. The
  frame playhead moves *within* the selected track. Input pane = the upstream track; output pane =
  this stage's track; the timeline shows the track's span + per-frame confidence/edit status.
- **`whole_clip`** (`ASSEMBLE, RESOLVE/EXPORT, RENDER`): no per-frame edit; show the aggregate (scene
  summary, exported file, rendered angle) with frame scrub only for preview.

## 5. Per-stage catalogue (the substance)

For each stage: what its **input** and **output** are, how each is **visualised**, the **params**, the
**manual** edit ops, the **AI** edit ops, and the **re-run cost**. Recon stages first, tail summarised.
This table *is* the spec for the layer renderers and the action set.

### 5.1 Recon + correction stages

| Stage (shape) | Input → viz | Output → viz | Params (source) | Manual edit | AI edit (model) | Re-run |
|---|---|---|---|---|---|---|
| **1 Decode** (`whole_clip`) | video uri | frames + `ClipRef` | fps, frame range, resolution (`ffmpeg`) | trim range | — | instant |
| **2 Detect** (`per_frame`) | frame JPEG | boxes `{cls,score,xyxy}` → coloured rects | `score_threshold=0.3`, class filter (`detection.py`) | draw / move / delete box, relabel class | **re-detect crop** (SAM box-refine / second detector), raise recall on a ROI | seconds (per frame) |
| **3 Track** (`per_track`) | boxes/frame | tracklets `{id,frames,bboxes,cls,team}` → id-coloured trails | ByteTrack thresholds, k-means k=2 (`tracking.py`) | merge / split tracklet, reassign id, fix team | re-cluster teams on appearance, re-associate an occlusion gap | seconds |
| **4 Stitch** (`per_track`) | tracklets | linked tracklets + `StitchReport` → link ribbons | `max_gap=12`, `max_center_dist=1.5`, `size_ratio=1.6` (`continuity.py`) | force-link / unlink two fragments | suggest links from motion+appearance | instant |
| **5 Identity** (`per_track`) | tracklets | jersey #, role → labels | (off by default) | assign jersey / role | OCR jersey number, GK/ref classifier | seconds |
| **6 Calibrate** (`per_frame`) | frame JPEG | homography `(3,3)` → **pitch-line grid** reprojected on frame | keypoint conf, HRNet model (`calibration.py`) | **drag pitch keypoints**, nudge homography (reuse the existing camera-adjust panel!) | re-run PnLCalib, PnP from clicked points | seconds |
| **7 Pose** (`per_track`) | tracklet crop + calib | SMPL-X `{global_orient(cam!),body_pose,transl,betas}` → **existing 3D editor + 2D skeleton** | backend **A/B** (`--pose-backend`), bbox×1.25 (`pose.py`) | **rotate joints (ships today)**, fix orientation, set root Z | **re-pose** A↔B, third backend, LLM "un-flip all inverted bodies" via OBSERVE | **gpu-minutes** (per subject) |
| **8 Ball 2D** (`per_frame`) | frame JPEG | ball pixel + score → marker | `score_threshold=0.1`, backend TrackNet/WASB (`ball.py`) | move / add / delete ball point | re-run WASB↔TrackNet, detect on ROI | seconds |
| **9 Ball 3D** (`per_frame`→traj) | ball 2D + homography | world xyz + `on_ground` → 3D arc | `max_speed=35`, `CONTACT_PX=140` (`ball_lift.py`) | toggle `on_ground`, set apex height | fit parabola to clicked apex | instant |
| **10 Coherence** (`per_track`) | poses | gap-filled/extended poses + `TEMPORAL_SMOOTHING` correction → **confidence ribbon** (1.0/0.3/0.2) | `max_fill_gap=12`, `smooth_window=5`, `decay=0.9`, `coast_max=10.5` (`coherence.py`) | toggle fill / extend / smooth per subject; edit a filled span | — | instant |
| **11 Physics** (`per_track`) | poses | corrected poses + `Correction`s → **per-gate ribbon + teleport marks** | **profile** + 16 gate thresholds (`config/physics.yaml`) | toggle any gate, edit any threshold live, mark/interpolate teleport | auto-tune ceilings from p95 (`ProfileUpdateProposal`, T4.b) | instant |

Notes that matter for the UI, straight from `docs/pipeline-en.md`:
- **Pose output is in the camera frame** (`pose.py:188`): `global_orient` is *not* rotated to world.
  The POSE inspector must show this honestly — a "camera-frame / world-frame" badge — because ~35% of
  bodies read inverted and that is *expected*, not a UI bug. The `orient_verticality` fix exists but is
  `enabled:false` in `default`; the Physics stage is where you'd toggle it and see the diff.
- **Coherence/Physics re-run is instant** (pure half, no torch) → these get **live** param sliders with
  auto-re-run. **Pose is gpu-minutes** → explicit "re-run" button, per-subject scope, async job +
  progress. The `rerun_cost` field drives this difference.
- **Corrections resolve at export** (`controller.export → resolved`), which is why `scene.json` shows 0
  corrections (`pipeline-en.md §13`). The Studio inspects the **layered** (pre-resolve) scene, so it can
  show the correction ribbons the exported file hides.

### 5.2 Export/render tail (summarised — `whole_clip`)

| Stage | Input → Output | Params | Edit | Re-run |
|---|---|---|---|---|
| **12 Assemble** | all stage outputs → `Scene` + `ConfidenceMap` | — | — | instant |
| **13 Resolve/Export** | `Scene ⊕ Corrections` → `scene.json` / glTF | format | — | instant |
| **14 anim_export** | scene.json → per-subject `.npz` (SMPL-X FK) | — | — | seconds |
| **15 Blender** | npz → rendered frames (Cycles) | cameras, samples=32, floodlit-night rig (`blender_animate.py`) | camera params, lighting | gpu-minutes |
| **16 ffmpeg** | frames → mp4/angle | crf=18, fps=25 | — | seconds |

## 6. Layout & wireframes — extend the grid, don't rewrite the app

The current app is a CSS grid (`poseannot/static/style.css`): `rows 44px 1fr 60px`, `cols 200px 1fr
44vw`, areas `toolbar / sidebar main right / timeline`. Studio adds **one row** (the stage rail) and
**re-roles** the existing panes. Nothing is thrown away.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ TOOLBAR  clip ▾  run ▾  branch ▾  frame [  30 ]  ◀ ▶   [Export]  [AI ▸]      user ▾ │  44px
├───────────────────────────────────────────────────────────────────────────────────┤
│ STAGE RAIL  ● Decode → ● Detect → ● Track → ○ Stitch → ○ Ident → ●Calib → ◍POSE →   │  40px
│             ● Ball2D → ● Ball3D → ◍Coher → ◍Phys → Assemble → Export → Render        │
│             (● clean · ◍ dirty/edited · ○ off · ⟳ running · red=diff)                │
├──────────────┬──────────────────────────────────────┬───────────────────────────────┤
│ CONTEXT      │  INPUT ⇆ OUTPUT  (wipe/toggle)        │  STRUCTURED / PARAMS           │
│ (220px)      │  main canvas: frame + stage layer     │  (minmax 420px, 46vw)          │
│              │                                       │  ┌── params (drawer) ────────┐ │
│ subjects/    │  ┌─ stage layer toolbar ───────────┐  │  │ score_threshold 0.30 ▉──  │ │
│ tracks list  │  │ [IN][OUT][DIFF]  edit: ▭ ✎ ✕ 🎯 │  │  │ profile: default      ▾   │ │
│ · #7 team A  │  └─────────────────────────────────┘  │  │ orient_verticality □→☑    │ │
│ · #12 ⚠flip  │                                       │  │        [⟳ Re-run from POSE]│ │
│ · ball       │      ▛▀▀▀ boxes / trails / grid /      │  └───────────────────────────┘ │
│              │      ▙▄▄▄ smplx / ball / ribbon        │  three.js 3D editor (POSE)     │
│              │       ↑ zoom/pan reuse existing        │   OR data table (boxes/homog)  │
├──────────────┴──────────────────────────────────────┴───────────────────────────────┤
│ TIMELINE (stage-scoped)  ▏measured▕▏0.3 fill▕▏edit▕▏teleport▕  playhead ▎  frame 30   │  64px
└───────────────────────────────────────────────────────────────────────────────────┘
```

- **Stage rail** (new): the DAG as a clickable strip; node glyph shows clean/dirty/off/running/diff.
  Clicking a stage loads its `StageView`. This is the "add all the steps to the UI" ask, literally.
- **Main canvas** (re-role of `.main`): shows the **selected stage's layer** over the frame. `IN/OUT`
  toggle; `DIFF` = before/after wipe slider after a re-run. All existing zoom/pan/overlay code is
  reused — we register new *layer renderers* per `DataSpec`.
- **Right pane** (re-role of `.right`): the **structured** view. For POSE it stays the three.js SMPL-X
  editor with the gizmo (ships today). For box/homography/ball stages it's a data table + handles. A
  **params drawer** docks at its top (collapsible; `c` already toggles the camera panel — extend the
  pattern), with the **[Re-run from here]** button whose label reflects `rerun_cost`.
- **Context sidebar** (re-role of `.sidebar`): the stage-relevant list (subjects for per_track stages,
  detections for detect, etc.), with ⚠ flags (e.g. "flipped", "teleport", "low-conf span").
- **Timeline** (re-role): **scoped to the selected stage** — colours mean measured / filled(0.3) /
  extrapolated(0.2) / edited(orange) / teleport(red), reusing the `editedFrames` colouring machinery
  and the confidence map (`pipeline-en.md §12`).

Per **temporal shape** the panes retune: `per_frame` → IN/OUT are the same frame, different layer;
`per_track` → sidebar drives track selection, timeline is the track span; `whole_clip` → canvas shows
the aggregate/rendered result, timeline is preview-only.

## 7. Interaction flows (concrete, end-to-end)

**Debug — "why is #12 lying flat at frame 30?"**
Select **POSE** → pick #12 → frame 30. Input pane: the crop RF-DETR/tracker fed the net. Output: the
3D body, flat, with a **`camera-frame`** badge. Rail shows POSE clean but the body wrong → the badge
says the tilt lives in camera coords (`pose.py:188`). Jump to **PHYSICS** → toggle `orient_verticality`
on (it's off in `default`, `pipeline-en.md §11.2`) → instant re-run (pure) → DIFF wipe shows the body
snap upright. You just localised a "bug" to a disabled gate, not a broken net — in two clicks.

**Experiment — "does WASB beat TrackNet here?"**
Select **BALL 2D** → params → backend `TrackNet → WASB` → `branch("ball-wasb")` → re-run BALL. Compare
view (§5) wipes the two ball tracks on identical pixels. Keep the winner; the branch that lost shares
every other stage's artifact by hash, so it cost only the ball re-run.

**Fix input — "detector missed the keeper at frame 12."**
Select **DETECT** → frame 12 → draw a `goalkeeper` box (an `edit_input` override) → the rail marks
DETECT…ASSEMBLE dirty → **[Re-run from DETECT]** (async, progress) → the keeper now tracks, poses, and
renders. The drawn box persisted as a provenance-stamped override; re-running the clip reproduces it.

**AI edit — "un-flip every inverted body in the clip."**
Toolbar **[AI ▸]** on POSE → the model calls `list_stages → get_io(pose, f) → observe → edit_output
(orient) → rerun(pose, subject) → observe → diff`, iterating per subject until the OBSERVE snapshots
read upright. Every AI edit is an `ai:<model>`-stamped override you can inspect, keep, or pop — the
model has the *same* action surface you do, nothing hidden.

## 8. AI / model editing — ADR-0008 realised

The requirement "edit input/output manually **or by AI**" is satisfied by one move: **the action API
(§3.4) is the MCP tool surface.** No parallel "AI path" — the model drives the exact endpoints the
buttons do. The loop:

```
        ┌──────────────── LLM (over MCP) ────────────────┐
        │ get_io(stage,f) → decide → edit_* / set_param  │
        │        → rerun(from_stage) → observe(frames)    │
        │        → read rendered overlay → judge → repeat │
        └───────────────▲───────────────────┬────────────┘
                        │ images            │ overrides (ai:<model>, confidence<1)
                   OBSERVE stage        overrides/<stage>.json  ── resolved into re-run
                (stages.py:38)                 │
                                        pipeline re-run (cache-miss only where changed)
```

- **Visual feedback** is `Stage.OBSERVE` (already reserved, `stages.py:38`) — it renders the per-frame
  overlay/multi-view the model needs to *see* its own edit, closing the loop the memory calls the
  "LLM-automation north-star."
- **Per-stage AI recipes** (from §5): detect→"raise recall in this ROI"; track→"re-cluster teams";
  calibrate→"refine keypoints from these clicks"; pose→"un-flip / re-pose backend B"; physics→"auto-tune
  ceilings from p95". Each is one `edit_*`/`set_param` + `rerun` + `observe`.
- **Honesty guardrail:** AI overrides are stamped `source: ai:<model>`, `confidence < 1.0`, and rendered
  in a distinct colour, so they never silently become ground truth (R-6). The human keeps veto (pop the
  row). This is the "auto-detect + manual override" rule as a UI invariant.

---

# PART II — IMPLEMENTATION PLAN

Dependency-ordered. Each phase is **independently shippable** and delivers eye-value before the next.
Ordered so the **highest debugging leverage lands first** (read-only glass box), and **cheap
interactive stages precede expensive ones** (tune coherence/physics/calib live before wiring GPU
re-runs). "Files" lists the primary touch-points.

### Phase 0 — Materialise the run (read-only glass box) · *highest value, lowest risk*
**Deliver:** run the pipeline once; every stage's input+output is dumped to a browsable bundle and
served per-frame. No editing yet — but you can finally *see* each step. This alone kills the black box.
- Implement **`DiskCache`** adapter (`adapters/cache/disk.py`) satisfying `core/ports/cache.py` — the
  key already exists (`content_key`); persist `cache.put` artifacts + write `manifest.json` of
  `StageRun`s. Wire it as the pipeline's cache (swap the memory fake).
- Bring `STITCH/IDENTITY/COHERENCE/PHYSICS` outputs into the bundle as artifacts (they run outside
  `run_cached` today — snapshot their results + reports where they execute in `pipeline.run` /
  `controller.run_reconstruction`).
- Backend: `GET /api/studio/stages` (manifest), `GET /api/studio/{stage}/io/{frame}` (typed I/O).
- **Risk:** artifact serialization for heterogeneous types (boxes vs homographies vs SMPL-X) — reuse
  `core/scene/serialization.py`'s tagged `__ndarray__/__enum__/__type__`.

### Phase 1 — Stage rail + read-only per-stage inspectors
**Deliver:** the rail UI; click any stage → see IN/OUT on the frame per-frame, for every stage.
- Frontend: stage rail component; `StageView` registry; **layer renderers** (boxes, trails, homography
  grid, ball marker, confidence ribbon) plugged into the existing `rebuildOverlay`. POSE reuses the
  three.js editor and 2D skeleton as-is.
- Re-role the grid (§6); stage-scoped timeline colouring from the existing confidence map + `/api/edits`.
- **Risk:** none structural — pure additive UI over Phase-0 endpoints. Reuses zoom/pan/projection whole.

### Phase 2 — Params editor + re-run engine for the *instant* stages
**Deliver:** live-tune `CALIBRATE, BALL-3D, COHERENCE, PHYSICS` params, auto-re-run (pure half, <1s),
DIFF wipe. This is where interactive experimentation first feels magical, at zero GPU cost.
- Backend: `set_param` writing a `params` override; **invalidation walk** over the DAG using
  `content_key` (miss ⇒ re-run downstream); re-run the *pure* tail in-process (coherence/physics already
  run in `controller.run_reconstruction`). `GET /api/studio/diff`.
- Frontend: params drawer bound to the real config schemas (`StitchConfig`, `CoherenceConfig`,
  `config/physics.yaml`, calibration cfg); DIFF slider.
- **Risk:** getting invalidation right — but it's just "re-key and check `cache.has`", which the design
  already guarantees is deterministic across processes.

### Phase 3 — Generalised output editing (the `Correction` pattern → all stages)
**Deliver:** edit any stage's *output* manually; persisted, undoable, resolved. POSE joint edit already
does this — widen it.
- Backend: extend `edits.py`/`edit_actions` to `side:"output"` ops per stage (add/move/delete box,
  reassign id, toggle on_ground, mark teleport…), each a provenance-stamped override row.
- Reuse `apply_and_persist_edit` → `rebuild_subject_cache` pattern (`poseannot/scene_state.py`) per
  stage. Timeline turns orange (exists).
- **Risk:** low — this is the shipped pose-edit path generalised.

### Phase 4 — Input editing + heavy re-runs (async jobs)
**Deliver:** edit a stage's *input* and re-run downstream through GPU stages (draw box → re-track →
re-pose; drag calib points → re-homography → re-pose).
- Implement a **real `JobQueue`** adapter (`adapters/jobs/threaded.py` first, process-pool later) —
  port + `JobState` exist; add progress + `job_status` polling. In-process on the pod, under the
  pipeline venv (the `--real-calib` precedent).
- Backend: `edit_input` overrides resolved into the stage thunk; `rerun(from_stage, subjects?)`
  scoped/partial (re-pose only the touched subject — pose is per-subject).
- Frontend: async progress on the rail (⟳), `rerun_cost`-aware buttons.
- **Risk (real):** GPU cost/latency — mitigated by **per-subject** scope, the content cache (only the
  changed path re-runs), and the page-cache prewarm already in `scripts/pod_ab_video.sh`. Be honest in
  the UI: re-posing is minutes, not interactive.

### Phase 5 — AI / MCP action surface + visual feedback loop
**Deliver:** the model drives the same actions with OBSERVE feedback (§8).
- Expose the action API as **MCP tools** (thin wrappers over the Phase 2–4 endpoints).
- Implement `Stage.OBSERVE` rendering (per-frame overlay / multi-view snapshot to `observe/`).
- Provenance/confidence stamping + distinct render colour for `ai:*` overrides.
- **Risk:** scope creep — keep the tool surface == the button surface; no bespoke AI logic server-side.

### Phase 6 — Branching + compare (any-stage A/B)
**Deliver:** `branch(from_stage)`; side-by-side/wipe compare of two branches on identical pixels —
the pose bake-off, generalised. Backlog item in `docs/poseannot-roadmap.md` ("Backend A/B compare")
falls out here for free.
- Backend: branch = new `overrides` file + shared content cache; compare reads two run_ids.
- Frontend: compare view (reuse the DIFF wipe; add split view + a branch switcher on the rail).

### Cross-cutting concerns
- **Re-run cost is the central UX constraint.** Bucket stages by `rerun_cost`; auto-run the instant
  ones, gate the gpu-minute ones behind explicit, scoped, async jobs. Never block the UI (R-10; the
  `JobQueue` port exists for exactly this).
- **Storage.** The content cache dedupes by hash across branches; a run bundle for the 60-frame clip is
  dominated by POSE meshes — reuse the FK-cache-on-demand trick (`scene_state.py`) rather than storing
  all verts. Keep artifacts, GC old branches.
- **Honesty (R-6) is load-bearing, not decoration.** Every override carries source + confidence; the UI
  colours measured vs filled vs extrapolated vs manual vs AI distinctly (the confidence map already
  exists). This is what makes the tool trustworthy for *judging* reconstructions, per the "honest
  status vs the GOAL" and "overlay: user is ground truth" disciplines.
- **Auth/deploy unchanged** — same uvicorn-on-pod, JWT, clip bundle model (`docs/poseannot-architecture.md`).
- **Supersedes** roadmap item `#45` ("raw-video → frame-range → auto scene.json — pipeline behind GUI")
  and folds in the two `poseannot-roadmap.md` bake-off backlog items.

### Suggested build order recap
`Phase 0 (see everything)` → `1 (rail + inspect)` → `2 (tune the free stages live)` →
`3 (edit outputs)` → `4 (edit inputs + GPU re-run)` → `5 (AI loop)` → `6 (branch/compare)`.
Ship and get eyes on it after **every** phase; Phase 0–2 already deliver a debuggable glass box with
zero GPU cost.

---

## Appendix — stage → code seam (where each re-run hooks in)

| Stage | Runs today in | Re-run entry for the Studio |
|---|---|---|
| Detect/Track/Calibrate/Pose/Ball/Assemble | `pipeline.run` via `run_cached` (`stages.py:78`) | swap `DiskCache`; re-submit the stage thunk on miss |
| Stitch/Identity | `pipeline.run` (`continuity.py`) | wrap as a `StageView`; snapshot + re-invoke in place |
| Coherence/Kinematic/Physics gates | `controller.run_reconstruction:203-292` (emit `Correction`s) | already layered/toggleable — re-resolve is instant |
| Resolve/Export | `controller.resolved` / `controller.export` | `resolve_export()` action |
| anim_export/Blender/ffmpeg | `scripts/*.py` | wrap as `whole_clip` StageViews (Phase 6+) |
| OBSERVE | reserved (`stages.py:38`) | Phase 5 — render snapshots for the AI loop |

**Verified against code 2026-07-09.** The load-bearing claim — that the pipeline is already a
content-addressable, cache-keyed, job-queued DAG with a reserved LLM-feedback stage and a layered
correction model — is checked in `stages.py`, `ports/cache.py`, `ports/jobs.py`, and `correction/`.
