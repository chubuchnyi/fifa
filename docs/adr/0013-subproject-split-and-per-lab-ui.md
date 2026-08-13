# ADR-0013 — Split by measurable question; every subproject owns its own UI

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** user + architecture
- **Related:** ADR-0001 (hexagonal core), ADR-0006 (swappable providers + provenance),
  ADR-0008 (visual feedback), ADR-0012 (rejected approaches); #140, #141, #135, #138

## Context

### The evidence that started this

`~/camlab` was cut out of this repo on **2026-08-11**. Three days later it holds 9635 lines,
89 tests that run in 6 s, and it answers its one question on every clip we own:

| clip | across | markings/frame | frames under 20 px |
|---|---|---|---|
| `fan` — 1080×608, phone from the stands | **1.82 px** | 6 | 120/120 |
| `broadcast` — 1920×1080 | **2.96 px** | 7 | 60/60 |
| `CRO_MOR_194948` — one hand anchor | 4.22 px | 9 | 120/120 |

No GPU, no neural network, no checkpoint: SIFT + MAGSAC, distance transform + LSD,
`scipy.optimize`, a k-d tree. 155 s for 60 frames at 1920×1080 on a laptop CPU.

Inside AVATAR the same question ran from **#61 → #119 → #140 → #141**. It was closed by the user's
eye on 2026-08-03 and *9 scenes out of 9 still carried an invented `fx = 772 @ 1280×720`*, because
`pod_real_e2e.sh` never called `apply_rigid_camera.py`. The capability existed, was tested, was
documented, and did not reach the run — the #141 defect.

So the split is not a preference. It is the only intervention on this problem that produced a
number that moved.

### Why camlab worked, stated precisely

Not "the camera is a separate stage". Because the question has a **check against the raw video
alone**: project the pitch model through the camera and the lines must land on the painted ones.
No scene, no poses, no render, no labels. camlab can be wrong and find out by itself — and it has,
repeatedly (a metric that could not report an error above 40 px, a "not radial" argument computed
about the wrong centre, a pincushion signature that was seven samples of noise).

### The UI observation (user, 2026-08-13)

`poseannot/` is **7380 lines, 2613 of them in one `index.html`, 38 endpoints**, serving pose
annotation *and* pitch-layout calibration *and* a 3D world scrubber *and* stage rerun *and* clip
upload *and* auth from one page. The user's verdict: **изжил себя, устарел, не информативен и
неудобен.**

camlab's viewer is **2730 lines total** — `app.py` 744, `index.html` 1060, `pitch_view.js` 697,
`style.css` 228 — for one question, and it is load-bearing rather than decorative:
`CRO_MOR_194948` is *the first clip solved from an operator's own anchor placed in that viewer*,
and getting there took three fixes in one day that only a person dragging the camera could have
found (the solver did not read hand edits; the anchor refit was position-locked; the centreline
extractor was not a thinning algorithm).

A UI built for one question is an instrument. A UI shared across six is a page nobody can change.

## Decision

### 1. Split by question-with-a-check, not by stage and not by module

A part earns its own repository when **its answer can be falsified against the raw video without
the rest of the pipeline**. A part that can only be judged from the final render is *assembly*, and
assembly stays in AVATAR.

### 2. The split as it stands

| | question | its own check | contract in → out | hardware |
|---|---|---|---|---|
| **camlab** (exists) | where was the camera, where pointed, what focal, per frame | projected markings vs the paint; mowing-stripe period under zoom | clip → `calib/<clip>.npz` schema 2 | CPU |
| **playerlab** (next) | which human is this, and where exactly does he stand | label-free duplicate/continuity criterion; silhouette + foot reprojection **through camlab's camera** | clip + camera npz + cached detections/poses → `tracks/<clip>.json` | CPU |
| **AVATAR** | what does it look like | the user's eye, and nothing else | both artefacts → `scene.json` → Blender → generative tail | GPU / pod |

**playerlab is one repository, not two.** #135 measured it: *"Every defect the user listed is
association or placement — none is a wrong pose on a correct crop."* Pose-on-a-good-crop is a black
box we call (SMPLest-X); association and placement are one metric seen from two sides — does the
posed body lie on its own pixels in every frame. camlab's 2 px camera is what first makes that
metric mean anything; before it, camera error swamped placement error.

> ⚠ **Amended 2026-08-13, the same day.** The #135 claim this paragraph rests on is **refuted**:
> t19 is `measured` on every frame f44–57 through a header, on a good crop, and the pose is flat —
> peak root-Z 0.837 m, ~9 cm above his own baseline against a real jump's ~40 cm. So
> pose-on-a-good-crop is not a black box that works; it has a measured failure mode. Whether that
> failure is *grounding* (already playerlab's) or the *pose backend* (outside it) is separable by
> measurement and **has not been separated** — see [`playerlab-spec.md`](../playerlab-spec.md) §7.
> The one-repository decision stands provisionally; the boundary is what is in question.

### 3. Every subproject owns its UI, and the UIs are deliberately not shared

This is the part that overrides ordinary good practice, and it is a user decision.

1. **Each lab builds the UI its own question needs**, tuned to judging and debugging *that*
   answer by eye. camlab needs the pitch drawn on the paint plus a drag/rotation gizmo and a
   per-frame verdict colour. playerlab needs per-track crop strips, the measured/imputed
   provenance timeline, and candidate duplicate pairs side by side. Nothing in those two overlaps.
2. **No shared UI library, no shared viewer, no shared component set between labs.** A lab may
   *copy* a file with an origin header (rule 5), never import one.
3. **AVATAR's UI is frozen, not refactored.** Keep `poseannot` running; spend nothing on improving
   it. It will be rebuilt later **from the UX the lab UIs prove works** — the labs are the research
   that the rebuild will be based on. Any proposal to "clean up `poseannot` first" is rejected by
   this ADR.
4. Cost accepted knowingly: three three.js viewers, three vendored copies, three sets of controls.
   In exchange each is ~2–3 k lines, one person can change one in an afternoon, and none of them
   grows into 2613 lines of accreted HTML.

### 4. The contract between parts is a file with a schema, not a Python import

The parent is **changed to fit the new schema**; no compatibility branch. Precedent: camlab's
schema 2 writes `focal_px` and `position` per frame, and `read_npz` refuses schema 1 *by name*
rather than guessing — collapsing to schema 1 costs 65 % of the accuracy on the fan clip
(1.69 px → 4.88 px, five frames of thirty leaving the 20 px band).

Every artefact carries its own **provenance inside the file** (`CameraTrack.source` is the first
field that does). AVATAR **refuses to run** on an artefact whose origin it cannot name, instead of
falling back with an `or`.

### 5. Copy, do not depend

Each lab installs, tests, runs and deploys with no other lab present. Every copied file carries a
header saying where it came from. **Do not hand-sync copies back.** Where a copy must not drift,
pin it with a golden test against the same real measurement both sides pin (camlab and AVATAR both
pin focal 4169.32 px, one optical centre over 60 frames, camera at (−2.29, −70.13, 17.22) m).

### 6. Each lab carries its own discipline files

`docs/STATUS.md` (what is true now), `docs/PROBLEM.md` (the question for someone who has not seen
the repo), `docs/findings/` (measurements *including the refutations*), and
`docs/findings/landmines.md` as the one place traps go. This travelled with camlab and is why its
retractions are visible rather than lost.

### 7. CPU-first wherever the question allows it

camlab's iteration speed is the absence of weights and GPU, not clever code. playerlab inherits
this by consuming **cached** detections and poses (4362 detections already on disk; the association
benches run in ~3 min on CPU).

### What is explicitly NOT split

- **Detection from tracking** — detections are a cached input; the boundary has no check of its own.
- **The paint detector from camlab** — its only consumer is the camera solve, and its error is
  already measured inside camlab (`worst spot` is 7.9× `across` on `fan` largely because of it).
  AVATAR's `scripts/detect_markings.py` is a duplicate of this and should collapse into camlab.
- **Render from the generative tail** — neither has a check; both are judged by eye.
- **Appearance (kit, numbers, texture)** — blocked by resolution, not by focus: the median player
  box is 28 × 72 px and #109 measured 0 of 23 crops usable for OCR. Revisit when that changes.

## Consequences

**Positive**

- One question at a time, with a number that can move without a pod run.
- The loop is minutes on a laptop instead of a GPU batch.
- Each UI stays small enough to be an instrument rather than a legacy page.
- A lab that is wrong finds out on its own evidence; camlab has retracted six of its own results.

**Negative / costs**

- **Duplication is real and paid for.** camlab already copies the pitch model, camera types and
  projection. playerlab will copy more. Three days to a 2 px camera on every clip is what it bought.
- Three UIs, three vendored three.js copies, three deploy paths.
- **#141 gets structurally worse.** Four instances of "a capability never reaches the run" happened
  inside *one* repository in two days. With three, the failure mode is cheaper to hit. Therefore
  these are obligations, not suggestions:
  1. every artefact records its provenance in the file;
  2. AVATAR refuses an unattributable artefact rather than substituting a default;
  3. AVATAR's only job is the end-to-end run, and it must consume a fresh export from each lab
     **within days**, not months. A lab at its local optimum that nothing consumes is the same
     defect as `fx = 772`.

## Alternatives considered

- **One repo with stricter module boundaries** — rejected on measurement. The camera question sat
  inside the monolith from #61 to #141, was declared closed by eye, and still shipped an invented
  focal in every scene. The same question in a separate repo was answered on all clips in three days.
- **Split, but share one UI or component library across labs** — rejected by the user 2026-08-13. A
  shared UI is exactly the process that produced the current 2613-line page: each new question adds
  a panel, none can be removed, and the result serves no question well.
- **Keep `poseannot` as a shared annotator service the labs call** — rejected: same accretion, plus
  it couples the labs' release cycles to it.
- **Split by pipeline stage (detect / track / pose / render as separate repos)** — rejected: stages
  without their own check (pose on a good crop, render) gain nothing from a boundary and lose the
  ability to be measured across it.
- **Refactor AVATAR's UI first, then split** — rejected: it would be a rewrite against requirements
  we have not learned yet. The labs are how we learn them.

## Checklist for cutting the next lab

1. State the question in one sentence, and the check that falsifies it **from the clip alone**.
   If there is no such check, stop — it is assembly.
2. `docs/PROBLEM.md` + `docs/STATUS.md` + `docs/findings/landmines.md` on day one.
3. Copy what is needed with origin headers; add the shared golden test that pins the copy.
4. Define the output artefact and its schema **before** the solver; give it a provenance field.
5. Build its own UI, for its own question, from scratch.
6. Wire the artefact into an AVATAR end-to-end run before the lab's first result is believed.
