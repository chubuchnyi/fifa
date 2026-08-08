# Pipeline I/O — proposed changes

Same document as [`pipeline-io.md`](pipeline-io.md), with the changes marked. Read that one
first; this only lists what moves.

Marks: **▲ CHANGE** an existing contract · **＋ NEW** something that does not exist ·
**◆ MEASURE** a diagnostic, no contract change.

Nothing here is built.

> ## ⚠ Rewritten 2026-08-08 (2). The first version's premise was wrong.
>
> It reasoned from *"the overlay drifts worse toward the frame edges, so the camera model is
> missing a parameter"* and spent four steps choosing which parameter. Measured since:
>
> **Every scene we have — nine of nine — carries the synthetic 772 px fallback, not a measured
> camera.** The observation was made through a camera that is wrong everywhere, at 5.4× the real
> focal and the wrong image size. There was no camera model applied to argue about.
>
> Full reasoning: [`findings/reply-camera-model-gap-2026-08-08.md`](findings/reply-camera-model-gap-2026-08-08.md).
> Board entry: **#140**.

---

## What is actually wrong, measured

| | |
|---|---|
| `scene.camera` on all 9 scenes on disk | `fx = 772.02 @ 1280×720`, `distortion = None` |
| the clip | 1920×1080, real focal ≈ 4200 |
| `camera_from_calibration` on the scene's own calibration | `focal_px = 4340.8` (sane), `reprojection_px = 471.1`, `realizable = False` |
| the calibration itself | 236/236 frames solved, median confidence 0.472, **no frame at 0.0** |

`controller.run_reconstruction` ends with `scene.camera = _measured_camera(...) or
_static_camera(...)`. The refusal is correct and deliberate (#61: a scene once carried two cameras
12686 px apart for months). **The silent substitution is the defect**, and `PlaneCameraFit` — the
only record of which camera you got — is held in memory and never serialized.

**Why it is not zoom, on our clip.** Reprojection is invariant to window length: 467 px at 30
frames, 480 at 60, 471 at 236. At 30 frames the drift is 0.8 %, which cannot produce 467 px.
Confidence filtering does not help either — 453 px on the top 10 %, the third confirmation of #126.

**Where the error lives.** Against the golden fit over frames 0–59: median **0.63 m**, p90
**3.22 m**, max **18.58 m**. Tail-dominated, and the tail sits at the far end of the pitch — the
top and edges of the frame. That is the original observation, and it is in the **calibration**, not
in the camera model's parameter count.

**The clean control:** `calib/*.npz` was produced by a *different* fitter
(`scripts/fit_rigid_camera.py`) on the same clip and succeeded, with one focal and a
mutation-checked camera position.

---

## What real broadcast cameras do — measured by the review, and it stands

From 89 WorldPose GT cameras ([`findings/camera-model-gap-2026-08-08.md`](findings/camera-model-gap-2026-08-08.md),
reproduce with `scripts/bench_camera_model_gap.py`):

| | measured | consequence for this plan |
|---|---|---|
| focal drift within a clip | **>5 % in 89/89**, median 44 % | per-frame focal is the dominant model term |
| camera translation | **0.000 m in 89/89** | ~~hypothesis~~ **deleted** |
| principal-point wander | ~1 px median | ~~step 2c~~ **deleted** |
| distortion at the corner | ~47 px | keep, second |

**Confirmed on our clip:** the focal runs 4083 → 4556 across windows, ~11 %, in line with the
review's 9.2 % median at this window size. So the model term is real — it is just not what breaks
the fit here.

---

## Step 0 ◆ MEASURE — re-run with the measured camera

**Nothing to build. It may dissolve everything below.**

`RIGID_CAMERA` (#129) appears nowhere in `scripts/pod_real_e2e.sh` and zero times in the 236-frame
run log, so every scene under discussion was built without it. On frames **0–59**, where
`calib/Colombia-1-0-Congo-DR1080p.npz` is valid and the golden fit holds, this is one run.

Then look at the overlay again. If it aligns, the whole camera-model discussion was about a model
that was never applied.

---

## Step 1 ◆ MEASURE — three residuals, not one

The first version binned **only the pitch-paint residual, by radius**. Two problems: the goal is
players, not paint; and a radius profile cannot distinguish a camera missing a parameter from a
camera that is absent — it reads "grows with radius" either way, and would have sent me to fit
distortion on a model that never ran.

| residual | source | what it separates |
|---|---|---|
| pitch paint vs drawn line | `pitch_evidence.classify(uv, …)` — exists, returns per-point distance | camera on the ground plane |
| **subject foot vs detector box bottom** | projected root + cached detections — both exist | camera vs per-player depth |
| **common-mode vs per-player scatter** | the same two arrays | all subjects displaced alike ⇒ camera; scattered ⇒ grounding or association |

The second and third were in the first version's *step 5*, costed at "ten lines" and ranked behind
days of UI work. That was the worst call in the document: they are the only test that separates the
two error classes.

**Bin only frames whose confidence clears `MIN_SOLVED_CONFIDENCE = 0.02`, and report the drop
count.** Confidence exactly 0.0 means the frame was never solved and its homography is a copy
carried from the last good one — that drifts with *time*, not radius, and mixing those frames in
manufactures the very signal being tested for. (On this clip no frame is at 0.0; on the vertical
clip it was 43 % of 355.)

---

## Step 2 ＋ NEW — why is the pipeline's calibration worse than `fit_rigid_camera.py`'s?

Same clip, same frames 0–59, two calibrations: one reduces to a single camera with a
mutation-checked position, the other reprojects at 480 px and is refused. **That gap is the binding
constraint, and no schema change touches it.**

| | |
|---|---|
| ◆ compare | the two homography sets point-by-point on the pitch — done once, median 0.63 m / max 18.6 m. Repeat per frame and per image region to locate the tail |
| ◆ check | different PnLCalib invocation, different post-processing, or a different world convention |
| ＋ likely outcome | a **robust** fit — reject outlier frames or regions — rather than a new parameter |

---

## Step 3 ▲ CHANGE — per-frame focal, then `k1`

Only after step 2, and now on a mechanism argument rather than a frequency one.

`camera_from_calibration` does not degrade when one focal cannot fit — it **refuses**, and the
fallback is silent. So on a clip that zooms enough, the single-focal model yields **no camera at
all**. That makes per-frame focal not an accuracy improvement but **the precondition for a measured
camera existing**.

### 3a. Per-frame focal — schema change

| | |
|---|---|
| `CameraTrack` | ▲ `intrinsics: CameraIntrinsics` (one, shared) → per-frame. Its own docstring already calls this a future refinement and says core need not change |
| `calib/<clip>.npz` | ▲ `focal` scalar → `(T,)` |
| `tests/e2e/test_golden_real_camera.py` | ▲ pins `focal 4169.32` and one optical centre for 60 frames. **Re-measure, never nudge** — it is mutation-checked and the only real measurement in the suite |
| consumers | ▲ `poseannot/camera.py`, `anim_export.py` `cameras.npz`, Blender |

### 3b. Distortion `k1` — no schema change

`CameraIntrinsics.distortion` **already exists** and is `None` on every solve we produce.

| | |
|---|---|
| `scripts/fit_rigid_camera.py` | ＋ one parameter in the optimiser |
| `calib/<clip>.npz` | ＋ key `dist: (k,)`; the writer already emits `width`/`height` that the current file predates |
| every projector | ▲ four must stay in step or overlay and export diverge again: `core/scene/projection.py`, `poseannot/camera.py`, `scripts/apply_rigid_camera.py`, `scripts/track_quality.py` |

Keeping those four in step is the real cost, not the optimiser change.

### ~~3c. Free principal point~~ · ~~camera translation~~

**Deleted.** 1 px median wander, and 0.000 m in 89/89 clips.

---

## Step 4 ＋ NEW — serialize which camera you got

The cheapest change in this document, and it would have caught #140 on day one.

| | |
|---|---|
| ▲ `Scene` or `CameraTrack` | ＋ carry `PlaneCameraFit` (`focal_px`, `reprojection_px`, `realizable`) into `scene.json` |
| ＋ any consumer comparing a scene to source pixels | refuse, or say so loudly, when the camera is synthetic |

Today the two are indistinguishable on disk. `track_quality.py` detects it by testing `fx ≈ 772` —
a magic number, because there is nothing else to test.

---

## Step 5 ＋ NEW — verticality and foot contact

Neither version of this document mentioned it, and the goal is "positions **and poses**".

#135 П5 measured the largest root-Z excursion **in a whole scene at 0.082 m** — nobody ever leaves
the ground. WorldPose GT says a real player's root ranges **0.23 m** per clip at the median and
rises **0.67 m** at p99 inside half a second. No camera fix touches this.

---

## Step 6 ▲ CHANGE — WorldPose, and ~~synthetic~~

**Correction to the first version:** it said the WorldPose video "is an agreement with FIFA, not
annotation work". **Wrong — 124 video files, 24 GB, at `models/worldpose/`.** All 89 clips have
footage *and* GT poses (our exact `PoseSequence` schema) *and* GT cameras with per-frame `K`, `R`,
`t` and five distortion coefficients.

~~The first version's step 3, `eval/synthetic_calib.py`~~ — **retired entirely.** WorldPose
measures the calibrator on real footage with a known answer and no domain gap, and measures pose
too. Synthetic keeps one narrow use: injecting a distortion or zoom profile WorldPose does not
contain, to test a solver's range. A follow-up, not a step.

---

## Step 7 ＋ NEW — export corrections as a training set

Unchanged, including the trap, which the review confirmed:

`edits.json` holds a **delta** (`PlaneTransformPayload.matrix`); a trainer needs the **absolute**
per-frame answer. And human corrections are made *through* the current camera — if that camera is
wrong, training on them teaches the model to reproduce the error. **After step 3, never before.**

With #140 known this is doubly true: corrections made on a 772 px scene are worthless as data.

---

## Step 8 ＋ NEW — controls in the browser, last

Today's calibration UI is a 4-DOF similarity on the pitch plane. Four degrees of freedom cannot fix
a wrong focal, a wrong tilt, distortion — or a camera that is not there. Point-correspondence
dragging (≥4 known pitch landmarks → homography → focal and pose) is the right control, and it is
also exactly the format step 7 exports.

Worth building only for what steps 0–3 leave behind.

---

## Order and cost

| step | cost | what it buys |
|---|---|---|
| 0 ◆ re-run with the measured camera | one run, no new code | may dissolve the premise entirely |
| 1 ◆ three residuals | half a day | camera vs player, which nothing else answers |
| 2 ◆ why our calibration is worse | ~1 day | the binding constraint, measured |
| 3a ▲ per-frame focal | days, re-measures a golden test | the precondition for a measured camera on a zooming clip |
| 3b ▲ distortion `k1` | ~1 day, four projectors in sync | ~47 px at the corner |
| 4 ＋ serialize the camera fit | hours | would have caught #140 on day one |
| 5 ＋ verticality | unscoped | the pose half of the goal, untouched by everything above |
| 6 ▲ WorldPose | already local | GT for calibrator and pose, no domain gap |
| 7 ＋ training-set export | ~1 day | after 3 |
| 8 ＋ UI controls | days | last |
