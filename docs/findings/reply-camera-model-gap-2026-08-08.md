# Reply to the camera-model review

Answer to [`review-pipeline-io-2026-08-08.md`](review-pipeline-io-2026-08-08.md) and its evidence
[`camera-model-gap-2026-08-08.md`](camera-model-gap-2026-08-08.md), from the author of
`pipeline-io.md` / `pipeline-io-proposed.md`. Written 2026-08-08.

**Verdict, after measuring rather than arguing:**

1. The review is right on every point it makes, and its zoom argument is stronger than it made it.
2. **Our clip does zoom** — ~11 % across 236 frames, confirming its WorldPose statistic transfers.
3. **Zoom is nevertheless not the binding constraint here.** The one-camera fit refuses just as
   hard on a 30-frame window, where drift is 0.8 %.
4. **The premise both documents were built on is not established.** The scene the observation was
   made on — and *every other scene we have*, nine of nine, including the reference scene for the
   #135 eye labels — carries the synthetic 772 px fallback, not a measured camera.
5. The dominant error is a **tail in the per-frame homographies**, not a missing parameter in the
   camera model. Per-frame focal and `k1` cannot repair it.

---

## 1. Conceded, without qualification

| review's point | status |
|---|---|
| WorldPose video is on disk, 24 GB, 89 clips with footage + GT poses + GT cameras | **verified.** 124 video files under `models/worldpose/`. My claim was wrong |
| Camera translation: 0.000 m in 89/89 → delete from the plan | accepted |
| Free principal point: 1 px median wander → delete from the plan | accepted |
| Distortion ~47 px at the corner is real | accepted, and it refutes the review's own optics objection, which it says |
| Step 1 measures pitch paint only; the goal is players. Add the foot-vs-box residual and the common-mode split | accepted, and see §4 — this was my worst error |
| Verticality (#135 П5, largest root-Z excursion in a whole scene **0.082 m**) is in neither document and belongs in the plan | accepted |
| WorldPose replaces synthetic calibration GT | accepted, and more completely than the review claims — see §5 |
| Check the camera's domain of validity first, "minutes of work" | accepted, and it is the single most valuable instruction in the review — see §3 |

**On the WorldPose error specifically.** My memory file on this dataset carried the sentence
*"check the disk before repeating either availability claim — it has now flipped twice"*. It had
flipped a third time and I repeated the stale claim without running `du -sh`. The review is also
right that the same claim sat in `pose-bakeoff-runbook.md` until 2026-08-07. That is a class of
error — inheriting a blocker instead of testing it — and the fix is a command, not a resolution.
The memory now carries the command.

---

## 2. The finding that reframes both documents

Both documents reason from one observation: *the drawn pitch does not sit on the painted pitch and
skeletons do not sit on players, worsening toward the frame edges.* Both then ask which parameter
the camera model is missing.

Measured on the scene the observation was actually made on, `out/res_ab236/f236_res896.json`:

```
scene.camera:  fx = 772.0  cx,cy = 640,360  @ 1280×720   distortion = None
the clip:                                      1920×1080, real focal ≈ 4200
```

**That is the invented fallback**, not a measured camera — 5.4× wrong in focal, and at the wrong
image size. `pipeline-io.md` §4 documents it as a hazard; it was live in the scene under review.

Why it is there, also measured:

```
camera_from_calibration(scene.field.calibration, 1920, 1080)
  focal_px         = 4340.8      # sane — within 4 % of the golden 4169.32
  reprojection_px  = 471.06
  realizable       = False  ->  camera = None
```

The field calibration itself is **fine**: 236 of 236 frames solved, median confidence 0.472, not a
single frame at the 0.0 "carried homography" value. PnLCalib worked. What failed is the reduction
of its free per-frame homographies to one camera, at 471 px of reprojection — so
`_measured_camera` correctly refused, and `scene.camera = ... or self._static_camera(scene)`
silently substituted a synthetic broadcast viewpoint.

The refusal is right and is there on purpose (#61: a scene once carried two cameras 12686 px apart
for months). **The substitution is the defect** — downstream, a synthetic camera is
indistinguishable from a measured one unless the caller checks `camera_fit()`, that fit is **held
in memory and never serialized**, and nothing in the viewer checks it.

### And it is not one scene — it is every scene we have

| scene | fx | size |
|---|---|---|
| `out/cue/scene_off.json` — **the reference scene the 24 eye labels were made on** | 772.0 | 1280×720 |
| `out/cue/scene_on.json` | 772.0 | 1280×720 |
| `out/res_ab/res{560,896,896_handover}.json` | 772.0 | 1280×720 |
| `out/res_ab236/f236_res{560,896,896_handover}.json` | 772.0 | 1280×720 |
| `out/vert137/scene.json` | 772.0 | 1280×720 |

**Nine of nine.** Every eye verdict, every criteria score, every A/B in this thread was made on a
scene with no measured camera. Two things follow, and they point in opposite directions:

- The verdicts that compare a scene **to the source pixels** — overlay alignment, pitch markings,
  skeletons on players — were never testing what we thought. That includes the observation that
  started both of these documents.
- The verdicts that compare a scene **to itself or to another scene** — identity churn, phantom
  counts, handover pairs, root-speed distributions, and the eye's ranking of 560 vs 896 vs
  896+handover — are untouched. They never used `scene.camera`. `scripts/track_quality.py`
  requires `--camera` for exactly this reason, and that is the only reason its numbers mean
  anything.

**Consequence for both documents.** The edge-worsening observation was made through a camera that
is wrong everywhere, not through a camera missing a parameter. Neither my ordering of the three
hypotheses nor the review's inversion of it is yet justified *on this clip*, because on this clip
no camera model was applied. The review's step 0 instinct was correct and the answer is worse than
it guessed: not "frame 236 is outside the npz's 0–59 span" — the scene never used the npz.

---

## 3. Where the review's argument is stronger than it made it

The review argues zoom on **frequency**: focal moves >5 % in 89/89 clips, so it beats distortion,
which is 1st in my plan.

The stronger argument is **mechanism**, and it comes out of §2. `camera_from_calibration` does not
degrade when one focal cannot fit — it **refuses**, and the fallback is silent. So on any clip
where the operator zooms enough, the single-focal model does not produce a slightly wrong camera;
it produces **no camera**, and the scene quietly gets a 772 px stand-in.

That moves per-frame focal from *"days of work, breaks a golden test, only if the operator
zoomed"* (my step 2b, ranked last) to **the precondition for a measured camera existing at all**.

### And the review's numbers predict our observation

Its §3 and §4 give focal drift by window and the position error it causes. Running them together
against what we measured:

| window | review's median drift | its position error at 40–60 m | as pixels at f = 4169 | measured here |
|---|---|---|---|---|
| 60 frames | 2.0 % | 0.4–0.8 m | ~40–80 px | golden fit **succeeds**, one focal, 4169.32 px |
| 236 frames | ~9 % | 3.3–5.0 m | ~340 px | fit **refuses**, reprojection **471 px** |

A single-focal model that fits at 60 frames and fails at 236, with the failure the right order of
magnitude for the review's own drift figures.

> ⚠ **§6 disproves the reading of this table.** The two rows compare *different calibrations* —
> the golden npz's and the scene's — not two window lengths of the same one. Running the scene's
> own calibration at 30, 60, 120 and 236 frames gives 467, 480, 476 and 471 px: **flat**. The
> window is not the variable. The zoom figure still transfers (our clip drifts ~11 %); the
> inference drawn here from it does not. Left in place because it is the reasoning the measurement
> was written to test.

---

## 4. Where I was wrong beyond what the review caught

It notes that the "is it the camera or one player" test sits in my step 5, costed at "ten lines"
and ranked behind days of UI work. That is worse than a mis-ordering: **it is the only test in
either document that separates the two error classes**, I priced it correctly, and I still put it
last. It moves to step 1.

Second: I wrote step 1 to bin the residual by *radius* only. Radius separates focal and distortion
from extrinsics. It does **not** separate them from a camera that is simply absent — which is the
actual state of this scene, and which a radius profile would have reported as "grows with radius",
sending me straight to fitting distortion on a model that was never used.

---

## 5. One place the review understates itself

It proposes WorldPose instead of synthetic for **pose**, keeping synthetic for calibration on the
grounds that a rendered white line transfers. But WorldPose carries **five GT distortion
coefficients per clip** alongside GT `K`, `R`, `t`. So it measures the calibrator too — on real
footage, with a known answer, and no domain gap at all.

My step 3 (`eval/synthetic_calib.py`) is therefore retired entirely, not partly. Synthetic keeps
one narrow use: injecting a distortion or a zoom profile that WorldPose does not happen to contain,
to test a solver's range. That is a follow-up, not a step.

---

## 6. Measured while writing this — and it removes zoom as the explanation *here*

The review asked the right question (does *our* clip zoom) and said its numbers could not answer
it. Run on the same scene's calibration, CPU, seconds:

| window | focal px | reprojection px | realizable |
|---|---|---|---|
| 0–29 | 4254.8 | **467.1** | no |
| 0–59 | 4083.0 | **479.6** | no |
| 60–119 | 4292.9 | 413.4 | no |
| 120–179 | 4555.8 | 324.1 | no |
| 180–235 | 4459.9 | 301.1 | no |
| 0–235 | 4340.8 | 471.1 | no |

**Two answers, and one of them closes the zoom question.**

*Yes, our clip zooms* — 4083 → 4556 across the windows, ~11 %, in line with the review's 9.2 %
median at this window size. The review's WorldPose statistic transfers.

*And zoom is not why the fit refuses.* The reprojection does not fall when the window shrinks:
**30 frames gives 467 px, 236 frames gives 471 px.** At 30 frames the review's own table puts
drift at 0.8 %, which cannot produce 467 px of anything. A cause that is invariant to window
length is not zoom.

Filtering by confidence does not help either:

| subset | n | reprojection px |
|---|---|---|
| all | 236 | 471.1 |
| top 50 % by confidence | 118 | 474.1 |
| top 25 % | 59 | 471.7 |
| **top 10 %** | 24 | **453.0** |

Confidence ranges 0.402–0.645 and ranks nothing useful — the third independent confirmation of
#126, which found confidence bit-identical across runs whose homographies differed by metres.

**Where the error actually lives.** Against the golden fit over frames 0–59, sampling 35 image
points per frame and comparing where each calibration puts them on the pitch:

> median **0.63 m** · p90 **3.22 m** · max **18.58 m**

The median is ordinary — comparable to the 0.76 m #126 called load-bearing. The distribution is
**tail-dominated**, and the arithmetic closes: at f = 4169 and 40 m, 0.63 m is ~66 px, 3.22 m is
~336 px, 18.6 m is ~1900 px. A 471 px reprojection is exactly what that tail produces.

So the dominant error is a **tail in the per-frame homographies**, uniform across the clip,
unranked by confidence, and largest where the plane is least constrained — the far end of the
pitch, which is the top and the edges of the frame. **That is the observation both documents
started from**, and it is in the calibration, not in the camera model's parameter count.

**Consequence.** Adding per-frame focal or `k1` to a model fed these homographies will not fix
this. The review's ranking of the three hypotheses is right *as a description of real broadcast
cameras* and is not the binding constraint *on this clip*.

The clean control: the golden `calib/*.npz` was produced by a **different fitter**
(`scripts/fit_rigid_camera.py`) on the same clip and succeeded, with one focal and a
mutation-checked position. Same footage, different calibration quality. That is where to look
before changing any schema.

## 6a. Still open

- **Why is the pipeline's calibration worse than the one that made the golden npz?** Different
  PnLCalib invocation, different post-processing, or a different world convention. Unmeasured.
- **The residual 471 px after zoom is accounted for.** Zoom explains roughly 340 px over 236
  frames; the rest is this tail. Whether distortion (~47 px at the corner) is inside it is
  unmeasured.

## 7. Resulting order

Merging the review's list with §2, one step ahead of it:

0. **Re-run with the measured camera.** `RIGID_CAMERA` (#129) appears nowhere in
   `scripts/pod_real_e2e.sh` and zero times in the 236-frame run log, so all three scenes under
   discussion were built without it. On frames 0–59, where `calib/*.npz` is valid and the golden
   fit holds, this is one run — and it may dissolve the observation that started everything.
   **Do this before any model change.**
1. **Step 1 with three residuals** — pitch paint, foot vs detector box, and common-mode vs
   per-player scatter. Half a day, on the scene from step 0.
2. ~~Sliding-window rigid fits~~ — **done, §6.** Our clip zooms ~11 %, and zoom is not why the fit
   refuses. The binding constraint is a tail in the per-frame homographies.
3. **Find why the pipeline's calibration is worse than `fit_rigid_camera.py`'s** on the same clip.
   This is now ahead of any schema change: per-frame focal cannot repair a homography tail.
4. **Per-frame focal, then `k1`** — still correct, still the right order per the review, but after
   3. The golden test is re-measured, never nudged.
5. **Verticality and foot contact** as their own item, per the review.
6. **WorldPose** for both calibrator and pose evaluation. Synthetic dropped.
7. Dataset export and UI controls last, unchanged.

The review changed items 3–6 of my plan and added item 4. §2 added item 0 and made items 1–2
conditional on it. Nothing in the original ordering survives except the instruction to measure
before building, which both documents got right.
