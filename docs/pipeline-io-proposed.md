# Pipeline I/O — proposed changes

Same document as [`pipeline-io.md`](pipeline-io.md), with the changes marked. Read that one
first; this only lists what moves.

Marks: **▲ CHANGE** an existing contract · **＋ NEW** something that does not exist ·
**◆ MEASURE** a diagnostic, no contract change.

Nothing here is built. This is the proposal to argue with.

> **Measured 2026-08-08 — three of the guesses below are now settled, and the order changes.**
> Against 89 WorldPose GT broadcast cameras: focal moves >5 % in **89/89** clips (median 44 %),
> the camera translates in **0/89** (exactly 0.000 m), distortion is **~47 px** at the frame
> corner, principal-point wander is ~1 px. So **step 2b (per-frame focal) is the dominant term and
> should be first**, 2a stays, and **2c and the camera-translation hypothesis should be dropped**.
> At our 60-frame fit that costs ~0.4–0.8 m of player position; at 240 frames, 2–5 m. Numbers,
> method and caveats: [`findings/camera-model-gap-2026-08-08.md`](findings/camera-model-gap-2026-08-08.md)
> · reproduce with `scripts/bench_camera_model_gap.py`.

---

## Why any of this

Observed by eye on `f236_res896`, 2026-08-08: the drawn pitch does not sit on the painted pitch,
and skeletons do not sit on players — **and both get worse toward the edges of the frame**.

Both errors together, growing with radius from the image centre, is the signature of the camera
model, not of individual players' depth. Our camera model is one focal for the whole clip, one
camera position, per-frame rotation, and no distortion. It structurally cannot express three
things that all produce exactly that radial signature:

1. zoom — the operator changing focal during the clip
2. camera translation — a rail, a crane, a nudged tripod
3. lens distortion

So the proposal is ordered to test that before building anything expensive.

---

## Step 1 ◆ MEASURE — split the pitch residual by radius

**No contract change. Measurable today with existing code.**

`poseannot/pitch_evidence.py::classify(uv, video, frame)` already returns, per drawn pitch point,
the distance to the nearest real paint (`NaN` where there is no evidence). It is fed `(N, 2)`
image pixels.

Add nothing; just bin those existing `(uv, dist)` pairs by radius from the principal point and
print the profile.

**One thing to get right or the measurement is worthless.** `FieldCalibration.confidence` of
exactly `0.0` means that frame was **not solved** — its homography is a copy carried from the last
good frame (`MIN_SOLVED_CONFIDENCE = 0.02`). On the vertical clip that was 43 % of 355 frames.
A carried homography drifts with the pan, so its residual grows with *time*, not with radius, and
mixing those frames in would fake exactly the signal we are testing for. **Bin only frames with
confidence above the floor, and report how many were dropped.**

| outcome | conclusion |
|---|---|
| residual flat with radius | not the lens or the focal. Look at extrinsics or at the pitch model |
| residual grows with radius | focal or distortion. Go to step 2 |
| residual grows with **frame index** | the solve drifts. A per-frame problem, not a lens one |

Half a day. It decides whether the rest of this document is worth doing.

---

## Step 2 ▲ CHANGE — give the camera the parameter it is missing

Only whichever one step 1 points at.

### 2a. Distortion — **cheap, no schema change**

`CameraIntrinsics.distortion: np.ndarray | None` **already exists** and is `None` on every solve
we produce. Fitting one radial coefficient `k1` needs:

| | |
|---|---|
| type | unchanged |
| `scripts/fit_rigid_camera.py` | ＋ one parameter in the optimiser |
| `calib/<clip>.npz` | ＋ key `dist: (k,)`; the writer already emits `width`/`height` that this file predates |
| every projector | ▲ apply distortion. There are at least four and they must agree or the overlay and the export diverge again: `core/scene/projection.py`, `poseannot/camera.py`, `scripts/apply_rigid_camera.py`, `scripts/track_quality.py` |

Keeping those four projectors in step is the real cost of 2a, not the optimiser change.

### 2b. Per-frame focal — **schema change**

`CameraTrack.intrinsics` is **one object for the whole track**. Its own docstring already calls
per-frame intrinsics a future refinement and says core does not need to change for it.

| | |
|---|---|
| `CameraTrack` | ▲ `intrinsics: CameraIntrinsics` → also accept `(T,)` of them, or add `focal_per_frame: (T,)` |
| `scene.json` | ▲ serialization follows the type, no wrapper change |
| `calib/<clip>.npz` | ▲ `focal` scalar → `(T,)` |
| `tests/e2e/test_golden_real_camera.py` | ▲ **pins `focal 4169.32` and one optical centre for all 60 frames.** It must be re-measured, not nudged |
| consumers | ▲ `poseannot/camera.py`, `anim_export.py` cameras.npz, Blender |

Bigger, and it invalidates a golden test that is currently mutation-checked. Do it only if 2a is
not enough.

### 2c. Free principal point — cheap, same shape as 2a

`cx, cy` already exist and are set to the image centre. Making them fitted parameters is one more
term in the optimiser and no type change at all.

---

## Step 3 ＋ NEW — synthetic ground truth for the calibrator

**The only way to know what the solver loses, rather than guess.**

Render a scene with a *known* camera — known focal, known distortion, known pose — and ask the
calibrator to recover it. `src/pitch3d/eval/synthetic.py` already generates synthetic scenes and
`eval/harness.py` already runs Condition A (GT camera) vs Condition B (our solve), so the frame
exists.

| | |
|---|---|
| ＋ `eval/synthetic_calib.py` | render pitch + players at known camera params, with deliberate distortion |
| ＋ metric | recovered focal vs true, recovered pose vs true, residual vs radius |
| contract | none — this is a test harness, not a pipeline stage |

Domain gap is small here: a white line on green is a white line on green. This is the half of
synthetic data that genuinely transfers. **Pose is the half that does not** — for pose the
appearance gap is real and the answer is WorldPose, not rendering.

---

## Step 4 ＋ NEW — export corrections as a training set

For "fine-tune the model", the current file is the wrong shape.

`edits.json` holds `Correction` records, i.e. a **delta**: `PlaneTransformPayload(matrix)` says
how far the operator moved the layout. A trainer needs the **absolute** answer per frame.

| | |
|---|---|
| ＋ `scripts/export_calib_dataset.py` | resolve corrections against the solve → per-frame corrected homography + the pitch keypoints it implies |
| ＋ output | `{frame, keypoints_image: (K, 2), keypoints_world: (K, 2), homography: (3, 3), source_clip}` |
| contract | none in core — a new consumer of an existing correction stack |

**The trap to state out loud.** Human corrections are made *through* the current camera. If the
camera model is wrong — which is the premise of this whole document — then corrections made on
top of it inherit that error, and training on them teaches the model to reproduce it. So this
step must come **after** step 2, never before.

And for **pose** specifically there is no need to build a dataset at all: WorldPose is 89 clips
of real World Cup 3D poses at 8 cm accuracy, already on disk at `AVATAR/WorldPose/`.

**Correction 2026-08-08: the video is not missing either.** All 89 clips have GT camera *and* GT
pose *and* footage on disk (`models/worldpose/`, 24 GB), and the GT poses are our exact
`PoseSequence` schema — `global_orient`, `body_pose`, `transl`, `betas`, 22 players. That makes
**step 3 above (synthetic calibration GT) redundant**: real broadcast beats a render, measures
both halves of the goal rather than one, and carries no domain gap.

---

## Step 5 ▲ CHANGE — controls in the browser, after the above

Today's calibration UI is a **4-DOF similarity on the pitch plane**: drag to translate, a turn
handle to rotate and scale, plus a typed panel, undo, and a measured `fit px · ok/off` readout.
It writes `FIELD_CALIBRATION` corrections that reach the export (#128).

Four degrees of freedom cannot fix a wrong focal, a wrong tilt or distortion. That is why the
alignment fight has been expensive: the control set does not contain the error.

| | |
|---|---|
| ＋ point correspondences | drag known pitch landmarks onto their image position; ≥4 gives a homography, and with the 105 × 68 model, focal and pose. **This is also exactly the format step 4 exports** |
| ＋ residual-vs-detections readout | project each subject's feet, compare to its detector box bottom-centre. Per-player pixel error, no human input |
| ＋ "is it the camera or one player" line | are all subjects displaced the same way, or one. This is ten lines and it decides which of the two problems you are looking at |
| ▲ existing 4-DOF drag | keep. It is right for the residual planar offset, once the camera is right |

---

## What I would not do

- **Per-frame calibration by hand.** A layout a metre out on one frame is a metre out on all of
  them; correcting one frame buys alignment there and a jump next door. The existing correction
  is whole-clip by default for this reason.
- **Synthetic data for pose appearance.** Weak transfer, and WorldPose already exists.
- **Building the annotation UI first.** If step 2 closes the edges, there is far less to annotate,
  and what remains is annotated on top of a camera that is right.

---

## Order and cost

| step | cost | what it buys |
|---|---|---|
| 1 ◆ residual by radius | half a day, no new code | tells you if any of the rest is needed |
| 2a ▲ distortion | ~1 day, several projectors to keep in sync | closes a radial error without a schema change |
| 2c ▲ free principal point | hours | same shape, sometimes the whole answer |
| 2b ▲ per-frame focal | days, breaks a golden test | only if the operator zoomed |
| 3 ＋ synthetic calib GT | ~1 day, harness exists | tells you what the solver loses, exactly |
| 4 ＋ training-set export | ~1 day | turns manual work into data — **after** step 2 |
| 5 ＋ UI controls | days | needed only for what steps 1–3 leave behind |
