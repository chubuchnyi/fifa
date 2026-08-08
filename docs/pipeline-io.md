# Pipeline I/O — what goes in, what comes out, in what form

`pipeline.md` shows the *shape* of the pipeline. This shows the **data contract**: for every
stage, the exact type, array shape and unit of what it consumes and produces, and which on-disk
files carry it.

Written 2026-08-08 by reading the code, not the docs. Where a number is invented rather than
measured it says so — that distinction is the point of the last section.

**The proposed changes to this document are in
[`pipeline-io-proposed.md`](pipeline-io-proposed.md).** Read this one first.

---

## 0. Units and frames — everything below depends on this

| | |
|---|---|
| World | **Z-up, right-handed, metres.** Pitch plane is XY at Z = 0. Gravity along −Z |
| Pitch | 105 × 68 m (`core/scene/units.py`, `FieldDimensions`) |
| Image | pixels, origin top-left, `(u, v)` |
| Rotations | axis-angle `(3,)` for SMPL-X, quaternion `(w, x, y, z)` for the camera |
| Time | integer frame indices, plus `fps` on the clip. Never seconds inside the core |
| Confidence | `[0, 1]` |

Two conversions bite people:

- **SMPL-X poses arrive in a camera frame with +y pointing DOWN.** `R_SMPLX_CAMERA_TO_WORLD`
  (`core/scene/frames.py`) maps `world = (x, z, −y)`. There is a second matrix,
  `R_SMPLX_CANONICAL_TO_WORLD`, for canonical +y-up output: `world = (x, −z, y)`. Both have
  det +1. The camera one treats the broadcast camera as level and leaves residual tilt to the
  camera solve. `smplx_to_world` re-origins on the pelvis first — SMPL-X's pelvis sits ~0.35 m
  off the model origin.
- **The solved camera is 180° rolled** relative to the raw video on legacy solves. Any consumer
  of raw pixels must rotate first. `CameraTrack.raw_frame_aligned` says whether that workaround
  is still needed; solves rebuilt by `scripts/recalibrate_camera.py` set it `True`.

---

## 1. The stage chain

`Stage` (`core/orchestration/stages.py`), in DAG order. Each stage is cached; the key is derived
from the clip hash + the stage's params + `model_version`, so changing a threshold invalidates
only what depends on it.

| stage | consumes | produces |
|---|---|---|
| `DETECT` | `ClipRef` | `Detections` |
| `TRACK` | `ClipRef`, `Detections` | `Tracks` (tracklets + teams) |
| — stitch | `Tracks` | `Tracks` + `StitchReport` (pure, not a cached stage) |
| — identity | `Tracks` | `Tracks` + `IdentityReport` (pure, opt-in) |
| `CALIBRATE` | `ClipRef` | `FieldCalibration` |
| `POSE` | `ClipRef`, `Tracks`, `FieldCalibration` | `dict[track_id, SubjectMotion]` |
| `BALL` | `ClipRef` | **`Ball2DTrack`** — 2D only |
| — ball lift | `Ball2DTrack`, `FieldCalibration`, motions | `BallTrack` (3D). Pure core, **not cached** |
| `ASSEMBLE` | `ReconstructionResult` | `Scene` — **called by the controller, not by `ReconstructionPipeline.run`** |
| — coherence | `Scene` | `Scene` + `CoherenceReport` (fills gaps, writes `provenance`) |
| — handover | `Scene` | `Scene` + `HandoverReport` (opt-in, `--handover`) |
| — physics gates | `Scene` | `Scene` + per-gate reports |
| `RENDER` / `EXPORT` / `OBSERVE` | `Scene` | images / files / `Observation` |

**Six of the twelve enum members are wired.** `RECON_ORDER = (DETECT, TRACK, CALIBRATE, POSE,
BALL, ASSEMBLE)` is the only order executed. `AMPLIFY`, `ENV`, `AVATAR`, `RENDER`, `EXPORT` and
`OBSERVE` are declared and have **no orchestration call site** — render/export/observe happen
through `controller.Application`, not through the stage runner.

`stitch` is off in `ReconstructionPipeline` (`stitch_cfg=None`) and **on** at the CLI
(`--no-stitch` to disable). Both statements are true at different layers.

**Cache key.** `content_key` hashes `{stage, input_hash, params, model_version}`. `input_hash` is
`clip_hash(clip)`, which covers `source_id`, `uri`, `width/height` and **only the first frame, the
last frame and the count** — not the full frame list. Two different frame subsets with the same
endpoints and length collide. POSE additionally folds the stitch and identity configs into its
params, so toggling either invalidates the pose cache.

---

## 2. The types

### Input

**`ClipRef`** — `source_id: str`, `uri: str`, `frames: (T,) int`, `width: int`, `height: int`,
`fps: float`. `uri` points at a video file or a frame directory. `frames` is which indices to
process, not necessarily contiguous.

### Perception

**`Detection`** — `bbox_xyxy: (4,) float` image px, `cls: str`, `score: float`.
**`FrameDetections`** — `frame: int`, `items: list[Detection]`.
**`Detections`** — `frames: list[FrameDetections]`.

The vocabulary is `player | goalkeeper | referee | ball`. **With the free COCO weights the map is
`{1: player, 37: ball}`, so every person becomes `player`** — goalkeepers and referees cannot be
labelled at all. The Roboflow sports checkpoint (`--detector-weights` + `--detector-classes
sports`) splits them.

**`Tracklet`** — `track_id: int`, `frames: (T,) int`, `bboxes_xyxy: (T, 4) float`, `cls: str`,
`team_id: str | None`.
**`Tracks`** — `tracklets: list[Tracklet]`, `teams: list[Team]`.
**`Team`** (lives in `core/scene/subject.py`, not perception) — `id: str`, `name: str | None`,
`color_rgb: (3,) | None` in 0..1, the mean kit colour of its members.

`team_id = None` means unassigned. `StitchConfig.require_same_team` treats it as a **wildcard**,
so a null label removes a constraint rather than adding one.

### Calibration

**`FieldCalibration`** — `homographies: (T, 3, 3) float` (image → world plane), `frames: (T,) int`,
`confidence: (T,) float` in `[0, 1]`, `keypoints: dict | None` (adapter-defined).

Direction is **image px → world pitch plane in metres**: `[x, y, 1]ᵀ ~ H @ [u, v, 1]ᵀ` with Z = 0.
Helpers `image_to_world(frame, uv)` and `world_to_image(frame, xy)` are on the type.

Confidence is **per frame**, and `MIN_SOLVED_CONFIDENCE = 0.02`. **Confidence exactly `0.0` means
the frame was not solved at all** — the homography is carried from the last good frame, or is
`eye(3)`. So a scene can carry a full `(T, 3, 3)` array in which a third of the frames are copies.
That is what produced the 2026-08-03 vertical-clip catastrophe: 43 % of 355 frames on a stale
homography.

#126 measured that confidence is bit-identical across runs whose homographies differ by 0.76 m
median / 3.67 m max, i.e. it is not predictive — and #136 now gates on it.

### Pose

**`SmplxShape`** — `betas: (n_betas,)`, typically 10 or 16; `body_model: SMPL_X`.

**`PoseSequence`** — `frames: (T,) int`, `global_orient: (T, 3)` axis-angle,
`body_pose: (T, J, 3)` axis-angle, `transl: (T, 3)` **world metres**,
`left_hand_pose` / `right_hand_pose` / `jaw_pose` optional, `provenance: (T,)`.

**`provenance`** is the honesty channel, and criteria #135 rest on it:

| value | meaning |
|---|---|
| `measured` | the model saw this frame |
| `interpolated` | filled between two measurements — limbs **do** move |
| `imputed` | no anchor on one side: posture frozen, root coasting. **Exactly 0.00 rad of limb travel** |

**`SubjectMotion`** — `shape: SmplxShape`, `pose: PoseSequence`.

### Ball

**`Ball2DTrack`** — what the `BallTracker` port returns: `frames: (T,)`,
`positions_2d: (T, 2)` px, `confidence: (T,)`.

**`BallTrack`** — the 3D result of `lift_ball_to_3d`: `frames: (T,)`, `positions_3d: (T, 3)` metres,
`height_confidence: (T,)` (Z is recovered by ballistics and is genuinely uncertain, so it is a
first-class field), `track_2d: (T, 2) | None` px, `mode: (T,)` of `BallMode`.

### Camera

**`CameraIntrinsics`** — `fx, fy, cx, cy: float` px, `width, height: int`,
`distortion: (k,) | None`.

**`PlaneCameraFit`** — what `camera_from_calibration` returns, and **the only way to tell a
measured camera from a synthetic one**: `camera: CameraTrack | None`, `focal_px: float`,
`reprojection_px: float`, `realizable: bool`. Reachable as `Application.camera_fit(scene_id)`.
It is held in memory only — **it is not serialized into `scene.json`**, so a scene on disk carries
no record of whether its camera was measured. Nothing in `poseannot` checks it.

**`CameraTrack`** — `intrinsics: CameraIntrinsics` (**one, shared by the whole track**),
`frames: (T,)`, `rotation_quat: (T, 4)` as (w,x,y,z) world→camera, `translation: (T, 3)` metres
world→camera, `estimated: bool`, `raw_frame_aligned: bool`.

Two consequences worth stating plainly:

- **Zoom is not representable.** One `intrinsics` for the whole track means a focal that changes
  during the clip cannot be expressed. The class docstring says so and calls per-frame intrinsics
  a future refinement.
- **Distortion is representable but never fitted.** The field exists and is `None` on every solve
  we produce.

### Scene

**`Scene`** — `id`, `episode_id`, `source_id`, `world_frame`, `field: FieldModel`,
`camera: CameraTrack | None`, `subjects: list[Subject]`, `teams`, `ball`, `corrections`,
`confidence: ConfidenceMap`, `render_assets`, `synth_views`, `run_log`.

**`Subject`** — `track_id: int`, `proposal: SubjectMotion`, `role: Role`, `team_id: str | None`,
`jersey_number: int | None`.

**Three layers, non-destructive (ADR-0002).** `proposal` is the raw model output; `corrections`
is a list of `Correction` records; `resolved` is computed by the correction engine and never
stored. Every editor — CLI, MCP, browser — appends to the same list.

**`Correction`** — `id`, `target: CorrectionTarget`, `frame_range` (**inclusive** start/end),
`mode`, `payload`, `enabled: bool`, `note`, `created_at`.

`TargetKind` has six members: `POSE_BODY_JOINT`, `ROOT_ORIENTATION`, `ROOT_TRANSLATION`,
`SHAPE_BETA`, `BALL_POSITION`, `FIELD_CALIBRATION`. `CorrectionMode` has four:
`CONSTANT_OFFSET`, `KEYFRAME_INTERP`, `REFIT`, `TEMPORAL_SMOOTHING`.

Payloads, and the delta-vs-absolute distinction that matters when reading them back:

| payload | carries |
|---|---|
| `OffsetPayload` | `delta` — a **relative** `(3,)` vector or axis-angle, or `(n_betas,)` for `SHAPE_BETA` |
| `KeyframePayload` | `key_frames (K,)`, `key_values (K, D)` — **absolute** values — and `interp: linear \| slerp` |
| `PlaneTransformPayload` | `matrix (3, 3)`, composed world-side as `H' = H @ matrix` |
| `SmoothingPayload` | `window` (odd), `method`, `sigma` |
| `RefitPayload` | `constraints: dict`, opaque to core |

A `FIELD_CALIBRATION` correction is a **similarity on the pitch plane: 4 degrees of freedom**
(translate x, translate y, rotate, uniform scale). It cannot express a focal, a tilt or
distortion — see [`pipeline-io-proposed.md`](pipeline-io-proposed.md).

---

## 3. On-disk artifacts

### `scene.json`

Custom JSON (ADR-0005), round-trips dataclasses/enums/numpy. Wrapper keys:

| wrapper | payload |
|---|---|
| `__type__` | dataclass — `{type, fields}` |
| `__ndarray__` | `{dtype, shape, data}` |
| `__enum__` | `{type, value}` |
| `__tuple__` | list |

A 236-frame, 38-subject scene is ~42 MB. Read it with
`core.scene.serialization.load_scene`.

### `calib/<clip>.npz` — the measured one-camera fit (#119 / #129)

| key | shape | meaning |
|---|---|---|
| `focal` | scalar | **one** focal for the whole clip, px. 4169.32 on the broadcast clip |
| `centre` | `(3,)` | camera position in world metres. (−2.29, −70.13, 17.22) |
| `rvecs` | `(T, 3)` | per-frame rotation, Rodrigues |
| `frames` | `(T,)` | frame indices |
| `world_to_image` | `(T, 3, 3)` | pitch-plane world (X, Y, 1) → image px. `K @ [r₁ \| r₂ \| −R·c]`. **The inverse of `FieldCalibration.homographies`** — `apply_rigid_camera.py` inverts per frame |

Pinned by `tests/e2e/test_golden_real_camera.py`, which is mutation-checked. **No distortion, no
translation over time, no zoom.** It is a pan-tilt-roll model on a fixed tripod.
`CameraTrack.translation = −R @ centre`. The current writer also emits `width`/`height`; this file
predates that and does not have them.

### Cached detections — `out/**/dets_*.npz`

`frame: (N,) int`, `boxes: (N,) object` of `(k, 4)`, `classes: (N,) object` of `list[str]`,
`scores: (N,) object` of `list[float]`. Written by `scripts/dump_detections.py`; every CPU probe
in `scripts/` reads this format, so a detector change is replayed downstream without a GPU.

### Blender hand-off — `anim_export.py` → npz + manifest

The exporter and the renderer talk only through `manifest.json`; an unknown or key-incomplete
artifact fails at export time, not mid-render (`adapters/blender/anim_contract.py`).

| file | required keys |
|---|---|
| `anim_subject_*.npz` | `verts (T, 10475, 3)`, `faces (20908, 3)`, `color`, `frames`, `alpha`, `provenance` |
| `ball.npz` | `frames`, `positions_3d`, `height_confidence`, `mode` |
| `pitch.npz` | `pitch_verts`, `pitch_faces`, `goal_verts`, `goal_faces` |
| `stadium.npz` | `verts`, `faces`, `colors`, `uv`, `tile` |
| `boards.npz` | `verts`, `faces`, `colors` |
| `lighting.npz` | `light_rgb` |
| `cameras.npz` | `names`, `frames` |

`provenance` travels all the way into the render, so an imputed frame can be drawn differently
from a measured one.

### Browser edits — `edits.json`

`{"corrections": [Correction, …]}`, whole-file atomic rewrite. Same `Correction` class the
pipeline uses. `scene.json` is never mutated.

---

## 4. Measured, or invented

The types do not distinguish these, so a fake run produces a file of exactly the same shape as a
real one. This table is the difference.

| thing | when it is real | what you get otherwise |
|---|---|---|
| Detections | `--detector rfdetr` | `FakeDetector` |
| Camera | `--calibrator keypoints` + PnLCalib weights | `FakeFieldCalibrator`: world = (pixel − centre) × 30 m / width. **Affine, no perspective at all** |
| Pose | `--pose gvhmr` + a real backend | `FakePoseEstimator`: `body_pose = zeros`, a T-pose on every frame |
| `Scene.camera` | `camera_from_calibration` returns `realizable=True` | an **invented** 772 px @ 1280×720 fallback, principal point dead centre — see below |
| Role | Roboflow sports weights | everyone is `player` |
| Team | kit k-means over the sampled torso | `None`, which downstream treats as a wildcard |

### The camera fallback is not hypothetical — measured 2026-08-08

`controller.run_reconstruction` ends with:

```python
scene.camera = self._measured_camera(scene, clip) or self._static_camera(scene)
```

`_measured_camera` returns `fit.camera`, which is `None` whenever a set of free per-frame
homographies cannot be reduced to one camera. That refusal is deliberate and correct (#61: a scene
once carried two cameras 12686 px apart for months). **The silent substitution is the problem** —
`_static_camera` builds a synthetic broadcast viewpoint, and downstream nothing distinguishes it
from a solve.

Checked across every scene on disk:

| scene | fx | size | |
|---|---|---|---|
| `out/cue/scene_off.json` — **the reference scene for the #135 eye labels** | 772.0 | 1280×720 | synthetic |
| `out/cue/scene_on.json` | 772.0 | 1280×720 | synthetic |
| `out/res_ab/res{560,896,896_handover}.json` | 772.0 | 1280×720 | synthetic |
| `out/res_ab236/f236_res{560,896,896_handover}.json` | 772.0 | 1280×720 | synthetic |
| `out/vert137/scene.json` | 772.0 | 1280×720 | synthetic |

**Nine of nine.** Every scene we have judged by eye, scored criteria against, or A/B-ed was drawn
with a camera at 772 px focal and 1280×720 against a 1920×1080 clip whose real focal is ~4200 px.

Why, on `f236_res896`, measured:

```
camera_from_calibration(scene.field.calibration, 1920, 1080)
  focal_px = 4340.8        # sane, within 4 % of the golden 4169.32
  reprojection_px = 471.1
  realizable = False
```

The calibration itself is solved on all 236 frames (median confidence 0.472, no frame at 0.0). The
reprojection is invariant to window length (467 px at 30 frames, 471 px at 236) and to confidence
filtering (453 px on the top 10 %), so it is neither zoom nor a rankable subset — it is a
tail in the per-frame homographies. Detail:
[`findings/reply-camera-model-gap-2026-08-08.md`](findings/reply-camera-model-gap-2026-08-08.md).

**What to do with this when reading any scene:** check `fx`. If it is 772.02 at 1280×720, the
scene has no camera, and anything comparing it to the source pixels is meaningless.
`scripts/track_quality.py` says so unprompted and is why its `--camera` flag is not optional.

**What survives and what does not.** Scene-to-source comparisons — overlay alignment, pitch
markings, skeletons on players — were never testing what they appeared to. Scene-to-scene
comparisons are untouched, because they never read `scene.camera`: identity churn, phantom counts,
handover pairs, root-speed distributions, and the eye's ranking of 560 vs 896 vs 896+handover.

Tracked as **#140**. The fix is proposed as step 4 of
[`pipeline-io-proposed.md`](pipeline-io-proposed.md) — serialize `PlaneCameraFit` into the scene,
so "measured or invented" stops being a magic number test.

Two more consequences that have cost sessions:

- A scene reconstructed without `--camera`/the rigid fit stores the 772 px fallback, whose field
  of view is so wide that **every subject lands inside the image** — which silently turns the
  in-frame test into a constant. `scripts/track_quality.py` prints a warning when it sees it.
- Without `--coherence` there is no `provenance` at all: a lost subject is dropped rather than
  held, and every track then reads "measured over its own span".
