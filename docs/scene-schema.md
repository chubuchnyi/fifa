# pitch3d — Canonical scene schema

The field-by-field spec of the canonical model in [`core/scene`](../src/pitch3d/core/scene).
This is the **single source of truth** (ADR-0002) and the **native save format** (ADR-0005):
self-describing tagged JSON that round-trips dataclasses, enums and `numpy` arrays losslessly.
All geometry is **Z-up, right-handed, meters**; the pitch plane is `Z = 0` (ADR-0003).

> Conventions: rotations on the body are **axis-angle** (SMPL-X native); the camera stores
> rotation as a **quaternion `(w, x, y, z)`**, world→camera. Arrays note their shape as `(T, …)`
> where `T` is the frame count of that track.

---

## 1. Containers — `scene.py`

```
Source → Episode → Scene → Project
```

### `Source` (FR-2)
| field | type | meaning |
|---|---|---|
| `id` | str | stable id |
| `uri` | str | path/URI to the clip or frame sequence |
| `kind` | `SourceKind` | `video` \| `frames` |
| `time_base` | `TimeBase` | fps + optional start timecode |
| `width`, `height` | int | source resolution (px) |
| `frame_count` | int | total frames |

### `Episode` (FR-3/FR-4)
A time-range selection on a source. `id`, `source_id`, `start_frame`, `end_frame`,
`origin` (`EpisodeSource`: `manual` \| `action_spotting`), optional `name`.
`n_frames = end_frame - start_frame + 1`.

### `Scene` — the canonical unit (FR-5..14, ADR-0005)
| field | type | meaning |
|---|---|---|
| `id`, `episode_id`, `source_id` | str | identity + provenance back to selection/clip |
| `world_frame` | `WorldFrame` | metric frame (Z-up, meters by default) |
| `field` | `FieldModel` | pitch dimensions + per-frame homography (world anchor) |
| `camera` | `CameraTrack \| None` | estimated broadcast camera |
| `subjects` | `list[Subject]` | tracked people, each with proposal SMPL-X motion |
| `teams` | `list[Team]` | team/officiating groups referenced by subjects |
| `ball` | `BallTrack \| None` | ball 3D trajectory with height confidence |
| `corrections` | `list[Correction]` | the non-destructive edit stack (subjects + ball) |
| `confidence` | `ConfidenceMap \| None` | per-frame/joint confidence + reprojection error |
| `render_assets` | `list[RenderAssetRef]` | pointers to derived render assets |
| `synth_views` | `list[SynthViewRef]` | ViewSynthesizer outputs (seams A & B) |
| `run_log` | `RunLog` | which models/params/costs produced this scene (NFR-7) |

Helpers: `subject(track_id)` → the `Subject`; `corrections_for(track_id)` → enabled corrections
targeting a subject (or the ball/global when `track_id is None`).

### `Project` (FR-1)
`id`, `name`, `sources`, `episodes`, `scenes`, `settings` (`Settings`), optional `created_at`.

---

## 2. World frame & units — `units.py`

- **`WorldFrame`** — `up_axis` (`UpAxis.Z` default), `units="m"`, `handedness` (right),
  `origin="field_center_on_ground"`, `meters_per_unit=1.0`. `gravity_vector()` returns
  `(0, 0, -GRAVITY)` for Z-up.
- **`GRAVITY = 9.80665`** m/s² — used by the ball 3D lift.
- **`FieldDimensions`** — `length=105.0`, `width=68.0` m (FIFA default).
- **`TimeBase`** — `fps=25.0`, optional `start_timecode`; `frame_to_seconds(i)`.
- **`Settings`** — project-level `extra: dict` (non-model knobs).

---

## 3. Subjects & motion — `subject.py`, `motion.py`

### `Subject`
`track_id` (stable tracker id), `proposal: SubjectMotion` (the non-destructive base), `role`
(`Role`: `player`/`goalkeeper`/`referee`), optional `team_id`, optional `jersey_number`.
**No resolved pose is stored here** — it is computed by the correction engine (ADR-0002).

### `SubjectMotion` = `shape` + `pose`
- **`SmplxShape`** — `betas (n_betas,)` (shape, shared across frames), `body_model` (`BodyModel`:
  `SMPL`/`SMPL-H`/`SMPL-X`, default SMPL-X).
- **`PoseSequence`** (per frame):
  | field | shape | meaning |
  |---|---|---|
  | `frames` | `(T,)` | frame indices |
  | `global_orient` | `(T, 3)` | root orientation, axis-angle |
  | `body_pose` | `(T, J, 3)` | body joint rotations, axis-angle (`J=21` for SMPL-X body) |
  | `transl` | `(T, 3)` | root translation, world meters (anchored by homography, FR-8) |
  | `left_hand_pose`, `right_hand_pose`, `jaw_pose` | optional | SMPL-X extras; `None` for SMPL/SMPL-H |

  Helpers: `n_frames`, `n_joints`, `frame_pos(i)`, `copy()`, and `PoseSequence.rest(frames, n_joints)`.

### `BallTrack` (FR-9, R-4)
| field | shape | meaning |
|---|---|---|
| `frames` | `(T,)` | frame indices |
| `positions_3d` | `(T, 3)` | world positions (m) |
| `height_confidence` | `(T,)` | confidence in the **Z** component, `[0,1]` (mono height is uncertain) |
| `track_2d` | `(T, 2)` opt | image-space track (px) |
| `on_ground` | `(T,)` bool opt | ground-contact flag (drives the ballistic segmentation) |

Related raw input: **`Ball2DTrack`** (`frames`, `positions_2d (T,2)`, `confidence (T,)`) — the
output of a `BallTracker` adapter *before* the core 3D lift. **`VectorCurve`** (`frames`,
`values (T,D)`, `label`) is a generic per-frame curve for uniform corrections.

---

## 4. Field & camera — `field.py`, `camera.py`

### `FieldModel`
`dimensions: FieldDimensions`, `plane_z=0.0`, `calibration: FieldCalibration | None`.

### `FieldCalibration` (FR-7, R-6) — the **world anchor** in mono
`homographies (T,3,3)` mapping **image px → field-plane meters**, `frames (T,)`,
`confidence (T,)` in `[0,1]`, optional `keypoints`. `image_to_world(frame, uv)` is a pure
projective transform (no model) used to put feet on the pitch and lift ball ground contacts.

### `CameraIntrinsics`
Pinhole `fx, fy, cx, cy`, `width, height` (px), optional `distortion`. `matrix()` → 3×3 `K`.

### `CameraTrack`
Shared `intrinsics` + per-frame pose: `frames (T,)`, `rotation_quat (T,4)` `(w,x,y,z)`,
`translation (T,3)` (all **world→camera**, meters), `estimated` (True for the broadcast cam,
False for a prescribed ViewSynthesizer trajectory). `CameraTrack.identity(intr, n)` for tests.

---

## 5. The three-layer edit model — `layers.py` (ADR-0002, FR-21/22)

```
proposal  (raw model, on Subject/BallTrack)
   ⊕ corrections  (list[Correction] deltas — the only thing an edit creates)
   = resolved  (computed by core.correction; baked empty on export)
```

### `Correction`
`id`, `target: CorrectionTarget`, `frame_range: FrameRange`, `mode: CorrectionMode`,
`payload` (one of the payloads below), `enabled` (toggle to compare/reset without deleting),
optional `note`, optional `created_at`.

- **`CorrectionTarget`** — `kind: TargetKind`, optional `subject_track_id` (`None` = ball/global),
  optional `joint_index` (**required** iff `kind == POSE_BODY_JOINT`).
- **`TargetKind`** — `pose_body_joint` \| `root_orientation` \| `root_translation` \|
  `shape_beta` \| `ball_position`.
- **`FrameRange`** — inclusive `[start, end]`; `frames()` → `arange`; validates `end ≥ start`.
- **`CorrectionMode`** (FR-22 a–d) — `constant_offset` \| `keyframe_interp` \| `refit` \|
  `temporal_smoothing`.

### Mode payloads
| payload | mode | fields |
|---|---|---|
| `OffsetPayload` | CONSTANT_OFFSET | `delta` — `(3,)` vector (transl/ball) **added**, or axis-angle offset **composed** onto rotations; `(n_betas,)` for β |
| `KeyframePayload` | KEYFRAME_INTERP | `key_frames (K,)`, `key_values (K,D)` (**absolute**), `interp` (`linear`/`slerp`, auto by target) |
| `RefitPayload` | REFIT | `constraints: dict` (opaque; passed to `PoseEstimator.refit`) |
| `SmoothingPayload` | TEMPORAL_SMOOTHING | `window` (odd), `method` (`moving_average`/`gaussian`), `sigma` |

### `ConfidenceMap` (FR-16/17, UX-4) — drives the "needs attention" list
`subject_frame_conf {track_id → (T,)}`, `subject_joint_conf {track_id → (T,J)}`,
`reprojection_error_px {track_id → (T,)}`, `field_homography_conf (T,) | None`.

---

## 6. Derived-asset & synthesized-view refs — `assets.py`

These are **pointers + provenance**, never the heavy payload (which lives on disk, addressed by
URI, reproduced from the cache — ADR-0004).

### `RenderAssetRef` (FR-11..13, NFR-7)
`id`, `kind: RenderAssetKind`, `uri`, `model: ModelInfo`, optional `subject_track_id`
(per-subject avatars), `extra: dict`. **`RenderAssetKind`**: `env_splat`, `env_nerf`,
`env_generative`, `avatar_textured` (#1), `avatar_generative` (#2), `avatar_gaussian` (#3),
`ball_texture`.

### `SynthViewRef` (FR-29..32, ADR-0007)
`id`, `seam: SynthViewSeam` (`A_render` \| `B_amplify` \| `B_inpaint`), `uri`,
`camera: CameraTrack` (prescribed, `estimated=False`), `model: ModelInfo`,
`frustum_overlap ∈ [0,1]` (low ⇒ likely hallucination, R-14), optional `subject_track_id`
(for `B_inpaint`), `editable` (**always False for seam A**, R-15), optional `note`.

---

## 7. Provenance — `provenance.py` (NFR-7, UX-7)

- **`ModelInfo`** — `name`, `version`, `backend: Backend`, `license`, `est_cost_usd`,
  `params: dict` (frozen; feeds the cache key, ADR-0004/0006).
- **`Backend`** — `local` (self-hosted GPU) \| `api` (cloud) \| `fake` (test double) \|
  `builtin` (pure-core math).
- **`RunRecord`** — one executed stage: `stage`, `model`, `cache_key`, `cache_hit`, `duration_s`,
  optional `note`.
- **`RunLog`** — append-only `records`; `add(record)`, `total_cost_usd()`.

---

## 8. Serialization — `serialization.py` (ADR-0005)

Tagged-JSON codec, stdlib `json` only, lossless round-trip:

| reserved key | encodes |
|---|---|
| `__ndarray__` | `{dtype, shape, data}` |
| `__enum__` | `{type, value}` (decoded before primitives, since enums subclass `str`) |
| `__type__` | dataclass `{type, fields}` |
| `__tuple__` | tuple |
| `__dict__` | dict with arbitrary (incl. non-string) keys |

API: `encode`/`decode`, `to_json`/`from_json`, `save_scene(obj, path)`/`load_scene(path)`. Every
serializable type is listed in the module's `_CLASSES` registry — **add a new dataclass/enum there
or it will not round-trip**. The exported scene is always the *resolved* scene (corrections baked).
