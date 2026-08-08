# Code map — where each subsystem lives

Verbatim §5 of `docs/STATUS.md` before the 2026-08-01 split. Grouped by the defect/lever that
drove the work, so entries read as "#203 world scale → these files".

`CLAUDE.md` carries the short version; come here for the full paths and the why.

---

- **Tracking / fragmentation (#202):** `src/pitch3d/adapters/models/tracking.py` (`ByteTrackTracker`,
  `min_track_frames`, `ByteTrackBackend.associate`), `src/pitch3d/core/orchestration/pipeline.py`
  (`stitch_cfg` gate), `src/pitch3d/core/orchestration/continuity.py` (`StitchConfig`,
  `stitch_tracks_with_report`), `src/pitch3d/core/orchestration/assemble.py` (one Subject per track_id),
  `src/pitch3d/app/cli.py` (`--no-stitch`, `run_dry_run`).
- **Calibration / world scale (#203):** `src/pitch3d/core/scene/field.py` (`image_to_world`),
  `src/pitch3d/adapters/models/pose.py` (`_ground_root`, foot=bbox bottom),
  `src/pitch3d/adapters/models/calibration.py` (identity fallback; `CameraModuleFieldCalibrator`),
  `src/pitch3d/core/scene/units.py` (`FieldDimensions` 105×68 m).
- **Cameras (#204):** `src/pitch3d/core/agent/viewpoints.py` (`standard_viewpoints` 63 m radius,
  `action_centroid`), `src/pitch3d/app/controller.py` (`_measured_camera` → `_static_camera`
  fallback, frozen frame-0), `src/pitch3d/core/scene/plane_camera.py` (#107 —
  `camera_from_calibration`: homographies → real `CameraTrack`, or `None` when none exists;
  `REALIZABLE_PX = 1.0` is the line between the two).
- **Render / pitch / goals (#205):** `src/pitch3d/adapters/blender/_cycles_script.py` (`_add_ground`,
  `_build_pitch`), `src/pitch3d/adapters/render/cycles.py` (`draw_pitch`), `src/pitch3d/core/scene/pitch.py`
  (markings). Goals: add a `_build_goals()` mesh (absent today).
- **Hybrid stadium backdrop (v1 step 3) + tinted mosaic (v1 polish):** `src/pitch3d/core/scene/stadium.py`
  (`stadium_bowl_geometry`, `fill_holes_by_copy`, `bowl_tile_loop_uvs` — per-loop wrap-safe tile UVs,
  `_rounded_rect_loop` — pure numpy), `src/pitch3d/adapters/render/stadium_backdrop.py` (`bake_backdrop_colors`
  — projective median bake = the per-vertex **tint**; `extract_crowd_tile` + `_busiest_window` — cut one clean
  crowd patch by edge-**density**; both auto-rotate each frame 180° to match the solved camera's rolled pixel
  convention), `tests/unit/test_stadium_geometry.py` (9 tests). Wired through `scripts/anim_export.py`
  (`PITCH3D_STADIUM_VIDEO` → `stadium.npz {verts,faces,colors,uv,tile}`) and `scripts/blender_animate.py`
  (`_add_stadium_mesh` — unit-mean tile image × vertex-colour tint → emission, mirror-tiled, so the crowd
  renders as repeated real spectators tinted by the measured regional colour).
- **Measured body texture (v2 lever 1):** `src/pitch3d/adapters/models/avatar.py` (`bake_body_vertex_texture`
  — image-level core, unit-testable; `measured_texture_from_clip` — clip wrapper: `_even_subset` frame pick,
  decode+resize+rotate180, delegate; both built on the existing `measured_vertex_texture` projective z-buffer
  sampler), `tests/unit/test_avatar_textured.py`. Wired through `scripts/anim_export.py` (`SOURCE_OK` gate →
  `vcolor,measured` in `anim_subject_*.npz`, unseen verts filled with kit colour) → `scripts/blender_animate.py`
  (BYTE_COLOR "Col" → `ShaderNodeVertexColor` → Principled Base Color, lit; flat-colour fallback).
- **Shared Blender node-graphs (v2 levers 2+3, the "B" layer):** `src/pitch3d/adapters/blender/
  scene_builders.py` (`build_grass_material` — mowing stripes + bump, `stripe_scale=0.1` ≈ 5 m bands;
  `build_stadium_lighting` — floodlit-NIGHT: dark world `light_rgb × sky_strength` + a ring of `sun_count`
  soft high SUN lamps tinted `light_rgb`; pitch3d-free, `bpy` as arg). Imported by file (sys.path shim) from
  BOTH `scripts/blender_animate.py` and `_cycles_script.py` (`_grass_material` delegates via `_scene_builders()`;
  `_cycles` keeps its OWN daytime sky/sun — it does not call `build_stadium_lighting`).
- **Light-from-clip (v2 lever 3, floodlit-night, auto+manual):** `src/pitch3d/adapters/render/lighting.py`
  (`estimate_light_color` — white-patch illuminant off bright near-neutral pixels, pure/unit-tested;
  `estimate_lighting_from_clip` — clip wrapper → the `lighting.npz` model; `NIGHT_*` measured defaults),
  `tests/unit/test_lighting.py`. AUTO: `scripts/anim_export.py` (`SOURCE_OK` gate → `lighting.npz`). MANUAL:
  `scripts/blender_animate.py` reads `lighting.npz` then lets `--light-rgb/--light-energy/--sky-strength/
  --sun-count/--sun-elevation/--sun-angle` override it, calls `build_stadium_lighting`, renders through the
  **Standard** view transform (broadcast-faithful colour, not AgX).
- **Deliverable video path (ADR-0011): virtual operator + export contract:** `src/pitch3d/core/scene/cameras.py`
  (`plan_virtual_cameras` — fixed in-bowl mounts broadcast/sideline/goal/top with per-frame look+fov;
  `action_track` — median centroid + bulk-quantile (q80) radius, straggler-robust, zero-phase smoothing;
  `project_normalized` — the framing contract check), `src/pitch3d/adapters/blender/anim_contract.py`
  (`SCHEMA_VERSION=1`, `write_manifest`/`load_manifest` — BOTH processes validate; pitch3d-free, imported by
  file on the Blender side), `src/pitch3d/app/anim_export.py` (`main(argv)`, flags > env > `.env`; writes
  subjects/ball/pitch/[stadium/lighting]/`cameras.npz` + `manifest.json`; `scripts/anim_export.py` = thin
  shim), `scripts/blender_animate.py` (manifest gate FIRST; `cameras.npz` → `sensor_fit=HORIZONTAL`,
  per-frame `_look_at` + `angle`, `clip_end=2000`; legacy bbox cams = fallback only; prints
  `BLENDER_ANIM_CAMS virtual-operator|static-legacy`), `scripts/video_defaults.sh` (single source of wrapper
  knob defaults), `tests/unit/test_virtual_cameras.py` (9), `tests/unit/test_anim_contract.py` (8),
  `tests/e2e/test_video_path_smoke.py` (4, gated on torch/smplx/SMPL-X models/Blender).

---

## Added 2026-08-08 — four subsystems this map had no entry for

Found while rewriting [`architecture.md`](architecture.md) §4: these exist, are load-bearing, and
appeared here zero times.

- **The browser annotator / Studio (`poseannot/`):** 12 modules at the **repo root, not under
  `src/`** — `poseannot/app.py` (37 routes), `scene_state.py` (FK cache, 5–22 s prewarm on a
  startup thread), `edits.py` (`Correction` rows → `edits.json`, atomic whole-file rewrite),
  `rerun.py` (12 correction gates as an ephemeral layer; 4 more declared unavailable because they
  need live-pipeline providers), `studio.py` (read-only stage manifest), `clips.py` (runtime clip
  switch, no restart), `pitch_evidence.py` (is there paint under the line we drew?),
  `camera.py`/`auth.py`/`config.py`/`video.py`. It imports **`pitch3d.core.*` only** — 25 imports,
  zero adapters. Mirror test: `tests/unit/test_gate_chain_parity.py`. Detail:
  [`poseannot-architecture.md`](poseannot-architecture.md).
- **Physics config (`core/config/`):** `physics.py` (`load_physics_config`, `PhysicsConfig`,
  `.summary()`; precedence shipped YAML → named profile → `PITCH3D_*` env → Python overrides, with
  per-scalar `.lineage`), `gates.py` (the per-gate dataclasses; **no package-internal imports**,
  which is what breaks the `physics.py` ↔ `core.correction` cycle). Data: `config/physics.yaml`,
  `config/player_priors.yaml`.
- **Two ids, one human (#135 П3/П2):** `src/pitch3d/core/orchestration/handover.py`
  (`merge_handovers`, `HandoverConfig`, `HandoverReport.suspect`),
  `src/pitch3d/app/controller.py` (runs it after `add_temporal_coherence`, before the physics
  gates), `src/pitch3d/app/cli.py` (`--handover`), `tests/unit/test_handover.py` (9,
  mutation-checked), probe `scripts/bench_handover_stitch.py`, A/B `scripts/view_handover_ab.sh`.
- **Benchmark harness (`src/pitch3d/eval/`):** `harness.py` (Condition A = GT camera isolates the
  pose net, Condition B = our PnLCalib — the A→B gap is what calibration costs), `synthetic.py`,
  `dataset.py` + `datasets_3dpw.py`/`datasets_soccernet.py`, `metrics.py` (`mpjpe_global`,
  `mpjpe_local`), `calib_metrics.py`, `novel_view.py` (R7/#99). **Outside the hexagon** — it
  imports `adapters.models.{calibration,pose}` directly, and nothing in the pipeline imports it.
- **Media ingest + player priors:** `src/pitch3d/adapters/io/` (ffprobe probe, `Source`/`ClipRef`,
  frame extraction), `src/pitch3d/adapters/profiles/local_json.py` (`LocalJsonPlayerStore` behind
  the profile port, backing `core/scene/player_profile.py`).
