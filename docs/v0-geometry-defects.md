# v0 geometry defects — real-clip novel-view reconstruction

**Goal context:** the deliverable is a faithful *novel-view* video of a real broadcast episode; the
agreed first bar is **v0 = correct geometry** (stable ~22 players, right placement/scale/poses, framed
cameras, pitch lines). See `feedback_results_over_process` / `project_goal_definition` in memory.

**Where these came from:** found by eye in the 300-frame render of the real clip
`samples/video/Colombia-1-0-Congo-DR1080p.mp4`, output at
`out/anim/video/{broadcast,sideline,top,goal}.mp4` (real CUDA models, 4 virtual cameras, 25 fps / 12 s
/ 1280×720). Frames inspected were extracted to `/tmp/anim_frames/*.png` (ephemeral).

**Tracked as tasks #202–#205.**

---

## D1 — Too many bodies / track-ID fragmentation  (task #202, user's #1)
- **Symptom:** top & sideline views show a swarm of many dozens of bodies, not ~22 players; the swarm
  densifies over the 12 s (`top_t11` > `top_t00`).
- **Evidence:** `top_t00`/`top_t06`/`top_t11`, `sideline_t06` (colored bodies strung along the horizon).
- **Comparison:** short real runs (`out/live_real`, `out/cuda`) produced 6 bodies → body count scales
  with clip length.
- **Hypothesis:** ByteTrack ID fragmentation — each occlusion / ID-switch spawns a new *persistent*
  subject; no re-id / track-merge / minimum-track-length gating.
- **Root cause (code):** `adapters/models/tracking.py:220-257` keeps every ByteTrack ID (no
  re-id / gap-bridge); the only gate is `min_track_frames`, default **1** (`tracking.py:153-169`). A
  fragment-stitch pass `stitch_tracks_with_report()` (`core/orchestration/continuity.py:157-223`) is
  wired at `pipeline.py:96-98` but only runs when `stitch_cfg` is non-None — it defaults to None (CLI
  `--stitch`, off). `assemble_scene()` (`core/orchestration/assemble.py:44-57`) then makes one Subject
  per track_id, no dedup / cap.
- **Fix:** stitch on by default (`pipeline.py` `stitch_cfg`) + raise `min_track_frames` to ~5–10 for a
  25 fps clip (`tracking.py:143`). Local, unit-testable, no GPU — the cheapest first win.
- **Note:** this run saved no `scene.json`, so the exact count is pending a re-run with export.

## D2 — Virtual cameras don't frame the action  (task #204)
- **Symptom:** broadcast & goal cameras render ~95 % empty field + sky; players are a faint speck
  cluster at the horizon.
- **Evidence:** `broadcast_t00/t06/t11`, `goal_t06`.
- **Hypothesis:** camera position / FOV / look-at is wrong or not derived from scene bounds (may be
  downstream of the scale bug D3).
- **Root cause (code):** `standard_viewpoints()` (`core/agent/viewpoints.py:99-145`) places cameras on
  a 63 m radius around `action_centroid()` (`viewpoints.py:74-89`, the mean subject root) — if D3
  collapses the roots, the camera targets empty pitch. The render also uses a single **static**
  broadcast camera frozen at frame 0 (`app/controller.py:479-490`); FOV is hardcoded
  (`viewpoints.py:49-52`), and there is no distinct "goal" camera in the Cycles path.
- **Fix:** mostly resolves once D3 is fixed (centroid becomes correct); then optionally a per-frame
  centroid-tracking camera instead of the frozen one.

## D3 — Depth collapse / wrong world scale  (task #203)
- **Symptom:** sideline shows bodies on a thin horizon line; top shows a tight blob — not spread across
  a 105×68 m pitch.
- **Evidence:** `sideline_t06`, `top_t06`.
- **Hypothesis:** calibration → ground placement collapses depth, or world units / scale are off.
- **Root cause (code):** the homography `H`. `FieldCalibration.image_to_world()`
  (`core/scene/field.py:39-51`) applies `H` to foot points; it is invoked per subject in
  `GVHMRPoseEstimator._ground_root()` (`adapters/models/pose.py:256-279`, foot = bbox bottom). If `H`
  is identity / degenerate (fallback at `calibration.py:333`; real path
  `CameraModuleFieldCalibrator.calibrate()` `calibration.py:422-455`), every foot maps to ~origin /
  pixel-space → no real spread across the 105×68 m pitch (`core/scene/units.py:60-63`).
- **Fix:** verify/repair calibration so `H` is valid for the clip (log `image_to_world` of a known
  foot point). This is the **deepest root** — it also drives D2. Likely needs a pod re-run with calib
  logging.

## D4 — Bare environment  (task #205)
- **Symptom:** green plane + grey sky; no pitch lines / markings / goals.
- **Evidence:** all views.
- **Impact:** reads as "specks on grass"; also removes the visual reference needed to judge placement.
- **Root cause (code):** ground + grass exist (`adapters/blender/_cycles_script.py:144-151`) and
  pitch-line geometry exists (`core/scene/pitch.py:66-100` → `_build_pitch()`
  `_cycles_script.py:155-176`; `draw_pitch` defaults True at `adapters/render/cycles.py:81`) — but only
  via the **Cycles** pass. The proxy / Workbench path draws no ground or pitch at all, so the 300-frame
  video likely used a non-Cycles path → bare plane. **Goal geometry is genuinely absent** (no goal mesh
  anywhere).
- **Fix:** render the video via the Cycles path (so existing pitch lines appear) + add a
  `_build_goals()` mesh for goal posts / nets.

---

## Status / next
- Recorded as tasks **#202–#205**; root causes located in code (above).
- **Cheapest first win:** #202 — a config flip (stitch on + `min_track_frames`↑); local, unit-testable,
  no GPU.
- **Deepest root:** #203 (homography) — fixing it also fixes #204; needs verifying `H` on the real clip
  (likely a pod re-run with calib logging).
- **#205** is partly a render-path choice (use the Cycles pass, which draws pitch lines) + new goal
  geometry.
- A single GPU re-run on the pod validates v0 end-to-end after the fixes, and must save `scene.json` so
  D1's count becomes measurable.
