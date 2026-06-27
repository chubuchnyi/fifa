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
- **Fix (LANDED 2026-06-27, local):** stitch made the **uniform default** across all reconstruction
  entrypoints — CLI flag flipped `--stitch`→`--no-stitch` (default ON), `pod_real_e2e.sh` +
  `pod_make_video.sh` default stitch on; the real ByteTrack path now sets `min_track_frames=2` in
  `app/wiring.py` to drop un-stitchable 1-frame singletons. Full suite green (557 passed).
- **Corrected understanding (important):** (1) the stitch pass reasons in **pixel space** (bbox centres;
  `max_center_dist` in bbox-widths — `continuity.py:34,79-81`), so #202 is **independent of #203's
  homography collapse**. (2) The tracker's `min_track_frames` filter runs **before** stitch
  (`pipeline.py`), so raising it too high *starves* stitch of fragments to re-link — hence keep it low
  (2) and rely on stitch's own post-link blip drop (`StitchConfig.min_track_frames=3`). (3)
  `demo_video.sh` **already** defaulted `STITCH=1`, so the 300f swarm appeared *with stitch on*.
- **Therefore — validation, not blind tuning:** the body count must be **measured** on a pod re-run
  (`scene.json` exists at `$OUT/export/scene.json`; read `len(scene.subjects)`). Only if it is still
  far above ~22 do we tune stitch gates (`max_gap`/`max_center_dist`) or add ByteTrack re-id — *after*
  seeing the number, never before.
- **VALIDATED 2026-06-27 (pod `zueopp6nzozxb7`, 48f real run → `out/val/export/scene.json`):**
  `len(scene.subjects) = **20**` (track_ids 1–18, 20, 31). Target ~22; the "swarm of dozens that grows
  over the clip" is **gone**. No further stitch/re-id tuning needed for v0. #202 CLOSED.

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
- **VALIDATED 2026-06-27 (pod `zueopp6nzozxb7`, `out/val/video/{broadcast,top}.mp4`):** with D3 fixed,
  the **broadcast** camera frames the whole pitch at a realistic oblique angle — players fill the
  action third (not a faint horizon speck), sky above, pitch lines visible; the **top** camera frames
  the full 105×68 m. The video render's cameras come from `blender_animate.py`'s `ctr`/`span` (pitch
  bounds folded in by #205), so no separate code change was needed. No per-frame tracking camera needed
  for v0. #204 CLOSED.

## D3 — Depth collapse / wrong world scale  (task #203)
- **Symptom:** sideline shows bodies on a thin horizon line; top shows a tight blob — not spread across
  a 105×68 m pitch.
- **Evidence:** `sideline_t06`, `top_t06`.
- **Hypothesis:** calibration → ground placement collapses depth, or world units / scale are off.
- **Hypothesis (original):** `H` is the identity/degenerate *fallback* (`calibration.py:333`).
- **Root cause (CONFIRMED 2026-06-27, local — not the identity fallback):** the defect render used the
  **`--calibrator fake`** path (the CLI default — `cli.py:361`), i.e. `FakeFieldCalibrator`
  (`adapters/fakes/perception.py:125-146`). That "proxy planar" calibrator is a **top-down orthographic
  toy**: `H` linearly maps image (u,v) → world with `scale = _FAKE_PITCH_SPAN_M / clip.width` and
  `_FAKE_PITCH_SPAN_M = 30.0`. So it (a) crams the *entire* frame into a **30 × 17 m** world box (the
  105×68 m pitch is unrepresentable) and (b) has **no perspective term** — an oblique broadcast can't be
  un-projected, so downfield depth folds onto a thin world-Y band. Real PnLCalib was wired
  (`adapters/models/pnlcalib_backend.py`) but is **opt-in and was OFF** in the default render path
  (`demo_video.sh` `REAL_CALIB=0`; `pod_real_e2e.sh` proxy unless `PNLCALIB_REPO` set). `image_to_world`
  (`core/scene/field.py`) and `_ground_root` (`adapters/models/pose.py`) are correct — they were just fed
  the toy `H`. **Numeric proof (this machine):** realistic broadcast feet → world span **22.7 × 7.3 m**;
  full-frame ceiling **30 × 17 m**. The y-axis (~7 m) IS D3's "thin horizon line / tight blob".
- **Fix (LANDED 2026-06-27, local):** make REAL PnLCalib the **default** in the render path (analogous
  to #202's stitch default). `pod_real_e2e.sh` now defaults `PNLCALIB_REPO=/workspace/repos/PnLCalib`
  and only uses the proxy when that staged repo is genuinely absent (the `-d` guard; force proxy with
  `PNLCALIB_REPO=`). `demo_video.sh` flips `REAL_CALIB`→1 by default (opt out with `--no-real-calib`).
  The backend's `make()` already defaults the weights to the pod paths. This is the **deepest root** —
  it also drives D2/#204 (cameras frame whatever the placement says). **VALIDATION (pod):** re-run the
  reconstruction (real calib now default), log `image_to_world` of a known foot point, and confirm the
  subjects spread across ~105×68 m in `scene.json` — only then is #203 closed.
- **VALIDATED 2026-06-27 (pod `zueopp6nzozxb7`, 48f real run; run echoed `calibration: REAL PnLCalib`):**
  subject-root world spread from `out/val/export/scene.json` (852 root samples across 20 subjects) =
  **X (length) 34.1 m** (−50.1 → −16.0), **Y (width) 40.0 m** (−29.4 → +10.7), **Z (pelvis) 0.69–1.01 m**.
  The fake-calib **22×7 m collapse is gone**: the y-axis (width) is now a real 40 m spread, not the ~7 m
  "thin horizon line", and pelvis heights are physically correct standing heights. Players cluster in one
  half (X all negative) — consistent with localized broadcast action, not a scale bug. #203 CLOSED.

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
- **Fix (LANDED 2026-06-27, local):** corrected understanding first — the real video render is
  `scripts/blender_animate.py`, which **builds its own Cycles scene from scratch** and simply never drew
  the pitch (only a bare grass plane); the pitch-line geometry that *did* exist lived only in the other,
  in-pipeline `cycles.py`/`_cycles_script.py` path, and a goal mesh was genuinely absent everywhere.
  So the fix is in the video path, not "switch to Cycles". Added measured `goal_frame_geometry()`
  (2 posts + crossbar, Laws dims) to pure core `core/scene/pitch.py` next to `pitch_line_ribbons()`;
  `anim_export.py` exports `pitch.npz` (line ribbons + goal frames in world m); `blender_animate.py`
  loads it, folds the pitch bounds into camera framing, and builds `pitch_lines` + `goals` meshes
  (bare-plane fallback if `pitch.npz` absent). 3 new tests (12 total in `test_pitch_geometry.py`).
  **Validated locally** with the Blender binary: top view shows the full markings, goal close-up shows
  correct posts+crossbar. Will appear automatically in the batched pod render.
- **VALIDATED 2026-06-27 (pod `zueopp6nzozxb7`, 48f real run):** `anim_export` logged
  `pitch: 2848 line-tris + 72 goal-tris (105x68 m) -> pitch.npz` (72 goal-tris = 6 boxes × 12, exactly
  `goal_frame_geometry`); the pod render (`out/val/video/{broadcast,top}.mp4`) shows the full markings
  (both penalty boxes, 6-yard boxes, centre circle + spot, penalty arcs, halfway line, touchlines) plus
  goal frames on the goal lines. The "specks on bare grass" defect is gone. #205 CLOSED.

## D5 — Ball lands outside the pitch  (task #206, surfaced during v0 validation)
- **Symptom:** the reconstructed ball sits **off the field** — well beyond the touchline and near/behind
  the goal line, ~30 m away from the player cluster.
- **Evidence (pod 48f run, `out/val/export/scene.json` → `BallTrack`):** `positions_3d` span
  **Y −39.4 → −37.8 m** (touchline is ±34), **X −53.6 → −37.9** (goal line −52.5), **Z 0.0 → 3.0**;
  player centroid (−33, −8) vs ball mean (−46, −38). `height_confidence` mean **0.25** (0.00 for many
  frames = the `low_ball_height` warnings in the run log); `on_ground` **2/48**; `track_2d` v≈450–525 of
  1080 (upper band of the frame), u sweeps 443→1650.
- **Hypothesis (root):** monocular height ambiguity — an airborne (or false/edge) 2D ball detection is
  un-projected onto the ground plane (z=0) via the homography, so it "shoots" past the real position to
  beyond the pitch edge. The system flags this honestly (low `height_confidence`), rather than fabricating
  a height. Code: `core/scene/field.py image_to_world` (ground-plane un-projection) + the ball lift in the
  ball backend / `assemble`.
- **Candidate fixes (diagnose locally first — `/tmp/val_scene.json`, no pod):** (a) pin ball to ground
  (z = ball radius) when `height_confidence` is low; (b) gate/clamp detections that un-project outside
  ±52.5 × ±34; (c) bias the ball toward the player cluster when 2D is ambiguous. Validate: ball positions
  land inside the pitch and near the action.

---

## Status / next — v0 PLAYER/PITCH geometry ACHIEVED 2026-06-27 (#202–#205); ball #206 OPEN
- **#202 / D1 — CLOSED:** body count = **20** (`out/val/export/scene.json`), no swarm.
- **#203 / D3 — CLOSED:** root spread **34 m (length) × 40 m (width)**, pelvis 0.69–1.01 m; real PnLCalib
  default; the fake-calib 22×7 m collapse is gone.
- **#204 / D2 — CLOSED:** broadcast + top cameras frame the action (`out/val/video/*.mp4`).
- **#205 / D4 — CLOSED:** full pitch markings + goal frames render (`pitch: 2848 line-tris + 72 goal-tris`).
- **Validation run:** ONE batched pod pass (pod `zueopp6nzozxb7`, 48 frames, real RF-DETR · ByteTrack ·
  PnLCalib · SMPLest-X · WASB), reused for the cheap top/broadcast re-render; pod STOPPED after. Local
  copies: `/tmp/val_video/{broadcast,top}.mp4`, `/tmp/val_scene.json`.
- **NEXT BAR → v1 (recognizability):** team kit colours (teams already split A/B), shirt numbers
  (OCR/roster), simple stadium backdrop.
