# pitch3d — STATUS (single source of truth)

<!--
  LLM COLD-START DOC. This file is the durable project tracker and the primary
  context-reload surface for a Claude Code session that starts with no memory.
  Claude Code sessions break without recovery and the context window compresses
  with data loss, so DO NOT rely on the CC task list (#...) or chat history.
  Everything project-relevant lives HERE, in git. Update this file and COMMIT at
  every meaningful step (decision, defect, status change, validation result).
  Keep it dense, structured, path/command-explicit — optimised for an LLM to
  rehydrate fast, not for prose.
-->

**Last updated:** 2026-06-28 · **Branch:** main · **Repo:** /home/chubuchnyi/AVATAR

---

## 0. TL;DR for a cold-start LLM

- **Goal:** from ONE broadcast clip → a realistic novel-view video of the *same* episode (different camera angle). Players look like originals (kit + shirt numbers), same realistic stadium. **Judged by eye.**
- **Mode:** results over process. Do NOT tick milestones / wire seams / pass tests on fake adapters. Only do work that makes the real-clip output visibly better.
- **Current focus:** **v1 (recognizability) COMPLETE (2026-06-28)** — v0 geometry DONE (2026-06-27). **Kit colours DONE** (measured, 10/10 split). **Shirt numbers DONE** (plate-on-back; 4/20 read, rest honestly blank). **Stadium DONE** (hybrid procedural bowl + measured crowd projection + copy-filled holes). **Next stage: v2 (photoreal).**
- **NEXT ACTION:** **v1 COMPLETE — stadium step DONE** (validated locally, no pod). Hybrid: a procedural seating **bowl** ringing the pitch wears crowd colour **measured** by projecting the clip onto it through the solved camera; stands the camera never saw are **copy-filled** from their mirror. Core `core/scene/stadium.py` (geometry + fill, 8 tests), adapter `adapters/render/stadium_backdrop.py` (median bake), wired through `anim_export.py` (`stadium.npz`, gated on `PITCH3D_STADIUM_VIDEO`) → `blender_animate.py` (emission vertex-coloured bowl). **KEY FINDING:** the solved broadcast camera is **rolled 180° vs the raw video** (whole reconstruction consistent in that rolled frame; bake auto-detects & rotates the frame before sampling — see §6). E2E rendered broadcast/goal/sideline/action — reads as the Colombia crowd around the pitch (`/tmp/val_frames_stadium/*`). **NEXT STAGE → v2 (photoreal): textured/Gaussian avatars + photoreal stadium + view-synth — scope with the user before starting.**
- **Target clip:** `samples/video/Colombia-1-0-Congo-DR1080p.mp4`

---

## 1. Goal (confirmed 2026-06-27)

From a source broadcast clip → a **realistic novel-view video of the SAME episode** (a different
camera angle), as faithful as possible. Players look like the originals (same kit + shirt numbers);
the stadium is realistic and the same as the source. **Judged by eye.**

- **Approximations OK** for exact numbers / exact stadium when unrecoverable from one clip — backstopped
  by **manual Blender editing** + **generative prompt-editing** (ADR-0008 LLM-over-MCP).

## 2. Staged bar (do in order; gate each on eye-judgement)

- [x] **v0 — correct GEOMETRY (ACHIEVED 2026-06-27).** Stable ~22 players (measured **20**), correct
  world placement/scale (root spread **34×40 m**), poses, virtual cameras that frame the action, pitch
  with lines + goals. Validated end-to-end on the pod (`out/val/`); see §6. First "good" result.
- [ ] **v1 — recognizability (CURRENT FOCUS).** [x] Team **kit colours** — measured from torso pixels;
  split repaired 19/1 → **10/10**; A=yellow, B=light-blue (validated 2026-06-27, see §6). [x] shirt
  **numbers** — plate-on-back; 4/20 read (OCR≈0 at this resolution → manual high-conf read), rest
  honestly blank (validated 2026-06-28, see §6). [x] **stadium backdrop** — hybrid procedural bowl +
  measured crowd projection + copy-filled holes (validated 2026-06-28, see §6). **v1 COMPLETE.**
- [ ] **v2 — photoreal.** Textured/Gaussian avatars + photoreal stadium + view-synth (the gated
  `avatars`/`viewsynth` heavy halves). The full stated goal; a long research stage.

---

## 3. v0 punch-list — the work right now

Detail + exact code root-causes: [`v0-geometry-defects.md`](v0-geometry-defects.md).
Found by eye in the 300-frame render of the real clip (`out/anim/video/*`, real CUDA models, 4 virtual
cameras, 25 fps / 12 s / 1280×720). That run saved **no `scene.json`** → body count is visual-only.

| ID | Defect | Status | GPU? | Root cause (file:line) | Next step |
|----|--------|--------|------|------------------------|-----------|
| #202 | Too many bodies (track-ID fragmentation); swarm grows over clip | **VALIDATED on pod ✓ (20 subjects)** | done | was: stitch OFF unless flag; tracker `min_track_frames=1` keeps 1-frame blips | DONE + VALIDATED 2026-06-27: stitch ON by default everywhere + `wiring.py` real ByteTrack `min_track_frames=2`. **Pod 48f real run (`out/val/export/scene.json`) → `len(scene.subjects) = 20`** (vs the old swarm of dozens; target ~22). The "swarm grows over the clip" defect is gone. **KEY:** stitch is PIXEL-space → independent of #203. No further tuning needed for v0. |
| #203 | Depth collapse / wrong world scale (players not spread across pitch) | **VALIDATED on pod ✓ (34×40 m spread)** | done | NOT the identity fallback: the default render used `--calibrator fake` = `FakeFieldCalibrator` (`adapters/fakes/perception.py:125`), a 30 m **top-down orthographic toy** `H` with NO perspective. Numeric proof: whole frame → 30×17 m world box; realistic feet → 22.7×7.3 m; the ~7 m y-axis is the "thin horizon" blob. Real PnLCalib was wired (`adapters/models/pnlcalib_backend.py`) but opt-in & OFF. | DONE + VALIDATED 2026-06-27: real PnLCalib is now the **default** render calib (`pod_real_e2e.sh` defaults `PNLCALIB_REPO=/workspace/repos/PnLCalib`; run echoed `calibration: REAL PnLCalib`). **Pod 48f scene.json root spread: X (length) 34.1 m, Y (width) 40.0 m, Z (pelvis) 0.69–1.01 m** — i.e. real-scale spread within the touchlines, NOT the fake-calib 22×7 m collapse; the ~7 m "thin horizon" is gone. Players cluster in one half (localized broadcast action), as expected. **Deepest root — also fixes #204.** |
| #204 | Virtual cameras don't frame the action (players tiny at horizon) | **VALIDATED on pod ✓ (broadcast frames the pitch)** | done | the VIDEO render (`blender_animate.py`) derives its OWN cameras from `ctr`/`span` of the loaded geometry — NOT `viewpoints.py`/`controller.py` (those drive the *in-pipeline* render, a different path). With #203's collapsed 22×7 m blob + a bare plane, the cams framed empty grass. | DONE + VALIDATED 2026-06-27: #205 folds the FULL pitch bounds into `ctr`/`span` + #203 puts bodies on the field. **Pod render eye-check (`out/val/video/{broadcast,top}.mp4`): broadcast camera frames the whole pitch at a realistic oblique angle (players fill the action third, not a horizon speck); top camera frames the full 105×68 m.** No per-frame tracking camera needed for v0. |
| #205 | Bare pitch (no lines / no goals) | **VALIDATED on pod ✓ (lines + goals render)** | done | the ACTUAL video render is `scripts/blender_animate.py` (builds its scene from scratch — only drew a bare grass plane); pitch lines existed only in the *other* (in-pipeline Cycles) path, and the goal mesh was genuinely absent | DONE + VALIDATED 2026-06-27: measured `goal_frame_geometry()` + `pitch_line_ribbons()` in pure core; `anim_export.py` writes `pitch.npz`, `blender_animate.py` builds `pitch_lines` + `goals`. **Pod 48f run: `anim_export` logged `pitch: 2848 line-tris + 72 goal-tris (105x68 m)`; render eye-check confirms full markings (boxes, circle, arcs, halfway) + goal frames on both views.** |
| #206 | Ball lands OUTSIDE the pitch (surfaced during v0 validation) | **VALIDATED ✓ (0/48→48/48 in-pitch); contact-anchored** | no (local) | monocular height ambiguity: the ball is airborne the whole window (image-v 449–525 above all feet 556–894) so ground-plane un-projection overshoots ~13 m, and the old lift drew a straight line between the two **frozen** WASB endpoints (both off-pitch). `field.py` + `ball_lift.py`. | DONE + VALIDATED 2026-06-27 (local, no pod) — **contact-anchoring** per user instruction: new pure-core `detect_ball_contacts()` finds frames where the ball 2D lands on a player's projected foot (new `FieldCalibration.world_to_image`), keeps one anchor/player, gates by plausible speed (≤35 m/s) to drop depth-spurious matches; `lift_ball_to_3d(motions=…)` pins ball XY to those feet + ballistic Z, falls back to mono projection if no contact; stale (frozen) frames excluded; wired in `pipeline.py`. **Re-run on `/tmp/val_scene.json`: ball in-pitch 0/48 → 48/48; recovered the pass t12@f10 (−36.3,−24.8) → t18@f19 (−32.8,−28.1).** 7 new tests; full suite green. **Visual VALIDATED:** top-down schematic (`/tmp/ball_fix_topdown.png`, old ball red off-pitch vs new blue on-pitch) + full Blender re-render (top + broadcast from `/tmp/val_scene_fixed.json`) — ball among players on the field. #206 CLOSED. |

**Validation — DONE 2026-06-27 (pod `zueopp6nzozxb7`, 48f real run, `out/val/`):** the four v0 *player/
pitch* defects validated end-to-end. #202 body count = **20** (scene.json); #203 root spread = **34×40 m**
real-scale; #204 broadcast/top cameras frame the action; #205 pitch lines + goals render. Numbers from
`scene.json`, eye-judged from `out/val/video/{broadcast,top}.mp4`. Pod STOPPED. **v0 player geometry:
ACHIEVED.** Validation also surfaced **#206** (ball off-pitch); diagnosed + fixed locally
(contact-anchoring) and **VALIDATED 0/48→48/48** — #206 now CLOSED, so **all of v0 geometry is done**.

---

## 4. Conventions & commands (rehydration cheat-sheet)

**Run from repo root** `/home/chubuchnyi/AVATAR`. Local Python env is `.venv` (CPU/core work needs no GPU).

```bash
# tests / lint / types (pure-core work is fully testable locally)
.venv/bin/python -m pytest                 # full suite (~557 tests baseline)
.venv/bin/python -m pytest tests/<path>    # focused
.venv/bin/ruff check <files>               # lint
.venv/bin/mypy <files>                     # types

# end-to-end CLI (real pipeline entrypoint)
.venv/bin/python -m pitch3d.app.cli ...    # see app/cli.py for args (e.g. --stitch, --real-calib)
```

**Lint policy:** repo has ~46 pre-existing ruff violations. Lint **only changed lines** — baseline-diff
with `git show HEAD:<file> | .venv/bin/ruff check --stdin-filename <file> -`. Don't "fix" unrelated
pre-existing violations.

**Git:** commit at every checkpoint (durability). Push uses a dedicated SSH key:
```bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/<fifa_key>' git push   # remote: git@github.com:chubuchnyi/fifa.git
```
(see memory `reference_github_push` for the exact key path.)

**GPU pod (RunPod):** ~$0.74/hr — **STOP it whenever not actively rendering**; the network volume
persists and restart is free. The pod repo `/workspace/fifa` is a **stale mirror**: it shows
already-pushed work as "uncommitted". Reconcile by byte-comparing vs pushed HEAD, then
checkout+rm+pull — never blind-commit on the pod. (memory `reference_pod_git_state`, `feedback_pod_cost`.)

**Architecture (ADR-0001 hexagonal):** pure `core/` (numpy/CPU, unit-tested) + `adapters/` (heavy ML
behind extras, lazy-imported via dotted-path injection, ADR-0006). Corrections are the sole edit path
(ADR-0002). LLM-over-MCP editing, human≡LLM (ADR-0008/0010). R-6 honesty: mark/interpolate, never
fabricate or silently hide.

**Working rules (from user feedback memory):**
- Results over process — only result-bearing work; no milestone flips / seam-wiring / fake-adapter tests.
- Verify the WHOLE decode→…→export path every iteration, not just the changed unit.
- Track everything here in `docs/STATUS.md` + commit; CC task list is at most a transient mirror.
- Be decisive when scope is known; don't over-poll or wait for obvious confirmations.

---

## 5. Code map (where the v0 work lives)

- **Tracking / fragmentation (#202):** `src/pitch3d/adapters/models/tracking.py` (`ByteTrackTracker`,
  `min_track_frames`, `ByteTrackBackend.associate`), `src/pitch3d/core/orchestration/pipeline.py`
  (`stitch_cfg` gate), `src/pitch3d/core/orchestration/continuity.py` (`StitchConfig`,
  `stitch_tracks_with_report`), `src/pitch3d/core/orchestration/assemble.py` (one Subject per track_id),
  `src/pitch3d/app/cli.py` (`--stitch`, `run_dry_run`).
- **Calibration / world scale (#203):** `src/pitch3d/core/scene/field.py` (`image_to_world`),
  `src/pitch3d/adapters/models/pose.py` (`_ground_root`, foot=bbox bottom),
  `src/pitch3d/adapters/models/calibration.py` (identity fallback; `CameraModuleFieldCalibrator`),
  `src/pitch3d/core/scene/units.py` (`FieldDimensions` 105×68 m).
- **Cameras (#204):** `src/pitch3d/core/agent/viewpoints.py` (`standard_viewpoints` 63 m radius,
  `action_centroid`), `src/pitch3d/app/controller.py` (`_static_camera`, frozen frame-0).
- **Render / pitch / goals (#205):** `src/pitch3d/adapters/blender/_cycles_script.py` (`_add_ground`,
  `_build_pitch`), `src/pitch3d/adapters/render/cycles.py` (`draw_pitch`), `src/pitch3d/core/scene/pitch.py`
  (markings). Goals: add a `_build_goals()` mesh (absent today).
- **Hybrid stadium backdrop (v1 step 3):** `src/pitch3d/core/scene/stadium.py` (`stadium_bowl_geometry`,
  `fill_holes_by_copy`, `_rounded_rect_loop` — pure numpy), `src/pitch3d/adapters/render/stadium_backdrop.py`
  (`bake_backdrop_colors` — projective median bake; auto-rotates each frame 180° to match the solved
  camera's rolled pixel convention), `tests/unit/test_stadium_geometry.py` (8 tests). Wired through
  `scripts/anim_export.py` (`PITCH3D_STADIUM_VIDEO` → `stadium.npz`) and `scripts/blender_animate.py`
  (`_add_vertex_colored_mesh` — emission-driven vertex colours so the crowd renders at clip brightness).

---

## 6. Progress log (newest first)

- **2026-06-28** — **v1 step 3 LANDED: hybrid stadium backdrop (procedural bowl + measured crowd).**
  Third/last v1 sub-task → **v1 recognizability COMPLETE.** Approach (agreed w/ user): "гибрид A + B,
  дыры дорисовываем копируя с имеющегося" = a procedural seating **bowl** ringing the pitch, given
  REAL appearance by projecting THIS clip onto its vertices through the solved camera, then
  **copy-filling** the stands the camera never saw (its own near side) from their long-axis mirror.
  **Mechanism:** pure-core `core/scene/stadium.py` — `stadium_bowl_geometry` (rounded-rect footprint
  `apron` m outside the touchlines, swept up+out through raked tiers; each vertex carries
  `(angle_frac, height_frac)`) + `fill_holes_by_copy` (uncovered vertex ← covered vertex nearest its
  `(x,−y,z)` mirror, fallback nearest covered). Adapter `adapters/render/stadium_backdrop.py` —
  `bake_backdrop_colors` projects every bowl vertex per frame and takes the per-vertex **median**
  RGB over the frames it was visible (median rejects a player/ball crossing a low vertex).
  `anim_export.py` builds+bakes+fills and writes `stadium.npz {verts,faces,colors}` (gated on
  `PITCH3D_STADIUM_VIDEO`); `blender_animate.py` loads it as a **vertex-coloured, emission** bowl
  (crowd renders at its measured brightness, lighting-independent) deliberately **excluded from the
  camera-framing bbox** (else it zooms every cam out until players are specks).
  **KEY FINDING (R-6, non-obvious): the solved broadcast camera is rolled 180° relative to the RAW
  decoded video.** The pitch model, the SMPL-X bodies AND the bowl all project onto the frame turned
  **upside down**, not as decoded — verified three ways: (a) overlaying projected pitch lines, (b)
  body root positions, and (c) bowl far-stand all land on the real features only in the rot-180 frame;
  (d) far +Y stand sampled **100 % green (pitch)** from the upright frame but **6 % green (94 % crowd)**
  from rot-180. The camera's image axes read `+u=world−X`, `+v=world+up` (image-up · world-up < 0).
  The whole reconstruction is **internally consistent** in that rolled frame, so nothing else needs
  changing — but the bake is the one place that samples raw video, so `bake_backdrop_colors`
  **auto-detects** the roll (`−R[1][2] < 0`) and rotates each decoded frame 180° before sampling.
  Any FUTURE raw-video consumer (validation overlays, jersey texturing) must do the same.
  **Validation:** 8 geometry unit tests pass (`tests/unit/test_stadium_geometry.py`); bake on the clip
  = **48 % covered** (far +Y sideline + ends seen, near −Y side the predicted hole → mirror-filled, 0
  black left); E2E rendered (`anim_export → stadium.npz → blender_animate`) from broadcast / goal /
  sideline / action cams — the bowl reads clearly as the **Colombia (yellow/red) crowd** wrapping the
  pitch, players composed in front (`/tmp/val_frames_stadium/*`, `/tmp/val_frames_action/*`). **Known
  v1 limitation:** the lowest far-stand rows sample the LED ad boards + grass margin (a light band at
  the foot of the stand) — that is genuinely what the camera saw there; clipping the bottom rows or a
  taller/steeper bowl is v1-polish/v2. Changed: `core/scene/stadium.py` (NEW), `tests/unit/
  test_stadium_geometry.py` (NEW), `adapters/render/stadium_backdrop.py` (NEW), `scripts/anim_export.py`,
  `scripts/blender_animate.py`. ruff+mypy clean on the new core/adapter; scripts ruff-clean (pre-existing
  blender_animate I001/E501 + anim_export numbers-block mypy left untouched). **v1 DONE → next is v2.**

- **2026-06-28** — **v1 step 2 LANDED: shirt numbers as a plate on the back.** Second v1 sub-task.
  **OCR reality (measured, R-6):** generic `easyocr` yields ≈0 on this clip's back-of-jersey digits
  (~20-30 px at 1080p), so I read them manually from upscaled, reprojected back-crops and assigned
  **only the high-confidence** ones: **#10** (track 1) & **#25** (track 5) Colombia/yellow, **#20**
  (track 8) & **#12** (track 17) Congo/blue. The other 16 backs were illegible in this 48-frame
  window → `jersey_number=None` (no fabrication). A real per-frame OCR/jersey model is deferred to
  v2/pod. **Render mechanism:** `anim_export.py` bakes, per numbered subject, a per-frame upper-back
  anchor (`0.62·spine3 + 0.38·neck + 0.19·back`, pushed proud so it floats on the curved skin), the
  posterior horizontal normal `back_dir`, and a luminance-picked contrast colour (dark on yellow,
  white on blue) into the npz. `blender_animate.py` builds a centred FONT plate per number and
  orients it each frame so its face points along `back_dir`; an **"action" camera** frames just the
  player cluster (the broadcast cam frames the whole 105×68 m pitch → players tiny). **Two bugs
  caught by eye-check** (close-ups `/tmp/num_closeup/closeup_t{1,8}.png`): the plate basis used
  `x = n×up`, which rolled the text 180° (upside-down **and** mirrored) — fixed to `x = up×n` so
  text-up = +world-Z; and a 0.12 m offset buried the plate in the mesh (only fragments poked through)
  — raised to 0.19 m. After the fix, **10 / 20 read upright, unmirrored, clearly on the right backs.**
  At the action-cam framing (~40 m span, all 20 players) digits are still small in the wide shot —
  per-frame legibility at distance is a v1-polish/v2 (UV jersey) concern, not a plate bug. UV-texture
  jersey explicitly **deferred to v2**. Changed: `scripts/anim_export.py`, `scripts/blender_animate.py`;
  `anim_export` ruff-clean, full path re-rendered without error. **NEXT v1 sub-task → simple stadium.**

- **2026-06-27** — **v1 step 1 LANDED: measured kit colours + repaired the 19/1 team split → 10/10.**
  First v1 (recognizability) sub-task. Two coupled bugs found while validating the real-clip teams: (1) the
  classifier collapsed **19/1** — `_assign_teams` ran euclidean k-means on **raw mean-HSV**, but OpenCV hue
  is circular (H∈[0,180]) and a single bright/shadow (high-V) torso becomes the farthest point, so it seeds
  a 1-vs-rest split instead of splitting on kit colour; (2) `Team.color_rgb` was **never set**, so the render
  fell back to an arbitrary tab10 palette (teams were A/B but not the real shirts). **Fix
  (measured-over-generative, pure core):** new `_hsv_to_feature` maps mean-HSV → a hue-aware, euclidean-safe
  chroma feature `[sat·cos(hue), sat·sin(hue), 0.25·val]` (respects circularity, downweights brightness) and
  `_assign_teams` now sets each `Team.color_rgb` to its cluster's measured mean (`_hsv_to_rgb01`, pure numpy);
  the heavy `_sample_appearance` backend samples a **central upper-torso patch**, rejects grass-green, and
  medians per-frame + across the first 8 frames. **Validated locally, no pod** (`/tmp/kit_measure.py`):
  reproject each subject's foot world-XY via `FieldCalibration.world_to_image` (the #206 `H⁻¹`) to sample
  real torso pixels → split is now **10/10** (was 19/1), **Team A = yellow (Colombia) RGB (0.689,0.651,0.275)
  H≈27**, **Team B = light-blue (Congo DR) RGB (0.302,0.524,0.647) H≈100**, consistent hues per cluster.
  Colours bake through to `anim_subject_*.npz` (t6,t14→yellow; t3,t11→blue); Blender re-render
  `/tmp/val_frames_v1/{top,broadcast}` + zoomed crops `/tmp/v1_zoom_{9,15}.png` show two clearly distinct
  team colours on the pitch. 2 new tests (`test_team_color_rgb_is_measured_from_the_cluster`,
  `test_brightness_outlier_does_not_collapse_the_split`); full suite **555 passed / 10 skipped**; ruff + mypy
  clean. **NEXT v1 sub-tasks:** shirt numbers (OCR where readable, else roster), simple stadium backdrop.
- **2026-06-27** — **#206 ball fix LANDED + VALIDATED → all of v0 geometry now CLOSED.** Root (confirmed
  locally on `/tmp/val_scene.json`, no pod): the ball is airborne the entire 48-frame window (its image-v
  449–525 sits above every player foot at v 556–894), so ground-plane un-projection overshoots ~13 m, and
  the old lift compounded it by treating the two **frozen** WASB endpoints (exact-duplicate pixels — the
  tracker lost the ball) as ground contacts and drawing a straight line between two off-pitch points. **Fix
  — contact-anchoring** (user's instruction "положение игроков, касающихся мяча, брать за основу"): new
  pure-core `detect_ball_contacts()` (`core/orchestration/ball_lift.py`) finds frames where the ball's 2D
  lands on a player's **projected foot** (new `FieldCalibration.world_to_image` = the `H⁻¹` of
  `image_to_world`), keeps one anchor per player, and gates by a plausible-speed test (≤35 m/s) to drop
  depth-spurious "nearest player" matches; `lift_ball_to_3d(motions=…)` pins ball XY to those feet,
  interpolates between, keeps gravity-parabola Z, and falls back to mono projection when no contact; stale
  frozen frames excluded; wired in `pipeline.py`. **Validated:** re-running the real core lift → ball
  in-pitch **0/48 → 48/48**, recovering exactly the kick→receive pass the user described — **t12 @ f10
  (−36.3,−24.8) → t18 @ f19 (−32.8,−28.1)** (~4.8 m, ~13 m/s, Z apex 0.16 m). Top-down schematic
  (`/tmp/ball_fix_topdown.png`): old ball red beyond the touchline (Y≈−38) vs new ball blue on-pitch on
  the t12→t18 line. Full Blender re-render (top + broadcast from `/tmp/val_scene_fixed.json`): ball among
  players inside the pitch. 7 new tests (`test_ball_lift.py` + `test_field_calibration.py`); ruff + mypy
  clean; full unit suite green. **#206 CLOSED → v0 geometry (#202–#206) fully done. NEXT BAR → v1
  (recognizability):** kit colours (teams already A/B), shirt numbers, simple stadium.
- **2026-06-27** — **v0 = correct GEOMETRY: ACHIEVED & VALIDATED end-to-end on the pod.** ONE batched
  48-frame real run (pod `zueopp6nzozxb7`: RF-DETR · ByteTrack · real PnLCalib · SMPLest-X · WASB →
  `out/val/export/scene.json`, then a cheap top/broadcast re-render via `REUSE_SCENE=1`). Measured: **#202
  body count = 20** (`len(scene.subjects)`; was a swarm of dozens) — CLOSED. **#203 root spread = 34 m
  (length) × 40 m (width)**, pelvis 0.69–1.01 m, run echoed `calibration: REAL PnLCalib` (the fake-calib
  22×7 m collapse is gone) — CLOSED. **#204** broadcast + top cameras frame the action (no horizon speck)
  — CLOSED. **#205** `anim_export` logged `pitch: 2848 line-tris + 72 goal-tris (105x68 m)`; render shows
  full markings + goal frames — CLOSED. Eye-judged from `out/val/video/{broadcast,top}.mp4` (local:
  `/tmp/val_video/`, scene `/tmp/val_scene.json`). Pod STOPPED ($0.012/hr storage only). All four v0
  tasks #202–#205 complete. **NEXT BAR → v1 (recognizability):** kit colours (teams already A/B), shirt
  numbers, simple stadium. ⚠️ Validation **surfaced #206**: the ball's `positions_3d` land OUTSIDE the
  pitch (Y≈−38 m vs touchline ±34; X up to −53.6 vs goal line −52.5; ~30 m from players), with
  `height_confidence` mean 0.25 / `on_ground` 2/48 — monocular height ambiguity (ground-plane
  un-projection of an airborne/edge detection). Diagnosable locally from `/tmp/val_scene.json`; new task
  #206.
- **2026-06-27** — **#203 root found locally + fix landed (no pod needed for diagnosis).** Traced the
  depth collapse to the *calibrator*, not a degenerate homography: the default render path uses
  `--calibrator fake` (`cli.py` default) = `FakeFieldCalibrator`, a **top-down orthographic toy** with
  `_FAKE_PITCH_SPAN_M = 30.0` — it maps the WHOLE frame into a 30×17 m world box (the 105×68 m pitch is
  unrepresentable) and has no perspective term, so oblique broadcast depth folds onto a ~7 m band.
  Proven numerically on this machine (realistic feet → 22.7×7.3 m world span). Real PnLCalib was wired
  (`pnlcalib_backend.py`) but opt-in & OFF. **Fix (mirrors #202):** made real PnLCalib the DEFAULT —
  `pod_real_e2e.sh` defaults `PNLCALIB_REPO=/workspace/repos/PnLCalib` (proxy only if the staged repo is
  truly absent; force with `PNLCALIB_REPO=`), `demo_video.sh` flips `REAL_CALIB`→1 (`--no-real-calib` to
  opt out). This is the deepest root and also addresses #204 (the video cameras already frame the full
  pitch via #205's bounds fold; #203 puts bodies on it). Both scripts `bash -n` clean; default-resolution
  logic unit-checked. Still must be VALIDATED on the pod (confirm spread in `scene.json`).
- **2026-06-27** — **#205 code done + validated locally.** The real video render is
  `scripts/blender_animate.py` (builds its scene from scratch — it only drew a bare grass plane), so
  the fix lives there, NOT in the in-pipeline Cycles adapter the original root-cause note guessed.
  Added measured `goal_frame_geometry()` (2 posts + crossbar at Laws dims: 7.32 m mouth, 2.44 m high,
  0.12 m square section) to pure core `core/scene/pitch.py`, beside the existing `pitch_line_ribbons()`;
  `anim_export.py` now writes `pitch.npz` (line ribbons + goal frames, world metres) and purges it with
  the other artifacts; `blender_animate.py` loads `pitch.npz`, folds the pitch bounds into the camera
  framing (so the whole field stays in frame), and builds `pitch_lines` + `goals` meshes (bare-plane
  fallback if absent). 3 new geometry tests (now 12 in `test_pitch_geometry.py`). Validated by rendering
  the fixtures locally with the Blender binary (top view = full markings; goal close-up = correct
  posts+crossbar). Full tree green (560 passed, 12 skipped); changed-line ruff clean; mypy clean.
- **2026-06-27** — **#202 local fix landed.** Made stitch the uniform default across ALL
  reconstruction entrypoints (CLI flag flipped `--stitch`→`--no-stitch`, default ON;
  `pod_real_e2e.sh` + `pod_make_video.sh` default stitch on) and set the real ByteTrack path's
  `min_track_frames=2` in `wiring.py` to drop un-stitchable 1-frame singletons. Discovered while
  diagnosing: stitch runs in PIXEL space (so #202 is *independent* of #203's homography), the tracker
  `min_track_frames` filter runs *before* stitch (so raising it too high starves stitch), and
  `demo_video.sh` already defaulted stitch on — i.e. the 300f swarm appeared *with* stitch on, so the
  body-count fix must be MEASURED on a pod re-run (read `len(scene.subjects)`), not blind-tuned.
  Full suite green (557 passed, 12 skipped); changed-line ruff clean; mypy clean. Body-count
  validation deferred to the batched pod run.
- **2026-06-27** — Reformatted this file into an LLM-friendly cold-start doc (dense sections +
  conventions/commands + code map). Pushed docs reset (7af66e9). Starting v0 punch-list #202.
- **2026-06-27** — Strategic reset: results over process. Defined the goal + v0→v1→v2 ladder. Inspected
  the first real 300-frame render (`out/anim/video`, real Colombia clip, real CUDA models) → found 4 v0
  geometry defects (#202–#205) and located their root causes in code. Created this tracker +
  `v0-geometry-defects.md`; added a results-first reset banner to `roadmap.md`.

---

## 7. Key references

- **v0 defects (detail + code root-causes):** [`v0-geometry-defects.md`](v0-geometry-defects.md)
- **Historical build log (M0–M4 = platform plumbing, NOT result quality):** [`roadmap.md`](roadmap.md)
- **M1 live state:** [`m1-status-and-plan.md`](m1-status-and-plan.md)
- **Memory (outside repo):** `feedback_results_over_process`, `project_goal_definition`,
  `feedback_durable_tracking`, `feedback_pod_cost`, `reference_github_push`, `reference_pod_git_state`.
