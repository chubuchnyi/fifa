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

**Last updated:** 2026-07-04 · **Branch:** main · **Repo:** /home/chubuchnyi/AVATAR

---

## 0. TL;DR for a cold-start LLM

- **Goal:** from ONE broadcast clip → a realistic novel-view video of the *same* episode (different camera angle). Players look like originals (kit + shirt numbers), same realistic stadium. **Judged by eye.**
- **Mode:** results over process. Do NOT tick milestones / wire seams / pass tests on fake adapters. Only do work that makes the real-clip output visibly better.
- **Current focus:** **v2 (photoreal) — STARTED 2026-06-28.** v1 (recognizability) COMPLETE; v0 geometry DONE. Plan «A через B, 1→2→3, свет из клипа» (port photoreal levers into the deliverable video path, share Blender scripts at the data layer). **Levers 1 (measured per-vertex body texture), 2 (grass-PBR via the shared `scene_builders.py`, the "B" refactor — ~5 m mowing stripes) + 3 (light-from-clip — floodlit-NIGHT, auto-detected colour + manual override) all DONE & eye-validated. The agreed 1→2→3 plan is complete.**
- **NEXT ACTION:** **CROWD STRUCTURE LANDED 2026-07-04 (§6): tier walkway/railing/offset
  aisles/top-fade overlay (`apply_stand_structure`; exporter flag `--crowd-structure` /
  `PITCH3D_CROWD_STRUCTURE`, default on) + `tile_gain` emission compensation — the renderer's
  unit-mean tile norm is scale-invariant, so darkening part of the texture silently BRIGHTENS
  seated rows (×1.41 → batch-1 amber "LED panels"). Reusable findings: ads boards OCCLUDE
  bowl-v < ~0.5 from the broadcast camera (stand features must sit at v ≥ ~0.55); measure
  stands with the stand-only ROI (y .17–.27, x .35–.65) — the wide crowd ROI is ~53 % sky.
  E2E batch 2: control stand V .224 S .486 (clip family), structure reads as architecture in
  control AND final. RESIDUAL (both batches): the generative tail re-saturates the stand to
  amber blobs (final S .94 vs control .49, clip .42); the v2v prompt's own "warm yellow and
  amber shirts" was the culprit. **Prompt-mute probe LANDED same day (§6): stand S .941→.737,
  V .298→.247, LED-panel glow gone — Wan AMPLIFIES stated colour adjectives, so state colours
  at the measured intensity. New best FINAL `out/struct_pod/tail1_pinned2.mp4`, sheet
  `out/struct_pod/stand_tail1_b2_clip.png`.** CROWD GRAIN fixed at the control same day (§6):
  quilt stitched at NATIVE tile px (`fan_scale`, default 1.0) + 16384×1024 canvas — control
  marble 7 px → 5-px speckle (clip ~2.7); pod E2E of the grain rides the NEXT batch. BOARDS
  TEXT landed same day (§6): measured «BANK OF AMERICA» strip cut from the clip (grass-boundary
  anchor; the aggregated camera is CONSTANT — projection only picks the run) and wrapped around
  the ring (optional tile/uv in boards.npz, normalize=False emission, u AGAINST ring order);
  local 1-frame render verified — brightest band, forward text, red logo, dark walkway. STRIPE
  CONTRAST landed same day (§6): measured end-to-end clip 1.015 vs control 1.089 → albedo ratio
  1.18→1.05 lands 1.019 (1.03 vanished into the denoiser floor). POD E2E BATCH #1 CONFIRMED
  same day (§6): boards text (readable "BANk … america" + red logo in the FINAL) and crowd
  grain (per-fan speckle, no marble) both survive the generative chain; new best FINAL
  `out/boards_pod/sideline_rgbnight_720p_pinned2.mp4`; pod DOWN. KIT ZONES landed same day
  (§6): the whole-body team-colour "morphsuit" is gone — SMPL-X LBS-derived garment zones
  (shirt/shorts/socks/boots/skin) + measured zone colours (per-team pooled medians with a
  grass gate + hue-mode estimator; REUSABLE NEGATIVE: the per-vertex sampler carries NO leg-kit
  chroma at broadcast distance, so THIS clip's shorts/socks/skin ride closeup-measured env
  overrides in `pod_make_video.sh`) + shirt-only TEAM_MASK (colour attribute reads back LINEAR
  → dim 0.1; >127 gates in kit-inject/hue-pin become shirt-only with zero consumer changes).
  Local render eye-verified: Colombia yellow/white/red + skin, Congo all-azure. POD E2E
  BATCH #2 CONFIRMED same day (§6): kit zones survive the full generative chain after two
  fixes — B-team azure overrides (pooled scene beats the fallback, `535fde4`) and a full-kit
  v2v prompt (bare "jerseys" wording made Wan paint shirtless torsos; now `DEFAULT_PROMPT`);
  stripes 1.05 clip-like. New best FINAL `out/kitzones_pod/sideline_tail2_pinned2.mp4`.
  Prompt polish is SATURATED (tail3 §6: colour negatives bleed across garments); stand-S
  residual small (.576 vs clip .462). Far-body softness PARKED — both levers lost: resolution
  (tail4 §6: 1536x864 unblocked via VAE tiling `a727d0e`, crispness equal + near-kit drift)
  and conditioning scale (tail5 §6: CS=1.15 drifts the whole frame bright-lime — CS leaks
  TONE, not just structure). 720p / CS=1.0 stay defaults; best FINAL still
  `out/kitzones_pod/sideline_tail3_pinned2.mp4`. GOAL CAM E2E confirmed (§6): novel-view #2
  works from behind the left goal, BALL visible in the final (f55) — sideline no-show was
  occlusion+size. Current: goal2 tail in flight — prompt-restore the near goal frame that
  grade3+Wan erase (twice-measured rule: name every large surface). Pipeline overview: `docs/pipeline.md`.** Previous lever same day (§6): CROWD TONE — knobs
  `PITCH3D_CROWD_EMISSION/CHROMA/TINT_SAT` = 3.6/0.15/1.35 + warm prompt wording; TWICE-MEASURED
  RULE: state the measured colour of EVERY large surface in the v2v prompt, else Wan's prior
  repaints it. GRASS TONE landed same day (§6): albedos in `scene_builders.py` + prompt;
  `TAIL_ONLY=1` on `pod_finish_batch.sh` = generative-stage iteration in ~15 min ≈ $0.2. Previous lever 2026-07-04 (§6): STADIUM PERIMETER — LED ad-board ring +
  walkway survive the whole chain; `out/boards_final/final_vs_clip.png`. Previous run 2026-07-03 (§6): fresh recon
  (PHYSICS=1, DEMO_EDITS=0) → quilt export → render → night-grade → Wan-VACE → SeedVR2 → mask
  pass → hue-pin, one command (`pod_finish_batch.sh`), all three levers eye-verified in the
  final: crowd non-periodic through the whole generative chain, kits pinned azure (this run's
  drift 244.2°→184.9°, auto target), physics continuous around the old teleport zone. **Player
  crispness FIXED same day (§6): v2v now runs at 1280×720 by default — 480p latents were
  repainting distant players as mush; A/B shows separated limbs/readable poses, no regressions.**
  **Cluster smear FIXED same day (§6): kit-colour re-injection into the control inside the
  team-mask AOV (grade3 was erasing kit identity in tight clusters → Wan hallucinated white/
  orange shirts) + a second hue-pin for team A (yellow) — both now defaults in
  `pod_finish_batch.sh` (mask pass moved before v2v).
  That run's FINAL: `out/anim_finish/sideline_rgbnight_kitinj_720p_pinned2.mp4`** (local).
  Historical context
  of the recipe: kit-boost at source +
  night-graded rgb control = team identity AND floodlit-night tone survive Wan-VACE in one pass**
  (yellow team locks hard, cyan reads as the second team with a cyan→blue hue drift; night look kept
  end-to-end — first variant ever). Chain of evidence: depth spike (restyle ✓, kits ✗) → rgb|gray A/B
  (control channel alone ✗ — the render's own kit signal was too weak) → kit-boost render + grade3
  night-grade (both ✓) → **(c) SeedVR2 720p upscale DONE 2026-07-03** (§6: sharper kits/bodies, night
  holds, ranking unchanged; `scripts/pod_seedvr2.sh`). **Full finishing chain now: recon → kit-boost
  render → night-grade → rgb-control Wan-VACE → SeedVR2 720p.** **Crowd kaleidoscope FIXED at source
  2026-07-03 (§6): non-repeating crowd QUILT** (`--crowd-mode quilt` auto-default, legacy `tile` +
  `--crowd-seed` manual; eye-validated on a local CPU render). **Kit-hue pin BUILT & VIDEO-validated
  2026-07-03 (§6): team-mask AOV pass (`--team-mask 1`) + `scripts/hue_pin.py` (still & video modes)
  undo the measured v2v/SeedVR2 drift — full 57-frame 720p pinned locally, 248.5°→183.5° with ONE
  clip-wide delta, kits azure again, everything else untouched.** All local levers are now DONE
  and the pod session is ONE command: **`bash scripts/pod_finish_batch.sh`** (recon→quilt
  export→render→night-grade→v2v→SeedVR2→mask pass→hue-pin; pin target auto-measured from THIS
  run's beauty render via `--target-from-frames`, `TARGET_HUE` env = manual override; validated
  locally: auto target 184.9° ≈ manual 183.5°). It bundles: quilt through v2v, hue-pin post-step,
  fresh PHYSICS=1 recon for the M3-9 eye-check. Artifacts: `out/kitboost/` local (480p+720p mp4s
  + judged stills), `out/quilt/` local (quilt render, masks, `pin_compare.png`, pinned 720p mp4s). **#207 player-physics gate (M3-9) BUILT 2026-07-03** — `core/correction/kinematics.py`
  (clamp impossible motion via the Correction seam; teleports MARKED not erased, R-6) **+ the root-cause
  coherence fix** (coast velocity capped at 10.5 m/s — a dying track's 43 m/s edge slid a ghost 10.9 m).
  Real-scene probe: speed/accel violations 22/999 → **0/0**, 10 raw teleports → **1 marked region event**
  (subj 1 f31, 8.7 m, n_intervals=8). Deliverable path defaults `PHYSICS=1` (`video_defaults.sh` →
  `--physics`); env knobs `PITCH3D_KIN_MAX_SPEED/MAX_ACCEL/TELEPORT`. Needs eye-validation on the next re-render;
  and deliverable re-renders now default `DEMO_EDITS=0` (a demo +10 cm offset + refit used to leak into
  every exported scene — fixed 2026-07-03). Spike artifacts local: `out/pod_adr11_check/v2v/` (broadcast+sideline mp4 + judged stills);
  weights + genfinish venv persist on the pod volume. Historical context — the 2026-06-28
  deliverable FAILED the eye-judgement (see §6 top): players = 5–10 px specks because the renderer's static
  bbox cameras framed the whole bowl from OUTSIDE; sideline = crowd wall; plus two wiring holes (COHERENCE
  unset→0 on direct pod runs, `PITCH3D_STADIUM_VIDEO` never wired officially). Fixed: exporter is now
  **`pitch3d.app.anim_export`** (CLI; writes **`cameras.npz`** — the virtual operator: fixed in-bowl mounts
  that pan/zoom with the action — and **`manifest.json`**, the versioned contract), the renderer validates
  the manifest FIRST and aims from `cameras.npz`; wrapper knob defaults single-sourced in
  `scripts/video_defaults.sh`. Old export dirs are refused by design — **RE-EXPORT on the pod**
  (`scripts/pod_make_video.sh` runs both halves). After the re-render eye-judgement, the known photoreal
  gaps remain: grass visible past the bowl, night exposure/tone, body-texture coverage ~7–11 %
  measured (crowd kaleidoscope FIXED 2026-07-03 — non-repeating quilt). Commit at every checkpoint
  (standing-authorized, NO push). **180°-roll** still bites any raw-pixel consumer (auto-detect
  `-rot[1][2]<0` + rotate before sampling).
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
- [ ] **v2 — photoreal (CURRENT FOCUS, started 2026-06-28).** Plan «A через B, 1→2→3, свет из клипа»:
  port the photoreal levers into the *deliverable video* path, sharing the two self-contained Blender
  scripts at the **data/contract layer**. [x] **Lever 1 — measured per-vertex body texture** (real broadcast
  pixels → posed SMPL-X → vertex colour; unseen → kit colour; validated, see §6). [x] **Lever 2 — grass PBR**
  via a shared pitch3d-free `adapters/blender/scene_builders.py` (the "B" refactor — one mowing-stripe
  node-graph, both render paths; ~5 m bands; validated, see §6). [x] **Lever 3 — light-from-clip**
  (floodlit-NIGHT, **auto-detected** floodlight colour + **manual** CLI override; dark world + ring of soft
  cool suns → even soft multi-shadows; validated, see §6). **The agreed 1→2→3 plan is complete.** Generative
  finishing (C) RULED OUT as *primary* (M2-0 spike); **2026-07-03 research reframes it as the recommended
  post-v2 lever** — structure-locked (depth+pose) v2v over our render, geometry stays the source of truth
  (see §6 top + [`research/2026-07-generative-render-landscape.md`](research/2026-07-generative-render-landscape.md)).

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
| #207 | **Unnatural player motion in the deliverable** (jitter + teleport-fast movement; user eye-report 2026-07-03) | **FIX BUILT 2026-07-03 (M3-9 gate + coherence coast cap); awaiting re-render eye-check** — `kinematics.py` gate on real scene: speed/accel viols 22/999→**0/0**, 10 raw teleports→**1 marked region** (subj 1 f31 8.7 m, n_intervals=8, conf 0.2 = coherence-extrapolated); ROOT CAUSE of the worst slide = coherence edge-coast inheriting a dying track's 43 m/s edge velocity → now capped at `coast_max_speed=10.5 m/s` (`CoherenceConfig`); deliverable defaults `PHYSICS=1`, knobs `PITCH3D_KIN_*` | no (probe runs anywhere) | user perception CONFIRMED by `scripts/motion_stats.py` on `out/anim_adr11/export/scene.json` (fps 29.97): TOTAL **32 speed- + 1083 accel-violation frames** over 23 subjects (limits 10.5 m/s, 8 m/s²); subj 1 sp_max **69.6 m/s**, 23 frames >10.5 (ID-swap teleports); typical ac_max 100–3186 m/s², turn up to 5370 °/s; **ball CLEAN** (p95 16.2 m/s, 0 >36). NOT an export bug: plumbing probe proved coherence MA(5) smoothing IS applied and survives resolve→save→load (synthetic 3769→558 m/s²) — `corrections=[]` in the exported file is the bake, by design. MA(5) is simply too weak for teleport-class errors (a 1-frame 1.8 m jump stays ~70× over the accel limit after MA5). Related: mux `FPS=25` vs source 29.97 plays the clip ~20 % slow — a separate small fidelity fix (the "too fast" feel comes from jitter, not fps). | **Kinematic plausibility gate — roadmap M3-9:** limits as attention items + limits-aware auto-corrections (velocity clamp / constrained smoother) via the ADR-0002 Correction seam; teleport spikes routed to identity/stitch review, not smoothed over. Probe on any scene: `python scripts/motion_stats.py --scene <export/scene.json>`. |
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

## 6. Progress log (newest first)

- **2026-07-05 (night run)** — **GOAL CAMERA E2E: novel-view #2 CONFIRMED through the full
  chain; BALL VISIBLE in the final; new residual = Wan erases the near goal frame.** Full
  per-camera chain (REUSE_SCENE=1, ~50 min ≈ $0.6): behind-goal mount (auto-placed at the
  action-side/left goal, 9 m up 5 m back) renders the same episode coherently — clusters,
  kits (Congo azure / Colombia yellow), stands, night tone all hold from the second angle.
  **Ball:** clear white round ball on the pitch at f55 (`goal_ball_zoom.png` recipe: crop
  700,350-1150,600); intermittent/dark-smudged mid-flight (heuristic heights, conf 0 — Wan
  repaints the flying ball inconsistently). The sideline "no ball" worry was occlusion+4px,
  as suspected. **Residual:** the beauty pass shows white posts+crossbar in the foreground;
  grade3+Wan REMOVE the goal frame entirely (thin near-camera structure, unnamed in the
  prompt). Iteration in flight (goal2, TAIL_ONLY): prompt gains measured-true "a football
  goal with thin white posts, white crossbar and a white net in the near foreground" per the
  twice-measured rule. Artifacts local: `out/kitzones_pod/goal_pinned2.mp4`, `goal_beauty.mp4`.

- **2026-07-05 (night run)** — **TAIL #5: conditioning-scale A/B (CS=1.15 vs 1.0) — LOSES;
  far-body-softness lever PARKED, second novel-view camera started.** By eye (4x zoom f1/f40
  vs tail3): far-body crispness unchanged, but the whole frame drifted BRIGHTER/lime — pitch
  tone moved away from the clip's dark floodlit night, and yellow shirts separate WORSE from
  the lighter grass. REUSABLE: CS also leaks TONE adherence, not just structure — raising it
  is not a free crispness dial. CS=1.0 stays default; `sideline_tail3_pinned2.mp4` remains
  best FINAL; `sideline_tail5_pinned2.mp4` kept for reference. Far-body softness: both
  mechanisms (resolution tail4, CS tail5) exhausted with no win — parked; at broadcast
  distance the clip itself is equally soft. NEXT: goal-camera run (full per-camera chain,
  REUSE_SCENE=1, `out/kitzones_goal.log`) — novel-view #2 + ball visibility (mount sits
  behind the action-side goal = left, where ball.npz ends).

- **2026-07-04 (night run)** — **TAIL #4: v2v resolution A/B (1536x864 vs 720p) — NO WIN,
  720p stays the default; VAE tiling landed as the unlock.** Three launches to get it running:
  1600x900 fails Wan's /16 check; 1600x896 and 1536x864 OOM in the **fp32 Wan VAE encode**
  (2.86 GiB pad alloc, fragmentation already eliminated by `expandable_segments` — genuinely
  short). Fix `a727d0e`: `vae.enable_tiling()` only above 720p (`pod_v2v_finish.py`), the
  validated 720p path stays byte-identical. Verdict by eye (4x zoom f1/f40, tail3 vs tail4):
  far-body crispness is **equal** — limbs/clusters resolve the same; near-body kit DRIFTED
  (Congo near player got grey shorts on f1); gen time +50% (~37 s/step vs ~25). Resolution
  lever closed. `out/kitzones_pod/sideline_tail4_pinned2.mp4` kept for reference. OPS lesson:
  check the v2v log right after "generating" prints — the /16 crash burned ~20 min of idle
  pod. Next: CS=1.15 (tail5, in flight) as the last cheap far-body lever, then a second
  novel-view camera + ball visibility on a goal-side angle.

- **2026-07-04 (night run)** — **TAIL #3: negative-prompt polish — trade confirmed, the prompt
  lever is SATURATED for kit micro-layout.** "bright white shorts" + negative "red trousers,
  red leggings, shirtless players, bare chest" (`c07a596`): shirts stay on (2nd clean run),
  red-trouser smear gone, BUT the negative bled into the red SOCKS (dimmer than tail2) and the
  shorts still render yellow-ish. REUSABLE: negatives act compositionally — a colour+garment
  negative dims that colour on OTHER garments too; don't stack colour negatives to sculpt a
  kit. Stand-S residual is now small: clip .462 / tail3 .576 (was .94 pre-mute) — dropped from
  the lever list. Best FINAL = `out/kitzones_pod/sideline_tail3_pinned2.mp4` (cleanest legs,
  no artifacts; tail2 kept as the stronger-socks variant). Ball: exported (60f, sane
  trajectory toward the left goal, conf 0 = heuristic heights) and rendered (0.11 m white
  sphere) but not visually confirmed in the sideline finals (likely cluster-occluded ~4 px) —
  recheck when a goal-side camera run happens. Next lever: far-body softness — tail4 = v2v
  1600x900 A/B in flight.

- **2026-07-04** — **POD E2E BATCH #2: kit zones + stripes 1.05 through the full generative
  chain; two fixes en route. New best FINAL `out/kitzones_pod/sideline_tail2_pinned2.mp4`.**
  Chain: fresh recon (23 subjects) → kit-zones export → 720p beauty → grade3 → shirt-only
  mask → kit-inject → Wan-VACE 720p → SeedVR2 → pins (~1.2h pod ≈ $0.9 incl. one killed
  restart + one TAIL_ONLY iteration).
  **Fix 1 (caught mid-batch, `535fde4`):** on the pooled pod scene Congo's post-grass samples
  pass the 40-sample floor, so the polluted line-white median (B shorts 0.765,0.702,0.71) BEAT
  the shorts→shirt azure fallback that the sparse local scene fell into — B legs would have
  rendered grey-white. Killed the render, pinned B shorts/socks to the f275-measured azure
  (0.189,0.52,0.688) in `pod_make_video.sh`, relaunched with REUSE_SCENE=1. npz probe
  confirmed both teams correct. LESSON: the fallback chain's outcome depends on scene
  DENSITY — measured overrides must cover BOTH teams, "unset = fallback" is not stable.
  **Fix 2 (tail iteration):** with the old prompt ("yellow jerseys / cyan blue jerseys") Wan
  painted several torsos as BARE SKIN and smeared red socks into red trousers — 4th
  confirmation of the twice-measured rule: the prompt named no shorts/socks/skin, so the
  prior repainted them. TAIL_ONLY=1 rerun (~15 min, $0.2) with the full-kit wording ("yellow
  short-sleeved shirts with white shorts and deep red socks … all sky-blue kit … dark-skinned
  players") restored every shirt; Congo reads all-sky-blue, Colombia yellow + red socks.
  Promoted to `DEFAULT_PROMPT` in `pod_v2v_finish.py`.
  Control (grade3 + shirt-only-mask kit-inject) kept the zones intact — the measured layer is
  clean end-to-end; stripes 1.05 read subtle/clip-like in beauty and control (eye).
  RESIDUAL for a next lever: Colombia shorts lean yellow instead of white at distance; red
  socks bleed up the thigh on 1–2 players; far bodies still soft at 720p latent scale.
  Sheets: `out/kitzones_pod/final_vs_clip.png` (tail1), zooms in `/tmp` regenerable from the
  pulled mp4s (`sideline_rgbnight_720p_pinned2.mp4` = tail1, `sideline_tail2_pinned2.mp4`).

- **2026-07-04** — **KIT ZONES: players get a real football kit — shirt/shorts/socks/skin/boots
  zones instead of the whole-body team-colour "morphsuit" (v2 face/limb lever, local).** Root
  cause of batch-#1's watercolour bodies: the measured per-vertex texture carries no kit LAYOUT
  at broadcast distance, and the flat fallback painted 90 % of vertices one colour — Wan then
  kept the morphsuit. Layout now comes from the body model itself: `smplx_kit_zones` (avatar.py)
  = dominant-LBS-joint per vertex → zone (spines/collars/shoulders→shirt; pelvis/hips→shorts
  with a 55 % hip→knee thigh cut; knee verts split by template height; ankles→socks; feet→boots;
  rest skin; real-model counts 6980/1890/715/632/258). Colours: exporter (`anim_export.py`) is
  now two-pass (pose+sample all → compose+write) so zone colours can pool PER TEAM.
  **Measured finding (kills naive auto): the vertex sampler carries NO kit chroma for legs at
  this distance** — raw pooled zone medians = grass green (0.43,0.57,0.24) for every zone of
  BOTH teams (legs are 20–40 px; a couple px of projection error lands on the lawn); after the
  grass gate they = line/LED white (~0.53–0.73 neutral); even the SHIRT zone's saturated-hue
  histogram is noise (team-A top bins 340°/190°/250° — no yellow mode). Auto estimator
  (`_zone_color_estimate`: grass gate → hue-mode among saturated, neutral-median branch) stays
  as the honest fallback chain (shorts→shirt, socks→shorts, skin→tan), and the designed manual
  override carries THIS clip: `PITCH3D_SHORTS_RGB_A=0.85,0.88,0.82` `PITCH3D_SOCKS_RGB_A=
  0.71,0.14,0.31` `PITCH3D_SKIN_RGB=0.32,0.26,0.20` measured off the clip's closeups (f30 leg
  bands, f275 tile classes) and defaulted in `pod_make_video.sh`; B unset on purpose — the
  one-colour fallback IS Congo's azure kit. **KIT CORRECTION: Colombia wears WHITE shorts + RED
  socks in this match** (earlier "navy shorts" note was a misread of trim/another player).
  Renderer: `--team-mask` is now SHIRT-AWARE — `zones` rides each subject npz (optional key),
  shirt verts carry the full team code, rest dim 0.1 (attribute reads back LINEAR: 0.35
  rendered ~160 = srgb_encode(0.35) and leaked past the >127 gates; 0.1 → ~89), so
  `control_kit_inject`/`hue_pin` key SHIRTS only with zero consumer changes. Local validation:
  1-frame broadcast render — yellow/white/red Colombia, all-azure Congo, dark skin heads/arms/
  thigh gap (`out/kitzones_diag/render_zoom.png`); eroded mask gate lands on shirts (beauty
  median under gate A=(0.875,0.886,0.137) yellow, B=(0.522,0.757,0.776) azure, off-body
  >127 px = 0). `PITCH3D_KIT_ZONES=0` restores the old fill. Tests: `test_kit_zones.py` (7 —
  synthetic-rig zone splits, real-model anatomy, grass/line pollution, fallbacks, env
  overrides). Pod E2E: rides batch #2 together with stripes.

- **2026-07-04** — **POD E2E BATCH #1 CONFIRMED BY EYE: boards text + crowd grain both survive
  the full chain** (fresh recon → export → GPU render → grade3 → mask → kit-inject → Wan-VACE
  720p → SeedVR2 → hue-pins; ~46 min pod ≈ $0.57, `out/boards_pod/batch.log`). **Boards:** the
  exporter reproduced the measured strip on the FRESH pod recon (939×48 ×19 — identical to the
  local adr11 run: the extractor is calib-run-robust), and the FINAL keeps a readable sponsor
  rhythm — "BANk … america" + red logo strokes on the far band
  (`out/boards_pod/final_f30_boardzone.png` vs beauty `beauty_f30_boardzone.png`; band zone =
  grass-boundary anchor, same trick as the extractor). **Crowd grain:** per-fan ~3 px speckle
  with dark aisles in the final stand ROI (`out/boards_pod/stand_final.png`) — no marble.
  Kit-pins worked (team A 63.5°→79.1°). **Residual:** final stand S 0.63 vs clip ≈0.42 (better
  than the amber-blob 0.94 and the mute-probe 0.74, still warmer than the clip) — keep on the
  candidate list. FINAL: `out/boards_pod/sideline_rgbnight_720p_pinned2.mp4` (local). Pod DOWN.

- **2026-07-04** — **STRIPE CONTRAST: mowing bands softened to the clip's measured near-flat
  level (v2 lever, local).** Metric: p90/p10 of the detrended smoothed grass-luma column profile
  (grass = uint8-HSV H 30–70, S>60, V>50; middle half of grass rows; 9-px smooth vs len/12
  trend). Measured: clip f100 = **1.015** (bands barely apart), graded control = **1.089**
  (~6× the clip's modulation), previous final `out/struct_pod/tail1_pinned2.mp4` = 1.060 (the
  generative tail preserves most of it). Albedo ratio (light/dark in `scene_builders.py`)
  1.18 → **1.05** around the SAME per-channel means (hue/value already clip-matched, §6
  grass-tone): 1.03 overshot to 1.003 — sub-denoiser-floor, stripes gone — so the tract is
  ~ratio^0.61 in ln-space, and 1.05 lands **1.019 ≈ clip's 1.015**, faint-but-there by eye
  (`out/stripes_tex/frame0_grade3.png`). Unit 618 pass. Renderer shares the constants via
  `scene_builders` — no other edit. Pod E2E: rides batch #2 (batch #1 `out/boards_pod` was
  already past its render stage with the old pair).

- **2026-07-04** — **BOARDS TEXT: the real "BANK OF AMERICA" LED strip, cut from the clip and
  wrapped around the ad-board ring (v2 lever).** The ring was flat white; the clip boards carry
  sponsor text + red logo. Now `extract_board_strip` (`stadium_backdrop.py`) cuts the strip,
  `_export_boards` (`anim_export.py`) adds OPTIONAL `tile`/`uv`/`tile_ext` keys to `boards.npz`
  (contract untouched — REQUIRED_KEYS are minimums, extras ride the manifest entries; graceful
  fallback to flat prior on extraction failure, reason printed), and the renderer wraps it with
  `normalize=False` (measured ABSOLUTE LED white + red logo hues skip the unit-mean norm and
  chroma pull), REPEAT extension, `PITCH3D_BOARD_EMISSION` (default 4.0).
  **Finding 1 (affects any future projection work): the pipeline scene.json CameraTrack is a
  CONSTANT AXIS-ALIGNED camera — aggregation keeps only POSITION from PnLCalib** (kitboost
  synthetic and adr11 real-calib both have rot0=[-1,0,0]; far-touchline projection exactly
  horizontal at y≈204/720p vs the real band slanting ~40 px and sitting ~200 px lower ≈ 19 board
  heights). Projection can place NOTHING pixel-exact; it only picks WHICH run/frame/span (straights
  in world coords; per frame+side ≥8 visible pairs, span ≥32 px, key=(depth//5 m, span) — widest-
  only had picked the useless NEAR straight). **Finding 2: anchor the band to the measured GRASS
  BOUNDARY** — plain LED-score argmax (val·(1−sat), box ~1 board height) locked onto the stand's
  white fascia rail TWICE; the fix is the topmost solid bright-green run per column (float-HSV hue
  70–170°, sat>.25, val>.3 — val gate excludes night hedges/dark seats), `_robust_quadfit` the
  boundary (3× MAD passes, 1.5 px floor, rides over goalposts/players), then LED argmax only in
  [g−3.2h, g+0.2h]. Raw-frame sampling: solved camera projects onto the 180°-rolled frame →
  point-reflect projected uv (letters upright), scale to native decode res (1280×720 grid →
  1920×1080). Measured: band 10.6 px, strip 939×48, mean rgb [.72 .69 .77], repeat =
  perim/(aspect·h) ≈ 19. **UV direction is AGAINST ring vertex order** (`adboard_loop_uvs`,
  `stadium.py`): ring runs toward −x on the far touchline, the upright-view strip toward +x —
  forward u rendered every board mirror-image (verified numerically: segment u vs screen x).
  Validation: unit 618 pass / 12 skip (+3 boards tests: contract keys, UV↔face order incl. wrap
  seam, quadfit outlier rejection); local exporter on `out/realcalib/scene_adr11.json` (real
  PnLCalib camera) + 1-frame CPU render on the grain_scratch export — band is the frame's
  brightest element, text reads FORWARD ("K OF AMERICA" legible at 720p), red logo hues intact,
  walkway stays dark. Artifacts: `out/boards_tex/{zoom_far_full,frame0_grade3}.png`. Pod E2E
  rides the NEXT batch together with the crowd-grain confirm.

- **2026-07-04** — **CROWD GRAIN FIXED AT THE CONTROL: quilt stitched at NATIVE tile
  resolution + 2× canvas — marble → per-fan speckle.** Gap: the clip stand is ~2.7-px per-fan
  speckle (720p-equivalent) but our control was 7-px smooth marble; the zoom sheet proved the
  grain dies in the CONTROL, not the tail (final 6 px ≈ control 7 px — Wan roughly preserves
  feature scale). Two root causes: (1) canvas — at 8192×512 the broadcast framing magnifies the
  quilt ~1.75×, so `CROWD_QUILT_SIZE` → 16384×1024 (one texel ≈ one 720p screen px); (2) the
  assembler sized patches as `height // 2` and UPSAMPLED the tile to fit, so canvas resolution
  cancelled out of the on-screen grain — doubling the canvas alone measured WORSE (7→9 px).
  `assemble_crowd_quilt` now stitches the tile at native px: new `fan_scale` param (quilt-px
  per tile-px, default 1.0 = measured clip grain; exporter `--crowd-fan-scale` /
  `PITCH3D_CROWD_FAN_SCALE`). Local 1-frame render after grade3: grain **5 px** (was 7 before,
  9 after the canvas-only bump), tone held (H 46 S .494 V .251 ≈ batch-2 ctrl V .224); zoom
  sheet `out/grain_iter2/grain_clip_vs_new.png` — the stand reads as crowd speckle, not
  marble; the residual 5 vs 2.7 px is largely 32-sample CPU denoiser smear. Unit test pins the
  decoupling (canvas size no longer sets grain; `fan_scale` does). **Pod E2E pending — bundle
  the grain confirmation into the NEXT batch instead of spinning the pod for one control-side
  change.**
- **2026-07-04** — **STAND SATURATION FIXED IN THE TAIL (prompt-mute probe): the colour-wording
  rule extended — Wan also AMPLIFIES the colours you state.** Batch 1/2 finals saturated the
  stand into amber "LED panels" (S .941) over a clip-exact control (S .486): the prompt itself
  asked for "fans in warm yellow and amber shirts". TAIL_ONLY probe reworded the crowd clause
  to "thousands of tiny individual fans in muted dark amber and brown clothing, crowd dimly lit
  and half in shadow" (now the `DEFAULT_PROMPT` in `pod_v2v_finish.py`) → final stand
  **H 46 S .737 V .247** (batch 2: H 43.8 S .941 V .298;
  clip stand-only: S ~.42 V ~.16–.20); grass held (H 66 vs 69). Eye: the LED-panel glow is
  GONE — the stand reads as a dark fine-speckled crowd with visible tier structure, closest to
  the clip so far; kits/grass unaffected. **RULE (third instance, sharpened): state the
  measured colour of every large surface AND state it at the measured intensity — Wan
  exaggerates colour adjectives ("warm yellow amber" → neon panels; "muted dark, half in
  shadow" → clip family).** New best FINAL: `out/struct_pod/tail1_pinned2.mp4`, 3-way sheet
  `out/struct_pod/stand_tail1_b2_clip.png` (probe/batch2/clip), full frame
  `out/struct_pod/tail1_full.png` (all local). Stand residual now: texture GRAIN — blobs
  coarser than the clip's per-fan speckle (quilt/tile-resolution lever, not tone). Cost:
  1 TAIL_ONLY ≈ $0.2; pod stopped after.
- **2026-07-04** — **CROWD STRUCTURE LANDED (walkway/railing/aisles/top-fade overlay +
  `tile_gain`); boards-occlusion + scale-invariance findings.** Gap: tone landed but the stand
  read as uniform TV-static; the clip shows architecture — stair aisles every ~0.03–0.09 frame
  width (3–5 px dark at 1080p ≈ 1.3 m), whole-rake luma fade top→bottom ~33–46 → 58–75
  (ratio ~0.53), tier break reading mostly as a bright railing line. Fix (7570c25, e1ae588):
  `apply_stand_structure()` in `stadium_backdrop.py` — walkway 0.28× at bowl-v 0.65 (±0.022),
  railing lip 1.6×, per-tier offset aisles (period 0.024 u, width 0.0035, 0.50×), whole-rake
  fade to 0.50 from v 0.20; overlaid by the EXPORTER (`--crowd-structure` /
  `PITCH3D_CROWD_STRUCTURE`, default 1; 0 = raw quilt — auto+manual rule). **Finding 1 (boards
  occlusion):** walkway at tier_v 0.35/0.48 never showed on screen; black-band diagnostics
  proved the ads boards hide bowl-v 0.2–0.5 from the broadcast camera — the visible stand is
  v ≈ 0.5–1.0, so tier features must sit at v ≥ ~0.55. Axis note: the quilt is stored
  screen-style (row 0 = stand TOP), the renderer flips it — bowl-v = 1 − row_frac. **Finding 2
  (scale-invariance trap):** the crowd shader normalises the tile to unit mean, so structure
  darkening ~30 % of texture mass silently brightened seated rows ×1.41 past the tuned
  emission — batch-1 final = oversaturated amber "LED panels" (stand S .941). Fix: exporter
  ships `tile_gain = mean(structured)/mean(raw)` in `stadium.npz` (0.716 here), renderer
  multiplies emission by it; unit test pins the mean-drop→tile_gain contract. **ROI hygiene:**
  the old wide crowd ROI (y .08–.22) is ~53 % sky and its median crossed into sky blue once the
  stand top dimmed — all stand measurements now use the stand-only ROI (y .17–.27, x .35–.65);
  clip stand there: V ~.16–.20, S ~.42. Batch 2 E2E (tile_gain): control stand V .224 S .486 =
  clip family (local iter6 V .235; top/bottom luma 25/86 vs clip 33–46/58–75). Eye: walkway +
  tier segmentation + top fade read as architecture in control AND final — the structure goal
  is met, lever CLOSED. **Residual (measured in BOTH batches): the generative tail re-saturates
  the stand to amber blobs — final S .941 vs control .486 vs clip .42; batch 2 ≈ batch 1 in the
  final despite the corrected control.** Prime suspect: the v2v prompt itself asks for "fans in
  warm yellow and amber shirts" (the twice-measured colour-wording rule, third instance?) →
  launched a TAIL_ONLY probe with muted wording ("thousands of tiny individual fans in muted
  dark amber and brown clothing, crowd dimly lit and half in shadow") — verdict next tick.
  Artifacts (local): 3-way sheet `out/struct_pod/stand_b2_b1_clip.png`, finals
  `out/struct_pod/batch1_pinned2.mp4` + `out/struct_pod/sideline_rgbnight_720p_pinned2.mp4`,
  local iters `out/struct_iter1..6/`, diagnostics `out/struct_diag*/`. Cost: 2 full batches +
  idle ≈ $1.5–2.
- **2026-07-04** — **CROWD TONE LANDED (render brightness/warmth + v2v prompt); the colour-wording
  rule GENERALIZED.** Gap: clip crowd is a warm amber mass (pure-crowd ROI median **V .188 H 48
  S .42**); ours was charcoal with confetti dots (V .043). Layer 1 — render (8bedf4c): the bowl
  tint is measured from the RAW clip (already a dark night broadcast) and grade3 darkens it AGAIN
  (the grass double-grade trap, same shape); on top, AgX desaturates bright emission to gray and
  grade3's `colorbalance bs=.12` paints dark gray blue. Three knobs in `blender_animate.py`,
  defaults measured over 5 local render→grade iterations: **`PITCH3D_CROWD_EMISSION=3.6`**
  (post-grade V .184 vs clip .188), **`PITCH3D_CROWD_CHROMA=0.15`** (tile hue-confetti → luma
  detail; real crowds vary in luma, not hue), **`PITCH3D_CROWD_TINT_SAT=1.35`** (headroom to
  survive AgX+grade). Grass bounce from the brighter bowl: none (H 81.9, ΔV .007). Pod E2E
  (REUSE_SCENE=1) confirmed the control exactly (V .184 H 69) — **but Wan output H 200 cold
  gray: second instance of the prompt-colour failure** (uncoloured "packed crowd" = prior wins,
  exactly like "deep green pitch"). Layer 2 — prompt (e5f6c9e): "stands densely packed with fans
  in warm yellow and amber shirts"; TAIL_ONLY rerun (~15 min) → **crowd V .161 H 52.2 S .57 vs
  clip .188/48/.42; grass held H 77.6**. Eye verdict: warm amber crowd in the clip's family,
  next to the prev gray-noise version the difference is night-and-day; A/B sheet
  `out/crowd_pod2/crowd_final_ab.png`, FINAL `out/crowd_pod2/sideline_rgbnight_720p_pinned2.mp4`
  (local). **RULE (now twice-measured): state the measured colour of EVERY large surface in the
  v2v prompt — any surface left uncoloured gets repainted from Wan's prior, overriding a
  clip-exact control. Check the prompt for colour words BEFORE blaming the model/control.**
  Residual: crowd reads as speckle, not a structured human mass (no tiers/aisles); V still ~.03
  under clip. Cost: 1 full REUSE_SCENE run + 1 TAIL_ONLY ≈ $0.9.
- **2026-07-04** — **GRASS TONE LANDED (two-layer fix: albedo at source + v2v prompt); TAIL_ONLY
  iteration mode for the generative stage.** Gap: render/final grass was neon emerald; clip's
  night-graded grass measures **H 81.9° S 0.651 V 0.557** (median HSV, pitch ROI, green gate).
  **Layer 1 — source albedo (44b2d83):** old emerald stripe pair (hue ≈115° linear) + grade3's
  gamma 0.75 (inflates S by crushing weak channels) → 4 measured render→grade iterations landed
  `GRASS_DARK/LIGHT_RGB = (0.292,0.42,0.102)/(0.346,0.495,0.121)` in `scene_builders.py`
  (post-grade H exact, V within 6%, stripe ratio softened 1.4→1.18); env override
  `PITCH3D_GRASS_DARK/LIGHT` in `blender_animate.py`. **BUT the pod E2E came back H 120 S 1.00
  V 0.85 — brighter emerald than before the fix:** control frames were clip-exact, so the
  generative stage repainted them. **Layer 2 — the v2v text prompt was the culprit (ea422e3):**
  `DEFAULT_PROMPT` in `pod_v2v_finish.py` literally asked for a "deep green pitch". Reworded to
  "muted yellow-green night grass with faint mowing stripes", pushed "vivid emerald grass,
  oversaturated colors" into the negative, added `V2V_PROMPT` env override in the batch
  (auto+manual rule). Rerun final measures **H 73.4 S 0.933 V 0.494** (Δhue vs clip −8.5° instead
  of +38°; V within 0.06) — by eye the neon is gone, grass reads as believable night turf; A/B/C
  sheet `out/grass_pod2/grass_prompt_ab.png`, FINAL
  `out/grass_pod2/sideline_rgbnight_720p_pinned2.mp4` (local). Residual: stripe contrast slightly
  stronger than clip, S still rich (0.93 vs 0.65) — park until a bigger lever demands it.
  **Lesson (pin the pattern): a measured control fix can be silently overridden by the
  generative stage — always re-measure AFTER Wan/SeedVR2, and check the text prompt for colour
  words before blaming the model.** **TAIL_ONLY=1 (ea422e3):** `pod_finish_batch.sh` now skips
  steps 1–4 and reuses a previous run's beauty/masks/control from the same `$OUT` — a
  prompt/CS/seed iteration is ~15 min ≈ $0.2 instead of ~55 min (validated by this very rerun:
  v2v 10 min + SeedVR2 + pins on 60 reused control frames; pins B 232.9→183.5, A 70.6→79.1).
- **2026-07-04** — **STADIUM PERIMETER LANDED: LED ad-board ring + dark walkway band; reordered
  one-command wrapper VALIDATED E2E.** Gap (clip vs render): the clip reads grass → bright white
  LED boards ("BANK OF AMERICA") → dark walkway → dense speckled crowd; our render ran grass
  straight into a pastel crowd wall. Fix is a geometric PRIOR (no clip measurement):
  `adboard_ring_geometry()` in `core/scene/stadium.py` — white emissive band (1 m) + dark
  walkway band (2.2 m) at +5 m outside the lines, exported as `boards.npz`
  (`--board-height/--board-offset`, `PITCH3D_BOARD_HEIGHT/OFFSET`, 0 disables), rendered as
  flat vertex-colour emission (`PITCH3D_BOARD_EMISSION`, default 4.0 — saturates the PNG so
  grade3 keeps the boards the brightest band in frame: 255→161 vs 30–90 neighbours). Local CPU
  eyeball validated geometry + post-grade brightness hierarchy BEFORE pod spend. First pod E2E
  died in `write_manifest`: `boards.npz` wasn't registered in the contract's `REQUIRED_KEYS`
  (the local eyeball synthesized the npz by hand and skipped exactly that path — lesson: local
  pre-checks must run the exporter, not hand-craft its outputs); fixed + contract test.
  **Batch: `BATCH_FINISH_OK` from ONE invocation of `pod_finish_batch.sh`** — first single-command
  run of the 2026-07-03 reorder (recon → export(+boards) → render → grade3 → mask AOV →
  kit-inject → v2v 720p → SeedVR2 → pin B → pin A auto-targets 46.6°→69.2°). Verdict
  (`out/boards_final/final_vs_clip.png`, f10/f30/f55 vs clip): the white LED strip reads along
  the perimeter and around the bend, the dark walkway separates the (now dark, speckled-yellow)
  crowd from the pitch — the broadcast-frame structure matches the clip; boards survive
  Wan+SeedVR2 intact. **Next gap by eye: GRASS TONE — ours is neon acid-green, the clip's is
  muted dark green.** Also: `docs/pipeline.md` added (the whole clip→final pipeline with mermaid
  diagrams, per user request). Artifacts local: `out/boards_final/` (FINAL + beauty + judged
  sheet), `out/boards_eyeball/` (local render + grade checks). Pod session ~55 min ≈ $0.7,
  **pod STOPPED**. Commits 7538e0a (ring), c8b0484 (contract fix).
- **2026-07-03** — **PLAYER-CLUSTER SMEAR FIXED: kit-colour re-injection into the v2v control
  (+ team-A hue pin).** Diagnosis (`out/clusters/control_vs_output_f10.png`): in tight clusters
  the night-graded control keeps clean separable geometry but grade3 all but erases kit colour —
  uniform dark-teal mannequins → Wan gets shape without identity and hallucinates white/orange
  shirts. Fix `scripts/control_kit_inject.py`: push H/S toward each team's kit colour inside the
  eroded team-mask AOV, keep V (shading/limb boundaries survive). Auto-target = histogram-MODE of
  the grade-surviving masked pixels (a circular mean is wrong here: grade3 splits team-A yellow
  into a yellow/olive bimodal — night 140–160°+60–70° vs beauty 60–80° unimodal — and the mean
  landed on green 139.8°); manual `TEAM_A_HSV`/`TEAM_B_HSV` used & validated this run
  ("65 0.85"/"185 0.95"). A/B (`scripts/pod_cluster_ab.sh`, same seed/steps/res, ONLY the control
  changed): **f10 far cluster — the white-shirt+orange hallucination on team B is GONE, all
  players azure; f30/f55 + mid crops: parity or slightly cleaner, no regressions.** Bonus: kit
  drift shrinks again (230.1° vs 235.8° pre-pin). Team-A residual (yellow renders ORANGE — the
  pin only covered B): second hue-pin pass, channel r, band 5–80°, sat-min 0.35 (faces stay out
  of the gate), target auto-measured 70.6° from the beauty render → f30 torso orange→yellow, no
  collateral. WanVACE `conditioning_scale` confirmed in diffusers 0.38 (default 1.0) — wired as
  `--conditioning-scale`/`CS` env, left at 1.0 in this A/B. PROMOTED into `pod_finish_batch.sh`:
  mask pass moved BEFORE v2v, `KIT_INJECT=1` default (TEAM_*_HSV unset = validated constants,
  set-empty = auto-measure), `PIN_A=1` second pin. The reordered one-command wrapper has not yet
  been re-run E2E as a single command (every step ran individually on the pod today — the next
  fresh finish run validates it). Artifacts local:
  `out/anim_finish/sideline_rgbnight_kitinj_720p_pinned2.mp4` (**new best FINAL**),
  `out/clusters/*` (inject check + judged A/B crops). Pod session ~40 min ≈ $0.5, **pod STOPPED**.
  Temporal probe (mean|max consecutive-frame abs diff @832×480): kitinj 5.00|10.91 vs hi720
  5.12|11.04 — the mask-driven injection adds NO flicker (marginally smoother, if anything).
- **2026-07-03** — **PLAYER-CRISPNESS LEVER LANDED: v2v at 1280×720 is the new default.**
  Diagnosis (`out/anim_finish/crisp_zoom_*_f30.png`, stage-by-stage crops of the same distant
  players): crisp in the 720p beauty render → MUSH after Wan-VACE at 832×480 (a 15–25 px player
  is 2–3 latent px after the VAE's 8× compression — the model cannot represent him and repaints
  a blob) → SeedVR2 only sharpens the mush. A/B (`scripts/pod_v2v_hires_ab.sh`, same night
  frames/seed/steps, ONLY resolution changed, SeedVR2 1:1 restore + pin on top): **720p resolves
  separated limbs and readable poses where 480p had merged blobs; a hallucinated crowd smear in
  the 480p cell is gone; foreground players also crisper; night look, quilt and pinned kits all
  hold — no structure breakdown from the 1.3B model at 720p.** Bonus datum: kit-hue drift shrinks
  at 720p (235.8° vs 244.2° before pin) — the prior repaints less. Cost: ~19.3 s/it × 30 steps
  ≈ 10 min vs ~3 min at 480p (fits 32 GB with cpu-offload, no OOM). PROMOTED: `pod_finish_batch.sh`
  now defaults `V2V_WIDTH=1280 V2V_HEIGHT=720 V2V_FLOW=5.0` (env knobs = manual override, old
  480p reachable). Artifacts local: `out/anim_finish/sideline_rgbnight_hi720_720p_pinned.mp4`
  (best-yet FINAL), `ab_hires_{far,mid,full}_f30.png` (judged). Pod session ~35 min ≈ $0.4,
  **pod STOPPED**. Multi-frame hardening (f10/f55 crops + probe): verdict holds — at f55 the
  480p cell's team-B tops came out brownish even after the pin (their drift differed from the
  clip median) while 720p wears correct azure; residual artifact both cells: tight player
  clusters still smear (orange blob f10). Temporal probe (mean|max consecutive-frame abs diff,
  identical motion): A480 4.90|11.28 vs B720 5.71|11.93 — more detail, no flicker blow-up;
  final say = watching the clip.
- **2026-07-03** — **FULL FINISHING CHAIN E2E ON THE POD — all three levers land in one run.**
  `pod_finish_batch.sh` executed on a FRESH reconstruction (PHYSICS=1, DEMO_EDITS=0, 60 frames,
  sideline): recon 310 s → quilt export → GPU render 60/60 → grade3 → Wan-VACE (57 f) → SeedVR2
  720p → team-mask pass 60/60 → hue-pin 57/57 (auto target measured 184.9° from THIS run's
  beauty render — matches the local 183.5° constant; this run's drift 244.2°, delta −59.3°).
  One bug found & fixed mid-run: INPUTS for `pod_seedvr2.sh` must be ABSOLUTE (it cd's into its
  own repo before `test -f`) — batch died between v2v and SeedVR2, fixed in
  `pod_finish_batch.sh` (388bac8) and resumed from SeedVR2 without redoing recon/render/v2v.
  Eye-verdict on `final_compare_f10/30/55.png` (unpinned top vs pinned bottom): **crowd quilt
  survives the whole generative chain — no kaleidoscope, non-periodic stands**; night look
  holds; pin turns team-B kits azure, yellow/grass/crowd untouched; beauty f29→f31 (the old
  teleport zone): positions continuous, no ghost slide — **M3-9 physics gate passes by eye**.
  Pod session ~50 min ≈ $0.6, **pod STOPPED**. Artifacts local `out/anim_finish/`:
  `sideline_rgbnight_720p_pinned.mp4` (FINAL), unpinned 720p, v2v 480p, `sideline_beauty.mp4`,
  compare stills; full set persists on the pod volume.
- **2026-07-03** — **KIT-HUE DRIFT MEASURED + PIN BUILT & VIDEO-VALIDATED (the cyan→blue lever,
  fully local).** Measured on team-B kit pixels (mask∩blue-gate, sideline f30): render hue
  **183.5°** (azure-cyan; clip truth ≈195°, measured team colour [0.367,0.514,0.647] = 208° flat) →
  after Wan-VACE **224.5°** → after SeedVR2 **251.3°** — the generative prior re-paints azure as
  deep "football blue", drift is real and AWAY from truth. Pre-compensating −68° would demand a
  GREEN source kit (grass collision) → rejected; built the AOV-mask design instead. (1) subject npz
  now carries optional `team` key ("A"/"B"/""); (2) `blender_animate.py --team-mask 1` renders the
  SAME cameras/frames as flat unlit team codes (A=red, B=green, other=blue; no lights,
  zero-strength world, stadium skipped, plates skipped, 1 sample) — pixel-aligned AOV masks;
  (3) new `scripts/hue_pin.py` rotates hue by one constant inside (dilated mask) ∩
  (hue-band 170–290 ∩ sat≥0.15), so grass/crowd/team-A stay untouched. Still-validated LOCALLY
  (no pod): mask f30 rendered on CPU, pin applied to the existing SeedVR2 720p still →
  **251.3°→182.1° ≈ render target**, eye-check `out/quilt/pin_compare.png` (before/after/render-ref):
  pinned kit matches the render reference, yellow team and grass unchanged. **VIDEO mode validated
  same day, fully local:** all 60 mask frames CPU-rendered (`out/quilt/mask_all/`, 1 sample —
  minutes), `hue_pin.py --video/--mask-dir` measures ONE clip-wide delta over all gated pixels
  (per-frame deltas would flicker) then applies it two-pass — on the existing SeedVR2 720p mp4:
  57/57 frames pinned, clip median **248.5°→183.5° (delta −65°)**; eye-check f10/f30/f55
  (`out/quilt/vid_pin_compare_f*.png`): team-B kits azure again across the whole clip, yellow
  team/grass/crowd untouched, no flicker by construction. **Auto target added same day:**
  `--target-from-frames DIR` measures the pin target from the run's own beauty render (same
  masks), so a fresh recon needs no hand-carried constant — local check: auto 184.9° vs manual
  183.5°, pin result identical by eye. **`scripts/pod_finish_batch.sh`** now bundles the whole
  finishing chain (recon→…→SeedVR2→mask→pin) into one pod command. **Pod untouched (STOPPED).**
- **2026-07-03** — **CROWD-QUILT VALIDATED: the stands kaleidoscope is dead at source.** Root cause
  was the mosaic scheme itself: ONE small measured crowd tile mirror-repeated 40×4 over the bowl
  (`tex.extension=MIRROR`) — the periodicity reads as a kaleidoscope, crisply since SeedVR2. Fix:
  new `assemble_crowd_quilt` (`adapters/render/stadium_backdrop.py`) stitches one large
  non-repeating 8192×512 texture from ~100 random crops of the SAME measured tile (random offsets,
  50 % x-flips, ±8 % gain jitter, Hann-feathered seams; placement wraps in x so the loop seam blends
  like any other); the bowl now unwraps continuously (`bowl_tile_loop_uvs` repeat=(1,1)) with
  extension REPEAT; the per-vertex measured-tint multiply is unchanged. AUTO default
  `--crowd-mode quilt`; MANUAL `--crowd-mode tile` (legacy mosaic) + `--crowd-seed N` re-roll (env
  `PITCH3D_CROWD_MODE`/`PITCH3D_CROWD_SEED`). `stadium.npz`: `tile` now uint8 in quilt mode (12.6 MB
  vs 50 MB) + optional `tile_ext` key; `blender_animate` reads both dtypes/exts, old exports keep
  MIRROR. Eye-verdict (LOCAL: anim_export on the kitboost scene + CPU Cycles f0/f30 sideline
  832×480, vs the old kitboost still): old = mirrored V-motifs + a horizontal repeat seam along
  every stand; new = varied non-periodic crowd, wrap seam invisible (quilt PNGs judged directly
  too). Texel density ~22 px/m ≈ 2.3× oversampling at 720p — slight softness vs the oversampled old
  tile accepted (v2v+SeedVR2 re-detail). Tests: `tests/unit/test_crowd_quilt.py` (coverage,
  determinism, non-periodicity, flat-tile neutrality, unwrap-UV); full unit + fakes-e2e green.
  Artifacts: `out/quilt/` (export, texture PNGs, 2 rendered frames). All LOCAL — **pod untouched
  (STOPPED)**; the next pod re-render picks the quilt up automatically.
- **2026-07-03** — **SeedVR2 720p UPSCALE VALIDATED (priority-2c): the winning v2v variant upscales
  cleanly 480p→720p; the quality ceiling is now the v2v pass itself, not resolution.** New
  `scripts/pod_seedvr2.sh` (standalone numz CLI, no ComfyUI/flash-attn; weights auto-download to the
  volume `/workspace/models/SEEDVR2`; venv `/workspace/venvs/seedvr2` torch 2.11 cu128): 3B fp16 on the
  32 GB RTX PRO 4500 Blackwell, `--resolution 720 --batch_size 33` (4n+1 temporal), 57 f per clip in
  minutes. Eye-verdict f10/f30/f55 vs 480p: **kits, bodies and markings clearly sharper** (1248×720),
  night tone holds (SeedVR2 slightly brightens the green), teams stay yellow vs blue; the un-graded
  control cell upscales equally sharp but keeps its day-bright drift — recipe ranking unchanged.
  Honest flip side: sharpness exposes the residuals — the crowd kaleidoscope mosaic and v2v body
  artifacts are now crisply visible (garbage in → sharp garbage out). **Deliverable chain as of today:
  recon → kit-boost render → night-grade → rgb-control Wan-VACE → SeedVR2 720p.** Next fidelity levers:
  pin cyan→blue kit drift (per-team semantic hint / AOV masks), fix the crowd mosaic at source
  (mirror-tiling → measured crowd texture), and eye-check the M3-9 physics gate on the next fresh
  recon (`PHYSICS=1` default). Artifacts local: `out/kitboost/sideline_{rgbnight,rgbboost}_720p.mp4`
  + stills. **Pod STOPPED.**

- **2026-07-03** — **KIT-BOOST + NIGHT-GRADE VALIDATED (priority-2a+b): the v2v recipe that keeps BOTH
  team identity AND the night tone is found.** Fresh full recon (kit-colour boost in `anim_export`
  vcolors: team A [.77,.73,.26]→[1,.91,0] yellow, team B [.37,.51,.65]→[.28,.61,.9] cyan; sat×1.6
  val×1.4) → sideline render → grade3 night-grade (`eq=brightness=-.28:contrast=1.12:gamma=.75:
  saturation=.9,colorbalance=bs=.12:bm=.06`) → two rgb-control v2v runs (57 f 832×480, 30 steps,
  ref=source f0). Eye-verdict on f10/f30/f55: (1) **the boosted render itself now carries legible team
  colours** (yellow vs cyan, vs the old grey bodies); (2) **v2v over night-graded control keeps the
  floodlit-NIGHT look end-to-end** — first variant ever (dark stands, dark striped grass, floodlit feel)
  — AND both teams stay distinct (yellow locks hard; cyan drifts to generic blue but reads as the second
  team); (3) v2v over un-graded boosted rgb re-confirms the day-bright drift (neon grass) even though
  kits lock — so the WINNING RECIPE = **kit-boost at source + night-graded rgb control**. Source-side
  colour strength was the missing signal, exactly as diagnosed (control channel alone couldn't pin it).
  Residuals for next pass: cyan→blue hue drift (semantic/per-team hint could pin it), crowd goes
  uniform-yellow (control's kaleidoscope mosaic leaks), 480p bodies still rough → (c) 720p + SeedVR2.
  Artifacts local: `out/kitboost/{sideline,sideline_rgbnight,sideline_rgbboost}.mp4` + judged stills
  `*_f{1,2,3}.png` (f10/f30/f55) + `night_grade_f30.png`; on-pod `out/anim_kitboost/`, `out/v2v/`.
  Batch script pattern: `/workspace/run_kitboost_batch.sh` (recon→render→grade→2×v2v). **Pod STOPPED.**

- **2026-07-03** — **rgb|gray CONTROL A/B (sideline, vs the depth run): NEITHER locks team identity, BOTH
  lose the night restyle — the control channel is not the lever; the render's colour signal is.**
  Same recipe as the depth spike (57 f 832×480, 30 steps, ref = source frame 0), only `--control`
  varies. Eye-verdict on f10/f30/f55 vs the CG input and the depth run: **rgb and gray keep the CG
  day-bright tone** (acid-green pitch, bright yellow crowd — the night look the depth run achieved is
  GONE) because luminance-carrying control locks tone; **and the kits still drift** — every player gets
  prompt-dressed yellow-shirt+blue-shorts (referee red survives), the cyan team vanishes in BOTH, i.e.
  the same identity failure as depth with none of its restyle win. **Sharpened diagnosis:** the v2v
  input's own kit signal is too weak to lock onto — near bodies read GREY (measured per-vertex colours
  sampled from a dark night clip + 7–11 % coverage; kit-fallback tint subtle at 480p), so identity
  currently lives only in the prompt, and no single global control video can pin per-player colours.
  **Refined priority 2:** (a) strengthen team colour at the SOURCE (saturate the kit fallback / raise
  kit-vs-measured blend in `anim_export` vcolors) so the render itself carries unambiguous team colours;
  (b) composite conditioning — depth (restyle freedom) + per-team semantic hint (AOV masks in the export
  contract) or a night-GRADED rgb control; (c) then 720p + SeedVR2. Artifacts local:
  `out/pod_adr11_check/v2v/sideline_{rgb,gray}.mp4` + `side_{rgb,gray,input}_f{10,30,55}.png`.
  **Pod STOPPED.**

- **2026-07-03** — **PLAYER PHYSICS: user's "players move unnaturally fast" CONFIRMED by measurement;
  root-caused; gate designed (M3-9) + TWO real bugs found, one fixed.** New probe
  `scripts/motion_stats.py` (per-subject speed/accel/turn vs human limits, both layers + ball) on the
  deliverable scene `out/anim_adr11/export/scene.json`: **TOTAL 32 speed- / 1083 accel-violation frames**
  across 23 subjects; subj 1 sp_max **69.6 m/s** with 23 frames >10.5 m/s (ID-swap teleports); typical
  ac_max 100–3186 m/s² (limit 8), turn up to 5370 °/s; **ball clean** (p95 16.2 m/s, 0 >36) — full row
  dump in §3 #207. Root-cause trail: (1) suspected "export drops corrections" — **DISPROVED** by a
  synthetic plumbing probe: `add_temporal_coherence → resolve_scene → save_scene → load_scene` keeps the
  smoothed values (ac_max 3769→558; `corrections=[]` in the file = the bake, by design, assemble.py:74);
  (2) so the deliverable IS smoothed — **MA(5) is just structurally too weak** for teleport-class errors
  (a 1-frame 1.8 m jump stays ~70× over the accel limit after MA5) → the right fix is the **kinematic
  plausibility gate, now roadmap M3-9** (limits→attention items R-6; limits-aware auto-corrections via
  the ADR-0002 seam; teleports→identity/stitch review); (3) **REAL BUG FOUND+FIXED: the golden-path demo
  walkthrough was polluting real deliverables** — cli steps 7/8c committed a dry-run **+10 cm root
  offset** (measured: exactly 0.100 m in the exported scene) **+ a REFIT** on subject[0]; new
  `--no-demo-edits` flag gates steps 4–8e, `DEMO_EDITS` env in `pod_real_e2e.sh`, **default OFF in
  `pod_make_video.sh`/`demo_video.sh`** (deliverables), ON elsewhere (seam coverage); e2e green both
  modes; (4) noted: mux `FPS=25` vs source 29.97 → deliverable plays ~20 % slow-motion (separate small
  fix; the "too fast" feel is jitter, not fps). Next deliverable re-render inherits: no demo offset, no
  demo refit.

- **2026-07-03** — **PRIORITY-1 SPIKE: structure-locked generative finishing WORKS — depth-locked Wan VACE
  turns the CG deliverable into night-broadcast footage.** New `scripts/pod_v2v_finish.py` +
  `scripts/pod_v2v.sh` (separate `genfinish` venv — torch 2.11 **cu128**, Blackwell sm_120 needs it;
  Wan2.1-VACE-1.3B-diffusers weights ~17 GB cached in `/workspace/hf`, both survive pod stop). Two runs
  (broadcast + sideline), 57 f 832×480 @ 30 steps, control = Depth-Anything-V2 over OUR rendered frames,
  reference = source-clip frame 0, ~5 min generation each on the PRO 4500. **Eye-verdict vs the render
  input: the three measured-base gaps close in one pass — day-bright tone → floodlit NIGHT bowl,
  kaleidoscope crowd → plausible dark crowd with yellow patches, flat grass → floodlit grass; LED ad
  boards appear; large sideline players stay coherent humans (no melted bodies / extra limbs in stills);
  camera, formation and motion are OURS (R-6 held).** Costs found: (1) **kit-identity drift: depth carries
  no colour, so per-player team colours come from prompt/ref and are NOT locked — sideline went mostly
  yellow, the cyan team nearly vanished** (exactly what research priority 2, AOV semantic/RGB conditioning,
  is for); (2) heavier edge vignette than the source; (3) hallucinated ad-board text; (4) 480p spike res —
  far-cam players blobby (SeedVR2 step still pending). Spike outputs local:
  `out/pod_adr11_check/v2v/{broadcast,sideline}_vace.mp4` + judged stills.

- **2026-07-03** — **POD RE-RUN attempt 2 (correct clip): `POD_MAKE_VIDEO_OK` — EYE-VERDICT: framing FIXED,
  the ADR-0011 virtual operator passes Gate 1.** Timing: recon 284 s (23 subjects + ball, continuity 25→23,
  refit on-anchor 59/60, max residual 0.85 m), export 28 artifacts (schema v1: 23 subjects + ball + cameras +
  lighting + pitch + stadium), render 240 PNG in ~8 min (OptiX + persistent data, c008241 paid off), 4 mp4s →
  local `out/pod_adr11_check/`, **pod stopped**. Judged vs source frames (f0/f30/f55 per camera):
  **broadcast** — elevated in-bowl main-stand cam, action tracked, zoom stable across the clip, players read
  as figures not 5–10 px specks; **sideline** — low pitchside, players large in frame; **goal** — behind-goal
  with the goal frame in shot; **top** — whole-pitch schematic with fixed fov **by design** (cameras.py:191),
  not a zoom bug. Same episode recognizable: formation cluster + drift over 60 f match the source. v2 levers
  survived the deliverable path: mowing stripes ✓, kit tints (yellow vs cyan distinguishable) ✓, crowd bowl ✓,
  `lighting.npz` measured & applied ✓. **Remaining visual gaps, in order: (1) night TONE — floodlight colour
  is measured+applied but frames read day-bright vs the dark floodlit source (exposure/contrast); (2) grey
  low-texture-coverage bodies on near cams (~7–11 % coverage); (3) crowd-mosaic kaleidoscope.** All three are
  exactly what the research's priority-1 structure-locked v2v finishing pass (Wan 2.2 Fun-Control/VACE +
  SeedVR2, source clip as night-look reference) is expected to address without contract changes.

- **2026-07-03** — **POD RE-RUN, attempt 1: crashed on the WRONG CLIP; root-caused & relaunched.** First
  ADR-0011 pod run used `pod_make_video.sh` defaults → `PITCH3D_CLIP=/workspace/clip.mp4`, which turned out
  to be a **stale daytime stock clip** (Pexels; byte-identical to `samples/video/15449383-hd_1920_1080_60fps.mp4`),
  not the target night match. Every symptom followed from that one mistake: PnLCalib found 0 keypoints (no
  soccer pitch → identity homographies, conf 0), 11 phantom subjects with ~445 m translation spans, fov pinned
  at 50°, body-texture coverage 0–4 %, and finally the loud crash in `anim_export` stadium backdrop
  («no crowd-band vertex is visible in any frame») — the contract refused to render garbage, which is ADR-0011
  working as designed. Eliminated en route: coherence edge-extension, commit regressions, stale `.env`, kp
  thresholds, CUDA, weights, PnLCalib repo drift. **The 2026-06-28 deliverable is NOT affected** (its scene
  `source_id = Colombia-1-0-Congo-DR1080p` — the framing eye-verdict stands). Relaunched with
  `PITCH3D_CLIP=/workspace/Colombia-1-0-Congo-DR1080p.mp4` (byte-identical to the local target sample) →
  `out/anim_adr11`. Candidate guard for later: wrapper-level clip-provenance check (expected duration/hash),
  since the default path silently reconstructs whatever `/workspace/clip.mp4` happens to be.

- **2026-07-03** — **RESEARCH: mid-2026 generative-model landscape vs. this task («как бы делали с нуля»)**
  → [`research/2026-07-generative-render-landscape.md`](research/2026-07-generative-render-landscape.md)
  (3 parallel web passes: char-swap/pose-driven video; re-camera+4D; HMR/avatars/env/finishing). **All three
  converge on reconstruct-then-condition — i.e. the existing measured core is the right architecture; the
  *CG-photoreal finish* is the replaceable part.** Facts: direct re-camera of broadcast tops out at ±10–30°,
  sub-HD, 81–121 f (monocular depth fails on telephoto/small players/grass/crowd); char-swap ceiling = 7
  large-in-frame subjects (22×~100 px is OOD for everything); BUT Vista4D (open, Netflix Eyeline) / GEN3C
  (NVIDIA) take an **explicit point cloud as conditioning** → we can feed OUR reconstruction; Wan 2.2
  Fun-Control/VACE (open) does depth+pose-locked v2v restyle = "generative renderer" over Cycles output;
  SAM 3D Body (Meta, open, promptable) fits the PoseEstimator port; LHM++/IDOL = one-shot SMPL-X-driven
  splat doubles from crops (stunt-double grade, not face-grade); SeedVR2/FlashVSR ≈ Topaz; night-relighting
  off-the-shelf. **Strategic reframe:** the Blender render's product becomes the *conditioning stack*
  (RGB+depth+normal+semantic AOVs) for a structure-locked finishing pass (ADR-0007 seam B — now buildable
  with open weights); photoreal levers stay as a better base plate, not the ceiling. Priority when v2
  re-render is judged: (1) Wan 2.2 Fun-Control v2v + SeedVR2 over existing frames (no contract change),
  (2) AOV passes in the export contract, (3) SAM 3D Body behind the port, (4) splat doubles, (5) Vista4D
  with our point cloud. Dead ends: end-to-end swap of the raw clip, pure T2V regen (violates R-6).

- **2026-07-03** — **EYE-VERDICT: the 2026-06-28 deliverable is UNWATCHABLE as broadcast → deliverable-path
  refactor (ADR-0011), E2E-verified locally.** Eye-judgement of the 4 mp4s vs the source clip (the pending
  NEXT ACTION; frames extracted with ffmpeg, judged multimodally): **broadcast/goal** show the whole stadium
  bowl from OUTSIDE with players as 5–10 px specks; **sideline** is a wall of crowd texture with zero players
  visible; the crowd mosaic MIRROR-tiles into a kaleidoscope pattern; the look reads day-ish vs the floodlit
  night clip; the grass plane runs past the bowl; the render covers 1.92 s of the 11.2 s clip (17 %).
  **Root causes:** (1) `blender_animate.py` derived four STATIC cameras from the bbox of everything loaded —
  with the 105×68 m pitch folded in, every deliverable camera sat outside the stadium (the eye-validated
  close-ups had come from the `action` camera, which was NOT in the deliverable set); (2) `COHERENCE` unset
  → `0` on direct pod runs, so the deliverable rendered RAW unsmoothed poses; (3) `PITCH3D_STADIUM_VIDEO`
  never wired in the official path → stadium/body-texture/lighting silently skipped. **Fix (ADR-0011),
  committed c8c1294 → fb048d7 → d2fc70a:** shared `scripts/video_defaults.sh` sourced by both wrappers
  (COHERENCE=1 everywhere; STADIUM_VIDEO defaults to the clip); **virtual operator** `core/scene/cameras.py`
  — fixed mounts INSIDE the bowl envelope (main stand @ halfway h=12 m, low pitchside, behind the action-half
  goal, overhead) that PAN (median action centroid — one idle keeper must not drag the aim) and ZOOM
  (bulk-quantile q80 radius + pad, fov fits BOTH image axes, zoom smoothed slower than aim so it never pumps)
  — 9 unit tests incl. "action fits in frame for every tracking camera" via `project_normalized`;
  **versioned contract** `adapters/blender/anim_contract.py` (manifest.json, schema v1, required-keys per
  artifact, validated on write AND read; 8 unit tests) — drift now fails in ms with a named cause instead of
  after a GPU render; **exporter → package** `pitch3d.app.anim_export` (argparse CLI, env as defaults;
  writes `cameras.npz` + `manifest.json`; purge covers them; script = shim); **renderer** validates the
  manifest FIRST, aims per frame from `cameras.npz`, `clip_end=2000` (bowl ~150 m > the 100 m default).
  **Verified E2E locally** (gated smoke `tests/e2e/test_video_path_smoke.py`, 4 green): dry-run scene →
  `anim_export.main` → manifest+cameras asserted → REAL Blender 5.1.2 render → `BLENDER_ANIM_CAMS
  virtual-operator` + PNGs; unmanifested dir REFUSED («Re-run anim_export»). **4-camera render eye-checked**
  (`out/smoke_virtualcam/`): broadcast frames the action from inside the bowl (lines legible), sideline =
  eye-level bodies, goal = down-the-pitch, top = full-pitch schematic. Full suite green; ruff/mypy clean on
  all new files. **NEXT:** re-export + re-render the real-clip deliverable on the pod through the new path
  (old export dirs are refused by design), eye-judge framing; then crowd-kaleidoscope / grass-past-bowl /
  night tone / texture coverage.

- **2026-06-28** — **FULL DELIVERABLE RENDERED on the GPU pod + the "100 % CPU / 0 % GPU" bug found & fixed.**
  Rendered the whole deliverable — **48 frames × 4 cameras** (broadcast, sideline, top, goal) @ **1920×1080,
  64 samples, 25 fps** — on the RTX PRO 4500 (Blackwell) pod, stitched one mp4 per camera, pulled all 4 to
  **`out/deliverable_video/{broadcast,sideline,top,goal}.mp4`** (gitignored). 192/192 PNGs (48/cam),
  `BLENDER_ANIM_OK`. **GPU-utilisation bug (user: «под 40 минут молотит на 100% cpu и 0% gpu — найди ошибку»)
  had 3 causes in `blender_animate.py`, fixed:** (1) **hybrid device — the main one:** the device loop enabled
  BOTH the GPU and the CPU (`dev.use = … in (chosen,"CPU")`) → Cycles path-traced on all 28 CPU cores
  alongside the GPU (the pegged-CPU symptom). Now **GPU-only** (`== chosen`). (2) **scene re-uploaded every
  render:** without `use_persistent_data` each `render.render()` tore down + re-uploaded the whole Cycles
  scene (BVH for ~16–20 deforming bodies) — VRAM collapsed to ~8 MiB between renders, GPU idled on CPU
  re-sync. Added **`sc.render.use_persistent_data = True`** → scene stays resident. (3) **CPU denoise:**
  default OpenImageDenoise ran on CPU per render; set **`sc.cycles.denoiser = "OPTIX"`** → denoise on GPU.
  Plus a per-frame **`bpy.context.view_layer.update()`** so persistent data re-syncs each frame's re-posed
  meshes (else it could reuse the prior pose). **Measured before→after:** GPU util ~0–25 % → **99 %**; VRAM
  8 MiB → **~2472 MiB resident**; CPU load (1 min) ~75 → **~1.3**; **~52 s → ~1.3 s per camera-render**; full
  192-frame render in **~4 min**. **Correctness verified** (persistent data isn't reusing stale geometry):
  md5 broadcast `frame_0000 ≠ 0024 ≠ 0047` (poses animate over time) and broadcast/top/goal `frame_0000` all
  differ (distinct camera views). Pod **STOPPED** after pull (cost). NEXT: eye-judge the 4 mp4s vs the clip,
  then attack the biggest remaining photoreal gap. File: `scripts/blender_animate.py` (4 edits).

- **2026-06-28** — **v2 lever 3 DONE: light-from-clip — floodlit NIGHT, auto-detected + manual override.
  The agreed 1→2→3 plan is now complete.** **Premise flip (R-6):** scoping the lever revealed the target
  clip is a **floodlit NIGHT match, not a sunny day** (measured from raw frames: no sky in frame, near-white
  pixels ≈ RGB [0.96,0.96,1.0] = neutral floodlights w/ faint cool tint, bright-vs-shadow grass only
  1.2–1.5× = soft even multi-directional shadows). So the original plan — recover a single sun's
  azimuth/elevation from a hard shadow + a Nishita blue sky — was **wrong**; there is no single sun and no
  blue sky. Reshaped to a **floodlit-night model**. **What shipped — both auto & manual (user ask: «нужна
  опция автоопределения и задания вручную освещения»):** (1) **AUTO** `estimate_light_color` (new
  `adapters/render/lighting.py`) — a **white-patch illuminant** estimate: keep the bright, *low-saturation*
  pixels (white kit / lines / bright grass — not the green grass or a red shirt, not the near-black sky),
  take a high per-channel percentile, peak-normalise → the floodlight colour. On the clip it measured
  **[0.969, 0.953, 1.000]** — independently reproducing the hand-measured WB (neutral, blue-peak cool).
  `anim_export.py` writes it (+ the night-model defaults) to **`lighting.npz`** under the `SOURCE_OK` gate.
  (2) **MANUAL** `blender_animate.py` reads `lighting.npz` as the baseline, then any `--light-rgb /
  --light-energy / --sky-strength / --sun-count / --sun-elevation / --sun-angle` flag overrides it (fallback
  = measured defaults baked into `scene_builders`). (3) **Model** new `build_stadium_lighting` in the shared
  `scene_builders.py` — a **dark world** (`light_rgb × sky_strength`) + a **ring of 4 soft, high SUN lamps**
  (wide 9° angle, elevation 65°, tinted `light_rgb`) → even low-contrast fill with faint *multi-directional*
  shadows. `_cycles_script` is untouched (keeps its own daytime sky — it's a separate formal demo, not the
  deliverable). **Tone-map finding:** Blender's default **AgX** view transform desaturated the floodlit grass
  to grey-green (render blue ≈0.43 vs clip 0.20) and lifted the sky; the clip is standard **Rec.709**, so
  `blender_animate` now renders through the **Standard** view transform → broadcast-faithful saturated colour
  (mowing stripes, crisp white lines, crowd bowl all pop). **Tuning:** `sky_strength` 0.06→**0.03** so the
  night sky (≈0.19 display) matches the clip's dark upper region (≈0.17), not a flat grey. **VALIDATED E2E:**
  re-export prints the measured colour → `lighting.npz`; AUTO render (`/tmp/val_frames_final/{broadcast,top}`)
  shows an evenly floodlit night pitch; MANUAL render (`--sky-strength 0.03 --light-rgb …`) confirmed both
  override paths; gated `test_cycles_render_produces_a_nonempty_frame` still passes (daytime path intact);
  4 new `test_lighting.py` + full unit suite green; ruff/mypy clean (only the 2 pre-existing
  `blender_animate.py` lints remain). Files: `src/pitch3d/adapters/render/lighting.py` (new),
  `tests/unit/test_lighting.py` (new), `src/pitch3d/adapters/blender/scene_builders.py`,
  `scripts/anim_export.py`, `scripts/blender_animate.py`.

- **2026-06-28** — **v2 lever 2 DONE: grass PBR in the deliverable video path, via the shared
  `scene_builders.py` (the "B" refactor).** The two self-contained Blender scripts now share ONE
  procedural grass node-graph: extracted `_cycles_script._grass_material` into a new **pitch3d-free**
  `src/pitch3d/adapters/blender/scene_builders.py` (`build_grass_material(bpy, stripe_scale=…)` — only
  `bpy`+stdlib, `bpy` received as an arg, never imported). **Import mechanism (the contract):** Blender's
  `--python` under `--factory-startup` does NOT put the script's dir on `sys.path`, so each consumer adds
  the module dir and `import scene_builders` **by file** — `blender_animate.py` via a top-of-file shim
  (`<repo>/src/pitch3d/adapters/blender`), `_cycles_script.py` via a lazy `_scene_builders()` shim (its own
  dir). One definition, two consumers → the paths can't drift; `_cycles_script._grass_material` now just
  delegates. **Tuning finding:** the verbatim M2-9 wave **Scale 6** averages to **flat green** on the full
  105 m pitch (sub-0.2 m stripes wash out under AA/denoise — why the old plane looked flat); **stripe_scale
  0.1** gives **≈5 m mowing bands** that read at broadcast distance (top view ≈22 stripes across 105 m).
  **VALIDATED:** Blender-gated `test_cycles_render_produces_a_nonempty_frame` (manifest `env=grass+lines+sky`)
  still passes through the shared builder — formal path intact; E2E video re-render (`/tmp/val_grass3/{top,
  broadcast}`) shows clear mowing stripes on the pitch + grass surround vs the old flat plane.
  **Lever 3 (sun-from-clip) still pending.** Files: `src/pitch3d/adapters/blender/scene_builders.py` (new),
  `src/pitch3d/adapters/blender/_cycles_script.py`, `scripts/blender_animate.py`.

- **2026-06-28** — **v2 STARTED · lever 1 DONE: measured per-vertex BODY texture in the deliverable video
  path.** Scope agreed with user: «A через B, 1→2→3, свет из клипа» — port the photoreal levers into the
  *video* export path (the actual deliverable), order **1 body-texture → 2 grass-PBR → 3 sun-from-clip**,
  sharing the two self-contained Blender scripts **at the DATA/contract layer** (both stay pitch3d-free /
  `--factory-startup`; the upstream compute lives in `anim_export.py` under the venv). Generative finishing
  (option C) stays RULED OUT as primary (M2-0 spike: hallucinates unmeasured detail) — kept only as a possible
  M3 last-mile backstop. **Lever 1 needed NO B-refactor — it is a pure data contract.**
  **Mechanism:** (adapter) `avatar.bake_body_vertex_texture` (image-level core, unit-tested) +
  `measured_texture_from_clip` (clip wrapper) project each subject's real broadcast pixels onto its **posed
  SMPL-X mesh** through the solved camera — front-facing + in-frustum + nearest-at-pixel (z-buffer), averaged
  over ≤12 evenly-spread reference frames; unseen verts stay `measured=False`. (Same **180° camera-roll**
  auto-detect+rotate as the stadium bake — `camera_pose`, `-rot[1][2]<0`.) `anim_export.py` fills the
  unmeasured verts with the flat **kit colour** (R-6: never black/fabricated) and carries `vcolor,measured`
  in each `anim_subject_*.npz`; `blender_animate.py` attaches them as a **BYTE_COLOR "Col"** attribute →
  `ShaderNodeVertexColor` → Principled **Base Color** (LIT by the scene sun, unlike the emission crowd), with
  a clean fallback to the flat colour when the keys are absent (older exports / no source clip).
  **VALIDATED (local, no pod):** E2E export of the Colombia clip baked **tex 6–11 %/subject** (front torso —
  the recognizability-critical region); Blender re-render of **broadcast + action** (`/tmp/val_frames_hi/*`,
  1920×1080) eye-check — Colombia reads cream with a **readable "10"** on the back, Congo reads blue, bodies
  show torso/shorts/leg tonal structure + contact shadows, **zero black artifacts**. Low coverage is fine:
  the fallback IS the correct kit colour, so no greyness — `max_frames` left at 12. **Levers 2 (grass-PBR via
  the shared `adapters/blender/scene_builders.py` — this is where "B" lands) and 3 (sun-from-clip) still
  pending.** Files: `src/pitch3d/adapters/models/avatar.py`, `tests/unit/test_avatar_textured.py` (+1 test,
  17 green), `scripts/anim_export.py`, `scripts/blender_animate.py`.

- **2026-06-28** — **v1 polish: tinted crowd MOSAIC (replaces the stretched per-vertex bake).** User ask:
  «вырезать именно трибуны и выстелать мозаикой, а не растягивать» → «тонированную мозаику». The median bake
  stretched ONE pixel per vertex, so the crowd read blurry. Now the bowl wears a **real crowd tile** repeated
  over it (high-frequency detail) **multiplied by** the per-vertex measured median (low-frequency regional
  **tint**) — each stand keeps its true colour but gains real spectator texture.
  **Mechanism:** (core) `bowl_tile_loop_uvs` turns the bowl's own `(angle_frac, height_frac)` param into
  per-loop tile UVs (× `repeat_around` / `repeat_up`), with a wrap-seam fix that lifts the one face-column
  bridging angle 1→0 so u never runs backwards. (adapter) `extract_crowd_tile` cuts one clean patch: pick the
  frame seeing the most mid-band covered verts, crop a robust percentile bbox, then `_busiest_window` shrinks
  to the most edge-**DENSE** sub-rect — crowd is uniformly busy (thousands of small edges) while the LED /
  FIFA signage / scoreboard panels are flat with only sparse high-energy text, so density (not energy) drops
  the signage. (blender) `_add_stadium_mesh` normalises the tile to **unit mean** (a neutral detail map) and
  multiplies it by the vertex-colour tint into Emission, `extension="MIRROR"` to hide repeat seams.
  **Validation:** 9 geometry tests pass (added `test_tile_uvs_are_per_loop_and_seam_free`); extracted tile =
  packed Colombia crowd, dominant **yellow** = real fan colours, **no signage** (`/tmp/crowd_tile_a.png`);
  E2E re-export `stadium.npz {verts,faces,colors,uv,tile}` (tile 429×144, 48 % covered) → broadcast + goal
  rendered: the bowl now reads as **textured spectators** wrapping the pitch, clearly sharper than the blur
  (`/tmp/val_frames_mosaic/*`). No tuning needed (repeat_around=40, repeat_up=4). Changed: `core/scene/stadium.py`,
  `tests/unit/test_stadium_geometry.py`, `adapters/render/stadium_backdrop.py`, `scripts/anim_export.py`,
  `scripts/blender_animate.py`. ruff+mypy clean on new core/adapter; scripts ruff-clean (pre-existing items left).

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
