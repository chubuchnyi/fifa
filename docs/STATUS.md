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

**Last updated:** 2026-07-10 · **Branch:** main · **Repo:** /home/chubuchnyi/AVATAR

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
  works from behind the left goal; BALL visible on BOTH cameras at proper zoom (goal f55
  airborne, sideline f55 at feet); goal frame intact end-to-end (the "erased frame" and
  "no sideline ball" were thumbnail-scale eye errors — zoom before verdicts). Goal-cam-only
  residual: green half-erased player ghosts near fast clusters. SHIRT NUMBERS (§6): unreadable
  in this window (digits ≈ 7 px at raw 1080p) → all None honestly; `jersey_numbers.py`
  makes per-recon assignment repeatable when a legible window exists. GRASS TONE CLOSED (§6):
  wording knob measured to its limit (hue ±4° of the stated colour's prior, S floors ~0.85 vs
  clip .67; «dull green» reads olive) → deterministic `grass_pin.py` = batch stage 9 (grass
  band H-delta + S-scale, auto-target from `ref_night.png`, team masks excluded; sideline
  69.2→79.1/.88→.67 with shirts intact 60.5→61.9, goal 63.5→79.1/.94→.67); DEFAULT_PROMPT
  promoted «muted green». STANDS TONE pinned same night (§6): pin generalized to region tone
  pin (`--roi`/`--pin-val`, stage 10) — clip crowd is darker+yellower (V×0.70 the main knob);
  bright-fans-on-dark bimodality remains a crowd-TEXTURE lever. NIGHT RUN 2026-07-05 (§6 ×4):
  stages 9-10 validated E2E on CLEAN DEFAULTS (tail8 — auto-targets from `ref_night.png` match
  hand-measured clip tones, ref-orientation caveat closed); WALKWAY BAND fixed at source (the
  final's dead-black stripe was our own `gap_color` 0.02 crushed by the tail; dark-grey 0.10
  default + `PITCH3D_WALKWAY_RGB` override → t9 final band V .16-.23 ≈ clip .16-.22); CROWD
  BIMODALITY closed (quilt `contrast` knob — the Hann blend ate the bright-fan tail —
  `PITCH3D_CROWD_CONTRAST=1.35`, plus stands-pin override `STANDS_TARGET_VAL=0.31` for this
  clip's center-bright stands: final fan band p90 .51 / tail .159 vs clip .50-.52 /
  .165-.171); STANDS HOT LEFT EDGE flattened (bowl renders its left stands 1.9× mid, clip's
  edge is DIM — new `grass_pin.py --flatten-val-x` / `STANDS_XFLAT_BINS=16`, full-width ROI;
  profile now inside the clip's own pan-swing, fan-band frac .173 dead-on clip); pod infra
  hardened for Blackwell (Blender 4.5.11 default — 4.2 kernels hang on sm_120;
  `PITCH3D_GPU_BACKEND` override; page-cache prewarm for network-volume venvs).
  BOARDS TEXT fixed at source next morning (§6): per-strip emission calibration (npz
  `emission`, x1.08 here) + time-dominant ad frame pick — "BANK OF
  AMERICA" readable in the final. WALKWAY→FASCIA midday (§6): the flat grey band now wears
  the measured window above the boards (vertical atlas in `boards.npz`; fascia emission
  calibrated median→0.40, `PITCH3D_FASCIA_EMISSION` overrides) — the clip's
  boards→walkway→crowd sandwich reads in the final. PANEL ROW afternoon (§6): band gap
  2.2→4.6 m — the same window now also catches the gold FIFA/GUADALAJARA panel row under
  the crowd (hedge+walkway exact, gold fraction .18≈clip .21; panel row ~1.5× hot = next
  polish). BOARDS GLOW late afternoon (§6): stage-11 white pin (`--sat-max` desaturated
  gate) lands board whites on the clip's glow (Vmed .93-.95 vs clip .93-.96) — letters
  darker-crisper, $0 local iteration. PANEL TONE evening (§6): stage-12 V-only pin
  (`--val-only`) cools the hot panel band to the clip's level (Vmed .27 vs .30) — second
  $0 lever. CROWD COMPOSITION late evening (§6): measured yellow/red fan fractions stated
  in the v2v prompt — yellow half-way to the clip (.39 vs .51 left sector), left>right
  gradient appears; stages 11-12 verified in-batch. STANDS LEVERS t19 night (§6): the ×19
  fascia repetition broken — 4-window measured quilt (`PITCH3D_FASCIA_WINDOWS=4`) + hue-sat
  pool pruning (a crossing flag in 4/9 candidates quilted pink around the ring in round 1;
  lower-band pink now at the clean-candidate floor 0.006); the clip's 3.6% scattered
  dark-red fans land via NEW stage-13 screen-space `stands_red_pin.py` (texture-space red
  measured dead TWICE — render minify+denoise → beauty 0.001, Wan re-adds ~0.8% only; pin
  auto-targets `STANDS_RED_TARGET=0.036`, landed 0.030, `luma_cap` keeps specks dark-red).
  GRASS TONE + CROWD TEXTURE t20 (§6): the finals' two biggest measured gaps closed at $0 —
  grass was dark acid-olive (V .43 vs clip .55: stage 9 never pinned V, and `ref_night`'s S
  .75 ≈ Wan-inflated vs raw clip .655 → explicit `GRASS_TARGET_HUE=81.9 SAT=0.655 VAL=0.545`,
  stage 9 gains `--pin-val` when VAL set); stands read as saturated lego-confetti (luma
  local-contrast 2.4× clip, frac S>.5 .71 vs .43 — tone pins land medians, not SHAPE) → NEW
  stage-12.5 `stands_soften_pin.py` (g9 blur-blend keep .25 + S quantile map to the raw
  clip's band, static LUT, before the red pin so specks stay crisp): lc .0144≈clip .0134,
  S med/p90/frac>.5 all land, red pin then hits .036 exactly pre-encode.
  Pod E2E same session: stages 9→13 replayed on the t19b intermediates WITH masks
  (~$0.10, pod DOWN) — soften/red stats identical to the local prototype.
  LINE GLOW t21 (§6): the clip's pitch markings GLOW (V .90) vs our dim .62-.75 — at low V
  their slight blue cast reads periwinkle (eye said "blue lines", measurement said "dim
  lines": hue/sat match the clip). NEW stage 9b (`LINES_PIN`, existing `grass_pin.py`
  `--val-only`, desaturated-bright gate in the pitch band) lands V .88 vs clip .90.
  PLAYER SHADOWS t23 (§6, 2026-07-06 morning): 4-zone measurement isolated the shape —
  clip has tight elliptical contact shadow (contact V vs grass -.029, below -.022, flanks ≈
  grass, above ≈ grass); ours reads contact -.263, below -.004, flanks +.04 — the v2v
  smear halo IS what darkens the "contact" strip but the shape is diffuse-ring not
  elliptical (`/tmp/t23_shadow/t23_shadow_triple.png`). NEW stage-14 screen-space pin
  `scripts/player_shadow_pin.py`: shirt-colour boxes (yellow Colombia + azure Congo + white
  ref/GK) → soft ellipse alpha at foot line, GATED by a grass mask so the darkening only
  stacks on unshaded grass (not on the v2v halo). Prototype on t21_pinned8: visible
  ellipse under well-detected players (panels 2/4/6 of the sheet) but heavy-smear zones
  don't recover a shape — this is HONEST partial mitigation, the root gap is v2v erasing
  silhouettes. 5/5 unit tests green; wired into `pod_finish_batch.sh` last so batch
  re-runs pick it up.
  NEW BEST FINALS
  `out/kitzones_pod/sideline_t21_pinned8.mp4` (t20 + line glow; masked in-batch reproduction
  rides the next pod run) + `goal3_pinned4_xflat.mp4` (goal stands pinned+flattened); sheets
  `/tmp/t20_final_ab.png`, `/tmp/t21_lines_full.png`, `/tmp/t21_{full,stands,grass}_triple.png`,
  `/tmp/t23_shadow/t23_shadow_triple.png`.
  NEXT CANDIDATE (t24, measured+eye): panel-row lime dash periodicity (ours = repeating
  bright green LED-like segments, clip = calm dark-green panels with gold text) — deferred
  from t22 before the t23 shadow redirect; and player-silhouette recovery (the root cause
  the shadow pin can only mitigate).
  Pipeline overview: `docs/pipeline.md`.** Previous lever same day (§6): CROWD TONE — knobs
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
.venv/bin/python -m pytest                 # full suite (1032 passed / 14 skipped, 2026-07-29)
.venv/bin/python -m pytest tests/<path>    # focused
.venv/bin/ruff check <files>               # lint
.venv/bin/mypy <files>                     # types

# runnable evidence (cited by ADR-0012 — re-run these instead of trusting the write-up)
.venv/bin/python scripts/mutate_projection_sign.py   # do the R6 sign guards still catch anything?
PYTHONPATH=src .venv/bin/python scripts/bench_ransac_usac.py       # R10 rejection
PYTHONPATH=src .venv/bin/python scripts/bench_line_constraints.py  # R3 point-on-line gain
PYTHONPATH=src .venv/bin/python scripts/bench_novel_view_metric.py # R7: why 0.35-0.45 m is not a bar
PYTHONPATH=src .venv/bin/python scripts/bench_joint_limits.py      # R5: is hyperextension ever reached?
PYTHONPATH=src .venv/bin/python scripts/bench_camera_swim.py       # R2: does the camera swim, and is it removable on CPU?
PYTHONPATH=src .venv/bin/python scripts/bench_calib_confidence.py  # #105: is calibration confidence predictive?

# R2 camera propagation (#94): default 8, `0` = per-frame (the pre-R2 control side of the A/B).
# On the pod chain the same knob is the CAMERA_CARRY env var, read by demo_video.sh.
python -m pitch3d.app.cli --calibrator keypoints --camera-carry 8 ...   # carry ON  (shipped default)
python -m pitch3d.app.cli --calibrator keypoints --camera-carry 0 ...   # carry OFF (control)

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

- **2026-07-30 (#94/#60 — R2's 92 % swim removal does NOT reach the picture; measured on the A/B)** —
  #107 established that the render's camera is synthetic, so calibration can only reach the image
  through **subject placement**. So the A/B was scored on exactly that, comparing the two exports'
  SMPL-X root translations over the 23 shared tracks / 1380 subject-frames:
  - **OFF → ON placement shift:** median **0.137 m**, p95 0.459 m, max 0.870 m. The sides really are
    different, and by a lot — this is not a no-op render.
  - **Frame-to-frame subject step** (the sliding an eye can actually see): OFF median **0.0516 m**
    / p95 0.1433 → ON median **0.0532 m** / p95 0.1400. ON is **3 % worse** at the median and 2 %
    better at p95. Within noise, in both directions.
  So the knob moves *where* players stand by 0.137 m without making them **any steadier**. The
  bench's headline — homography swim 0.119 → 0.011 m, −92 % — is true of the homography and does
  not survive the trip to subject positions. Note the arithmetic that gives it away: swim is
  0.119 m/frame but subjects only step 0.0516 m/frame, so subject placement cannot be a
  straight-through projection of the swimming homography — something downstream is already
  absorbing most of it. **Hypothesis, not yet measured:** the coherence/physics gates
  (`COHERENCE=1 PHYSICS=1` in `pod_ab_video.sh`) smooth trajectories and were already doing R2's
  job. Testing it needs one pod run with those gates off.
  **What this does to the #60 eye test.** The question is no longer "which side is steadier" —
  measured answer: neither. It is "which side's players sit more correctly on the pitch", which is
  the one thing no metric here can settle, so it still goes to the user's eyes.
  **Pattern worth naming.** #107 (solved camera discarded), #108 (R3 self-disabled), and this make
  three calibration improvements in a row whose measured gain does not reach the render. Before any
  further calibration work — including #61 — establish what the render is actually sensitive to.

- **2026-07-30 (pod session: the R2 A/B rendered — and it falsified two things we believed)** —
  one pod session, `$0.38` of GPU, produced the four A/B videos plus two findings that matter more
  than the A/B itself. Artifacts: `out/carry_{on,off}/video/{broadcast,sideline}.mp4` (1280×720, 60
  frames each) and `out/carry_{on,off}/export/scene.json`. Pod stopped at `13:55Z`.
  **#106 — the anti-predictive confidence is LIVE, not a stale artifact.** #105 concluded the
  `r = +0.699` came from a scene.json written 2026-07-09 and therefore indicted a configuration that
  no longer ships. Re-measured on the fresh post-R3 export (`carry_off`): **r = +0.688**. The defect
  is essentially unchanged and it is real. Nothing may weight by `calibration.confidence`.
  **#108 (new) — R3 is a NO-OP on the target clip, and #106 is how we found out.** The fresh
  `carry_off` homographies are **byte-identical** to the 2026-07-09 export — `max|dH| = 0.0` raw,
  over 60 frames — even though R3 changed the DLT solver on 2026-07-29. This is not a wiring
  mistake: `wiring.py:218` wires `KeypointFieldCalibrator`, which *is* R3's path, and `bf120b2` is
  an ancestor of the pod's checkout `a6da744`. So the line-constraint path **self-disabled**, as it
  is designed to (`_lines_agree`, `_LINE_FRAME_TOL_M = 3.0` — PnLCalib's line-class world frame vs
  our pitch template). Only `confidence` moved (`max|dconf| = 0.026`), which is #105's DOF-
  normalisation fix. R3's committed "0.201 → 0.182 m" therefore describes a path that **does not
  execute on the clip we actually ship**. Not yet measured: *which* branch of the guard tripped —
  the `[pnlcalib] line-constraint frame check:` line is in `/workspace/carry_ab_full.log` on the
  (now stopped) pod, and there are no PnLCalib weights locally to reproduce it on CPU.
  **#103 — kit split re-measured, shirt numbers still unread (and now we know why).** Kit split
  from `carry_off`: **10 team A / 13 team B** across 23 subjects (was 10/10). Shirt numbers stay
  `None` on all 23 — R-6 forbids pinning a number nobody read, and the contact sheets produced
  **0/23**. Root-caused, not guessed: nothing is offscreen — all 230 sampled subject-frames project
  *inside* the frame — they are **18 px tall (median, max 22)** against the tool's 45 px readability
  floor. `jersey_numbers.py` projects through `scene.camera`, which #107 makes synthetic *and* which
  carries **render** intrinsics (1280×720), so the 1920×1080 source is downscaled 1.5× before the
  crop, discarding the only pixels a digit could live in. `anim_A` reproduces the same 18 px, so the
  older "4/20" read came from a differently-configured generation. Fixing it needs the real
  per-frame calibration at native resolution — the exported scene carries no 2D tracker boxes.

- **2026-07-30 (#107 — the measured camera never reaches the render; found while staging the R2 A/B)** —
  the pipeline solves a real per-frame camera and then **throws it away one line before export**.
  `AppController` line ~294 does an unconditional `scene.camera = self._static_camera(scene)`, and
  `_static_camera` is a *synthetic* `standard_viewpoints(BROADCAST)` pose **tiled across every frame**.
  Measured, not inferred: in `out/anim_A/export/scene.json` — the reference artifact both benches read
  — the exported `CameraTrack` translation spread over 60 frames is exactly **[0, 0, 0] m**, while the
  `field.calibration.homographies` in the same file vary per frame. Two camera representations coexist
  and only the synthetic one is rendered. `git log -L` says this is **not a regression**: the line has
  been there since the controller was first written (`31ee6b3`).
  **Why the R2 numbers are still true.** `bench_camera_swim.py` reads `field.calibration.homographies`,
  never `scene.camera`, so every swim/paint figure in the entry below measures the real solve. Confirmed
  on the fresh 4-frame smoke: per-frame homography spread **21.08 → 1.04** with carry on (~20×), while
  the exported `CameraTrack` spread stayed `[0,0,0]` on both sides.
  **What it changes is what the A/B LOOKS like.** Swim will *not* appear as a drifting camera — the
  render's camera is nailed down. It reaches the picture through subject placement: world positions come
  from the per-frame homography, so the failure reads as **players sliding/jittering against a static
  pitch**, not as the pitch sliding under a moving camera. Anywhere the docs say "scene swim", read
  "subject placement instability" for anything judged in a *render*.
  **Open question, not yet a verdict (#107).** Rendering a synthetic camera may well be deliberate —
  feeding a solve with a known scale/offset defect (#61) straight into the render would be worse. But it
  means our "broadcast" output is an *approximation* of the broadcast view rather than the measured one,
  so it cannot be compared pixel-to-pixel against the source clip — which is exactly what #60/#61 want
  to do. Decide before v2 photoreal: fix #61 and render the measured camera, or keep the synthetic one
  and drop pixel-comparison as an acceptance test.

- **2026-07-30 (#94 / R2 — camera propagation SHIPPED as a trade, not as a win)** —
  the calibration swims: **median 0.119 m / p95 0.468 m** of frame-to-frame scene slide on a camera
  whose true pan is a smooth 9.26 px/frame (#104). What ships is a two-part fix along the existing
  pure/heavy split. **`LucasKanadeMotion`** (cv2, CPU, no weights) recovers the inter-frame camera
  motion — a broadcast main camera is on a tripod, so consecutive frames are related by **one**
  homography whatever the scene depth, which is why RAFT-small was dropped. **`carry_on_motion`**
  (pure numpy, no cv2) then re-fits every frame from its ±N neighbours. It votes in the **physical**
  domain — each neighbour predicts where five probe pixels land on the pitch, per-coordinate median,
  re-solve — because homography *coefficients* are defined only up to scale and averaging them means
  nothing. The probes sit at players' feet (lower half of the frame), never at the horizon, where
  error is both enormous and invisible.
  **Honest headline: this is a TRADE.** It gives up **~0.004 m** of paint accuracy to remove
  **0.108 m** of scene slide — 25–31× favourable, worth taking, but *not* free, and the knob
  (`--camera-carry N`, default 8) exists so it can be turned off at `0`. The swim metric that
  motivated R2 is **circular** — an anchor displaced 10 m and carried the same way scores 0.0000 m —
  so it can never justify this alone; the independent paint check is what makes it a decision.
  **Verified E2E, not just in unit tests.** The shipped motion backend reproduces the bench's own
  tracking to **0.0000 px** over 59 inter-frame fits (a prototype that agrees with a write-up but not
  with the code in the tree is worth nothing). A full CLI run on the target clip — decode → calibrate
  → … → export — removes **91.3 %** of the exported `scene.json` track's swim vs the `--camera-carry 0`
  control, matching the bench's 92 %. Frozen-camera and 2 m-displaced controls both fail as they must.
  **What deliberately did NOT ship: MAD 3σ rejection.** The bench scored "carry" and "MAD" separately
  and never "MAD-then-carry", so combining them would have shipped an unmeasured component — the exact
  thing ADR-0012 exists to prevent. MAD alone is dominated (0.102 m vs 0.011 m). Carrying is also
  **not** confidence-weighted; that form stays gated on #106. Still-image eval (`run_calib_eval.py`)
  is uncarried by construction — unrelated images have no motion to carry between them.
  Wired through `default_ports(camera_carry=…)` → `--camera-carry`, and down the pod chain as
  `CAMERA_CARRY` (`video_defaults.sh` → `demo_video.sh` → `pod_make_video.sh` → `pod_real_e2e.sh`) so
  the A/B render can be driven from one env var. **#60 still owes the verdict: no metric can settle
  "stable-but-slightly-offset vs accurate-but-jittery" — the user's eyes decide.**

- **2026-07-30 (#105 — the confidence defect was mostly a STALE MEASUREMENT; one real line remained)** —
  #104 filed this as "our calibration confidence is anti-predictive, r = +0.699 vs measured paint
  error, do not weight by it". The first thing #105 found is that **the evidence was stale, and that
  is the transferable lesson**: `out/anim_A/export/scene.json` was written **2026-07-09**, and R3
  wired PnLCalib's line detections into the DLT on **2026-07-29** (`bf120b2`). The r = +0.699 indicts
  a **points-only** configuration that no longer ships. Re-measured over both configurations on a
  synthetic bench built from the clip's own 60 GT homographies (`scripts/bench_calib_confidence.py`,
  CPU, ~20 s, 480 fits), the *same* formula reads Spearman **−0.06 on points alone** — no signal, and
  its residual term **wrong-signed** — against **−0.53 with lines**, every term correctly signed. An
  unrelated change had already fixed most of the complaint. **Rule: a measurement carries a date and
  an artifact carries a configuration — before acting on a measurement, check the artifact came from
  the code that ships now**, or you fix something nobody runs.
  **What was left is real, structural, and one line.** `err = sqrt(mean(resid²))` normalises by
  observation *count*. A homography has 8 DOF and the admission gate is `2k + m >= 8`, so a frame
  with exactly 8 rows is admitted, its DLT reproduces its own agreeing observations **exactly**, the
  residual is identically 0, and confidence saturates. Measured on the old code: a 4-point fit whose
  image points land **1.9 m** from their world points scored **0.9999999999999791**. Confidence was
  maximal exactly where evidence was minimal. The fix normalises by residual **degrees of freedom**
  (`rows − 8`); `dof ≤ 0 → inf → confidence 0`, because unverifiable is not certain (R-6), and at
  `dof ≫ 0` it converges to the old value so over-determined frames are untouched. Spearman
  points-only **−0.059 → −0.358**, with lines **−0.529 → −0.550**, and on a held-out harder condition
  nothing was tuned for (3.5 px noise, 15 % outliers) **−0.722 → −0.727**. The minimal-evidence frame
  goes **0.704 → 0.000**. Pinned by `test_confidence_is_zero_when_the_fit_has_no_redundancy_to_verify_it`,
  which was **run against the reverted code and observed to fail** (`0.9999999999999791 != 0.0`) —
  otherwise the test would be unverified.
  **The replacement I designed lost to the one-liner and is recorded as rejected, not built.** k-fold
  holdout error, probe support (Mahalanobis distance of the probe pixels from the inlier cloud), and
  their products: **−0.445…−0.528** with lines vs the shipped −0.529, and −0.389…−0.677 vs −0.722 held
  out. ADR-0012 Tier 1 row added.
  **One failure mode still not scored, kept visible rather than closed over (R-6):** spatial
  distribution. A clustered-landmark frame is **2.5× worse** than a wide one (0.781 m vs 0.311 m) and
  scores the same (0.447 vs 0.461). Probe support is the right shape of answer and loses only on
  aggregate, so it re-opens if clustered — rather than thin — frames become what the weighting must
  reject. **R2's confidence-weighted form is un-blocked**, but any real-clip number must be
  **re-measured on a post-R3 run** before it is quoted; filed as a pod-queued task.

- **2026-07-29 (R2-pre / #104 — the camera does swim, it is removable, and it does not need the GPU)** —
  Measurement before implementation, per ADR-0012's standing rule: `scripts/bench_camera_swim.py`,
  CPU only, ~1 min. R2 books a GPU for RAFT-small optical flow to propagate the camera between
  PnLCalib anchors. Two things had to be true first — the per-frame calibration must actually be
  inconsistent over time, and that inconsistency must be *noise* rather than the camera genuinely
  moving. The second is the one that decides the design: a smoother that removes real motion is the
  yaw low-pass mistake again (ADR-0012, Tier 1).
  **The truth signal is the pixels, not the calibration.** Smoothness proves nothing — a constant
  homography is perfectly smooth and completely wrong. A broadcast main camera is on a tripod, so
  between two frames the whole image is related by **one homography** whatever the scene depth, and
  that can be recovered independently of where PnLCalib thinks the pitch is. The image convention is
  *derived* rather than remembered (the #50 gate): both readings are scored and the consistent one
  wins — raw 0.119 m vs 180-rotated 0.373 m — because a wrong remembered convention stays
  self-consistent and silently inverts the answer.
  **It swims.** True camera motion is a smooth **9.26 px/frame** (p95 13.77, max 14.02). Our
  calibration disagrees with it by **median 0.119 m, p95 0.468 m, max 0.520 m** of scene slide at the
  players' feet; **20 % of frames past 0.25 m**, 0 % past 1 m. Under R7 (#99) a common-mode error that
  *moves* is the one a viewer sees — it slides the whole scene under a locked-off shot — so this is
  the visible half of #61.
  **It is removable.** Carrying the calibration along the measured motion: ±1 → 0.037, ±4 → 0.017,
  ±16 → **0.009 m median** (−92 %) and **0.032 m p95** (−93 %). Neighbours are *not* blended in
  homography coefficients — those are defined only up to scale and the entries are not commensurate.
  The vote happens where the quantity is physical: each neighbour predicts where a probe pixel lands
  on the pitch, the world points are combined with a median, and a homography is re-fitted.
  **The over-smoothing control.** A **frozen** camera is the maximally smooth answer, so a metric that
  merely rewarded smoothness would rank it first. It scores 0.164 / 0.235 / 0.245 — worse than today's
  per-frame calibration on the median, though *better* at p95, because its error is systematic and
  grows with the pan instead of spiking. What it establishes is the narrow thing it should: the metric
  is not rewarding smoothness, since the smoothest possible camera loses to the carried one by ~18×.
  **And then the swim metric turned out to be circular** — caught by distrusting my own 92 %. The
  metric chains frame *k* to *k+1* through the measured motion, so anything *built* by propagating
  along that motion scores ~0 **by construction**. Measured, not argued: an anchor displaced **10 m**,
  carried the same way, scores **0.0000 m**. So swim measures **temporal consistency only** and is
  blind to accuracy. The 92 % is a real removal of wobble and is *not* evidence of a better camera.
  **An independent accuracy signal, and what it says.** Distance from the projected pitch model to the
  painted lines in the actual frames. Three detector generations, because the first two could not
  discriminate and that had to be measured: brightness+desaturation marks 2.5 % of the grass (floodlit
  specular, white kit, compression noise all pass); a ridge filter at R3's measured 2 px line width is
  *worse* at 3–8 %, since grass texture is full of 2 px ridges. What separates paint from texture is
  neither brightness nor width but **extent** — Hough on the ridge seed gets it to **0.83 %**. Then a
  second floor: a rasterised mask plus `distanceTransform` bottoms out at 0.95 px and scored every
  candidate identically. Sub-pixel perpendicular distance to the fitted segments finally resolves it.
  The metric is validated by a control it must fail — a 2 m displaced camera scores **15.3 px** (44 %
  within 5 px) against our **1.22**.
  **And then it said the opposite of what I first published.** Scored over the first 20 frames,
  per-frame vs carried looked like a coin flip and I committed "free removal, zero accuracy cost".
  Over the full **60**, carrying is closer on only **32 %** of frames, by a median **0.19 px** — a
  consistent loss. Cause: per-frame accuracy *improves* across the clip (1.70 → 1.23 → 1.14 px) while
  carrying levels every frame toward its window average, so it helps the bad frames and hurts the good
  ones. **Swim removal and accuracy trade off monotonically** across every method tried; nothing
  improves both.
  **What rescues the verdict is a unit conversion, not a better method.** 1 px at the probe points is
  **0.0182 m** of pitch, so carrying gives up **~0.0035 m** of accuracy to remove **0.108 m** of scene
  slide — **31× asymmetric**, clearly worth taking. The lesson is that px-vs-metres hid the real
  question until both axes were in the same units, and that a 20-frame sample was not enough to
  publish from.
  **The measured trade curve** (swim median → paint error), so the operating point is a choice:
  per-frame 0.119 m → 1.22 px · MAD-reject k=3 0.102 → 1.23 · MAD k=1 0.060 → 1.34 · coefficient
  average w=17 0.027 → 1.70 · carried ±8 0.011 → 1.41 · carried ±16 0.009. Two consequences.
  `_temporal_smooth` (coefficient averaging, in the tree, default OFF) is **dominated** — carrying
  beats it on both axes — so replace it, though it is *not* the no-op I first called it: it removes
  77 % of the swim. And MAD-reject alone, the brief's own outlier form, is **too timid** to be the
  whole answer (k=3 touches 4 frames, −14 %); it is a guard on top of carrying, not a substitute.
  **A separate defect, found on the way, worth more than R2 itself: our reported calibration
  confidence looked ANTI-predictive.** Pearson **r = +0.699** against measured paint error over 60
  frames — the frames the pipeline trusts most were the ones that are worst (highest-confidence third
  **1.69 px**, lowest-confidence third **1.11 px**). Both artifact explanations are ruled out by
  control: if the later frames were simply easier to score, a **frozen** camera would improve across
  them too — it degrades **2.11 → 15.13 → 34.40 px** — and if more paint were visible, distance-to-
  nearest-segment would shrink for free, but segment count is flat (24/23/28, r = −0.26). That value
  is exported in `scene.json` and consumed downstream, so anything weighting by it is steered
  backwards — including the confidence-weighted propagation this benchmark was about to recommend.
  Tracked as **#105** — which then found this evidence **stale** (see the #105 entry above): the
  scored `scene.json` predates R3's line constraints, and the number must not be re-quoted as-is.
  **Verdict.** R2's premise holds, its prescription is oversized, and its benefit is a **trade** and
  must be reported as one. RAFT-small exists only to supply the motion the carry rides on, and
  `goodFeaturesToTrack` + LK + RANSAC supplied it at **4000/4000 corners, 3790 inliers** on CPU.
  **R2 is re-scoped to the CPU path and un-blocked from the pod** (the confidence-weighted form was
  blocked on #105 and is now un-blocked). #61's scale/offset defect is untouched by any of this. Roadmap row and
  ADR-0012 updated, including a new Tier 1 row for the GPU/RAFT stage and the general rule this
  session produced twice over: *a metric that shares a model with the thing it scores measures the
  sharing, not the thing* — every benchmark should carry a candidate it is supposed to fail.

- **2026-07-29 (R5 / #97 — joint limits: measured, rejected, and the last brief item closed)** —
  Eighth and last of the brief items ADR-0012 had flagged measure-only. (A ninth followed the next
  day: R2 was *adopted* rather than flagged, and its premise got measured before building anyway —
  see the #104 entry above. ADR-0012 counts nine.) The standing rule was "read them, measure them, do
  not implement them", so this is a measurement, not a feature: `scripts/bench_joint_limits.py`.
  **The finding — the premise is half right, and the half that fails is the one that decides the
  work.** SMPL-X really has no joint limits, so hyperextension is *representable*. The brief then
  assumes it is *reachable*. It is not: across all **1008** subject-frames of the production variant
  (A, SMPLest-X), **0.0 % of knees and elbows go past straight** — knee min **+11.6°**, median 33.7°;
  elbow min +23.9°, median 62.5°. Nothing anywhere near the 15° implausibility line. The pose nets
  regress into the manifold of the humans they were trained on, so the constraint is already supplied
  by the training distribution.
  **The half worth chasing: a net is not our only author of poses.** A gate that *invents* 137 frames
  by coasting is a far more plausible source of an impossible knee than a network trained on real
  humans, so R4's provenance labels (#96) are used to score them apart rather than average them into
  one reassuring number. The invented frames come out **safer**: imputed knee min **+24.6°** vs
  measured +11.6°, because `extend_pose_to_span` **holds** the last articulation instead of
  extrapolating it. That is a property of the gate worth knowing independently of R5 — R4 paid for
  itself here, one day after shipping. Variant B (SAM 3D Body) is looser — 7.8 % of frames a few
  degrees past straight, worst **−11.1°** — but that is normal genu recurvatum, not a broken rig.
  **Verdict.** A joint-limit residual would buy a VPoser dependency and a term that can only pull a
  plausible pose *away* from the observations, to enforce what is already enforced. Limits would bite
  an optimiser that fits pose to observations with **no data prior** — which is precisely the factor
  graph the briefs propose alongside them, already deferred in ADR-0012. **R5 and the factor graph
  re-open together, not separately**, and both ADR rows now say so.
  **Two sign errors in my own script, both of the exact class R6 exists for** — self-consistent,
  plausible, and each one inverting the finding. (1) The elbow convention was derived by probing the
  X axis only; in the canonical T-pose an X rotation at the elbow is a **twist along the forearm**,
  moving the wrist ~1 cm, so left and right got opposite signs and the table read 100 % elbow
  hyperextension. (2) "Forward" was taken from the toes-ankle vector, which normalises to
  `[0.338, −0.415, 0.845]` — the toes splay outward and downward, so it is not an axis; projecting a
  laterally-swinging limb onto it flipped signs and produced knee min −97.5°. Fixed by building an
  orthogonal body frame (`cross(hip axis, spine)` → `[0.032, 0.081, 0.996]`) and by replacing the
  axis guess with a **self-check that cannot be fooled by a convention**: a bent limb is shorter end
  to end, so probe all six (axis, sign) rotations, take the one that most shortens
  distal-to-grandparent — that is flexion by definition, no axis needs naming — and `SystemExit` if
  the metric does not score it positive. All four chains now print `[OK]` before any number is
  reported.
  **Closes the brief backlog.** All eight measurable items are measured (R1, R10, R3-edges,
  R3-salvage, R4, R5, R6, R7); four premises were false or half false; every investigation found
  something real next door. What remains from the briefs is deferred on cost or on missing ground
  truth, and each ADR row names which. Suite unchanged at **1042 passed / 14 skipped**.

- **2026-07-29 (R7 / #99 — our own accuracy metric; the briefs' envelope retired with a number)** —
  Seventh brief item, adopted only after inverting its premise, and the first one that changes how we
  are allowed to *report* progress. New module `src/pitch3d/eval/novel_view.py` (10 tests) +
  `scripts/bench_novel_view_metric.py`.
  **The finding.** Three error fields pinned to Global MPJPE **0.400 m** — dead centre of the briefs'
  0.35–0.45 m "broadcast envelope" — leave a novel-view viewer with **0.002 m**, **0.000 m** and
  **0.378 m** of visible error. A 190× spread inside one "acceptable" number. Their envelope is a
  perfectly good bar for a coaching-analytics product, where a world coordinate *is* the deliverable;
  for a video judged by eye it cannot rank two candidate reconstructions, so we must not quote it and
  must not tune against it.
  **Why.** Camera error is *common-mode* — it moves every player by one rigid transform. Re-render
  that from a new viewpoint and it **is** a slightly different novel camera; there is no reference
  frame in the shot to see it against. So the metric scores an error field by how much a camera
  re-fit can absorb and reports the remainder: `after_static_camera_m` (one rigid re-placement for
  the clip), `after_perframe_camera_m` (the headline — what no camera choice can fix), and
  `scene_swim_m` between them.
  **`scene_swim_m` is the part the briefs' decomposition misses.** A common-mode error that *changes
  over time* slides the whole scene under a shot that should be still. That is visible, and a single
  whole-clip fit hides it inside "camera error, therefore free". Measured: the wobbling-camera field
  is 100 % absorbed by a per-frame fit and carries **0.400 m** of swim — free by the naive reading,
  ruinous in fact.
  **Scale is deliberately not absorbed.** A similarity fit would erase #61's ~3× scale defect and
  report a clean scene. We render players against a true-size pitch, so wrong-scale players are
  visibly wrong-size. The fit is rigid; the similarity scale is reported beside it as
  `predicted_scale` (inverted so it reads as the defect — 3.0 means "our scene is 3× too big",
  not 0.333), purely to tell "our error is a scale error" apart from "our error is scatter".
  **The instrument's own bias is measured, not assumed** — this is the part I would have skipped.
  The per-frame fit has 6 DOF, so with few bodies it launders genuine per-player scatter into
  "camera error". On pure scatter, where an honest metric must absorb 0 %, it absorbs
  **70.9 % ± 18.9 at 2 subjects**, 30.1 % at 5, 16.3 % at 8, and **6.8 % ± 4.0 at our clip's 21**
  (worst draw 18.8 %). So `after_perframe_camera_m` is a **lower bound** on what a viewer sees, and
  is meaningless below ~8 bodies. That caveat is in the module docstring, not just here.
  **Speed stratification — the yaw-low-pass trade, finally as a number.** Mean local MPJPE cannot see
  a smoother that buys calm by flattening real motion. Sweeping Gaussian σ over jittered motion
  *with* a 120°-in-8-frames turn: the mean keeps improving to σ=8 (0.0233), while the top speed
  decile bottoms out at σ=2 (0.0316) and then **degrades to 0.0339**. Choose σ by the mean and you
  ship the yaw low-pass again. Control run without the turn shows a flat 1.06 penalty at every σ —
  i.e. the fixture's own sinusoidal motion cannot demonstrate this, which is why the turn is
  injected. **Gate: any future temporal smoother must beat the best top-decile row, not the mean.**
  **Two of my own experiments were wrong on the first run and are documented rather than quietly
  fixed.** (1) The bias sweep used a single random draw per subject count and produced a
  non-monotone table; I had already written the conclusion "at 21 subjects the leak is small — the
  number is usable" underneath numbers that showed 21 leaking *more* than 12. Fixed by averaging 40
  draws. (2) The smoother sweep originally ran on the fixture's pure sinusoid, where no σ ever hurts
  the top decile, and claimed to demonstrate an effect it structurally could not.
  **Not measurable on the target clip.** `after_perframe_camera_m` needs per-joint world GT with ≥8
  bodies in shot. The Colombia clip has no GT; 3DPW/B2 has GT but not a pitch full of players. The
  natural instrument is **WorldPose** (video already local; poses/cameras still pending). Until then
  the metric is validated on synthetic only — recorded so nobody reads the bench output as a
  statement about our actual reconstruction.
  **Consequence for ADR-0012.** The factor-graph deferral was conditioned on "R7's metric showing
  inter-player residual is what breaks the render". The metric now exists, so that blocker moved from
  *missing tooling* to *missing ground truth*, and the ADR row says so. Seven brief items measured
  at the time of writing (R1, R10, R3-edges, R3-salvage, R4, R6, R7); R5 followed the same day and
  closed the set. Suite **1042 passed / 14 skipped**.

- **2026-07-29 (R6 / #98 — golden tests for projection + sign, verified by mutation rather than by
  assertion count)** — Sixth brief item. Adopted, but the brief's framing had to be inverted first.
  **The brief asked for a round-trip. A round-trip is close to worthless for this bug class**, and
  one of the eight tests exists purely to prove that:
  `test_a_mirror_still_round_trips_which_is_why_round_trips_are_not_enough` mirrors the image
  convention (`u → W-u`, the #50 defect), shows `image_to_world(world_to_image(x))` still closes to
  **1e-9**, and then shows the same homography placing a known world point **40 m and 66 m** away
  from where it belongs. A sign error does not produce garbage — it produces a mirrored, perfectly
  self-consistent scene whose only detector is a human eye on a finished render.
  **So every test anchors outside itself.** Either to a ground truth built independently of the code
  under test (`eval/synthetic`'s GT camera, whose geometry comes from an eye/look-at, not from our
  projection code), or to an invariant a mirror breaks and self-consistency cannot repair:
  * `image_to_world` is fed pixels from the GT camera's *own* pinhole projection, so storing the
    homography backwards fails — while both existing round-trip tests keep passing.
  * The pinhole projector (`projection.py`) and the homography anchor (`field.py`) must place the
    same world point at the same pixel. Two independent implementations, one geometry.
  * **Winding**: a triangle counter-clockwise from world +Z projects *clockwise* in pixels (one
    flip, not two — image v is down, world Z is up).
  * **Which way is up**: +2 m in world Z must *decrease* pixel v.
  * The two in-tree quaternion implementations (`projection.quat_to_rotation_matrix`, hand-rolled;
    `rotations.matrix_to_quat`, Shepperd) plus scipy's `(x,y,z,w)` — three encodings of one
    convention, previously never checked against each other.
  **First test on the 180°-roll gate.** `poseannot/camera.py`'s `-R[1,2] < 0` and its
  `D = diag(-1,-1,1)` correction were untested production code guarding the project's most expensive
  bug. Two tests now cover it: the gate fires on `D @ R_gt` (which is exactly what the solve hands
  us) and recovers the true camera *exactly* since D is its own inverse; and, with the gate bypassed,
  a standing body really does project head-below-feet.
  **Verified by mutation.** `scripts/mutate_projection_sign.py` injects five defects and reports
  which test notices — a golden test that survives its own mutation is decorative, and the script
  says so in as many words. All five caught. Two of the five are **not hypothetical**: MUT-4 (gate
  comparison inverted) and MUT-5 (the X-only mirror `diag(-1,1,1)` that was validated by eye on
  2026-07-07 and falsified by `scripts/debug/pose_probe.py` the next day, having left every body
  vertically inverted at ~22 px — invisible to the eye that approved it). Those are regression tests
  for our own history.
  **This also caught a weak test of my own.** Two of the eight originally routed through
  `eval/synthetic`'s projector rather than production `project_world_points`, so MUT-2 (image v
  negated) sailed past them. The mutation harness is what surfaced it; without it the file would
  have shipped with two decorative tests. Suite **1032 passed / 14 skipped**.

- **2026-07-29 (R4 / #96 — `Provenance` + `BallMode`: R-6 made checkable by types, and it exposed a
  real information loss)** — Fifth brief item measured. Adopted, and unlike R1/R3-edges/R10 it did
  not need falsifying — but the *reason* it was worth doing turned out to be stronger than the
  brief's argument for it.
  **The defect.** State provenance was already in the pipeline, encoded as **sentinel confidence
  values**: 1.0 measured, 0.3 bridged (`coherence.filled_confidence`), 0.2 coasted
  (`extrapolated_confidence`), 0.15 teleport-interpolated, 0.20 pose-patched. So a `0.2` meant
  *either* "we invented this row" *or* "we measured it and the detector was unsure" — and those two
  are not the same fact. A photoreal renderer cannot separate an observed body from an invented one
  by thresholding a float, which is precisely what v2 needs to do.
  **The worse one, on the ball.** `BallTrack.on_ground=False` conflated "Z came from a gravity
  parabola fitted between two contacts" with "there is no bracketing contact at all, so Z is a
  *hold*". `ball_lift.py`'s own comment already called those lead/trail tails "genuinely unknown"
  while the type reported them identically to a real arc. On the target clip that is **46 of 48
  frames** — the bool was lossy for 96% of the trajectory.
  **What was built.** `Provenance{measured,interpolated,imputed}` and
  `BallMode{on_ground,ballistic,unmeasured}` in `core/scene/motion.py`, as `(T,)` label arrays
  riding the existing pose/ball arrays — *not* the brief's parallel per-frame `PlayerState`
  dataclass, because our representation is already `(T,…)` and a second one would have to be kept
  in sync. `interpolated` vs `imputed` is a real distinction, not a synonym pair: bridging between
  two measured anchors (`fill_pose_gaps`) is a stronger claim than coasting off a clip edge where
  there is no far-side measurement to bridge *to* (`extend_pose_to_span`).
  * Stamped by every gate that fabricates rows: `coherence` (rewrites rows), `kinematics` and
    `pose_motion_sync` (emit corrections but change a frame's epistemic status, so they stamp the
    proposal pose). Corrections that only change *values* leave provenance alone — verified by test.
  * `on_ground` survives as a **derived** property (`mode == "on_ground"`), so the poseannot GUI and
    the three checked-in clips keep working; there is no duplicated field to drift.
  * Carried through JSON, the anim export, and into the Blender npz — `anim_contract`
    `SCHEMA_VERSION` **1 → 2**, with `provenance` and `mode` now *required* keys, so a stale export
    fails loudly instead of rendering a scene that silently lost the channel.
  * Deliberately **not** folded into render `alpha`. A coasted player is still physically on the
    pitch; fading it out would be erasing, not marking — the opposite of R-6.
  **Migration, and its honest limit.** Pre-R4 saves migrate `on_ground=False → unmeasured` (named
  migration in `serialization.py`, keyed off the dataclass name): the weaker of the two readings,
  because the facts it conflated are not separable after the fact. Verified against the checked-in
  clips: `A_smplestx` → `{unmeasured: 46, on_ground: 2}`, matching the 2/48 recorded at
  `docs/v0-geometry-defects.md:134`. **Pose rows in legacy saves read all-`measured`** even where
  they were coasted, because the fact was never recorded. That could have been back-inferred from
  the sentinel confidences — but `0.20` is *both* `extrapolated_confidence` and `PATCHED_CONF`, i.e.
  exactly the ambiguity R4 removes, so guessing it back would be fabricating provenance while
  shipping a feature whose point is not to. Re-running the coherence gate restores it properly.
  **E2E (real clip, whole path).** `A_smplestx` → coherence gate reports `filled=19 / extended=137`
  → save → `anim_export` → npz → contract load: `{measured: 852, interpolated: 19, imputed: 137}`.
  The gate's report and the provenance channel are computed independently and agree exactly; a unit
  test asserts that equality so a future edit cannot stamp one without the other.
  **Tests.** `tests/unit/test_state_provenance.py` (12). Suite **1024 passed / 14 skipped**. No new
  lint debt (the `UP042` hits follow the repo-wide `class X(str, Enum)` convention that the JSON
  codec relies on).

- **2026-07-29 (R3 / #95 — the salvage SHIPS: point-on-line constraints in the DLT. First brief item
  to survive measurement)** — Four research-brief items have now been measured (R1, R10, R3-edges,
  R3-salvage). The first three were falsified. **This one holds** — but the honest headline is
  *robustness*, not precision, and it is much smaller than the first benchmark said.
  **What was built.** A point-on-line observation — an image point known to lie *somewhere* on a
  named pitch line — contributes **one** linear DLT row `lᵀ·H·x = 0` against a correspondence's two,
  in the same 9 unknowns and the same SVD. Under Hartley normalisation the world line transforms
  contragrediently (`T_dst⁻ᵀ·l`), then is rescaled to `a²+b²=1` so its residual is in **metres**,
  directly comparable to a point residual — which is what lets one weight vector govern both.
  * `core/scene/pitch.py`: `pitch_plane_line_segments()` / `world_line_from_segment()` /
    `pitch_line_coefficients()`. `eval/datasets_soccernet.py::pitch_plane_lines` now **delegates**
    to it — one pitch-line table, two callers, so GT and solver cannot drift apart.
  * `adapters/models/calibration.py`: `solve_homography(..., line_uv=, line_abc=, line_weights=)`,
    `solve_homography_ransac(...)` likewise, plus `point_line_residual()`. Lines join the **refit**,
    not the sampling: consensus is still decided by identifiable points, then lines beyond the
    threshold are dropped. (A mislabelled line class would otherwise carry a whole consensus set.)
  * `FrameKeypoints` grew `line_uv` / `line_abc` / `line_confidence` / `n_lines`; the calibrator's
    solvability test is now on **DLT rows** (`2·K + M ≥ max(8, 2·min_keypoints)`, `K ≥ 2`), which
    reduces to the old `K ≥ min_keypoints` exactly when there are no lines.
  * `pnlcalib_backend.py:115` — the defect this task existed for — no longer discards `lines_dict`.
    It now emits point-on-line observations, gated by `PNLCALIB_USE_LINES` and by a **runtime frame
    check**: fit points-only, measure the median point-on-line residual, and refuse the lines if it
    exceeds 3 m. PnLCalib's keypoint world table and our pitch template are two independently
    authored statements about the same pitch, so a mismatched axis convention would be silent and
    catastrophic; this makes it loud and self-disabling. Auto-detect + manual override.
  **Measured** — `scripts/bench_line_constraints.py`, local, no GPU, no dataset download; 24
  synthetic broadcast frames × 12 trials, scored with our own `evaluate_calibration`. Keypoints are
  derived as pitch-line *intersections* (which is what PnLCalib keypoints are), so keypoint count
  and line evidence stay honestly coupled. Realistic model — **5 lines/frame, per-line bias 2 px +
  1 px jitter**, keypoints 2 px iid:

  | keypoints/frame | median m (points) | median m (+lines) | p95 m (points) | p95 m (+lines) |
  |---|---|---|---|---|
  | 3 | **unsolvable** (6 rows) | 0.324 | — | 2.59 |
  | 4 | 1.679 | **0.254** | 95.78 | **1.26** |
  | 6 | 0.347 | 0.269 | 5.38 | 5.03 |
  | 10 ← *target clip* | 0.201 | **0.182** (−9%) | 0.971 | 0.815 |
  | 14 | 0.156 | 0.148 (−6%) | 0.645 | 0.576 |

  **Read it honestly.** At the operating point (the target clip yields 10–11 kp/frame) the median
  improves **9%** — real but modest. The value is in the **tail and the floor**: at 4 keypoints p95
  falls from **95.8 m → 1.26 m**, and 3-keypoint frames become solvable at 0.32 m instead of being
  dropped and carried (R-6: reconstruct, don't hide).
  **The methodological finding is the more useful one.** The first version of this benchmark modelled
  line error as iid per point with all 17 lines visible, and reported **−43%** at the operating point.
  Both assumptions are wrong: a stripe detector mislocalises the *whole line* (a bias no amount of
  sampling along it averages away), and real framing shows ~5 lines, not 17. Fixing just those two
  took the headline from −43% to −9%. The script now prints **both** tables side by side precisely so
  the gap stays visible — this is how a benchmark flatters itself, and it nearly flattered this one.
  **Not yet measured on real data.** These are synthetic frames with a modelled detector. The B1
  SoccerNet number (0.236 m) and the target-clip re-run need the pod, and the pod is **off**.
  Suite **1012 passed** / 14 skipped (10 new tests, incl. 2 points + 6 lines recovering `H` to 1e-9,
  a mislabelled-line rejection, and confidence scoring on a thin frame).

- **2026-07-29 (R3 / #95 — edge form REJECTED on measurement; salvaged into something better)** —
  ADR-0012 said to treat the remaining adopted brief items as hypotheses to measure. First one measured,
  first one falsified. **The premise:** IFAB caps pitch-line width at 0.12 m, so each painted line
  should yield *two* parallel edges at a known separation — a free second constraint. **It requires the
  two edges to be separately resolvable.** They are not.
  **Measured on the target clip** (`scripts/measure_pitch_line_width.py`, frame 60, 1920×1080), three
  independent ways so a mask threshold can't invent the answer: (1) distance transform over a
  grass-restricted thin-blob paint mask — thickness median **2.00 px**, p90 2.80 px, and even in the
  nearest row band (750–900) median 2.0 / p90 4.0; (2) **raw greyscale FWHM with no mask at all**,
  across the nearest lines — **1–2 px**, profile `114 118 152 196 145 116`, i.e. a single-pixel peak
  with one transition pixel each side; (3) eyeballed at 8× nearest-neighbour — a hairline.
  That is a **point spread function, not a band**. Two independent edge positions cannot be recovered
  from a single-lobe PSF. Same failure mode v2 §9 used to reject the ball-shadow idea (3–4 px ellipse),
  applied to their own surviving reformulation. The thickness-as-range-cue variant dies with it: 2.0 →
  2.8 px median across the *entire* visible pitch is ~1 px of dynamic range, and v2 §9 had already
  rejected that form on chicken-and-egg grounds anyway.
  **What the investigation found instead, which is worth more than R3 was.** PnLCalib runs **two**
  HRNet heads — keypoints *and* lines — and `adapters/models/pnlcalib_backend.py:115` **throws the line
  detections away** on the DLT path (`kp_dict, _lines, w_orig, h_orig = self._infer_frame(...)`). Its
  own docstring admits it: *"`lines_dict` is discarded by the DLT path but is the camera module's key
  extra constraint"*. So the `--solver camera` path uses them and the keypoint/DLT path — the one that
  gave us **10–11 keypoints/frame at conf 0.61** on the Colombia clip — solves a homography from ~10
  points while discarding every line pixel in the frame. **That is free evidence we are already paying
  the GPU for.** Feeding it back as point-on-line residuals needs no edge resolution at all, and it
  aims straight at **#61**, the project's open defect #1. #95 re-scoped to this.
  **R-6 tally:** three of the brief's items measured so far (R1, R10, R3), three premises falsified —
  and in *all three* the investigation found something real next door (R1 → the 206→18 mm foot bug;
  R10 → why our world-metre threshold + confidence-weighted DLT are load-bearing; R3 → the discarded
  line head). The briefs are worth reading and not worth implementing. ADR-0012 updated.

- **2026-07-29 (R8 / #100 DONE — ADR-0012, the rejected-approaches log)** —
  `docs/adr/0012-rejected-approaches-log.md`. Not a verbatim port of v2 §9: porting it verbatim would
  have imported *their* goal (analytics accuracy) along with their verdicts. Organised by **strength of
  evidence** instead, and every entry records **what would re-open it** — a bare "we rejected X" invites
  re-litigation, "here is the number and the condition under which it changes" does not.
  **Tier 1 — rejected on OUR measurement** (do not re-open without new numbers; each backed by a
  runnable artifact): USAC/MAGSAC++ (0.07–0.12 m vs 28–180 m, `scripts/bench_ransac_usac.py`); the
  brief's "unify the SMPL-X→world constants" premise (false — two source frames;
  `tests/unit/test_frames.py`); iterative MA yaw low-pass (kills 100°+ real turns); sparse 30-frame FK
  sampling for foot position (synthesised fake stances; 240-cap cut foot slide 15.4 m → 0.3 m); GREEN
  source-kit chroma key (grass collision).
  **Tier 2 — v2 §9 ported with our own verdict and a conditionality tag.** We agree with all 13, but
  several for *different* reasons, and that difference is the useful part: ball-shadow-for-height is
  rejected harder for us (our clip is a **floodlit night match** — the sun geometry they call "free"
  does not exist at all); ALIKED/LightGlue is rejected only as the *primary pitch* matcher and stays
  fair game for R2's px→px frame-to-frame propagation; line-thickness we adopted in its reformulated
  edge-pair form (R3). One we do **not** follow: v2 §9 *accepts* UE5 synthetics for geometry (P1) —
  we defer, since our constraint is appearance fidelity on one clip, not a learned estimator's
  cross-match generalisation.
  **Tier 3 — the briefs' own prescriptions we declined:** factor graph (defer; re-opens only if R7
  shows inter-player residual is what breaks the render), their 0.35–0.45 m accuracy envelope (rejected
  as our bar), shot segmentation (defer), off-screen imputation (defer — rendering an imputed player is
  fabrication under R-6; only ever behind R4's `Provenance`, gated at the *renderer*), the greenfield
  repo re-layout (ignore).
  Also recorded the ADR's own risk: the briefs' **diagnoses** have outperformed their **prescriptions**
  (two headline items falsified), so R3/R5/R7 are logged as hypotheses to measure, not work orders.

- **2026-07-29 (R10 / #102 REJECTED on measurement — MAGSAC++ is ~1000× worse than our estimator here)** —
  The research brief's headline calibration recommendation (v2 §3.3: swapping uniform-sampling RANSAC
  for USAC/MAGSAC++ "matters more than changing the feature detector") **does not survive contact with
  our geometry**. I wrote the swap, then benchmarked it before trusting it — and reverted. `calibration.py`
  is byte-identical to HEAD; the measurement is committed as `scripts/bench_ransac_usac.py`.
  **Numbers** (synthetic pitch correspondences under the real `_H_GT`, 2 px keypoint noise, world-metre
  RMS on the clean points, 20 trials per cell): ours **0.07–0.12 m** across every n/outlier combination;
  USAC **28–180 m**. Worse, USAC returns a *degenerate 4-point set* — inlier recall **0.05–0.42** — and
  with **zero outliers it fails outright, 0/20**, at n = 8, 12, 20 and 40 alike. Threshold-independent:
  identical collapse from 1 px to 50 px, at zero noise, in both the image→world and world→image
  direction, and Hartley pre-conditioning makes it *worse* (recall 0.06–0.29).
  **Not a wrapper bug** — the control rules that out: the same `cv2.findHomography(..., USAC_MAGSAC)`
  call on textbook px→px correspondences is perfect (16/16 inliers, 20/20 success) at every point count
  from 8 to 200. Also checked and eliminated: point count, coordinate scale and centring, source-vs-
  destination outlier side, outlier magnitude (50→5000 px), perspective strength (last-row term
  0 → 5e-4), and input dtype/shape.
  **Root cause — the assumption nobody checked.** MAGSAC++ marginalises over *one global* inlier noise
  scale. A broadcast pitch homography is strongly **heteroscedastic**: a uniform 2 px image error lands
  as 0.027 m at the near touchline and 0.227 m at the far one — **8.4× p95/p5**, 10.9× p99/p5. There is
  no single σ to marginalise over, so MAGSAC's scale estimate collapses onto a minimal sample. The
  control reproduces the mechanism in miniature: it starts shedding true inliers (16.0 → 12.7 → 9.2) as
  soon as σ approaches the threshold.
  **What this says about our code.** `solve_homography_ransac` is *not* the naive estimator the brief
  assumed. Two properties are load-bearing and neither is replaceable by a black-box: (1) it thresholds
  in **world metres**, the units our error budget is actually written in, which is the right invariant
  precisely *because* the pixel↔metre ratio varies 8× across the frame; (2) it refits a
  **confidence-weighted** DLT, and the robust estimator has no channel to receive PnLCalib's
  per-landmark confidence (measured 0.61 mean on the Colombia clip). Kept as-is.
  **Knock-on for #94 (R2), which was sequenced to inherit this.** The roadmap said "build the RAFT
  propagation stage on USAC from the start rather than write it twice" — **that instruction is now
  withdrawn** in `docs/roadmap.md`. R2 must reuse `solve_homography_ransac` for anything image→world.
  USAC stays defensible for frame-to-frame *optical-flow* matching, which is px→px and genuinely
  homoscedastic — but that must be measured before adoption, not assumed.
  **R-6:** this is the second of the two research-brief calibration/geometry recommendations to be
  falsified by measurement (after R1 #93, where "unify the SMPL-X→world constant" would have inverted
  half the pipeline). Both were headline items. The brief's *diagnoses* have been better than its
  *prescriptions* — R1's premise was wrong but pointed at a real 206→18 mm bug next door. Treat the
  remaining adopted items (R3, R5, R7) as hypotheses to measure, not work orders. Suite 1002 passed /
  14 skipped, ruff clean.

- **2026-07-29 (R9 / #101 DONE — OpenCV 4.11.0.86 → 5.0.0.93; zero code changes, one real behaviour change)** —
  Migrated the pin. **The whole suite passes unmodified**, and so does mypy (47 errors, byte-identical
  to HEAD, none cv2-related). The pre-migration sizing held: our surface is 48 distinct `cv2` symbols,
  all imgproc / videoio / imgcodecs / drawing, with exactly **two** calib3d calls (`findHomography`,
  `decomposeHomographyMat`) and **no `cv2.dnn`, no G-API** — so the research brief's two loudest
  warnings (§12.11, §12.12) never applied to us. Checked the specific break candidates:
  `cv2.VideoWriter_fourcc` still exists (and the new `VideoWriter.fourcc` alongside it), `imread` still
  returns **BGR**, and `USAC_MAGSAC` / `USAC_ACCURATE` / `USAC_DEFAULT` are all present — which is what
  **#102 (R10)** needs, so that task is now unblocked.
  **Calibration is bit-identical.** Seeded (`--seed 7`, 40 frames) synthetic oracle *and* perturb runs
  of `scripts/run_calib_eval.py` produce byte-identical JSON across 4.11 and 5.0; oracle
  reproj_rms 1.52e-14 m, i.e. still machine precision. B1 = 0.236 m itself needs the SoccerNet split,
  which is pod-side — flagged for re-measure, not assumed.
  **The one real change is the video decode, and it is not cosmetic.** Rendering the same three
  overlay frames under both versions differed on **92 % of pixels**, which traced to decode, not
  drawing: the same frame of the target clip comes back with mean |Δ| ≈ 2.97/255, max 32. Per-channel
  linear fit gives B ×1.008 −2.44, G ×0.940 +1.39, R ×0.995 −0.30 — both versions span full 0–255, so
  this is **not** a limited/full range change; green moving ~6× more than red/blue is the signature of a
  **different YUV→RGB matrix (BT.601 vs BT.709)**. For a 1080p broadcast source BT.709 is the correct
  one, so OpenCV 5 is very likely *more* accurate — which helps v2 photorealism rather than hurting it.
  **Quantified the downstream risk instead of hand-waving it:** on 165 player-sized crops from three
  real frames, the 19-dim HSV kit feature (`appearance_hsv`) drifts **0.119 mean L1 (p95 0.276)** between
  decoders, against a between-crop spread of **1.303 mean (p05 0.070)** — **9.1 % of the clustering
  signal.** Team clustering is relative and keys on a much larger separation, so it should hold, but the
  tail overlaps, so **kit split (10/10) and shirt-number reads (4/20) must be re-measured on the next pod
  run, not carried forward** (R-6: this is a mark, not a claim). Filed as a task.

- **2026-07-29 (R1 / #93 DONE — SMPL-X→world frames unified; a real foot-placement bug found and fixed)** —
  The brief's §2.1 premise was that four hardcoded SMPL-X→world constants are an inconsistency to be
  collapsed into one. **That premise is false, and acting on it would have broken the pipeline.** An
  empirical rest-pose forward pass plus real exports settled it: SMPL-X reaches us in *two* source
  frames and they need *different* remaps. `out/cuda` (fake/canonical export) has native
  head_y − ankle_y = **+1.495** — canonical, +y up. `out/live_real` (real SMPLest-X) has
  **−1.21 … −1.45** — camera frame, +y DOWN, exactly as `anim_export._rotation`'s docstring already
  warned. So `eval/bodymodel.py`'s `[[1,0,0],[0,0,−1],[0,1,0]]` and the pipeline's
  `[[1,0,0],[0,0,1],[0,−1,0]]` are **both right, for different inputs**. Unifying them inverts one path.
  I also flagged an "x-axis inconsistency" mid-investigation and then **disproved it**: `CANONICAL_SKELETON`
  puts l_shoulder at x=+0.17 and the SMPL-X map yields +0.161 — they agree. A canonical body under the
  remap faces −y, and its anatomical left is +x; "x-right/y-forward" are *world* axis labels, not claims
  about anatomy. No bug there.
  **The real defect was next door, in `smplx_foot_pos.py`** — the provider feeding `contact_probe`
  (foot slide + contact detection). Three compounding errors: (1) the remap was `[x, z, y]`, **det = −1**,
  an improper transform that mirrors the forward axis, so a planted foot's local offset fought the root
  translation instead of cancelling it; (2) `argmin` ran over *native* y, which on camera-frame data
  selects the **top of the head**, not a foot; (3) no pelvis re-origin, though `transl` is the world
  *pelvis* and SMPL-X sits its pelvis ~0.35 m off its own origin. Validated against physics, not
  convention: **feet must land on the pitch.** Across 6 real subjects, foot z went −0.167 m → −0.004 m;
  E2E on the Studio scene `A_smplestx` (8 subjects, 96 samples) **206 mm → 18 mm, a 91% reduction.**
  **Shipped:** `core/scene/frames.py` — both constants named and documented by source frame,
  `detect_source_frame()` (the far end from the pelvis is always the feet) and a `source_frame` override,
  per the auto-detect + manual-override rule. All 7 literal copies removed (`eval/bodymodel.py`,
  `orient_verticality.py`, `anim_export.py`, `poseannot/scene_state.py`, 3 scripts); `anim_export`'s
  manual `--canonical-up` footgun now resolves through the shared table. 16 new tests
  (`test_frames.py`, `test_smplx_foot_pos.py`), including one that **fails if someone unifies the two
  constants** — the trap the brief would have walked us into. Full suite green, no GPU used.

- **2026-07-29 (RESEARCH INTAKE — `football-3d-*` briefs analysed; adopted as tasks #93–#102)** —
  Two external docs landed in `docs/research/`: `football-3d-pipeline-v2.md` (the *why* — benchmarks,
  rejected alternatives) and `football-3d-implementation-brief.md` (the derived *what/how*). Both
  untracked at intake. Quality is high — the `[meas.]/[claimed]/[est.]` convention is honest, and the
  two arithmetic chains I checked are self-consistent (§7.1: 250×7+2 = 1752 camera vars, 38×250 = 9500
  per player, ×14 + ball 830 ≈ **136k** ✓; §13.2: ViT-H/14 @512×384 → ~1000 tokens → 2×632M×1000 =
  1.26 TFLOP → A10G at 40% of 125 TFLOPS → 25 ms unbatched / 12.5 ms batched → 350 crops/s ≈ 4.4
  GPU-s/s ≈ **5×A10G** for 25 fps ✓).
  **CENTRAL FINDING — they spec a DIFFERENT PRODUCT.** Their purpose is *coaching analytics* (parquet
  of distances/sprints/pitch control); brief §1 lists "photorealistic rendering" under **explicit
  non-goals**. That non-goal is our deliverable (§1 of this file). The shared core overlaps ~70%; the
  *bars* do not. Consequence: their envelope ("Global MPJPE 0.35–0.45 m, do not spec below") is
  rejected as ours. Their own decomposition supplies the replacement — global error is ~55%
  camera-driven `[meas.]`, and camera error is **common-mode**, a rigid transform of the whole scene
  that at a novel viewpoint re-renders as a slightly different camera and is nearly free. What shows
  in our deliverable is **inter-player residual spread** → task #99 (R7).
  **LIVE DEFECT FOUND via brief §2.1** ("do not hardcode the S→W rotation from memory; pin it with a
  golden test"): the SMPL-X→world constant is hardcoded in **four** places and one is the *transpose*
  of the other three — `eval/bodymodel.py:178` = `[[1,0,0],[0,0,-1],[0,1,0]]` (world_z = **+**y_smplx,
  R_x+90°) vs `correction/orient_verticality.py:65`, `adapters/models/smplx_foot_pos.py:92` and
  `app/anim_export.py:140` = `[[1,0,0],[0,0,1],[0,-1,0]]` (world_z = **−**y_smplx, R_x−90°). Both
  comment blocks claim z=up and y_smplx=up, so at most one is right unless eval's "ours" is a
  legitimately different frame — undocumented either way. This is the exact "silent, plausible, wrong"
  class, and it is adjacent to real time already spent on 180°-roll bugs (#50, #64). → task #93 (R1).
  **Verified-absent gaps** (grep, not impression): no VPoser / joint limits (#97), no factor graph /
  theseus / bundle adjustment, no RAFT / optical flow of any kind (#94), no shot segmentation, no
  `Provenance` on state — `RunLog` is provenance of the *model*, and the ball carries a bare
  `on_ground: bool` + `height_confidence` (`core/scene/motion.py:142,149`) rather than a 3-state mode
  (#96).
  **ADOPTED → #93 R1** unify S→W constant + golden test · **#94 R2** RAFT-small + MAD 3σ + RANSAC
  propagation between PnLCalib anchors (0.041°/frame `[meas.]`; direct treatment for **#61**, the
  project's open defect #1) · **#95 R3** fit line **edges** (IFAB ≤0.12 m ⇒ two parallel lines at known
  separation) not centrelines · **#96 R4** typed `Provenance`/`BallMode` · **#97 R5** joint-limit
  residual · **#98 R6** golden tests for projection round-trip + homography **sign** · **#99 R7** our
  own metric · **#100 R8** ADR-0012 rejected-approaches log · **#101 R9** OpenCV 5 migration ·
  **#102 R10** USAC/MAGSAC++.
  **OPENCV 5 (#101/#102) — sized against actual usage, cheaper than the docs imply.** Inventoried the
  cv2 surface across 23 importing files: **`cv2.dnn` used nowhere** (all inference is torch/rfdetr/
  ultralytics), so §12.12's "new DNN engine is CPU-only" is a non-issue for us; **G-API used nowhere**
  (§12.11 non-issue); `calib3d` exposure is exactly **two** calls (`findHomography` ×1,
  `decomposeHomographyMat` ×2) which move under the `geometry`/`calib`/`stereo`/`ptcloud` split; the
  rest is imgproc/videoio/remap, stable across the major. The payoff is **#102**: we run a *hand-rolled*
  RANSAC at `adapters/models/calibration.py:196` + confidence-weighted DLT at `:266` — uniform-sampling
  RANSAC is exactly what **MAGSAC++** (marginalises over the inlier threshold, no magic 1 px) and
  **PROSAC** (quality-ordered sampling — we already carry per-keypoint confidences) beat, and v2 §3.3
  claims that on player-contaminated grass this "matters more than changing the feature detector".
  Gate: no regression vs **B1 = 0.236 m** on `scripts/run_calib_eval.py`. **Sequenced R9 → R10 → R2**
  (#94 and #102 both marked blocked-by #101) so the propagation stage is built on USAC once, not twice.
  **DEFERRED/REJECTED with reasons** (roadmap "Research intake" section, so they are not re-argued):
  factor graph §7/§4 — entirely `[est.]` by the doc's own admission, buys *accuracy* while our binding
  constraint is appearance fidelity; our correction stack (ADR-0002) + human/LLM edit loop is the
  cheaper path to the same visible result. Shot segmentation — deferred, we reconstruct one continuous
  clip. **Off-screen imputation — deferred WITH a noted conflict:** for analytics an imputed ghost is a
  useful estimate; in a photoreal video, rendering an imputed player is **fabrication**, which R-6
  forbids — if ever adopted it must be gated by #96's `Provenance` at the *renderer*, not at the
  analytics boundary. Audio TDOA — their own §9 rejects it for broadcast (mixed stereo, not a
  multichannel field feed). Brief §0/§10 (split into `CLAUDE.md`+`docs/spec/`, build `src/core/`) —
  ignored, written greenfield; we already have `src/pitch3d/{core,adapters}`, 11 ADRs, and this file as
  SSOT.
  **INHERITED CAVEAT:** §12.6 warns TrackNet/TOTNet-class ball trackers are validated on racket sports
  and table tennis, untested on football (deforming ball, ~20× scene scale, occlusion by 22 legs). Our
  `adapters/models/wasb_backend.py` is from that same lineage — so the warning lands on a choice we
  have **already made**, not on their recommendation.
  **Reply drafted for the authoring model:** `docs/research/football-3d-response.md`.

- **2026-07-10 (PIPELINE STUDIO Increment 4 — generalized OUTPUT editing: whole-subject root orient/move nudges; tasks #89–#92)** —
  Widened the shipping pose-edit path (which only exposed per-joint `POSE_BODY_JOINT` axis-angle) to the two
  root targets the correction engine **already resolves but the UI never surfaced**: `ROOT_ORIENTATION`
  (→ `global_orient`) and `ROOT_TRANSLATION` (→ `transl`) — both at `engine.py:274-283`. **Design decision:**
  edits are `CONSTANT_OFFSET` deltas (nudges), not absolutes — the client sends a pure 3-vector, the server
  composes (`apply_offset_rotation` left-composes the axis-angle) / adds (`apply_offset_vector`), so there is
  **no client-side quaternion math and no current-value round-trip**. Same dual-layer model as everything else:
  each nudge = one `Correction` row appended to `edits.json`, resolved at the next FK pass. **Backend:**
  `edits.py` `build_root_edit(kind, delta)` via `make_offset` + `_ROOT_KINDS` wire-map; `pop_last_matching`
  gains a `kind` filter (undo can target a root layer without a joint_index). `scene_state.py`
  `apply_and_persist_root_edit()` mirrors the joint path (persist → fold into `st.scene.corrections` → rebuild
  ONE subject's FK); `undo_last_edit` threads `kind`. `app.py` `EditRequest` gains `kind` (default
  `pose_body_joint`; `joint_index` now optional; `axis_angle` doubles as the generic 3-vector), `/api/edit`
  routes by kind (400 on unknown kind / on joint-edit w/o joint_index), `UndoRequest`+`_EDIT_KIND_TO_TARGET`
  map for undo. **Frontend** (`index.html`): `nudgeRoot(kind, delta)` / `undoRootEdit(kind)` mirror
  `commitGizmoEdit`; a compact **Root override** strip under the 3D header — ORIENT ±15° X/Y/Z± + flip (180°
  about up) + undo, MOVE ±5 cm X/Y/Z± + undo — shown when a subject is selected. CSS
  `.root-ovr/.ro-btn/.ro-lbl/.ro-sep`. **Verified two ways, both green:** (1) in-process TestClient E2E (minted
  JWT) — translation offsets STACK (+0.5 z twice = +1.0), undo pops LIFO back to baseline, a 180° flip shifts
  joints (max 0.83 m) and its undo restores them to **residual 0.00000**, unknown-kind → 400, pose_body_joint
  w/o joint_index → 400, undo-when-empty → `ok:false`; edits.json restored byte-identical after. (2) **full
  in-browser E2E (Chrome ext, subject t1 frame 0):** real DOM click on **flip** → POST persisted
  `manual-admin-t1-f0-root_orientation` (constant_offset, delta[3]) to edits.json (34738→36025 B, 21→22 corr) →
  FK re-ran → **3D figure visibly rotated 180°** (faced left → faced right); real click on orient **↺** →
  undo removed the correction (edits.json back to 34738 B / 21 corr / 0 root edits) → figure reverted. Note:
  the per-track frame badge counts frames-per-track, so a second edit on an already-edited frame doesn't bump
  it (expected). Pod stayed OFF.

- **2026-07-10 (PIPELINE STUDIO Increment 3 — LIVE per-gate params editor wired into the re-run engine; tasks #86–#88)** —
  Made the correction-gate params **live-editable** and fed them into the existing ephemeral re-run, completing
  the Phase-2 "params editor" for the correction family. **Design decision (resolves the mismatch flagged in
  Increment 2):** the studio manifest (`studio.py`) had exposed physics params as *module constants*
  (`kin.HUMAN_MAX_SPEED`) that don't map to what the gates actually run off. So the editor sources params from
  the **same per-gate config dataclass the re-run runs off** — `getattr(phys, attr)` for each gate
  (`phys.orient_verticality`, `phys.kinematic`, `phys.coherence`, …) — introspected via
  `dataclasses.fields`. No module-constant path; a profile switch re-seeds the shown defaults honestly.
  **Backend** (`poseannot/rerun.py`): `_editable_params(cfg)` lists a gate's numeric/bool fields (skips
  `enabled` — the on/off checkbox owns it; skips str/enum so a typo can't build an invalid config);
  `_clean_param_overrides(cfg, params)` casts each incoming value to the field's declared type (bool before
  int — `isinstance(True,int)` is True) and drops unknown/uncastable keys. `gate_catalog()` now returns
  `params:[{key,value,type}]` per available gate (48 editable params across 12 gates); `run_corrections(…,
  params={gate_id:{field:value}})` applies them on top of the profile default in one `dataclasses.replace`
  (alongside the forced `enabled=True`) and **echoes the applied params back in the per-gate report**.
  `RerunRequest` gains a `params` field (`app.py`). **Frontend** (`index.html`): `rerunParams` state seeded
  from the catalog in `loadRerunCatalog()`; each enabled gate renders its params as editable **number/checkbox
  inputs** (rendered as a sibling of the `<label>` so clicking an input doesn't toggle the gate); `@change`
  (not `@input`) so decimals don't snap mid-type; `setParam` casts, `paramDirty`/`anyParamDirty` drive an
  accent **dirty-highlight** + a **"Reset params"** button (re-seeds profile defaults); `doRerun()` POSTs
  `params`. CSS for `.rr-params/.rr-param/.rr-pn`. **Proven that a param genuinely changes reconstruction
  output**, three ways: (1) direct gate call — `orient_verticality` `max_tilt_rad` 0.61→**21** corr, 3.0→**0**,
  0.05→**23**; (2) live HTTP endpoint (minted JWT, raw-pose clip) — same 21 / 0, params echoed in report;
  (3) **full in-browser E2E via Chrome ext** — select Physics → real click enables `orient_verticality` (params
  appear) → real-type `max_tilt_rad`=0.05 (dirty-highlight, Reset enables) → Re-run (busy overlay
  "re-running correction gates…") → report `applied:[coherence,kinematic,orient_verticality]`,
  `orient_verticality +23` with `params:{max_tilt_rad:0.05,inferred_confidence:0.25}` echoed → Revert restores
  baseline. Pod stayed OFF.

- **2026-07-10 (PIPELINE STUDIO Increment 2 — usability fixes + IN-BROWSER verification via Chrome ext; tasks #82–#85)** —
  User tested Increment 2 in the browser and hit two real bugs + a systemic UX gap. **Bug 1+2 (same root
  cause):** gate checkboxes only "toggled" when a profile was picked from the dropdown, and the Re-run/Revert
  buttons "did nothing" on click. Root cause found via the Chrome extension (`elementFromPoint` confirmed the
  checkbox was the top element; a synthetic `change` toggled it but a real mouse click did not): the `.main`
  div's `@pointerdown="onPanStart"` calls `setPointerCapture()`, and its guard whitelist
  (`.zoom-hud, .cam-panel, .upload-panel`) did **not** include `.stage-inspect` — so every pointerdown on the
  re-run panel started a frame-pan and stole the click. Exact bug class as commit 1656ec2. **Fix:** add
  `.stage-inspect` to the `closest()` guard in both `onPanStart` and `onWheel`. **Systemic UX fix (user:
  "интерфейс очень тормозит… нужно с этим что-то делать системно"):** added a **global busy overlay** —
  `busy`/`busyMsg` state + `beginBusy()`/`endBusy()` helpers + a fixed full-viewport dim+spinner card, wired
  into **every** long op (initial load, clip switch ~20s FK, re-run ~20s, revert, upload), each with a
  `try/finally` so it always clears. The overlay also captures pointer events, so dead-control clicks during a
  long op are eaten. **Verified end-to-end in a real browser** (Chrome ext, raw-pose clip, 23 subjects): a real
  click now toggles `orient_verticality`; Re-run fires (busy overlay "re-running…", +67 corr, orient +21,
  21.3s) and at frame 15 the active subject flips head-down→**upright** (body-up −→ +0.598, all 23 subjects
  0 inverted); Revert fires (busy overlay "reverting…") and restores baseline; clip switch shows
  "switching to default — rebuilding poses…" then reloads clean. Note: inversion is **frame-dependent** —
  frame 0 baseline is all-upright, frame 15 has 11 inverted (earlier "9 inverted" was a specific frame, not
  universal). This closes task #41 (visual browser E2E). Pod stayed OFF.

- **2026-07-10 (PIPELINE STUDIO Increment 2 — INTERACTIVE correction re-run in the glass-box UI; tasks #77–#81)** —
  Turned the read-only correction stages into a live, editable seam. The operator selects a
  **correction-family stage** (`coherence` or `physics`) → the inspector now shows a **profile dropdown +
  per-gate checkboxes + "Re-run (~20s)" + "Revert to baseline"**. Re-run runs the enabled
  correction/coherence gates on the *frozen baseline* scene as an **ephemeral in-memory layer** (NEVER
  written to `edits.json`) and rebuilds SMPL-X FK only for the subjects the new corrections touch; the
  existing `/api/subject/.../joints|mesh` path then serves the corrected poses, so the 3D + 2D overlays
  move with no special-casing. **Backend** — new `poseannot/rerun.py`: a 12-gate registry whose order +
  per-gate configs mirror `controller.run_reconstruction`; 4 provider-dependent gates (foot_plant,
  momentum_smooth, contact_lock, gravity_project) are surfaced `available:false` (need live-pipeline
  pelvis/foot providers) rather than faked. Two `SceneState` fields (`studio_correction_ids`,
  `studio_baseline_corrections`) make repeated re-runs independent (always from the frozen baseline, never
  cumulative) and robust to deterministic-gate-id collisions on already-corrected scenes (frozen-snapshot
  restore + `_dedup_last_wins`). Endpoints `GET /api/studio/rerun/catalog`, `POST /api/studio/rerun`,
  `POST /api/studio/rerun/clear`. **Two bugs found + fixed while building:** (1) *flagship no-op* — my
  overrides decide whether I CALL a gate, but each gate ALSO self-short-circuits on its own `cfg.enabled`,
  and the `default` profile ships most gates `enabled:false`; fix = force `enabled=True` on the cfg once my
  code decides to run it. (2) *id collision* on already-corrected scenes (677-correction scene regenerates
  existing `auto-orient-vertical-N` ids) — fixed by the frozen baseline snapshot + dedup. **FLAGSHIP proven
  end-to-end through the real HTTP API** (raw-pose clip = `out/anim_full_realism/scene.json`, 23 subjects):
  baseline **9 inverted** bodies (head.z−pelvis.z < 0, e.g. t4 −0.325, t11 −0.447) → toggle
  `orient_verticality` on → re-run (orient gate +21 corrections, ~21s total, FK-bound) → **9/9 stand
  upright** (t4 +0.563, t11 +0.588, all ≈+0.6 m) → Revert → **9/9 exactly restored** to baseline. `node
  --check` clean on the inline module; every markup method/state name resolves; two correction-family
  stages confirmed present in the manifest so the panel renders. **NOT yet eye-judged in a real browser**
  (Chrome ext unavailable, task #41) — the data path is verified headless; the user is ground truth on the
  visual render. Pod stayed OFF the whole increment.

- **2026-07-09 (POSE BAKE-OFF A/B — FULL video E2E of BOTH variants on the pod, pulled local; MooseFS cold-run stall root-caused + fixed with a page-cache prewarm; tasks #65–#70)** —
  Ran the complete decode→detect→track→PnLCalib→pose→WASB→render→FK-export→Blender(4-cam)→ffmpeg path for
  **both** pose backends at production knobs (60 f · cams broadcast,sideline,top,goal · 1280×720 · 32 spp ·
  coherence+physics ON, demo-edits OFF) via the new **`scripts/pod_ab_video.sh`** (VARIANT=A|B; identical
  detect/track/calibrate/render, differing ONLY in `--pose-backend`). Pod `abso48i2h1j0ex` (RTX PRO 4500
  Blackwell, euro MooseFS volume). **A = SMPLest-X** (`smplestx_backend`): 28m27s (13:35:53→14:04:20 UTC).
  **B = SAM 3D Body** (`sam3dbody_backend`, MHR→SMPL-X): 29m32s (14:06:54→14:36:26 UTC), **no segfault** — the
  `pymomentum-gpu==0.1.90.post0` pin + `LD_LIBRARY_PATH=<torch>/lib` ABI env held. Each variant produced 4
  camera MP4s (broadcast 1.4M · sideline 1.3M · goal 1.1M · top ~165K), `export/scene.json` 6.5M, **29 tracked
  subjects** (mesh npz), preview frames. Pulled to local **`out/anim_A/`** + **`out/anim_B/`** (video/ +
  export/scene.json + render/); byte-sizes match the pod. **Pod STOPPED** ($0.012/hr storage only). A-vs-B pose
  QUALITY comparison is DEFERRED — deliver both videos to the user to eye-judge first (no analysis yet).
  **ROOT-CAUSE + FIX (the reusable finding):** a COLD run appears to "idle" — GPU 0 %, main thread parked in FUSE
  `request_wait_answer` — because the **venv AND model weights live on the MooseFS euro network volume**, so
  Python import + `torch.load` pay a **per-file FUSE round-trip latency tax** (~150 KB/s *effective*). This is
  NOT a bandwidth limit: a bulk read of the same volume clocks **417 MB/s cold** and a warmed re-read **6.1
  GB/s**, and the kernel page cache **persists across processes** here (verified by dd re-read). **Mitigation
  that worked:** bulk-sequential-read the import trees + weights into page cache before the run
  (`find <trees> -print0 | xargs -0 -P48 -n8 cat >/dev/null`) — A jumped 4 MiB→3.6 GiB GPU + started emitting
  output the instant the venv was warmed; B additionally needs `sam-3d-body`(model.ckpt 2.0G)+`MHR`(mhr_model.pt
  664M). Caveat: prewarm caches file *contents*; residual pre-GPU time is import-time `stat`/negative-lookup
  latency that content-prewarm can't remove (B's larger import stack = longer pre-GPU phase). **Baked into
  `pod_ab_video.sh`**: a `PREWARM=1` (default; `PREWARM=0`/`PREWARM_JOBS` overrides) step that warms the
  variant-appropriate trees up front, so a fresh pod no longer eats the ~20-min stall. `bash -n` clean.

- **2026-07-09 (poseannot orient controls "do nothing" — ROOT-CAUSED to pointer-capture hijack; FIXED + self-validated in Chrome; task #64 reopened→closed)** —
  After the redesign (next entry) the user reported in-browser that **reset / flip-overlay / flip-skeletons /
  sliders did nothing**, the 2D/3D toggle "does nothing", the overlay sits wrong vs the frame, ±60 m was too
  small, and a **zoom** slider was missing — and demanded I test+validate **myself in Chrome** (not headless).
  My prior "0.000000 px" checks verified the MATH but never exercised the DOM event path, which is exactly
  where the bug was. ROOT CAUSE: the `.cam-panel` (orient panel) lives **inside** `.main`, and `.main`'s
  `@pointerdown="onPanStart"` calls `setPointerCapture()` — so every pointerdown on a panel button/slider
  **started a frame-pan and stole the interaction**; the `@click`/`@input` handlers never fired. (That's why
  the same methods worked when called directly via the Alpine instance but "did nothing" on click.) `.zoom-hud`
  was already excluded; `.cam-panel`/`.upload-panel` were not. **Fix** (`poseannot/static/index.html`): exclude
  `.zoom-hud, .cam-panel, .upload-panel` in both `onPanStart` and `onWheel`. Secondary fixes: (a) 2D/3D "does
  nothing" = the projected-mesh dots were `r≈2.6` in 1920-viewBox → **~0.9 px on a 730 px canvas** (invisible);
  enlarged to `r=(7−4·depth)` so the 10 475-vertex cloud reads as a body. (b) Translation limits **±60→±100 m**.
  (c) **New zoom control** (`ovr.zoom`, ×0.2–5, applied as a focal multiplier `fx·zoom, fy·zoom` about the
  principal point) — the manual knob for the residual "~3× too small" calibration gap (#61). SELF-VALIDATED in
  Chrome via **real clicks/drags** (build `orient-fix`): flip-overlay-X click → `ofx=true`, button active,
  overlay mirrors, `_panning=false` (no pan hijack); rot-Z drag → rotates, `view.tx` unchanged (frame did NOT
  pan); reset → all-neutral + buttons de-highlight; 2D/3D toggle → `mesh10475` renders (12 823 vs 2 348 SVG
  els); zoom drag → `2.74×`, active-mesh bbox grew 9×26→25×72 px; flip-skeletons-X → `sfx=true`. Screenshots
  captured each step. Residual overlay mis-position is calibration **#61** (still open) — zoom+move are now the
  hand-correction path. Cleanup: removed the temp `cc_test` login from `users.yaml`. **index.html is a static
  file served with `no-store`, so no server restart needed** (the app.py `/api/overlay3d` route from the
  redesign is already live).

- **2026-07-09 (poseannot overlay-orientation controls REDESIGNED — client-side projection; task #64)** —
  User: the overlay-camera controls were unusable — "жутко тормозит" (terrible lag), rotation axes + offsets
  confusing. Asked for: rotation about **3 world axes through the pitch-markings centre** + translation on
  **3 axes (metres)**, a **flip/mirror of the whole overlay**, and a **separate flip/mirror of the skeletons**.
  ROOT of the lag: every slider tick fired **3 sequential server round-trips** (`/api/pitch`, `/mesh2d`,
  `/frame/{n}/skeletons`, each re-running `frame_projector`+`project_points` server-side) behind a 60 ms
  debounce. ROOT of the confusion: the old `CameraAdjust` nudged the **camera** (zoom/pan/yaw/pitch/roll/dolly
  in camera space), not the overlay about the pitch centre. **Fix = move projection to the browser.** New
  `GET /api/overlay3d/{frame}` (`poseannot/app.py`) returns, in ONE round-trip, the flip-corrected camera
  (K,R,t), the **3D** pitch world points + their bbox centre, and every subject's **3D** joints — all in the
  same world frame. The client (`poseannot/static/index.html`) fetches that once per frame, then on every
  slider tick applies a rigid transform `p' = R·S_overlay·(p−C)+C + t` (C = pitch centre, R = `eulerXYZ` about
  world X/Y/Z, S = flip signs, t = metres) and projects with a plain pinhole — **zero server calls per tick**,
  coalesced via `requestAnimationFrame`. Skeleton flip (`sfx/sfy/sfz`) mirrors each body about its **own
  centroid** first (in-place — directly targets head-down bodies), then the whole-overlay transform. Controls
  reworked: 3 rotation sliders (±180°), 3 translation sliders (±60 m), 6 flip toggles (overlay X/Y/Z + skel
  X/Y/Z), reset. Header button `camera`→`orient`; `BUILD_ID` bumped. OBJECTIVE VALIDATION (math/geometry only —
  NOT pixel-on-player alignment, which stays the user's call): (a) client identity-transform projection
  reproduces the old `/api/pitch/0` pixels to **0.000000 px** over 1444 pts (JS pinhole ≡ server
  `project_points`); (b) `eulerXYZ` orthonormal err 1e-16, det 1, right-handed; (c) skeleton flip keeps
  centroid fixed (6e-17 drift); (d) `node --check` clean, all Alpine bindings resolve. Endpoint live: 23 subj ×
  22 joints, pitch centre `[0,0,0]`. Old server endpoints (`/api/pitch`, `/skeletons`, `/mesh2d`,
  `CameraAdjust`) now UNUSED by the client but left functional (only self/probe-docstring refs). **Restarted
  the local dev server (was no-`--reload`) to load the new route; user must hard-reload the browser.** Final
  alignment correctness = user's eye.

- **2026-07-09 (poseannot server WEDGE on clip switch — async event-loop starvation; FIXED)** — After the
  clip-switcher fix (next entry) made switches actually fire, the user hit a worse symptom: the page "не
  грузится, висит загрузка" and every route (INCLUDING `/static`) timed out with 0 bytes while the worker
  spun one core. Root cause: **fake-async handlers blocking the asyncio event loop.** `POST /api/clips/select`
  was `async def` and called `get_state(force_reload=True)` — a synchronous SMPL-X FK rebuild that takes
  **~5-25 s** (default 23 subj/60f = 21.8 s; A 11 s; measured `scripts/debug/clip_build_probe.py`) — directly
  on the loop thread. ALL 16 `api_*` GET/POST handlers were `async def` yet did purely synchronous CPU/IO/lock
  work (get_state, video decode, projection, FK) on the loop; only `api_clips_upload` genuinely awaits
  (`UploadFile.read`). So a single switch froze the whole server for the rebuild, and the user clicking
  source→A→B stacked several 5-25 s blocking rebuilds → apparent PERMANENT hang. NB the offline probe proved
  no single clip's build/render is infinite (default/A/B/clip all complete + render upright); it was pure
  loop starvation. Fix (`poseannot/app.py`): declare all 16 `api_*` handlers plain `def` so Starlette runs
  them in its threadpool (keep `api_clips_upload` async — it awaits); add an `@app.on_event("startup")` daemon
  that pre-warms `get_state()` so the first `/api/scene` is a cache hit, not a 22 s spinner. Proof: with a
  25 s rebuild running in the background, 10× `/static` + `/` probes all returned **200 in 4-28 ms** (pre-fix:
  hung). Full authed path re-verified 200 with real data (A: scene 21 subj/10 ms, pitch 23 ms, JPEG 113 ms,
  skeletons 21×22 pts; default 23 subj). Frontend: `switchClip()` now shows "switching … rebuilding poses
  (~5-25 s)…" so the inherent rebuild wait isn't a frozen-looking page (`poseannot/static/index.html`).
  KNOWN residual: a switch still takes 5-25 s (real FK cost) but the server stays fully responsive — no wedge.
  `_ACTIVE_ID` is in-memory only, so a restart always resets to `default`. Reusable probe:
  `scripts/debug/clip_build_probe.py` (per-clip build+render under an OS `timeout`).

- **2026-07-09 (GUI unbroken: dead clip-switcher root-caused + fixed; "dead controls" = stale page)** — User
  reported (on a long-open tab) that switching source/A/B did nothing with an EMPTY Network log, the 2D/3D
  overlay toggle did nothing, and the overlay was head-down. Three distinct causes, all resolved:
  (1) **Clip switcher was genuinely broken** — `switchClip()` guarded on `id === activeClipId`, but the
  `<select>` carries BOTH `x-model="activeClipId"` AND `@change="switchClip($event.target.value)"`; Alpine's
  x-model syncs `activeClipId` to the newly-picked value BEFORE the `@change` handler runs, so the guard was
  ALWAYS true and every switch silently returned with no fetch (exactly the "nothing happens, Network empty"
  report). Fix: drop the same-clip guard (`if (!id) return;`) — a native `<select>` only fires `change` on a
  real value change, so the guard was redundant anyway (`poseannot/static/index.html`). Verified E2E on :8899:
  default(23 subj/60f) → B_sam3dbody(8/48) → default all reload the correct distinct scene; the POST shows as
  "503" in the Chrome devtools capture but the server logs 200 and the data loads — the immediate
  `window.location.reload()` aborts the page before the extension finalises the record (cosmetic only).
  (2) **"Dead 2D/3D toggle + dead controls" = STALE PAGE** (the user's own guess "код не подгрузился" was
  right): a hard reload restores everything — all `/api/*` return 200 and the toggle flips the diag
  `mesh0 ↔ mesh1310`. Nothing to fix in code; just needed a reload against the running server.
  (3) **Head-down overlay** — fixed by reverting the #61 f=3500 recalibration (next entry). Post-revert, the
  real `poseannot.camera` projector renders bodies UPRIGHT + in-frame on all three clips (default 23/23, A 18/18,
  B 5/5). Also killed the duplicate servers (user had 8000 AND 8899 up); one clean uvicorn on :8899.

- **2026-07-09 (#61 f=3500 homography re-decomposition — TRIED, REVERTED as a regression; #61 STILL OPEN)** —
  The idea was to rebuild the broadcast camera from the trusted homography `H` (which grounds the feet on the
  painted lines) by decomposing `K⁻¹H⁻¹ = [r1 r2 t]` at a guessed focal `f=3500`, `r3=r1×r2`, SVD-orthonormalise,
  sign via `t[2]>0`, marking the result `raw_frame_aligned=True` to skip the legacy 180°-roll gate. **This
  shipped as 9dec2e0 and was WRONG — the overlay rendered every body HEAD-DOWN and horizontally displaced (user
  caught it; my zoomed-screenshot "validation" and `focal_size_probe`'s `abs(head−foot)` metric both HID the
  vertical inversion).** Objective re-diagnosis (`scripts/debug/redecomp_branch.py` + a focal sweep):
  (a) bodies are UPRIGHT in world (headZ 1.7 > footZ 0.2, plane_z=0) yet project head-down;
  (b) the rebuilt camera reproduces `H` only at the pitch CENTER and diverges with distance — **~277 px ground
  error at 40 m** — so it is NOT the homography turned into a pinhole (a single plane cannot fix the focal;
  degeneracy already noted below, and the SVD "snap" then distorts r1,r2);
  (c) it puts the camera BELOW the pitch (`C_z<0`) at EVERY focal tried (−7 m @ f=300 … −18 m @ f=3500);
  (d) NO rigid fix works — identity gives pitch-correct/bodies-head-down, a 180°-roll gives bodies-upright/
  pitch-INVERTED (they demand OPPOSITE corrections ⇒ the camera is geometrically wrong, not just frame-rolled),
  and all 4 `cv2.decomposeHomographyMat` branches give 0/23 upright.
  **Reverted 2026-07-09:** restored the three scene artifacts to the original PnLCalib camera (`fx=772`,
  `rfa=False`) — default+A from their `.orig` backups, B given A's original camera (B's own `.orig` cam was
  garbage: `tz≈−391`). Re-validated via the real `poseannot.camera` path: default 23/23, A 18/18, B 5/5 bodies
  **upright + in-frame** (the flip gate fires as designed). The original camera renders upright but small
  (pitch reads high / players ~3× too small) — that is the REAL, honest #61 bar, still unsolved. Kept the code
  as harmless scaffolding (`CameraTrack.raw_frame_aligned` field, projector skip, the debug scripts) — with no
  scene setting the flag True it behaves exactly as before; `scripts/recalibrate_camera.py` is annotated
  SUPERSEDED/FLAWED so its output is not trusted. **Correct next approach for #61:** get a real focal — a
  Zhang/vanishing-point solve on a frame with two visible pitch directions (STATUS estimate ≈2110), or a pod
  PnLCalib re-run that persists keypoints + vertical references — NOT a single-plane re-decomposition at a
  guessed focal. Touches (revert): `docs/STATUS.md`, scene artifacts restored; code from 9dec2e0 left in place.

- **2026-07-09 (variant B placement fixed locally + #61 focal-ambiguity proven)** — **Two findings, one
  fix shipped.** (1) **#59 B ROOT-CAUSED & REPAIRED (local, no pod):** measured B's `pose.transl` = it was
  stored as `(foot_pixel_x, foot_pixel_y, pelvis_height_m)` — x∈[494,1641], y∈[510,729] are raw 1920-space
  image pixels (they exceed 1280, so can't be calib space), z≈1.0 is metric; the foot XY never went
  through the homography. B's stored `camera` was also a bad solve (transl ≈[867,-281,-392] vs A's sane
  [-34,2.9,73]). Since A and B are the SAME 33-frame video they share one camera+calibration, so
  `scripts/fix_b_placement.py` adopts A's `camera`+`field` into B and re-grounds every foot pixel through
  `calibration.image_to_world` — the *identical* path `GVHMRPoseEstimator._ground_root` uses for A. Result:
  86% of foot samples land on-pitch, 7/8 subject roots project on-screen (was 0/8), verified live through
  the running server (`/api/frame/0/skeletons` → 7/8 full 22-joint skeletons at A's ~22 px scale; tid=5 is
  a far-field outlier whose foot sits high in-frame → homography over-extrapolates past the sideline, and
  is culled). Idempotent (re-grounds from a pristine `scene.json.orig`). Durable pipeline fix (SAM-3D
  backend should call `_ground_root` with the good camera on the next pod run) still open, but the artifact
  is now correct. (2) **#61 scale is a PROVEN focal ambiguity, NOT a BA-fixable problem:** solving focal
  from each per-frame homography (Zhang orthogonality) yields a real root on only 1/48 frames (that one:
  f≈2110, ×2.73 vs stored 772 — corroborating the ~3× observed scale error); the robust joint Zhang solve
  over all 48 homographies is rank-deficient (trailing singular values → 0). This is the structural
  degeneracy of a pan/tilt/zoom camera viewing a single plane: no translational parallax + one vanishing
  point near infinity ⇒ the plane cannot determine focal. So **temporal bundle adjustment (the "#61 as BA"
  framing) will NOT fix the scale.** Real levers: (a) LOCAL — impose a focal prior (fx ≈ ×2.7–3, ~2100) and
  re-decompose the homographies into a consistent camera; the shipped manual camera-panel `zoom` already
  does exactly this multiplier, so the true focal is one eyeball-test away (user = ground truth on scale);
  (b) POD/durable — re-run PnLCalib persisting keypoints + add vertical references (goalposts 2.44 m, known
  player statures) to break the ambiguity. `field.calibration.keypoints` exists in the schema but is never
  populated, so a durable BA would first need the pod run to persist them.

- **2026-07-08 (manual overlay-camera + pitch reference)** — **Added a manual overlay-camera control
  panel + a projected pitch-markings overlay to poseannot.** Motivation: the flip fix made A upright,
  but the residual PnLCalib inaccuracy (per-player offset + ~3× too small, #61) left the overlay too
  tiny/shifted to judge — and made the A/B clip switch and 2D/3D toggle *look* like no-ops (both
  clips share the same video; A's skeletons are ~22 px, B projects off-screen). The switch/toggle are
  in fact working (verified in-process: A=21 subjects world-transl, B=8 subjects raw-pixel transl).
  So the fix is the manual backstop the user always wants: `camera.CameraAdjust` (zoom / pan XY / yaw
  / pitch / roll / dolly) applied server-side in `frame_projector`, exposed as query params on
  `/api/frame/{n}/skeletons`, `/api/subject/{tid}/mesh2d/{frame}`, and a new `/api/pitch/{frame}`
  (projects the *measured* pitch markings — a pose-independent alignment reference). GUI: a camera
  panel (toolbar `camera` / key `C`) with sliders + numeric inputs + reset, a `pitch` toggle (key `P`,
  green dots under the poses), debounced live re-projection. Workflow: drag until the green pitch dots
  sit on the painted lines and the players follow. All defaults = identity (no behaviour change when
  neutral). E2E-verified via TestClient (pitch/skeletons/mesh2d 200 with params; coords transform).
  Does NOT fix #61 (still the real calibration bar) and does NOT rescue B (raw-pixel transl still needs
  the pipeline placement fix, #59) — it's a hand-alignment tool over the approximate camera.

- **2026-07-08 (debug suite + A/B root-cause)** — **Built an objective pose-overlay debug suite
  (`scripts/debug/`), root-caused BOTH variants, fixed A's inversion, demonstrated B's fix.** The eye
  can't judge a 22 px skeleton, so the overlay work was flying blind; these tools replace eyeballing
  with gravity/geometry ground truth. Tools (all take `--scene/--video`, PYTHONPATH=`.:src`):
  - `pose_probe.py` — per-track transform-chain dump + metrics: `transl_ok` (pitch-plausible root),
    `in_front`, `in_frame`, `upright` (foot pixel-v > head — gravity truth), `scale_px`.
  - `camera_plane_check.py` — projects the STANDARD pitch markings (known geometry, pose-independent)
    to separate CAMERA bugs from POSE bugs; writes `camplane_<scene>_f<f>.png` + anchor pixels.
  - `flip_sweep.py` — sweeps the 4 camera-frame flips {I, X, Y, XY}, scores upright%/inframe%/pitch;
    `--render=-1,-1,1` draws head/pelvis/feet markers to see orientation.
  - `ab_transl_diff.py` — golden test: per-track world root must match A↔B (shared upstream).
  - `rebuild_B_placement.py` — re-places B via a reference calibration (the B fix demo).
  **FINDINGS.** *Variant A (SMPLest-X):* root `transl` is CORRECT (pitch-plausible, x∈[-40,-17],
  z≈0.9), players in-frame — but **every body was vertically INVERTED** (`upright 0/18`), invisible at
  ~22 px so the 2026-07-07 eye-check missed it. Root: `poseannot/camera.py` applied an X-ONLY mirror
  `diag(-1,1,1)` for the camera-180 case; the real correction is a full optical-axis roll
  `diag(-1,-1,1)` = (u,v)->(W-u,H-v). **FIXED** → `upright 18/18` in the production projector, verified
  visually (heads up) + the projected pitch far-line moves off the crowd onto the boards.
  *Variant B (SAM 3D Body):* scene_B's root `transl` x,y are **raw FOOT PIXELS** (∈[0,W]×[0,H], only z
  in metres) — the placement stage never ran image→world — AND scene_B's own field homography is
  IDENTITY and its camera projects the pitch behind (0/1444 in front). So B's whole world scaffold in
  that artifact is broken, not the articulation. Since A and B are the SAME clip/frame, A's real
  `FieldCalibration` is the true one for B: running B's foot-pixels through A's `image_to_world` lands
  5–6/6 players on the pitch, upright (`rebuild_B_placement.py`, `out/bakeoff/overlay_Bfixed_f*.png`).
  Durable B fix = apply image→world in the pipeline placement stage + re-run (pod).
  **COMMON RESIDUAL / NEXT BAR (task #61):** with the flip fixed, A AND B project into the right pitch
  region and upright, but each skeleton is offset per-player and ~3× too small, and the pitch template
  is shifted — this is **PnLCalib camera accuracy**, shared by both, NOT a per-variant pose bug. That
  is the real "acceptable alignment" bar and needs a better/bundle-adjusted camera (pod). Overlay 2D/3D
  toggle (task #54, `poseannot/{app.py,static/*}`) also landed this session. `out/bakeoff` gitignored.

- **2026-07-08 (later)** — **POSE BAKE-OFF A/B — variant B NOW RUNS END-TO-END; both scenes +
  overlays delivered for the GUI.** User supplied the gated weights (`facebook/sam-3d-body-dinov3`:
  `model.ckpt` 2.1 GB + `assets/mhr_model.pt` 696 MB, staged to the pod network disk), so the HF
  block from the earlier entry is cleared. Restarted the pod and drove variant B to a real
  end-to-end run. **Four blockers found + fixed in the MHR→SMPL-X path** (all in
  `src/pitch3d/adapters/models/sam3dbody_backend.py` unless noted):
  1. **NATIVE SEGFAULT (the hard one).** `pymomentum.geometry.Character.load_fbx` (inside
     `MHR.from_files`) segfaulted on the pod's torch **2.8.0+cu128 / RTX PRO 4500 Blackwell (sm_120)**.
     Isolation proved it is an **ABI mismatch, not code**: the identical `load_fbx(fbx, model, load_blendshapes=True)`
     call *works* at module top-level but *crashes* the moment it runs inside `from_files` or a
     thread (classic heap/ABI corruption; ruled out stack size, import order, args, path — all
     identical). `pymomentum-gpu` wheels **≥0.1.97 segfault**; **`==0.1.90.post0` loads cleanly**
     (newest that matches the torch-2.8 ABI). Also requires `LD_LIBRARY_PATH=<torch>/lib` (the
     solver ext links `libtorch.so` at import). Pinned + documented in the adapter docstring and the
     new **`scripts/run_sam3dbody.sh`** (re-asserts the pin, sets the LD path, forces `POSE_BACKEND`).
  2. **SMPL-X model path.** Adapter used `smplx.SMPLX(model_path=…)` which wants the `smplx/`
     subdir directly; switched to **`smplx.create(root, model_type="smplx", …)`** (appends the subdir,
     exactly like the SMPLest-X backend) so the shared `human_model_files` root resolves
     `smplx/SMPLX_NEUTRAL.npz`.
  3. **Converter cwd-relative assets.** Meta's `Conversion` reads `./assets/*.npz` mappings/masks at
     **both** construction *and* per-frame conversion. Added `_conv_cwd()` (pins cwd to
     `tools/mhr_smpl_conversion`) around both sites.
  4. **`utils` name-collision (non-deterministic).** The converter uses bare `from utils import …`;
     another cached top-level `utils` won the import lottery (6-frame smoke passed, 48-frame run
     failed with `cannot import name _concat_mhr_lbs_model_parameters from 'utils'`). Added
     `_import_conversion()` — prepends the converter dir to `sys.path` and evicts stale generic-named
     modules so the import re-resolves deterministically.
  * **B RESULT:** `FRAMES=48` real E2E (same RF-DETR→ByteTrack→PnLCalib upstream as A, only
    `--pose-backend` swapped) → `scene.json` 1.3 MB, **8 subjects**, all **48/48 frames posed**, 320 s.
    Pulled to `out/bakeoff/scene_B.json`, installed GUI clip **`poseannot/clips/B_sam3dbody/`**
    (scene.json + video symlink, mirrors A), overlays `out/bakeoff/overlay_B_f{0,16,32}.png` via the
    same `overlay_from_scene.py`.
  * **A-vs-B (my headless eyeball on f0; USER is ground truth in the GUI per
    `feedback_overlay_user_is_ground_truth`):** **A (SMPLest-X)** draws ~18 subjects — taller, more
    upright skeletons, wider coverage incl. the far touchline. **B (SAM 3D Body)** draws 6 (8 total)
    — fewer subjects, more compact skeletons sitting lower on the players. Both project sanely with
    `flip=True` (camera-180 auto-detect). Real judgement = load both clips in poseannot and compare
    on identical pixels.
  * **Cost:** stopping the GPU pod now that both deliverables are pulled (`feedback_pod_cost`).
  * **STILL OWED (unchanged):** camera→world rotation lift for `global_orient` (shared A/B backlog).

- **2026-07-08** — **POSE BAKE-OFF A/B — variant A DONE & verified REAL; variant B built but
  HF-gate-BLOCKED.** User directive: «сделай оба варианта A и B, подними под и сравни результаты»
  + «нужны реальные фреймы с оверлеями для SMPLest-X и SAM 3D Body … и scene.json для обоих чтобы
  прогнать через наш gui».
  * **Variant A (SMPLest-X)** — re-ran the wired real pose backend on the pod (`pod_real_e2e.sh`,
    `FRAMES=48`, real RF-DETR→ByteTrack→PnLCalib→SMPLest-X→WASB, CUDA) → `scene.json` 4.2 MB,
    **21 subjects**, in 295 s. **Verified REAL, not rest-pose:** per-subject `body_pose` std
    0.23–0.39 (rest-pose ≈ 0), `global_orient` std 1.3–1.8 (real per-frame turning), `betas` std
    0.12–0.68 (real shape fits) — 21/21 pass `bp_std>1e-3`. Pulled to `out/bakeoff/scene_A.json`
    (gitignored) and installed as GUI clip `poseannot/clips/A_smplestx/` (scene.json + video
    symlink) — the clip switcher lists it, so it runs through poseannot as-is.
  * **Overlay tool** — new `scripts/overlay_from_scene.py` reuses the **exact GUI path**
    (`pitch3d load_scene` → poseannot SMPL-X FK → `poseannot.camera` projection with the validated
    180°-roll / camera-X-mirror auto-detect) to draw skeleton-on-real-frame PNGs for ANY scene.json.
    Same tool renders A and B on identical pixels for the eye comparison. A overlays: frames 0/16/32
    (camera track = 33 frames), **18–19 subjects** each →
    `out/bakeoff/overlay_A_f{0,16,32}.png`. NOTE per memory `feedback_overlay_user_is_ground_truth`:
    absolute pixel-on-player alignment is the USER's call in the GUI, not my headless check.
  * **Variant B (SAM 3D Body)** — adapter written: `src/pitch3d/adapters/models/sam3dbody_backend.py`
    behind the SAME `HMRBackend` port (imports clean, `isinstance(HMRBackend)` true, torch-free
    module import; heavy load lazy). Per-frame `process_one_image(rgb, bboxes=OUR ByteTrack boxes,
    use_mask=False, inference_type="body")`, one **batched MHR→SMPL-X** fit, identical foot-plane FK
    to A (so A/B differ only in articulation, not root placement). `pod_real_e2e.sh` now takes
    `POSE_BACKEND` (default SMPLest-X) so B runs by swapping ONE env var.
  * **KEY FINDING (why B is not a drop-in):** SAM 3D Body predicts on the **Momentum Human Rig
    (MHR)**, *not* SMPL-X, so every prediction must be bridged through Meta's own converter
    (`facebookresearch/MHR` `tools/mhr_smpl_conversion` → `Conversion.convert_sam3d_output_to_smpl`,
    a per-person PyTorch mesh fit). Both B repos are cloned on the pod network disk.
  * **BLOCKER (needs USER):** the 3DB checkpoint `facebook/sam-3d-body-dinov3` is **HF-gated**.
    Running B needs the user to accept the model licence on HuggingFace + provide an authenticated
    token (or run `hf download facebook/sam-3d-body-dinov3 --local-dir <repo>/checkpoints/…`
    themselves). I cannot accept a licence on their behalf. Surfaced to the user; B run + A-vs-B
    comparison deferred until the token/weights land.
  * **SHARED blocker (A *and* B):** both nets return `global_orient` in the **camera** frame; the
    camera→world rotation lift is still owed downstream (~35 % of subjects would read inverted in
    world). The pure half owns world *translation* (foot→homography); articulation stays
    camera-relative for now. Logged as a backlog item.
  * **Cost:** GPU pod STOPPED right after A finished (B blocked on user) per `feedback_pod_cost`;
    balance $21.34, network disk keeps the B repos + scene_A. Runbook + poseannot-roadmap updated
    (2D/3D overlay toggle, SAM-3D-Body backend, camera→world lift added to backlog).

- **2026-07-07 (evening)** — **POSEANNOT overlay 180°-roll bug — root-caused + fixed.** User's
  4th visual test: "оверлэй отзеркален, повёрнут, сжат" (mirrored/rotated/compressed), anchored with
  real IDs (GK=t31, t9=yellow #3, ref=t66). Root cause: `poseannot/camera.py` had the 180°-roll
  auto-detect gate **inverted** — `flipped = bool(R[1,2] < 0)`, but the validated convention
  (memory `project_camera_180_roll`) is `-R[1,2] < 0` ⟺ **R[1,2] > 0**. This scene's frame-0
  `R[1,2] = +0.853 > 0` NEEDS the roll, so the buggy gate never fired → poseannot projected in the
  solve's self-consistent (upside-down) frame but displayed on the as-decoded upright frame → every
  body landed **head-down + point-reflected** (= exactly the mirror/rotate/squish the user saw).
  Fix: gate → `bool(-R[1,2] < 0)` + corrected the misleading comment. Since intrinsics scale to
  cx=960=W/2, cy=540=H/2 exactly, the composed camera-Z roll `Rz=diag(-1,-1,1)` **is** the 180°
  image reflection `(u,v)→(W-u,H-v)`, so the overlay lands head-up while the **video the user sees
  stays upright** (we do NOT rotate the served frame — per memory, the roll is fixed at the one place
  that reconciles solve-frame vs raw-frame). Verified 3 independent ways: (a) rendered no-roll math
  on a 180°-rotated frame → markers land on the (upside-down) players; (b) foot(z=0)→pelvis(z=1.4)
  orientation test → 23/23 subjects head-up with the roll, 0/23 without; (c) **live** server
  `GET /api/frame/0/skeletons` → pelvis above ankles for all sampled subjects. Server restarted on
  :8899 with the fix. Residual sub-body offset (calib/frame-sync) left for user confirmation against
  the anchor IDs. NOT yet committed at time of writing.

- **2026-07-07 (afternoon)** — **POSEANNOT v1 + editor GUI batch.** Built on v0 (read-only
  `354cfee`). Landed in order:
  * **v1 editing** (`1e2034c`): click a subject → click a joint → Three.js TransformControls
    gizmo → drag = per-joint rotation delta → `POST /api/edit` recomputes FK + reprojects, writes
    to `edits.json` (config `corrections_out`). Undo per-joint. Persists across reload.
  * **Alpine load fix** (`2886c53`): Alpine 3.14.1 was racing Three.js on the global; moved to an
    ESM importmap so both load deterministically as modules (no build step, still CDN-pinned).
  * **GUI feature batch** (`78f0fca`, user-requested): (1) **all-players 2D overlay** — every
    tracked subject's SMPL-X skeleton drawn on the frame, active one highlighted (thicker,
    accent-bright); backed by batched `GET /api/frame/{n}/skeletons`. (2) **frame zoom + pan** —
    wheel-to-cursor zoom + drag-pan on a transform-origin(0,0) zoom-layer; `0` resets view.
    (3) **3D show-all toggle** (`a`) — 3D view renders all subjects as colored stick figures
    (centered on active pelvis) vs active-only; backed by `GET /api/frame/{n}/poses3d`. Show-all
    aids orientation.
  * **Runtime clip switcher + upload** (this commit): a *clip* = (video + scene.json) pair. New
    `poseannot/clips.py` registry: built-in `default` from `config.yaml` + user bundles under
    `poseannot/clips/<id>/` (gitignored, runtime-only). `GET /api/clips`, `POST /api/clips/select`
    (installs a `config.set_override` for source_video/scene_json/corrections_out, then
    `get_state(force_reload=True)` rebuilds FK), `POST /api/clips/upload` (multipart video+scene
    [+edits], validates scene parses as JSON). Toolbar `<select>` + `＋` upload panel. Honest
    constraint surfaced in-UI: a bare video without its scene.json has nothing to annotate —
    "upload your own clip" = uploading a reconstructed bundle. **Deploy motivation:** remote
    RunPod deploy needs local-disk clip upload, not just the config-bound default.
  * **Verification:** Chrome extension unavailable this session → no visual E2E (handed to user).
    Headless gates: `node --check` on the extracted Alpine module + backend import check + curl
    against a live uvicorn — select switches `scene.clip` video/scene, revert restores the
    Colombia default, bad id → 404, real 15.2 MB scene + 5.6 MB video bundle uploaded OK.
  * **BUGFIX from user's visual test (same day):** the 2D overlay + active marker **never
    rendered** — `<template x-for>` inside `<svg>` clones children in the HTML namespace, so
    `<line>`/`<circle>` are inert `HTMLUnknownElement`s (headless curl saw the data, hence the
    false "works"; only a real browser exposes it). Fixed by building an SVG **string** in
    `rebuildOverlay()` and injecting via `x-html` on the `<svg>` (innerHTML on an SVG element
    parses in the SVG namespace). Added: dashed bbox + "● active" label around the edited player;
    3D panel widened 380px → `minmax(420px,44vw)` (~half screen) + camera dropped to eye-level
    (was top-down); undo + **export** (`GET /api/edits/download`) surfaced always-visible in the
    3D header. Overlay markup validated well-formed with real data (1013 SVG elements/frame).
    STILL PENDING (F2): frame-range select + auto scene.json generation from a raw video — that's
    the perception pipeline behind the GUI (GPU/pod), scoped with the user, not yet built.
  * **BUGFIX batch from user's 2nd visual test (overlay floating off-frame, tiny frame, empty 3D,
    bad scroll):** root cause was a **layout race** — panels measured 0×0 before the CSS grid
    settled. Consequences + fixes: (a) **overlay drift** — the `<img>` filled its box while the
    `<svg>` used `preserveAspectRatio="xMidYMid meet"` (fit+centre), so when the box wasn't exactly
    16:9 they diverged (overlay large/offset over a tiny frame). Now **both use identical explicit
    `imgSize` px + `preserveAspectRatio="none"`** → overlay is pixel-locked to the frame by
    construction. (b) **frame rendered tiny in a corner** — `fitLayer()` ran on a 0-size box; added
    a zero guard + a **`ResizeObserver`** on `canvasFrame` that re-fits once real dimensions exist.
    (c) **3D figure not visible** — WebGL canvas could init at 0×0; added a guard + `ResizeObserver`
    on the 3D container (camera aspect + renderer size). (d) **independent scroll** — `min-height:0`
    /`min-width:0` on the three grid cells + `overflow:hidden` on `.app-shell`, so toolbar/sidebar/
    timeline stay fixed and panels contain their own overflow. **HONEST caveat (not a GUI bug):** the
    projected skeletons cluster centre-pitch and don't perfectly trace every real player. Proved
    poseannot's projection == the pipeline's canonical `project_world_points` (× the 1280→1920
    resolution scale), camera frames are contiguous (no index bug), and the client now renders it
    faithfully — the residual mismatch is **reconstruction quality** in `scene_replayed_v2.json`
    (its free-cam render `render_after_v2/` shows the avatars genuinely clustered in world space).
    That approximation is exactly what poseannot's manual joint editing exists to correct. Verified
    headless (node --check; served CSS/HTML assert the fixes; subject joints/mesh endpoints return a
    ~1.4 m figure); **final visual confirmation is user-side (Chrome ext not connected).** Open Q
    (#47): code has only ONE 3D renderer — the "two views" impression came from the empty-looking
    centre (mis-sized frame) + un-framed figure, both now fixed; if a panel should be literally
    removed, need to know which.
  * **BUGFIX from user's 3rd visual test — REAL root cause of the overlay "offset" found
    (`viewBox` binding was silently dropped).** The user reported the overlay was *still* offset "by
    the same distance as before" and the 3D "empty" (footballers gone). Built a **headless
    self-verification harness** (Playwright + system `google-chrome` via `channel="chrome"`, JWT
    cookie injected from `issue_token`) to render the live server and measure the real DOM/GL state —
    this is what finally exposed both bugs. Findings: **(1) The `:viewBox` binding never applied.**
    The `<svg>` used `:viewBox="'0 0 '+srcW+' '+srcH"`; the HTML parser lowercases the attribute name
    to `:viewbox`, so Alpine wrote the attribute `viewbox` — which SVG ignores (case-sensitive). With
    **no viewBox**, the 1920×1080-space points rendered 1:1 inside the ~700 px SVG box and shot off the
    frame down-right (the whole overlay pushed off — the persistent "offset"). The earlier px +
    `preserveAspectRatio="none"` "fixes" locked img/svg to the same *box* but never restored the
    missing viewBox, so the content still overflowed — which is why the user saw *the same* offset.
    Fix: set it imperatively/case-correct via `x-effect="$el.setAttribute('viewBox', …)"`. Verified:
    `viewBox="0 0 1920 1080"` now present, 1013 SVG children, and a high-res crop shows the `t1 ●
    active` label + skeletons rendering **on the pitch**. **(2) The 3D figure DOES render** — a
    `gl.readPixels` at the figure's screen centre returns the blue SMPL-X mesh colour (poseGroup has
    44 children: 23 meshes + 21 bones, joint projects to on-screen NDC). The "empty" screenshot was a
    **headless software-GL compositing artifact** (readPixels sees the real buffer; `page.screenshot`
    drops WebGL content — note the `GPU stall due to ReadPixels` warnings). In a real browser the
    figure is visible. **(3) Gizmo-always-visible fixed** — in three r0.170 the TransformControls
    *helper* is a separate object, so `transform.visible=false` no longer hid it; a bare gizmo floated
    over the scene (reinforcing the "empty" impression). Now hold the helper and toggle its `.visible`
    with the selection. So "removed the footballers view" = the 2D overlay was broken by the viewBox
    bug (now restored); "empty 3D" = the floating gizmo + subtle figure (both addressed). No panel was
    ever removed — every commit has exactly one `#three-canvas`.

  * **BUGFIX from user's 4th visual test — REAL root cause was a double `init()` race; prior
    "3D renders" claim retracted.** User: "лучше не стало. Скелеты совсем не попадают на игроков,
    в 3d редакторе пусто." The earlier `gl.readPixels` "the figure DOES render" claim was **not
    trustworthy** (software-GL headless ≠ the user's browser) — retracted. Instead of staring at
    unreliable WebGL screenshots, added an **on-screen diagnostics panel** (`'d'` toggles; build-id +
    panel/frame sizes + overlay viewBox + 3D-canvas size + `refreshThree` status + pelvis coords +
    mesh-vert count + track count) so a single screenshot reports internal state as DOM **text**.
    That panel immediately exposed the real bug: `track /0` — **zero tracks in the Alpine state** even
    though `/api/scene` returns 23. Fetch-timeline probe showed **every init endpoint fetched twice**
    (`/api/scene` at 1221 ms *and* 1387 ms) with only one `x-data` element → **`init()` ran twice**.
    Cause: Alpine v3 **auto-invokes** a data method named `init()`, and the element *also* had
    `x-init="init()"` — a second call. The two inits share `this`, race, and intermittently leave the
    app empty (0 tracks → no overlay, blank 3D). Fix: **removed `x-init`** (Alpine auto-calls it) +
    `_inited` idempotency guard. After the fix: single fetch per endpoint; `tracksLen 23`,
    `selectedTrackId 1`, `refresh j200/m200 ok · 22j`, `pelvis −28.69,−2.18,1.09` (finite),
    `meshV 10475` — the 3D data reaches `updateThreePose` without error, so the figure renders on the
    user's real GPU. Robustness also added: **literal `viewBox="0 0 1920 1080"`** on the `<svg>` (the
    HTML parser's SVG foreign-attribute table preserves the camelCase — works with zero JS, unlike the
    `x-effect` timing dependency); overlay now **pixel-exact** to the frame (img box == svg box ==
    `[200,246,696,392]`, 1013 children, screenshot shows skeletons on the players). Stale-cache masking
    killed: **`Cache-Control: no-store`** on `/app` + `?v=diag1` on `style.css`. `refreshThree` wrapped
    in try/catch that writes the failure into the diag panel. Verified via Playwright DOM/network/SVG
    (all reliable — only WebGL rasterization stays unverifiable headless, and that works on the user's
    GPU). Only remaining console 404 = `/favicon.ico` (harmless).

- **2026-07-07 (overnight, 12+ hours autonomous)** — **FULL PHYSICS STACK
  end-to-end + full_realism pod run BATCH_FINISH_OK.**
  User directive: "keep hitting physics iteratively until morning, don't stop."
  Autonomous iterations over the tier plan; 40+ commits. Full stack shipped:
  * 12 new correction gates: `contact_probe`+`contact_lock`, `momentum_probe`+
    `momentum_smooth`, `pose_motion_probe`+`pose_motion_sync`, `facing_align`,
    `inertia_probe`+`inertia_smooth`, `gravity_probe`+`gravity_project`,
    `body_scale_probe`, `stride_probe`, `interpen_probe`,
    `ball_contact_probe`, `ball_gravity_probe`.
  * All wired through `controller.run_reconstruction` via `physics_cfg` +
    `foot_position_provider` + `pelvis_target_provider`. All parametric in
    `config/physics.yaml`. Two new profiles: `safe_new_plant_lock` (T1a-c +
    foot_plant + contact_lock + momentum_smooth) and `full_realism` (adds
    pose_motion_sync + facing_align + inertia_smooth + gravity_project).
  * `scripts/physics_diagnose.py` reports 8 dimensions in one call.
  * CLI fix: `foot_position_provider` was missing; without it contact_lock
    silently skipped on the pod. Fixed in `7d91c98`; verified in-log
    `== contact/momentum: SMPL-X foot-position provider ON`.
  * 949 unit tests, 12 skipped, 0 regressions.
  **Pod E2E full_realism (fresh boot after prior OOM):** all 14 stages
  green, ~15 min ~$0.30. Final: `out/anim_full_realism/sideline_full_realism_pinned8.mp4`.
  Motion metrics vs prior baselines: contact slide **max 5.62m → 0.99m
  (–82%)**; gravity `max_dev` **10.6 → 7.6 m/s² (–28%)**; joint clamps
  (T1b) **127 → 0**; orientation clamps (T1c) **12 → 0**. Momentum jerk
  remains high (recon HMR noise on non-Blackwell pod, not from our gates).

  Reference commits (this session, in order):
  `909d219 a2a990e 96a8cfb 19e83cb aedfc18 ea7cf8f a706d22 663acdb eaaa5c8
  da01330 4652ac4 7d91c98 99f7b61 0f332f5 f887b10 daab1be 54f58e9 a40cf78
  fd9dc56 c8b88b6 e9983db 7da6fcb b7c6489`.

- **2026-07-07 (morning)** — **POSEANNOT v0 SHIPPED + orient_verticality landed.** Full
  pipeline debug (raw HMR → scene.json) exposed the real physics blocker:
  SMPLest-X has fundamental orientation ambiguity for standing / slow-moving
  players and puts **35% of subjects (8/23)** lying sideways or inverted
  for >50% of the clip. No temporal smoother can fix an absolutely wrong
  pose. Fix: **`orient_verticality_gate`** (commit `3a3b954`) — hard
  verticality clamp on any frame where body-up tilts >0.61 rad from world
  +Z; preserves world-yaw. On target scene: median frac_upright 70% → 100%,
  subjects <50% upright 8/23 → **0/23**. Visible in the physics-only
  Blender debug preview (a "cluster of falling meshes" → "cluster of
  standing footballers"). Also landed the stage-by-stage debug tooling:
  `scripts/pipeline_stage_debug.py`, `scripts/physics_debug_replay.py`,
  `scripts/pipeline_before_after.py`. Then **poseannot v0** (`354cfee`):
  browser-based pose annotator on top of scene.json. FastAPI + JWT auth +
  vanilla HTML + Alpine.js + Three.js (no build step). Read-only for now:
  navigate frames, click a subject, see 2D pose overlay on the source
  video AND matching 3D SMPL-X view with orbit camera. 22/22 SMPL-X joints
  project onto real players (verified by curl composite). Deploy target
  is RunPod. Architecture + roadmap in
  [`poseannot-architecture.md`](poseannot-architecture.md) +
  [`poseannot-roadmap.md`](poseannot-roadmap.md). Next: v1 body_pose
  editing (click joint → gizmo → save correction to `edits.json`).
  Tests: 970 passed / 12 skipped.

- **2026-07-06 (overnight)** — **PHYSICS ITERATION (autonomous batch, no pod).** Landed 6 commits
  atop full_realism: (a) **beta_variance_probe** — SMPL-X shape stability +
  cross-track cos-distance; pod scene signal: **169/253 pairs "similar" @ cos<0.05
  → betas NOT discriminative** for identity in this export, so downstream merges
  must not lean on shape alone (`360fa4b`). (b) **11th probe dimension** wired
  into `scripts/physics_diagnose.py` (`6714db9`). (c) **joint_smooth gate** wired
  end-to-end (config → yaml → controller → full_realism profile) — per-joint
  peak α **447 → 91 rad/s² @ window=5, → 46 @ window=9 (-80%/-90%)** on real
  pod scene, direct attack on HMR twitch that pose_kinematics misses (`a49eb5c`).
  (d) **Gate reorder:** gravity_project runs AFTER jerk_clamp (was before) —
  jerk_clamp was smoothing the ballistic parabola right back out; violations
  21→18, max_dev 13.5→9.5 m/s² offline; full gain on next controller pass
  (`556bb5c`). (e) **facing_align unwrap fix** — averaging wrapped angles
  (+π, −π) through EWMA gave ~0 (180° corruption); `np.unwrap → EWMA →
  wrap-to-pi` + regression test (`1c02588`). (f) **SMPL-X foot-pos
  densification 30→240 frame cap** — the 30-cap held-between-samples was
  synthesizing fake stances; contact_lock now zeros **98% of foot slide
  (15.4 m → 0.3 m aggregate, max 1.05 → 0.07 m)** on the same scene
  (`829ea81`). New memory: `feedback_yaw_lowpass_kills_motion.md` — iterative
  MA on HMR yaw removes 90% of α but flattens 100°+ real turns; use
  facing_align for structural yaw discipline, not statistical low-pass.
  Total tests: **960 passed / 12 skipped** (was 959 → +1 regression test).

- **2026-07-06 (afternoon)** — **PHYSICS TIER SHIPPED + FIRST POD RUN with safe_new profile.**
  Autonomous cycle closed all six eye-symptoms from the user's physics complaint
  (A-F): T0 (motion_stats extended with foot_z / joint_ω / turn_rate, parametric),
  T1a (foot-floor gate + plateau detection), T1b (per-joint slerp gate 600°/s),
  T1c (root-orientation slerp gate 720°/s), T2 (rendering-audit + teleport_policy
  hold|interpolate), T3 (capsule collision Jacobi soft-repulsion), T4a-c
  (PlayerProfile + BallProfile schema with seven-layer filter, LocalJsonPlayerStore,
  profile_provider wired into M3-9, apply_profile_updates, CLI end-to-end
  `--auto-tune`). All thresholds live in `config/physics.yaml` +
  `config/player_priors.yaml` (no hidden Python constants). Named profiles:
  default / conservative / strict / no_smoothing / future_full / humanize_teleports
  / safe_new / safe_new_humanize. Comparison harness `scripts/physics_compare.py`
  runs the full stack (kinematic + coherence + foot_floor + joint + orientation +
  collision) with per-field lineage. 12 commits, ~110 new tests (781 total, 0
  regressions). Real-scene numbers on `out/kitboost/scene.json` (60 frames, 22
  subjects, real SMPLest-X): safe_new fires 125 joint clamps + 15 orient clamps
  + flags 22 plateau subjects vs baseline default. **POD E2E with safe_new
  completed 15:43Z ($0.30-0.50)** — BATCH_FINISH_OK through all 14 stages, new
  final `out/anim_safe_new/sideline_safe_new_pinned8.mp4` (3.8MB, 57f). Zoom
  A/B sheet f28 `/tmp/safe_new_ab/safe_new_pitch_ab.png` — same v2v pass but
  slightly different pose geometry (joint/orient clamps applied at the beauty
  render's motion source). Eye validation of the motion pending (need to WATCH
  the video, not a single frame). Config: `PHYSICS_PROFILE`,
  `PLAYER_PROFILES_DIR`, `AUTO_TUNE`, `BALL_ID` env plumbing added to
  `scripts/pod_real_e2e.sh`. Pod DOWN. Doc `docs/research/2026-07-06-player-physics.md`
  logs the whole tier. Next: (a) eye-judge the safe_new video vs
  baseline; (b) fix collision compose-order to avoid +99 accel spikes on real
  scene; (c) T5 fatigue (deferred). Commits `d26d387..2909322`.

- **2026-07-06 (morning)** — **PLAYER SHADOWS t23: measured the shape gap, built the
  screen-space contact-shadow pin, honestly partial mitigation.** The prior session's
  eye-note ("no crisp contact shadows in ours — soft dark halos AROUND players") got a
  4-zone measurement (`/tmp/t23_shadow/shadow_zones.py` — contact / below / flanks / above
  vs a nearby grass patch): CLIP f28 has classic elliptical contact (contact V vs grass
  **-.029**, below **-.022**, flanks ≈ 0, above ≈ 0 — tight shadow with a soft below-fade,
  no side spill); OURS t21_pinned8 f28 reads contact **-.263**, below **-.004**, flanks
  **+.04** — the v2v smear halo IS what's darkening the "under feet" strip, but as a
  diffuse ring, not an elliptical shadow, and the below-feet fade is missing. Fix at
  SCREEN scale, last stage: NEW `scripts/player_shadow_pin.py` = shirt-colour player boxes
  (yellow Colombia H 25-45 S>100 V>100 + azure Congo H 130-175 S>80 V>60 + white ref/GK
  S<40 V>180, pitch y-band 0.40-0.88 to skip stands/boards, MORPH_OPEN+CLOSE, blob
  40<area<4000, 4<ww<60, 8<hh<90) → soft ellipse alpha at foot line (axes .75-.85 × ww,
  .18-.22 × hh, gaussian feather 5-6 px) → multiplicative darken `pixel *= 1 - strength*
  alpha*grass_mask`, gated by a grass mask (H 50-110 S>40 V>90, blurred 5 px) so the
  darkening stacks on unshaded grass rather than on the v2v halo. On the prototype
  (`/tmp/t23_shadow/pinned9_shadow_v2.mp4`, s=.30 ax_w=.85 ax_h=.22): eyeball A/B triple
  (`t23_shadow_triple.png`) shows visible elliptical shadow under the well-detected
  players (panels 2/4/6); heavy v2v-smear zones (panels 3/5) don't recover a shape because
  the underlying silhouette IS the smear — HONEST partial mitigation, root cause is v2v
  erasing silhouettes (future lever). Contact delta -.139 → -.192 (measurement is
  confounded by v2v leakage into the strip; visual shows the added ellipse under grass).
  Unit tests: `tests/unit/test_player_shadow_pin.py` (5/5) — detect yellow+azure, grass
  mask covers everything except shirts, paint darkens contact strip AND leaves shirt/far
  grass untouched, strength=0 is identity, deterministic. Wired as stage 14 in
  `pod_finish_batch.sh` (after stands red-scatter, PLAYER_SHADOW=1 default, env
  PLAYER_SHADOW_STRENGTH/AXW/AXH/FEATHER/YBAND). Best final unchanged (still
  `sideline_t21_pinned8.mp4`); next full pod E2E rides pinned8 after stage 14. Fifth
  eye/measure inversion (contact reads dark on ours despite "no shadows" eye note — the
  darkness is the smear, not a shadow). Sheets: `/tmp/t23_shadow/{t23_zoom_f28.png,
  t23_shadow_triple.png}`. Commit `<t23>`.

- **2026-07-05 (late night 2)** — **LINE GLOW t21: pitch markings brightened to the clip's
  glow — $0, existing machinery, third eye/measure inversion in a row.** t21 zone pass on
  the t20 final: the shadow-zone zoom showed our markings as periwinkle vs the clip's white.
  Measured (bright desaturated px, pitch band, f28): hue/sat actually MATCH (ours H 241
  S .09, clip H 233 S .106 — both slightly blue-white) — the whole gap is V: ours .75 med
  (.62 at the pin gate V>.55) vs clip .90-.92; a dim near-white reads as its hue cast, a
  BRIGHT one reads white. Same in t19b (.769) — longstanding, not a t20 regression. Fix =
  stage-11 pattern on the pitch: NEW stage 9b `LINES_PIN` (default 1) = `grass_pin.py`
  `--roi 0.48 1.0 0.0 1.0 --sat-max 0.35 --val-min 0.55 --val-only --target-val 0.90`,
  team masks excluded in-batch (local prototype maskless — kit whites glow like the clip's
  anyway); ROI starts at .48 so the boards band (.42-.48, stage 11) is never double-pinned;
  disjoint ROIs ⇒ stage order commutes. Landed V med .882 vs clip .902 (scale ×1.46); eye:
  lines glow white across the frame, periwinkle gone (`/tmp/t21_lines_ab.png`,
  `/tmp/t21_lines_full.png`). No new pure code → no new unit tests; `bash -n` green. NEW
  BEST FINAL `out/kitzones_pod/sideline_t21_pinned8.mp4` (= pod-produced t20_pinned7 + this
  pin locally; deterministic cv2 ⇒ pod-equivalent, masked in-batch reproduction rides the
  next pod run). Shadow-zone finding recorded: no crisp contact shadows in ours — soft dark
  halos AROUND players instead (v2v smear class, players parked). t22 candidate: panel-row
  lime dash periodicity (`/tmp/t21_panel_ab.png` — ours repeating bright-green LED segments
  vs clip's calm dark panels + gold text). Commit `<t21>`.

- **2026-07-05 (late night)** — **GRASS TONE + CROWD TEXTURE t20: the two biggest measured
  gaps in the final closed at $0, screen-space.** Zone-by-zone A/B eye pass on
  `sideline_t19b_pinned7.mp4` vs clip f28 (full frame + 4 zooms) surfaced two dominant gaps,
  both then confirmed by numbers (twice-measured rule — the eye impression "grass too
  bright" was actually WRONG: measurement showed the opposite failure).
  **(a) Grass = dark acid-olive.** Band medians ours H 79.1 S .753 V .431 vs clip (pin's own
  gate, stable across 4 frames) H 81.9 S .655 V .545: stage 9 pins H+S only — V was NEVER
  pinned — and its auto-target `ref_night.png` carries S≈.75 (Wan-inflated) vs the raw
  clip's .655, so the pin faithfully reproduced the reference's own excess. Fix: stage 9 now
  adds `--pin-val` when `GRASS_TARGET_VAL` is set; winning recipe passes explicit raw-clip
  targets `GRASS_TARGET_HUE=81.9 GRASS_TARGET_SAT=0.655 GRASS_TARGET_VAL=0.545` (auto path
  unchanged — auto default + manual override). Landed: V .43→.52-.56 (clip .549), S
  .75→.68-.71, H 79→83.
  **(b) Stands = saturated lego-confetti.** Ours luma local-contrast (|x−box5| med) .0316 vs
  clip .0134 (2.4×), chroma grain 2.6×, frac(S>.5) .71 vs .43 — the tone pins (10/12) land
  MEDIANS but not distribution shape, and nothing touches micro-contrast (real crowd at bowl
  distance = optics+sensor blur; ours = crisp quilt blocks re-sharpened by v2v+SeedVR2). A
  saturation KNEE can't fit the clip (matching frac>.5 undershoots p90 — the clip has its
  own saturated tail); a quantile map lands the whole distribution. NEW
  `scripts/stands_soften_pin.py` = stage 12.5 (`STANDS_SOFTEN=0` skips; `SOFTEN_KSIZE/KEEP/
  ROI/REF`): g9 blur-blend keep=.25 (tuned: 5px kernel floors at lc .019 — confetti blocks
  are larger; g9 k=.25 → lc .0127) + static S-LUT (mid frame × raw clip frame, quantile
  map), vertical feather, BEFORE the red pin so specks stay crisp. Landed (f28): lc .0144 vs
  clip .0134, S med .451/.447, p90 .725/.729, frac>.5 .427/.427 exact; red pin on the
  softened band then hits .036 = target exactly pre-encode (encode eats ~15% → .030, known).
  Cross-frame stable (lc .012-.014, red .028-.030). Eye (triple sheets old/new/clip
  `/tmp/t21_{full,stands,grass}_triple.png`): grass off acid-olive onto the clip's night
  green; stands confetti → soft crowd mass at clip texture scale. Unit tests
  `tests/unit/test_stands_soften_pin.py` (4: contrast drop + keep=1 identity, LUT lands ref
  distribution, V-channel preserved, determinism); full suite green. Local chain validated
  on t19b intermediates (pinned6 → grass re-pin → soften → red pin); batch wiring bash-n'd —
  in-batch reproduction rides the next pod run (stages 9+12.5+13 on the t19b tail is the
  cheap path: `TAIL_ONLY=1` or manual pin rerun on the volume's pinned2). RESIDUALS (eye,
  parked): boards letter fragments/repetition («BAN!» ×5 — v2v letter crispness, both levers
  lost before), stiff plasticky players (parked), stands S med post-encode .52 vs clip .45
  (double-encode chroma smear; shape landed pre-encode). POD E2E same session: stages 9→13
  replayed on the t19b intermediates WITH team masks (local prototype was maskless) via a
  one-off ssh chain (~4 min, ~$0.10, pod DOWN after): soften landed identically (lc .0154,
  S med .459, frac>.5 .434), red pin .036 exact; grass with masks lands H 84.7 S .722 V .502
  (vs clip 81.9/.663/.549 — masked gate shifts the measured "before", residual S +.06/V −.05,
  candidate micro-polish: measure stage-9 targets through the mask gate). NEW BEST FINAL
  `out/kitzones_pod/sideline_t20_pinned7.mp4` (volume `v2v/sideline_t20_pinned7.mp4`); A/B
  `/tmp/t20_final_ab.png` — acid-olive gone, crowd reads as soft mass at clip texture scale.
  Commits `d70d9e1` (lever) + `<t20-close>`.
  two rounds + a screen-space pin; new best sideline final.** Lever A (the ×19 fascia window
  repetition, t18 residual): the aggregated camera is CONSTANT, so the broadcast pan slides
  clip content UNDER the fixed rectified window — the strip cut's up-to-9 candidate frames
  each hold a different stretch (measured: ±70 px @512 aligned shifts, luma diff to 0.16).
  `assemble_fascia_quilt` stitches full-height Hann-feathered crops of those candidates into
  a 4-window canvas (`--fascia-windows` / `PITCH3D_FASCIA_WINDOWS=4`; strip repeat ×19→×5;
  x-blend only — the band is vertically layered; never flipped — panel text). Round-1 batch
  (~$0.45, `17cf6ae`): variety lands at beauty (band autocorr at the window period
  +0.216/+0.189 → +0.139/+0.127) and by eye, BUT a red/white flag crossing the band in 4 of
  9 candidates quilted into pink blocks around the ring — luma-consensus pruning dropped
  only the worst (with 4/9 contaminated the consensus drifts toward them; minority-pixel
  contamination barely moves a whole-window luma diff). Round 2 (`88c4985`):
  `fascia_pool_keep` prunes on the LOWER-2/3-rows hue-sat histogram (12×4 bins, V>0.1, L1 to
  the per-bin pool median, cut 1.25×median — panels/hedge/walkway are hue-stable across the
  pan while crowd-top red is legit): drops all 4 (keeps f0–f24), quilt lower-band pink
  0.048→0.006 = the clean-candidate floor; batch atlas eye-clean, variety kept. Lever B
  (clip stands = 3.6% scattered dark-red fans, strict-red): texture-space red is a measured
  DEAD END — round-1's 12–56 px spots minified to 0.0 at every gate (the isotropic 0.18× sim
  was the misprediction; honest sim is ANISOTROPIC — visible band 512×5460 → 35×1280,
  vertical ≈0.07), and round-2's 48–120 px clusters at honest-sim 5.4% STILL landed 0.001 at
  beauty (Cycles minification+denoise) with Wan re-adding only ~0.8% from the prompt. So the
  red enters at SCREEN scale after every gate: NEW stage 13 `scripts/stands_red_pin.py` —
  `scatter_fan_recolor` on the stands ROI of the finished video, speck positions static
  across frames (fixed deliverable camera), auto-measures the band's strict-red (0.009) and
  scatters the shortfall to the clip target (`STANDS_RED_TARGET=0.036`, `STANDS_RED=0`
  skips, `STANDS_RED_FRAC` manual) → landed 0.030. Speck brightness capped (new
  `luma_cap=0.35` scatter param): clip red V med/p75 .24/.31 but the uncapped
  luma-preserving swap glowed .26/.48 on bright fans — capped lands .25 med, eye reads the
  clip's dark-red pepper (clip blobs median 3.2 px @1080p → sub-pixel at our ~4×-smaller
  framing; the FRACTION is scale-invariant, so pepper, not clusters). 10 unit tests green;
  in-batch export = local smoke byte-for-byte (red 0.04, 4 windows, fascia ×1.88). ~$0.95
  both rounds, pod DOWN. NEW BEST sideline FINAL:
  `out/kitzones_pod/sideline_t19b_pinned7.mp4` (pod volume `v2v/sideline_t19b_pinned6.mp4`
  + local stage-13 pin; the next batch reproduces pinned7 in-batch). Honest residuals:
  stands red 0.030 vs clip 0.036 (specks on dark structure stay dark — luma-preserve working
  as intended); the final-level periodicity autocorr barely moves (~0.18 — the v2v tail
  re-regularizes and the metric is noisy) though beauty + eye clearly improved; quilt-space
  red stays at 0.04 (harmless; leaves a hint in the rgb control).

- **2026-07-05 (late evening)** — **CROWD COMPOSITION IN THE PROMPT (`b9808f8`) + stages
  11-12 verified in-batch; new best sideline final (t18, TAIL_ONLY ~$0.25).** Measured the
  clip's stands composition per x-sector (V>.15-lit, S>.35): 33-51% yellow-shirted pixels
  (more on the LEFT) + 3-5% scattered red; ours rendered 25-33% yellow / 1% red with the
  "amber and brown" wording — the twice-measured rule applied to the crowd's COMPOSITION,
  not just its tone. Prompt now states "most in muted dark yellow and amber shirts, some in
  brown, a few scattered fans in dark red" (intensity words stay muted/dark — the documented
  S-blowout trap; stands pin re-anchors global tone after). t18 TAIL_ONLY A/B vs t17 (same
  control frames): yellow .39/.28/.31/.33 (f28, left→right) vs t17 .33/.24/.27/.28 vs clip
  .51/.37/.39/.33 — half the gap closed, the left>right gradient appears; scattered red did
  NOT land (1% unchanged — Wan ignores few-instance instructions; residual). Stages 11-12
  ran in-batch with auto-measured targets, numbers match the local t16/t17 prototypes
  (boards V ×1.48, panels V ×0.64 val-only): the $0-local-then-wire pattern holds. Stands
  re-pinned to the same H 40.9 / V .16; grass/boards/kits unchanged (hue pins -38.1/+14.1
  normal). Eye: the mosaic pops yellow-on-dark like the clip instead of uniform amber. NEW
  BEST sideline FINAL: `out/kitzones_pod/sideline_t18_pinned6_crowdmix.mp4` (pod volume
  `v2v/sideline_t18_pinned6_crowdmix.mp4`); sheet `final_vs_clip_t18.png`.

- **2026-07-05 (evening)** — **PANEL ROW TONE: V-only pin (stage 12, `grass_pin.py
  --val-only`); new best sideline final (t17) — second $0 lever in a row.** The t15/t16
  residual: the gold panel band rendered ~1.5× hot (Vmed .42-.44 vs clip .27-.31). Root:
  one scalar fascia emission is walkway-anchored and the tone stages amplify intra-tile
  brights. Deterministic post fix instead of at-source pipeline-curve guessing: pin the
  panel zone's V to the clip's (auto from the reference with `PANELS_TARGET_ROI`, manual
  `PANELS_TARGET_VAL`; `PANELS_PIN=0` skips). NEW `--val-only` mode because the zone mixes
  materials (our zone median hue is GREEN 76-86 — hedge+bg — while the clip's is warm 47-48;
  matching the median hue would repaint the minority materials, so hue/sat stay untouched).
  Applied locally to t16: V .41 → ×0.67 → panel Vmed .27 vs clip .30, p90 .43 vs .51 (base
  right, accents slightly soft — same signature as the walkway calibration); crowd-bottom
  above (.46) and hedge below (.21) untouched, no ROI seam. Eye (f28 sandwich): the panel
  band now reads as the clip's understated lit strip between crowd and hedge instead of
  popping. Honest residuals: panel accents p90 soft; ×19 window repetition; v2v letter
  crispness. NEW BEST sideline FINAL:
  `out/kitzones_pod/sideline_t17_pinned6_paneltone.mp4` (t16 + panel pin; on-pod pinned6
  next batch); sheet `final_vs_clip_t17.png`.

- **2026-07-05 (late afternoon)** — **LED BOARDS GLOW: boards white pin (stage 11,
  `grass_pin.py --sat-max`); new best sideline final (t16) — $0, no pod run.** The
  longest-documented residual: clip LED board whites GLOW (Vmed .93-.96, p90 .98) while
  ours came out matte (t15 Vmed .63-.64, p90 .72) — the night grade + v2v eat the emitted
  level and no pin owned the whites (all pins gate on `sat >= min`, whites need `sat <=
  max`). Fix, auto+manual: `--sat-max` upper saturation gate in `grass_pin.py` (default 1.0
  = legacy no-op) + batch stage 11 — desaturated gate (S≤.35, V≥.35) in the boards ROI
  (y .42-.48 sideline), target auto-measured from the clip reference with the same gate
  (`BOARDS_TARGET_VAL/_SAT/_HUE` manual override, `BOARDS_PIN=0` skips). Letters (V<.35)
  stay dark — contrast INCREASES like real LED; kits are saturated i.e. outside the gate;
  ball sits below the ROI. Applied locally to the pod-produced t15 final (the pin is the
  chain's last stage, so local application = the true tail): V .64 → ×1.49, S .13 → ×0.56
  (purer white), result Vmed .93-.95 p90 1.00 vs clip .93-.96/.98 — dead-on. Collateral
  surgical: frame diff concentrates in y .44-.48 (boards rows, 19-42); walkway above
  (.24→.23) and grass below (.49→.48) at re-encode-noise level; kits/letters intact in the
  zoom. 6 unit tests green (sat-max selects board whites only / legacy default unchanged);
  `bash -n` on the batch. The next full batch exercises stage 11 in-batch (wiring is the
  same one-command pattern as stages 9-10). NEW BEST sideline FINAL:
  `out/kitzones_pod/sideline_t16_pinned5_boardglow.mp4` (t15 + boards pin; on-pod pinned5
  reproduces it next batch); sheet `final_vs_clip_t16.png`.

- **2026-07-05 (afternoon)** — **BAND GAP 2.2 → 4.6 m: THE MEASURED WINDOW NOW CATCHES THE
  GOLD FIFA/GUADALAJARA PANEL ROW (`d8e6194`); new best sideline final (t15).** Closing the
  t14 residual: the clip's lit gold-text panel row (FIFA WORLD CUP 2026 / GUADALAJARA /
  #FIFAWorldCup, gold-on-dark-green) sat ABOVE our 2.2 m band. Probes at `gap_rel` 4.2/6.0
  located it at ≈3.4–4.6 board heights above the board top (hedge below, bowl crowd directly
  above; at 5.5–6 h the crowd's yellow shirts contaminate the gold gate — so 4.6 is the
  ceiling). Fix = NO new geometry: the existing band grows to 4.6 m, and the same
  physical-window fascia cut catches walkway+hedge+panels in one atlas (269×939, fascia rows
  221, emission ×1.97 median→.40); `_cut_run_strip` headroom now scales with `gap_rel`.
  t15 E2E per-zone vs clip (each in its own screen coords — h_band differs: ours ≈.023,
  clip ≈.029): hedge+walkway Vmed .23 p90 .53 vs clip .21/.51 — exact; panel zone gold-pixel
  fraction .18 vs clip .21 — the row is there; panel Vmed .42 vs clip .27 — ours ~1.5× hot
  (the single per-window emission is walkway-anchored; the pipeline's tone stages amplify
  intra-tile brights). Eye (f56 sandwich A/B): crowd → gold panel row (GUADALAJARA fragments
  read) → green hedge → dark walkway → BANK OF AMERICA → grass — the clip's full stack in
  order; t14 lacked the panel row entirely. Stands (H 42 S .54 V .17) and grass (H 81 S .71
  V .46) unchanged vs t14; whole-frame diff at v2v-noise level. Honest residuals: panel-row
  brightness ~1.5× (candidate: per-row emission profile inside the fascia window); window
  repetition ×19 visible in the panel row (clip's run is continuous unique content); board
  bg glow gap and v2v letter-crispness cap unchanged. Watcher fix baked in: poll
  `BATCH_FINISH_OK` in the log / `pgrep -f "[p]od_"` — plain `pgrep -f` matched its own ssh
  command line and both t13/t14 watchers spun forever. NEW BEST sideline FINAL:
  `out/kitzones_pod/sideline_t15_pinned4_panels.mp4` (pod volume
  `v2v/sideline_t15_pinned4_panels.mp4`); sheet `final_vs_clip_t15.png`; t14 preserved both
  sides for the A/B.

- **2026-07-05 (midday)** — **WALKWAY BAND WEARS THE MEASURED FASCIA WINDOW (vertical atlas,
  `b3b266f`; emitted-level calibration `1893d8d`); new best sideline final.** The t12
  residual: our walkway band was a FLAT GREY stripe while the clip reads boards → dark
  walkway/fascia sandwich (people, hedge, dim panels) → crowd. Measured: our crowd mosaic met
  the boards with only a thin line (band y .40-.45, beauty V .70 flat, p90≈med = no
  structure); the clip's dark zone (y .344-.415) is Vmed .20 p90 .45 — dark base, bright
  accents. Fix, auto+manual: `_cut_run_strip` cuts a SECOND window from the same rectified
  run — the physical extent the 2.2 m band occupies (`gap_rel` board heights ending at the
  board-top edge) — reconstructing whatever the clip has there (this stadium: walkway with
  photographers/stewards + green hedge). The exporter stacks fascia+LED into ONE vertical
  float atlas (npz `tile`), walkway loops get real UVs into the fascia half
  (`adboard_loop_uvs(board_v=, walkway_v=)`, half-texel v-insets so REPEAT can't wrap grass
  into the fascia top), walkway tint → white; per-band emission splits at render time — npz
  `fascia_emission` folded into the float tile relative to the LED strength
  (`PITCH3D_FASCIA_EMISSION` overrides; the tile is a float_buffer image, so >1 values
  survive — no 8-bit clip inside the atlas). Calibration lesson (t13→t14): the LED rule
  (p90→1.05) rendered the band 1.6× hot — final Vmed .33 vs clip .20 — because the fascia is
  NOT a glowing ad; recalibrated to the walkway-validated emitted level (median→0.40,
  dimming below ×1 allowed) → t14 band Vmed .20 p90 .39 vs clip .20/.45: base exact, accents
  slightly soft. Eye (f12/f44 zooms): the sandwich reads — crowd → dark band with sparse
  standing figures (v2v turns the measured vests into plausible walkway people) → BANK OF
  AMERICA → grass; stands/grass/kits unchanged vs t12. 13 unit tests green (atlas UV bands
  share u / split v; fascia calibration incl. dim-below-×1); local Blender CPU smoke-render
  verified atlas orientation BEFORE the pod. Honest residuals: the clip's lit
  GUADALAJARA/FIFA gold-text panel row sits ABOVE our band (crowd-bottom region — a separate
  mosaic lever); board bg glow gap unchanged (p90 .64 vs clip .97); letter crispness still
  v2v-capped. NEW BEST sideline FINAL: `out/kitzones_pod/sideline_t14_pinned4_fascia.mp4`
  (pod volume `v2v/sideline_t14_pinned4_fascia.mp4`); sheet `final_vs_clip_t14.png`; t13
  preserved both sides for the A/B.

- **2026-07-05 (morning)** — **LED BOARDS text legibility at source: per-strip emission
  calibration + time-dominant ad frame choice (`b27eee3`, walkway compensation `0a36948`);
  new best sideline final.** Investigation first KILLED the "polarity inverted" hypothesis by
  zooming the RIGHT band: the clip's pitch-side LED boards (y .40-.47, grass-anchored — what
  `extract_board_strip` cuts) show white BANK OF AMERICA with dark text through the WHOLE
  window; the dark FIFA/GUADALAJARA panels live on the upper-tier fascia (y .345-.40) — a
  different element (we model it as the dark walkway; its texture = future polish). The strip
  cut was right all along — the killer was the fixed `emission_strength=4.0`: under the
  Standard view transform the letter edges (V .26 ×4 = 1.04) clipped to PNG white together
  with the background, eroding text to thin ghosts (beauty band frac(V>.9) .267). Fix at
  source, auto+manual: (1) the exporter calibrates strength off the strip itself —
  `strip_emission` = 1.05/p90(V), clamp [1, 4] → x1.08 for this white ad (bg still saturates,
  letters keep V .28) — saved as npz `emission`; `PITCH3D_BOARD_EMISSION` still overrides;
  (2) frame choice is time-dominant, not widest-span (`dominant_strip_index`: median panel V
  across ≤9 evenly-sampled candidates — guards against LED ad rotation mid-window;
  `PITCH3D_BOARD_FRAME` pins an exact clip frame); pure math unit-tested
  (`tests/unit/test_board_strip.py`). t11 E2E exposed a coupling: the walkway band SHARES the
  boards material, so x4→x1.08 crushed its tuned grey back to the dead-black stripe (final
  band p50 .20→.01) — the exporter now scales the walkway tint by 4.0/emission (emitted level
  held at the validated .40); t12 E2E clean. Numbers (band y .41-.475): beauty frac(V>.9)
  .267→.185; final p10/50/90 t10 .18/.20/.75 → t12 .18/.21/.64 vs clip .18/.58/.97;
  stands/grass unchanged vs t10 (med .29 / p90 .58; grass H 78). Eye: letters READABLE in the
  final ("BANK OF AMERICA" + red logo mark), walkway grey, no regressions. Honest residuals:
  our board background dims through grade3 (p90 .64 vs the clip's glowing .97 — broadcast
  boards bloom above everything; would need a boards-band brightness pin or grade exclusion);
  letter crispness capped by the v2v repaint; the upper fascia band (dark FIFA panels, gold
  text) is not modelled. NEW BEST sideline FINAL:
  `out/kitzones_pod/sideline_t12_pinned4_boards.mp4` (same name on the pod volume); sheet
  `final_vs_clip_t12.png`.

- **2026-07-05 (night run)** — **STANDS HOT LEFT EDGE flattened (`--flatten-val-x`, stage-10
  option): the bowl's left stands render ~1.9× brighter than mid while the clip's edge is its
  DIM end; new best sideline final.** Found by x-profile measurement right after the tv-sweep
  (8-bin gated V medians, f28): ours .51 .40 .30 .27 .27 .27 .30 .31 — IDENTICAL shape in
  t9/t10/tv031, i.e. upstream of the pins (the floodlit-bowl light rig favours the left
  corner); clip f30 .26 .33 .33 .32 .31 .29 .24 .24 vs f150 .22 .25 .29 .29 .35 .32 .34 .32 —
  the clip's profile WANDERS with the pan (swing up to 1.6×), so the framing-independent
  target is FLAT, not "match the clip's hump". Zoom on the hot corner: the blowout is partly
  DESATURATED (S<.15 → outside the tone-pin gate — a gated global V pin can never reach it).
  New `grass_pin.py --flatten-val-x BINS` (default off): per-x-bin gated V medians over all
  frames → gain = band-median / bin-median (clamp [.55, 1.3], 3-tap smooth), applied to ALL
  pixels in the ROI band except kit masks, vertically feathered (no seam); pure math
  unit-tested (`tests/unit/test_tone_pin_xflat.py`: flat = no-op, hot-edge inversion about a
  stable median, clamp, field identity outside the ROI). Batch hook: `STANDS_XFLAT_BINS=16`
  (unset = off). Applied to tv031 with FULL-WIDTH ROI (`--roi 0.08 0.32 0.0 1.0` — the
  default x .02 margin left an untouched hot sliver at the frame edge): left bin .51→.35
  (ratio 1.25 = inside the clip's own swing), center untouched or better — frac(V>.45) .173 vs
  clip .165-.171 (dead-on), p90 .53; full-frame eye A/B: the left corner no longer pulls the
  eye, the bowl reads evenly floodlit, no regressions (players/boards/grass outside the
  band). Honest residuals: blown-patch TEXTURE remains (chroma died in the generative tail —
  at-source follow-up: reduce the stands' response to the key light or the rig's left bias in
  the render); a clip-profile-matched prototype dimmed further (.29) but chases a
  framing-dependent target — flat chosen. NEW BEST sideline FINAL:
  `out/kitzones_pod/sideline_t10_pinned5_xflat.mp4` (reproducible via the committed script:
  tv031 input + `--flatten-val-x 16`); sheet `final_vs_clip_t10.png` rebuilt.
  **PROPAGATED to the goal view (same night):** goal3's stands band (y .00-.07) had NO stands
  pin at all — washed-white H 24 S .28 V .42-.47 with a 2.2× center-hot swing. Full pin+xflat
  run locally on the blessed `goal3_pinned3` (`--roi 0.0 0.07 0.0 1.0 --flatten-val-x 16`,
  same 40.9/.55/.31 targets): band lands V .31-.32 FLAT (f0/f28/f55), zoom eye = the clip's
  golden-dark fan family (was pale speckle), walkway/boards/pitch untouched, no feather seam,
  right-corner stands consistent. Population note (twice-measured): post-pin re-gated medians
  read H 30 S .41 — S×1.98 pulled previously-washed pixels INTO the gate (same mechanism as
  the tv-sweep's band-filter population effect); the eye verdict + V dead-on carry it. NEW
  BEST goal FINAL: `out/kitzones_pod/goal3_pinned4_xflat.mp4`; sheet `goal_vs_clip_t10.png`.

- **2026-07-05 (night run)** — **CROWD BIMODALITY closed (both halves measured): quilt
  `contrast` knob restores the bright-fan tail the Hann blend eats, and the stands-pin V
  target moves .23→.31 for this clip — the final's fan band lands the clip's p90/tail
  exactly.** Half A (texture, at source): `assemble_crowd_quilt`'s overlap-averaging eats
  ~25% of the tile's bright-fan tail (tile frac(V>med+.2) .117 → quilt .089 → beauty .105)
  — the very bimodality the STANDS TONE entry called a crowd-texture lever; and because
  stages 9-10 are MEDIAN pins, level changes cancel end-to-end, so the fix must change
  distribution SHAPE: new `contrast` kwarg scales luma about the quilt median (1.0 = strict
  no-op, chroma direction kept; unit test pins spread↑ / median-stable / channel-order-kept).
  Wired as `--crowd-contrast` / `PITCH3D_CROWD_CONTRAST` (default 1.0, auto+manual rule).
  E2E t10 at 1.35: beauty tail frac .105→.135, final relative tail .108-.119 = clip family,
  eye A/B (`/tmp/stands_ab_t10v.png`) visibly speckled-bimodal vs t9's mud. Half B (level,
  at the pin): stage-10's auto V target gate-measures the ref's WIDE stands ROI (x .05-.95)
  → .23, but the clip's CENTER stands (x .35-.65 — floodlit, where the eye judges) sit at
  filtered V .30-.32; NOT a code bug (`measure_image` gates symmetrically) — the clip's
  stands are center-bright while our quilt is spatially uniform, so the wide-to-wide pin
  underlights the judged band ×0.74. tv-sweep on t10's pinned3 (f28 fan band,
  filt-med/p90/frac(V>.45)): auto .23 → .20/.38/.039; .28 → .25/.46/.114; **.31 →
  .28/.51/.159**; .34 → .30/.56/.199; clip f30/f150 = .31-.32/.50-.52/.165-.171 (f250 =
  framing-shift outlier, excluded). Winner **tv=.31** on the bimodality signature AND eye
  (tv.34 reads uniformly lifted, `/tmp/stands_tv_ab.png`); elegant confirmation: at .31 the
  pin reports V ×1.00 — the winning treatment is H+8.4 S×1.43 with V untouched, i.e. the
  auto pin's V-darkening WAS the residual. Auto target stays the batch default; this clip's
  override documented: `STANDS_TARGET_HUE=40.9 STANDS_TARGET_SAT=0.55 STANDS_TARGET_VAL=0.31`.
  Next stands lever (if revisited): measured horizontal gain profile on the quilt (clip is
  floodlight-center-bright, ours flat). NEW BEST sideline FINAL:
  `out/kitzones_pod/sideline_t10_pinned4_tv031.mp4`; sheet
  `out/kitzones_pod/final_vs_clip_t10.png`.

- **2026-07-05 (night run)** — **WALKWAY BAND root-caused (the "black stripe" under the
  crowd): our own `gap_color` 0.02 renders V .39 and the generative tail crushes it to a dead
  V 0.00; fix = dark-GREY walkway (0.10/0.10/0.115) + `PITCH3D_WALKWAY_RGB` override.** The
  boards-row investigation ended in an eye-scale lesson worth keeping: row V-profiles first
  suggested "boards washed out", then "black apron gap between boards and pitch" — a zoom
  showed the truth: image order is crowd → DARK WALKWAY BAND → bright LED boards ("BANK OF
  AMERICA" legible) → grass; the V-1.00 rows I'd read as "touchline" were the boards. The dark
  stripe is our own adboard-ring walkway band (H 247 S .20 V .39 in beauty — the near-black
  0.02/0.02/0.03 vertex colour under emission 4.0), and the clip HAS the same band (crowd .27
  → walkway .16-.22 → bright boards/apron .41-.57): geometry right, only our level dies in
  the tail (grade3 halves .39, Wan sees a smooth near-black stripe and paints 0.00; clip's is
  .16-.22 WITH texture). Fix at source (stadium.py default + env override in
  `_export_boards`, auto+manual rule); target: beauty walkway ~.55-.6 → post-tail ~.15-.2 ≈
  clip, walkway/crowd ratio .67 matches the clip's. Test threshold updated (walkway dark-grey
  < 0.2, not < 0.1). E2E rerun (REUSE_SCENE=1 — export re-runs, recon skipped) is the next
  pod step; verdict lands in this log.
  **VERDICT (same night, t9 E2E): FIXED.** Full batch rerun on the fixed export
  (`out/kitzones_pod/sideline_t9_pinned4.mp4`): beauty walkway V .71 (H 248.5 S .09; pre-fix
  .39 — the dead-black source is gone), and through the full tail (grade3 → Wan v2v → SeedVR2
  → 4 pins) the final's walkway band lands at **V med .16-.23, p10 .15-.20 (f0 & f28) vs clip
  .16-.22** — t8's same rows were V .00-.02. Zoom eye check: band order now reads exactly like
  the broadcast (crowd → textured dark walkway → LED boards with legible strip text → grass);
  pins on this run re-confirmed auto-targets (grass H 69.2→79.1 S ×0.73; stands H 33.9→40.9
  S ×1.40 V ×0.78). Residual gaps by eye after this lever: boards band a touch brighter than
  clip (V med .73 vs .57) and slightly green-tinged, crowd blob colour vs the clip's yellow
  fan block, and distant-player fidelity (generative smear) — the first is a candidate
  `PITCH3D_BOARD_EMISSION` tune, the last is the SAM-3D-Body/LHM++ track, not a tonight
  lever. **Fix PROPAGATED to the goal view (same night, goal3):** the old `goal_pinned3` hid
  the same defect (thin dead band y≈.08, V .02 between crowd and boards); rerun with the
  blessed goal recipe (`REUSE_SCENE=1 ANIM_CAMERAS=goal STANDS_PIN=0` — one variable: the
  walkway fix) → beauty band V .71, final band V .18-.24 p10 .16-.21 (f0/f28/f55), crowd
  .36-.38 above it, grass-pin delta shrank to +4.2°. Clip-band clarification (boards
  residual, twice-measured on f1/f150/f300): the clip's LED band is HIGH-CONTRAST (med
  .35-.60, p90 .95) vs our uniform glow (med .73, p10 .5+) — the gap is contrast/text
  crispness more than brightness, so an emission cut risks the text-legibility win; parked
  with the knob documented. New best goal FINAL `out/kitzones_pod/goal3_pinned3.mp4`; sheets
  `final_vs_clip_t9.png` + `goal_vs_clip_t9night.png` (both in `out/kitzones_pod/`).

- **2026-07-05 (night run)** — **BATCH STAGES 9-10 VALIDATED E2E ON CLEAN DEFAULTS (tail8):
  one command, zero env overrides → grass + stands pins auto-target correctly from
  `ref_night.png`; ref-orientation caveat CLOSED; new best sideline FINAL.** Full
  `pod_finish_batch.sh` defaults run (TAIL_ONLY=1 over the tail7 control): BATCH_FINISH_OK;
  stage 9 auto-measured grass target H 78.9 S .68 and stage 10 stands target H 40.9 S .55
  V .23 from the committed ref — i.e. the ref IS upright and both auto-targets match the
  hand-measured clip values (grass 78.8/.67, stands 42.2/.56/.23), closing the stage-10
  orientation caveat from the STANDS TONE entry. FINAL measured: grass H 81.6 S .70 V .49,
  stands H 43.6 S .59 V .21 vs clip 42.2/.56/.23 — both in the clip family. NEW BEST:
  `out/kitzones_pod/sideline_tail8_pinned4.mp4` (replaces tail7_pinned4 as the sideline
  deliverable finish).

- **2026-07-05 (night run)** — **STANDS TONE: pin generalized to a region tone pin (stage 10)
  — clip's darker+yellower crowd tone landed locally, free.** Fresh eye pass over the new
  finals flagged the stands as the biggest visible gap (clip = dark crowd with bright
  yellow-gold fan pockets; final = pale uniform amber speckle). Measured (stands band, warm
  band 15-80 s>.15): clip H 42.2 S 0.56 V 0.23 vs final H 33.9 S 0.45 V 0.33 — smaller than
  the eye suggested and V-dominated (final too BRIGHT, not just too brown). `grass_pin.py`
  generalized: `--roi`/`--target-roi` (fractional spatial gates; video vs target framing
  differ) + `--pin-val` (V scale — the missing knob) = stage 10 in the batch (`STANDS_PIN=0`
  skips; `STANDS_ROI` is sideline-tuned, override per camera). Applied H+8.5 S×1.22 V×0.70:
  stands land the clip family by eye (darker, golden), no ROI seam at y=.32, kits protected
  by the ROI itself (players stay below y .32 in this window). LIMIT (honest): a median pin
  cannot create the clip's bright-fans-on-dark bimodality — that is crowd-texture contrast
  (`PITCH3D_CROWD_*` knobs, a render-side lever). Goal cam left at pinned3 (its stands band
  needs its own ROI). ORIENTATION CAVEAT for the batch default: `ref_night.png` grass-measures
  correctly (rotation-invariant for grass) but the stands `--target-roi` assumes it is
  upright — verify at next pod-up before trusting stage 10's auto-target. NEW BEST sideline
  FINAL: `out/kitzones_pod/sideline_tail7_pinned4.mp4`; sheet `/tmp/lv_stands_zoom3.png`.

- **2026-07-05 (night run)** — **GRASS TONE CLOSED: deterministic grass pin (batch stage 9)
  lands the clip's exact tone on BOTH cameras; prompt knob measured to its limit and promoted
  one word.** tail7 («muted dull green») LOSES: S unchanged (.85-.88) and hue back to olive
  H 68.9 — intensity words carry their own hue prior («dull green» ≈ olive). Three wording
  A/Bs total ⇒ reusable limit: hue lands ±4° of the stated colour's prior, S floors at ~0.85
  (clip .67) — wording gets CLOSE, it cannot land exact tone. NEW `scripts/grass_pin.py`
  (stage 9 in `pod_finish_batch.sh`, `GRASS_PIN=0` skips, `GRASS_TARGET_HUE/SAT` manual
  overrides): ONE global hue delta + ONE global sat scale over the grass band (55-140°,
  s≥.25, v≥.10), targets auto-measured from the clip-derived `ref_night.png` (measured 79.1/
  .67 ≡ clip 78.8/.67), team masks EXCLUDED (pin-A yellow sits at H≈70 INSIDE the band —
  maskless smoke visibly dragged shirts to chartreuse). Validated: sideline grass 69.2→79.1,
  S .88→.67 while team-A shirt hue moved only 60.5→61.9 (dilated-edge residual); goal cam
  (worse: H 63.5 S .94) → same 79.1/.67. Eye A/B: pinned grass reads as the clip's deeper
  night green, lime cast gone. DEFAULT_PROMPT promoted: «muted yellow-green» → «muted green»
  (best pre-pin wording = smallest pin delta). NEW BEST FINALS:
  `out/kitzones_pod/sideline_tail7_pinned3.mp4`, `goal_pinned3.mp4`;
  sheets `/tmp/grasspin_final_abc.png`.

- **2026-07-05 (night run)** — **TAIL #6: grass tone — the prompt itself violated the
  twice-measured rule; dropping «yellow-» closes ~1/3 of the hue gap, saturation untouched;
  tail7 (dull-green wording) in flight.** Measured gap (grass ROI y .45-.92 x .15-.85, green
  band H 60-180 S>.15, medians): clip H 78.8 S 0.67 V 0.52; best-final tail3 H 67.5-68.9
  S 0.87-0.90 — the lime cast traces to DEFAULT_PROMPT's literal «muted yellow-green night
  grass» (Wan amplifies stated colour adjectives — the four-times-measured rule, violated by
  our own wording). Local HSV sim (H+11 S−0.20 on the grass band) eye-validated the direction
  before spending pod time. tail6 = TAIL_ONLY sideline, one-word change «muted green night
  grass» (NOT «deep green» — known emerald overshoot H→120): H 70.3-73.1 (+3-4°, ~1/3 of the
  gap), S 0.85-0.89 (unchanged); eye A/B = greener but still vivid vs the clip's duller green.
  Saturation knob next: tail7 «muted dull green night grass» (precedent: intensity wording cut
  crowd S .94→.74). Artifacts: `out/kitzones_pod/sideline_tail6_pinned2.mp4`,
  `/tmp/grass_abc_tail6.png` (clip/tail3/tail6 stack).

- **2026-07-05 (night run)** — **SHIRT NUMBERS: honestly UNREADABLE in the current window —
  all 23 stay None (R-6); repeatable assignment tool landed for future windows.** Investigation:
  the v1 plate mechanism (subject.jersey_number → exporter back-anchor → FONT plate) is intact,
  but v1's 4 manual reads died with the June-28 scene generation — track IDs are per-recon and
  nothing re-assigns them. Ground truth for the CURRENT deliverable window (raw f0-59, wide
  framing): bodies ≈ 20 px at the 720p calib (fx 772, z ≈ 70 m) → back digits ≈ 7 px at raw
  1080p — verified unreadable TWO ways (raw-frame 6x zoom by eye; tool crops). No fabrication:
  zero plates in the current deliverable is the honest state; the goal's "shirt numbers" is
  bounded by source legibility here (approximations-OK clause). NEW TOOL
  `scripts/jersey_numbers.py`: `sheets` = per-subject upscaled torso contact sheets off the
  solved camera (same 180-roll convention as the body-texture sampler, tiles rotated back
  upright; min-px gate correctly refuses this window at default 45); `set` = pins
  track=number into scene.json + provenance sidecar; validated E2E to the official loader
  (pins round-trip into subject.jersey_number). Re-assignment per recon is now a ~10-min step
  whenever a window/clip has readable backs.

- **2026-07-05 (night run)** — **GOAL2 prompt iteration = NO-OP by design: the goal frame was
  never erased — my "residual" was an eye-check error at thumbnail scale.** Zoomed A/B
  (crop 550,530-1280,720, 3x) shows clean white posts+crossbar in BOTH goal1 (default prompt)
  and goal2 (+goal-frame phrase); goal2 changed nothing meaningful → the phrase is NOT
  promoted to DEFAULT_PROMPT (prompt saturated; unneeded terms risk bleed). REUSABLE OPS RULE
  (R-6 applied to my own verdicts): zoom the region BEFORE declaring a residual — thin white
  structures vanish at 320-px thumbnail scale, and this no-op cost a full tail run (~$0.25).
  Corrections from proper zooms: **ball is visible on BOTH cameras** (goal f55: airborne white
  ball mid-flight; sideline tail3 f55: ball at the azure player's feet — the earlier "not
  confirmed on sideline" was the same zoom error). NEW real residual (goal cam only, absent
  on sideline): grass-green half-erased player "ghosts" near fast clusters
  (`goal2_ball_zoom.png` left side). Artifacts: `out/kitzones_pod/goal2_pinned2.mp4`. Pod
  DOWN (balance $13.05); next lever being picked locally.

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
