# pitch3d — STATUS (current state only)

<!--
  This file is VOLATILE STATE: what is true right now and what to do next.
  Keep it under ~150 lines. It is read in full at the start of every session.

  It is NOT the place for evidence, measurements or reasoning:
    - how to work here, commands, rules  -> CLAUDE.md
    - per-item detail / root causes      -> docs/findings/
    - what happened and when             -> docs/archive/status-log-2026-07.md
    - where code lives                   -> docs/code-map.md

  Update + commit at every meaningful step. Chat history and the CC task list do
  not survive a session; this file does.
-->

**Last updated:** 2026-08-07 · **Repo:** /home/chubuchnyi/AVATAR · **Target clip:**
`samples/video/Colombia-1-0-Congo-DR1080p.mp4`

---

## 1. Goal

From one source broadcast clip → a **realistic novel-view video of the SAME episode** (different
camera angle), as faithful as possible. Players look like the originals (same kit + shirt numbers);
the stadium is realistic and the same as the source. **Judged by eye.**

Approximations are OK where one clip cannot recover the truth — backstopped by manual Blender
editing and generative prompt-editing (ADR-0008, LLM-over-MCP).

## 2. Staged bar (in order; each gated on eye-judgement)

- [x] **v0 — correct geometry** (2026-06-27). 20 players, root spread 34×40 m, pitch lines + goals,
  cameras that frame the action. Validated end-to-end on the pod.
- [x] **v1 — recognizability** (2026-06-28). Kit colours (10/10 split), shirt numbers (honest blanks
  where illegible), hybrid stadium backdrop.
- [ ] **v2 — photoreal** (started 2026-06-28). The agreed «A через B, 1→2→3, свет из клипа» plan is
  **complete**: lever 1 measured per-vertex body texture, lever 2 grass PBR via shared
  `scene_builders.py`, lever 3 light-from-clip (floodlit night, auto + manual override).
  Work since then is the generative finishing tail — see §3.

## 3. Current focus

**v2 finishing tail**, iterated on the pod as numbered batches (t1…t23). The chain is one command:
`bash scripts/pod_finish_batch.sh` — recon → quilt export → render → night-grade → Wan-VACE →
SeedVR2 → mask pass → screen-space pins (stages 9–14).

Best finals: `out/kitzones_pod/sideline_t21_pinned8.mp4` (sideline) ·
`goal3_pinned4_xflat.mp4` (goal).

**Next candidate (t24):** panel-row lime-dash periodicity (ours reads as repeating bright green LED
segments, the clip is calm dark-green panels with gold text), and player-silhouette recovery — the
root cause that the stage-14 shadow pin can only mitigate.

**OVERNIGHT 2026-08-07 → 08-08 (autonomous).** Queue + full report:
[`work-plan-2026-08.md`](work-plan-2026-08.md). Five items closed, three of them by *refuting*
what the plan assumed:

| | outcome |
|---|---|
| **W1** detector resolution | ⚠ **corrected 2026-08-08.** The first measurement counted players per frame (+2 %) and concluded the knob was useless. Measured on identity instead — 236 frames, two clips — **896 beats 560**: mid-pitch identity events 89 → 61 (broadcast) and 90 → 58 (fan), raw tracklets 70 → 56 and 76 → 54, for 1.5× the detector. Going higher is worse: at 1512 the detector finds more boxes than at 560 and makes more identity errors (97 vs 89). The best value depends on the clip, so it lives in `config/detector_resolution.yaml` keyed by clip name, not in the code. Override with `--detector-resolution` |
| **W2** #137 `team=None` | **fixed** — the identity gate un-teamed everything it cleaned. 23-of-27 unlabelled → 0 of 29 |
| **W3** stitch on П3 | **refuted where the plan put it** — pre-pose the rule joins two different shirts 3 times in 6. Do not relax the 2D stitcher |
| **W3+W4** the merge | **built, OFF by default** (`--handover`). On the judged scene: **24 → 21 subjects**, the three pairs the eye named, **183 mannequin frames gone**. ⚠ **Needs the eye:** `bash scripts/view_handover_ab.sh` |
| **W9** WorldPose GT | **89 clips, 2.4 M samples.** Speed ceiling 10.5 m/s never fires (real max **9.74**); accel ceiling **8 m/s² clips real football** (p99 = 8.35 on a 100 ms average); a real root ranges **0.23 m** per clip where our best whole scene is 0.234 |
| **W10** #103 BT.709 | **closed** — the clip declares `bt709`, so OpenCV 5 is right and OpenCV 4 was wrong. Changes **0 of 32** kit readings |
| **W5** selective mask propagation | **premise confirmed, deliberately not built.** Per *frame* the assignment margin is worthless (lift **0.86–1.10×** against a random trigger of the same size — 73 % of rows already sit near an event). Per **track** it is strong: a breaking track's median margin is **0.127 vs 0.739**, and firing on 12 % of rows catches **65 %** of the tracks that break (**5.3×** lift), 21 % catches 86 %. So ~4–5× cheaper than the always-on cue — but the cue's own ceiling is 14 %, so this buys **cost, not quality**. Build it when there is a stronger propagation to spend the saving on. **Extended 2026-08-10, two additions.** (1) The row margin is blind to births by construction (29 of 78 events had no row); the **column** margin sees them — all 40 births matched their own column, median **0.079–0.125 vs 0.705**, lift **2.4–3.8×**. So the trigger is not structurally blind to our dominant event type. Trap fixed en route: a symmetric ±2 window reaches past the birth, where the newborn track matches itself perfectly and every birth reads confident. (2) The rival rule *"fire when no track claims the detection"* measures **0.0 %** — a birth's best cost is an ordinary **0.176–0.221 against 0.250** for every column. **Our mid-pitch births are contested, not orphaned:** the right track was available and the assignment gave it to a competitor, which puts the defect in **allocation**, not in evidence. And W5's verdict assumed only the *frequency* of propagation moves — SMP also changes **when the mask is seeded**, which is the diagnosed cause of the 14 % ceiling ([findings §"go online"](findings/human-physics-requirement-2026-08-06.md)). Whether online seeding lifts that ceiling is the one unmeasured thing and needs the GPU pass. [reply](findings/reply-occlusion-stack-2026-08-10.md) |
| **#141** capabilities that never reach the run | **The structural defect behind #140, and it is not about cameras.** Four instances in two days of: a capability exists, is tested, is documented, and silently does not reach the run. `apply_rigid_camera.py` never called by `pod_real_e2e.sh` (**fixed** `400e400`); `broadcast_crop.py`'s per-segment contract collapsed to one rect on vert137 (**live**); its zoom warning rediscovered by measurement two days later; and the fan clip run with no crop at all on 2026-08-09, which died on a singular homography (**fixed** `dfc1075`). Two entry points apply different fixes and **nothing in scene.json records which**. `CameraTrack.source` is the first field that does, for one stage. **Plan re-ordered around this:** capability manifest → no silent `or` → one entry point → clip class as input → solvability gate before reconstruction. Per-frame focal DROPPED (0.6 % over 236 frames, 2.35 px paint residual). [reply](findings/reply-architecture-brief-2026-08-09.md) |
| **#140** camera never reaches the scene — **duplicate of #61, fixed by wiring** | 9 of 9 scenes carried `fx = 772 @ 1280×720`. **Not a new defect:** `scripts/apply_rigid_camera.py` has carried the full diagnosis since #119 (*"the closest realizable pinhole is still 525 px away... exactly what 'the ground marks are right but the players are not' looks like from the UI"*). What is missing is that `pod_real_e2e.sh` never applies it — `RIGID_CAMERA` is wired into `pod_make_video.sh`, `pod_physics_ab.sh`, `pod_129_ab.sh` and not into the script every scene here was built with. **Applied to the 60-frame scenes; overlay residual 240.0 px → 8.0 px median**, matched subjects 124 → 1088, common-mode 230 → 5.9 (`scripts/bench_overlay_residual.py`). Confirmed by eye. Remaining residual grows 6.2 → 15.7 px centre-to-edge, which is where distortion becomes testable. **Next: wire RIGID_CAMERA into pod_real_e2e.sh.** [reply §9](findings/reply-camera-model-gap-2026-08-08.md) |
| **W13** kit reader | **the yellow band contained the pitch.** H 18–48 vs grass at H 39–40 → **64.9 % of every frame read "yellow kit"**. Four claims retracted, incl. one that predates this session |

**2026-08-09 — the fan clip reconstructed itself, and the scenes turned out to be half invented.**

The pipeline now measures its own framing (`adapters/io/framing.py`, `--crop auto`) instead of
being handed a file someone cut with ffmpeg. On the raw portrait clip: crop `1080×608+0+1294`,
grass 29 % → 82 %, and PnLCalib solved **120/120 frames** where the raw frame solved **0 of 8**.
No crash — the singular-homography fix holds. 234 s on `demorig`.
The camera still refuses: the closest realizable pinhole is **12 382 px** away, so the scene
carries the synthetic 772 px stand-in and *says so in the log*. Positions on the pitch are real
(they come from the homography); a novel view of this clip does not exist. That is the predicted
handheld case, not a regression.

Root Z came out **measured on all 32 subjects** — SMPLest-X reports `pelvis_above_foot` itself, so
the #142 constant never fired here. The FK provider is the net for backends that do not.

**What the run exposed is bigger than the run.** 51 % of its subject-frames are `imputed`, so I
measured the scene the eye has actually been judging. `f236_res896`: **38 subjects, all 236 frames,
median 37 % measured, worst 2 %** — one subject has 5 real frames and 231 held. **47.9 % of its
subject-frames sit further than 12 frames from ANY measurement**, 11.9 % further than 120, worst
228. `extend_to_span` runs every subject to the full clip *and* raises the interior cap to the full
span, so neither edge nor middle was bounded. Decay bounds the coasted **distance** and
`coast_max_speed` the **velocity**; nothing bounded the **time**. R-6 says a lost subject is never
blinked out — it does not say the claim of presence never expires.
`CoherenceConfig.max_extend_frames` bounds it (`cec52ae`, 12 tests, mutation-checked; chain
`physics.yaml` → `PITCH3D_COH_MAX_EXTEND` → `--max-extend-frames`; `MAX_EXTEND` on the pod script).
**A/B measured on the fan clip** (same detections, same calibration, one knob apart; 195 s):

| | A unbounded | B cap 12 |
|---|---|---|
| subject-frames | 3840 | **2254** (−41 %) |
| of them measured | 1766 | **1766** — not one measurement lost |
| of them imputed | 1955 | **369** |
| subjects | 32 | **32** — nobody deleted |
| shortest subject | 120 frames | 27 frames |
| frames >12 from any measurement | 1588 | **2** |
| worst such distance | 105 | **13** |
| measured share of the scene | 46 % | **78 %** |

⚠ **Needs the eye:** `bash scripts/view_extend_ab.sh`. **Default stays unbounded** until then —
this changes what every scene contains, so it is not mine to switch on. Register:
[`findings/landmines.md`](findings/landmines.md).

**The one open question for the morning.** The merge `t10 → t77` is almost certainly **wrong**, and
the eye called it right on 2026-08-07. At f55–57 t77's box holds a **yellow** Colombia player while
t10 is **blue #8**; `team_id=B` on t77 is the only mislabel in 24 and is what let it past the team
gate. Independently, the rebuilt t10 lands **0.05 m from t5** — yellow #25, whom t77 probably
belongs to. The other two merges look clean. Judge in the A/B above.

**THE ACTIVE THREAD 2026-08-07 — player correctness, not rendering.** Standing user constraint:
**no full final render until poses and physics are settled**; judge in the `/app` pitch panel
(#134) instead. The user judged all 24 tracks of `out/cue/scene_off.json` by eye on 2026-08-07 and
asked for the criteria to be written down — done, and they now score **20/21 against that eye**:
[`findings/track-correctness-criteria-2026-08-07.md`](findings/track-correctness-criteria-2026-08-07.md),
probe `scripts/track_quality.py`, labels `findings/track-labels-2026-08-07.json`. Headline: an
`imputed` frame is a **frozen mannequin** (exactly 0.00 rad of limb motion while the root coasts
metres), so the phantom the eye sees is already flagged in the scene and needs no new model. Every
defect the user listed is **association or placement — none is a wrong pose on a correct crop**.
The user then corrected one verdict (t20) and that correction *changed the criterion*, so it is
worth reading §П2 before adding to this. See #135.

**Vertical fan clip — re-run twice on the pod 2026-08-07, and the catastrophe is gone.** The
2026-08-03 run was unusable (43% of 355 frames carried a stale homography, roots **3080 m** apart,
one subject at **100 416 m/s**; detail in [`findings §3.3`](findings/open-items-2026-08-01.md)).
After #136's four fixes, same input and same calibration (202/355 measured either way): root spread
**58.4 × 86.4 m**, max speed **405 m/s**, off-pitch placements **344 → 0**. Every bit of that is the
code refusing to place what it never measured. Scenes `out/vert136` (fixes 1–3) and `out/vert137`
(+ fix 4), ~50 min of GPU, ~$0.6. Honest verdict on the result: **5 tracks `OK`, 22
`OK_UNMEASURED`** — no phantoms, but most players are held through stretches with no ground plane,
because 153 of 355 frames never solve one. 405 m/s is still wrong; that is ordinary monocular depth
error now, not arithmetic. Full write-up: [`criteria §9`](findings/track-correctness-criteria-2026-08-07.md).

**Before the next pod run (audited 2026-08-03).** The chain itself is proven — `out/pod_0801b` is
23 subjects, 60/60 solved. #128 and #129 are both wired now, so the render carries the layout registration AND the one
fitted camera (`RIGID_CAMERA=1` default; #129 closed by eye 2026-08-06).
Then: all 5 pods are EXITED; start one of the
four mounting `/workspace`, **not** `jd9syxkau3rqzm` (mounts `/runpod` — only `pod_real_e2e.sh`
resolves the volume by content, `pod_finish_batch.sh:50` hardcodes `cd /workspace/fifa`); reconcile
the stale pod mirror against `87889c5` (**plus** the hand-patched `src/pitch3d/app/cli.py` from the
#130 hotfix — `git checkout --` it before pulling); and confirm `repos/PnLCalib` is really staged,
since `pod_real_e2e.sh:78` falls back to the proxy calibrator (#203 depth collapse) by printing a
line, not by failing.

Lever-by-lever history: [`archive/status-log-2026-07.md`](archive/status-log-2026-07.md).

## 4. Open board

One line per item. Reasoning, measurements and root causes live in
[`findings/open-items-2026-08-01.md`](findings/open-items-2026-08-01.md).

**The ordered work queue and its running report live in
[`work-plan-2026-08.md`](work-plan-2026-08.md)** — W1…W12, each with the measured fact that put it
there and the number that closes it. That file is the queue; this table is the board.

**Cross-thread verdicts — what we validated, what came out, what is owed — are indexed in
[`findings/research-ledger-2026-08-07.md`](findings/research-ledger-2026-08-07.md).** Read it before
re-opening any question below: it lists 14 ideas already measured and refuted (do not re-try them),
the gates that are built and never judged, and the claims in these docs that have no number behind
them. It is also where two conclusions this table used to state wrongly are corrected.

| ID | Item | Status |
|----|------|--------|
| #61 | Camera-calibration accuracy (offset + ~3× scale) | **CLOSED 2026-08-03 by the user's eye** |
| #119 | Re-solve the calibration as ONE camera, not 60 free homographies | **CLOSED 2026-08-03 by the user's eye** |
| #60 | Re-run overlays + verify acceptable alignment | **CLOSED 2026-08-03** — the eye-check passed and closed #61/#119 with it |
| #112 | Drag the pitch layout to correct the homography | **works** (user, 2026-08-03); its controls do not — split out as #127 |
| #127 | The layout gizmo is twitchy and shows nothing until you let go | fixed 2026-08-03 — live preview + shift-fine + typed panel; **awaiting the user's hands** |
| #128 | Hand-made calibration never reaches the render | **CLOSED 2026-08-03** — the export reads `FIELD_CALIBRATION` *and* the annotator's sidecar. Verified on the real scene: 11 drags merged, both camera halves agree 0.0000 px, pitch moves 63–207 px |
| #129 | `apply_rigid_camera.py` (the one camera, #119) is called by no pod script | **wired 2026-08-05, awaiting the eye.** `pod_make_video.sh` now applies the fit to the scene between reconstruction and `anim_export`, so mesh, render and export all see the one camera. `RIGID_CAMERA=0` reverts to the pre-#129 chain. The fit covers frames 0-59 and the chain's default is 60, so it fits exactly; a longer run must refit first (the script refuses rather than guesses). **A/B done 2026-08-05.** Both arms reconstructed identically (24 subjects, 60/60 measured), so only the camera differs: control carries an **invented** fallback (`controller.py:654`) at **772 px @ 1280×720, principal point dead centre**; #129 carries the **measured** fit at **4169.3 px @ 1920×1080**, `raw_frame_aligned` False→True. Normalised for resolution that is **3.6×**, i.e. #61's "3.9× too small" measured on both sides. **Limitation found by the same measurement:** roots are bit-identical (34.0×40.5 m) — the fit lands after pose grounding, so it fixes the camera and everything baked through it, not player placement. **CLOSED 2026-08-06 by the user's eye — "стало лучше"** on `out/cmp_129/side_by_side.mp4`. `RIGID_CAMERA=1` stays the default. |
| #130 | A subject shorter than the clip sank the whole run (`IndexError` after 22 min of GPU) | **CLOSED 2026-08-03** — the observation frame is now the middle of that subject's *own* track. Mutation-checked regression test |
| #131 | A run reports `confidence mean=0.28` and never says 43% of it is *carried*, not measured | **CLOSED 2026-08-03** — every run now prints `N/T measured, M carried`, mean over measured frames only. Report, not gate: the drift judgement stays the caller's |
| #132 | Player crossings break ByteTrack IDs and fuse per-crop poses (occlusion) | **CLOSED 2026-08-05, both halves.** *Pose half: fusion is real but rare — **this row's earlier claim ("not reproduced") is withdrawn**, and so is the single-pair measurement behind it.* The 40-pair stratified sweep supersedes it: fusion happens on **2/40** pairs, both at cover 0.41–0.49, ~**12 fused meshes per 236 frames**; below cover 0.31 every pair is clean. It was missed because the investigation ranked candidates by **box IoU**, and the fusing case has box IoU **0.126** — a small distant player passing *behind* a near one. **The decision still stands** (the typical contaminated crop is fine at 80% of the mesh on its own player, and identity churn is **5×** more frequent), so **PromptHMR is not adopted** and the SAM 3 branch stays out — but "the failure does not occur" was false, and the cancellation of Stage B rested on it. Re-justify or re-open: see [`research-ledger §5.1`](findings/research-ledger-2026-08-07.md). *Tracking half: fixed* — kit colour was sampled from a track's first 8 frames only, making a mid-track player swap undetectable by construction; `split_on_kit_change` takes tracks carrying two humans **9 → 0**, and turns the dormant stitcher from 1 merge into 14. Three follow-ups then measured empty (stitch gates ≤2 ids, NMS ≤1 id) and one of my own claims was corrected (the identity gap is a third, not double). *Shot-cut detection built and then fixed* — the first threshold was bin-dependent and shredded a 2-shot clip into 36; now scale-free, CLI guard ON by default with `--no-shot-guard`. *Render cost checked 2026-08-05:* run both ways through the real detector+tracker over 48 frames, the split creates 6 extra fragments and the stitcher merges exactly 6 back — **same 21 subjects, identical 848 subject-frames, no short-lived subjects**, so it relabels without churn. **Still unseen in a novel-view render (needs a pod run).** Detail: [`findings`](findings/occlusion-pose-research-2026-08-04.md) |
| #134 | **Pitch-3D scrubber in the UI** — judge poses without rendering | **built 2026-08-06** at `/world`, linked from the app toolbar. Whole pitch in 3D: measured markings + goals + turf from `core/scene/pitch.py` (same geometry the overlay and render use, so the views cannot drift), every subject's 22-joint skeleton at the current frame coloured by team, frame scrubber with play, ids and pelvis trails, free orbit/zoom/pan. Two new endpoints: `/api/world/geometry` (static) and `/api/world/{n}/skeletons` (batched 3D — the per-subject joints endpoint is for the single-player editor and would spend the scrub in HTTP). **Folded into the app 2026-08-07**, which reversed the "kept off `index.html`" call above: the viewer's core is now `poseannot/static/world_view.js`, mounted by BOTH `/world` and the app's right panel via a `3D pose | pitch` switch, so neither carries a second copy. In the app the middle panel is already the source frame, so pitch mode is frame-vs-3D side by side, synced; clicking a body selects that track everywhere. Column widths drag (two grid gutters writing `--rail-w`/`--right-w`, persisted); `overlays: on/off` (H) strips every drawn layer for a bare frame and says so on the frame. `/world` keeps its own copy of the frame panel for the A/B overlay case. Navigation: right/middle-drag or WASD pans (0.6 m/frame, shift 2.0), single click selects a player everywhere, double click re-centres the orbit on him, F fits. Verified in a real browser, no console errors. Verified: 23 markings / 6 uprights, frame 0 gives 23 subjects (A 10 / B 13) with joint z 0.06-2.13 m — feet on the turf, heads at 2 m — and the client JS passes `node --check`. **Standing request: no full final renders until poses and physics are settled** — use this. |
| #133 | **Human physics: filter motion, solid bodies, no popping, limbs in sync** | opened 2026-08-06 by the user, who noted it had been discussed before and **lost**. Most of it is already built and switched off; `joint`+`orientation` turned on today (violations 118+11 → 0, 97.8%/98.5% of real angular travel kept, **passed the eye** on the pod A/B). Still off: `collision`, `pose_motion_sync`, `contact_probe`, `foot_plant`, `gravity_project`, `identity`. *Identity half diagnosed and attacked:* 78 mid-pitch births/deaths, **96%** with an unclaimed detection a median 6–23 px away — so the boxes exist and the missing thing is an association **cue**. Three cheap fixes measured **null** (match threshold both ways, detector threshold). **McByte's mask cue now built and measured:** Cutie propagates one mask per track (236 frames in 686 s on GPU, 90× CPU), `MaskCue` applies the mm1/mm2 discount by wrapping supervision's `matching.iou_distance`. Same detections, same stitch: **mid-pitch events 28 → 24**, kit changes 10 → 10 (no cost), 2 fewer ids allocated, 48 vs 52 gap-frames bridged. **Works but weak — 14% against a 96% ceiling**, most likely because the two-pass design seeds masks from pass-1's broken tracks; McByte is online for exactly that reason. NOT the default — enable with `PITCH3D_MASK_CUE=<track_masks.npz>`. Scenes for the eye: `out/cue/scene_{off,on}.json` via `POSEANNOT_SCENE_JSON=… uvicorn poseannot.app:app --port 800X` → `/world`. Detail: [`findings`](findings/human-physics-requirement-2026-08-06.md) |
| #135 | **When is a reconstructed player correct?** — criteria defined, measured, scored | **defined 2026-08-07** on the user's own per-track verdict (24 tracks, `out/cue/scene_off.json`), then **revised the same day by his correction on t20**. Seven признаки: **П1 shape** (measured frames span the clip → `FULL` ⇒ correct, 12/12). **П2 an imputed run is a phantom ONLY when another track is measuring the same human** — first drafted as "in-frame = phantom", and t20 overturned it: t20 is a real player *occluded* by 15 and 17 (measured: the highest occlusion in the scene, 93–100 % covered at f0–9), so *why* we failed to measure is a diagnostic, not the verdict. Off-frame / occluded / simply missed all keep the player, marked (R-6); only the duplicate half of a merge is dropped. **П3 handover** (a HEAD dying where a TAIL is born, ≤14 frames, ≤6 m = one human) — and it must be an **assignment**, one partner each nearest-first, or t20 gets swept into t25's merge at 2.09 m. **П4 twin** (every interpenetration in the scene is a phantom parked inside a measured player — first measurement of #133's "solid bodies"). **П5 vertical** (largest root-Z excursion in the WHOLE scene is 0.082 m, so *«17 не прыгнул»* is a missing degree of freedom, not a t17 defect). **П6** an invented prefix *holds* the first measured position (t25 travels 5 cm in 24 frames) instead of extrapolating — the t25 placement bug, which stitching `15→25` **deletes** rather than repairs. **П7 kit read off the video pixels** (`--kit`), because the user rightly stopped trusting the reconstruction's colours. ⚠ **Re-measured 2026-08-08 (W13): the reader's yellow band contained the pitch** (H 18–48 vs grass at H 39–40, so 64.9 % of every frame read "yellow kit"). Corrected — grass rejected before the median, yellow kept below it: `team_id` agrees with the shirt on **23/24**, t3 does **not** flip (it is blue until its box walks onto a yellow player at ~f32), and the one real mislabel is **t77**, yellow while labelled B — confirmed in the crop, and exactly what let the wrong `10→77` merge past the team gate. Score **20/21**. Open: `15→71` (eye) vs `15→25` (geometry 0.85 m / 2-frame overlap, both blue), user re-checking frames 24–30; the kit rules out «20 стал 71» (t20 yellow, t71 blue). ~~Also open: `split_on_kit_change` did not cut t3/t12/t13/t17~~ — that was the same broken reader; those tracks do not measure two kits (W13). **Work list in §6 of the findings.** **Generalisation-tested on a second clip 2026-08-07** (`14604731_1080_1920_30fps.mp4`, fan phone video, portrait, different stadium and kits — §8 of the findings): the признаки transfer and immediately name that clip's dominant defect — **one camera whip at f38 breaks six identities at once** (t1-t5/t9/t16/t17 die at f31-34, t60/t63/t71/t73/t75/t76 are born at f36-38, and П3 pairs them all at 2.0-4.5 m / +4..+8 frames). Three holes in the criteria fell out of that run and are fixed: **`--coherence` is off by default and without it the criteria are blind** (no imputation is written, a lost subject is dropped, and a 3-frame fragment scored `OK · FULL`; the script now says it cannot see rather than printing OKs), the HEAD/TAIL shape taxonomy was fragile (П3 now pairs on measured-run endpoints, guarded by "simultaneously measured >4 frames = two humans"; the reference's three pairs are unchanged), and the off-frame verdict now warns unconditionally without `--camera`. Also measured: **the shot guard cannot tell a handheld whip-pan from a cut** — histogram distance 0.334 vs clip median 0.049 and threshold 0.250 (= the floor), costing 22 of 60 frames; `--no-shot-guard` is the escape hatch. **That run was NOT full-fidelity and the findings say so explicitly (§8, corrected):** only detect+track+stitch+coherence were real; `FakePoseEstimator` writes `body_pose = zeros` (**max |angle| 0.0000 rad** — a T-pose on every frame) and `FakeFieldCalibrator` is a plain affine scale of the image (30 m across the frame width, **no perspective**), so on that clip the frame numbers and id churn are real while the metre distances are 72–162 px of image adjacency, the vertical test is a constant, and П2's frozen-mannequin basis is vacuous. `track_quality.py` now prints that banner itself on any all-zero-`body_pose` scene. **Real pose was not runnable on this box**: SMPLest-X hardcodes `.cuda()` and the local torch is CPU-only. That is now a staging problem, not a hardware one — `4ceca0b` put the chain in Docker on a **local RTX 4080** (48 frames, `--device cuda`, 61 s); weights are still unstaged there, so calibrate/pose/ball remain fakes until they are. |
| #136 | **Social-media footage: four defects that made a phone clip unusable** | **all four fixed 2026-08-07**, each measured before and after on real files, and the lot validated by two pod runs (see §3 and [`criteria §9`](findings/track-correctness-criteria-2026-08-07.md)). **(1) Grounding never read calibration confidence.** `GVHMRPoseEstimator._ground_root` called `image_to_world` on every frame; `calibration.py:718` writes `h = last_good, conf = 0.0` on a frame it could not solve; nothing between them looked at `confidence`. That is the 3079.7 x 3079.7 m root spread and the 100 416 m/s subject from 2026-08-03 — and `require_solved_calibration` only refuses a run that solved ZERO frames, so 43 % carried sailed through. Now `FieldCalibration.solved_mask()` drops unsolved rows **before** grounding, coherence marks them `imputed` (R-6), and a subject with no solved frame at all is reported by id. Floor 0.02 = "was the plane solved at all"; `--min-calib-confidence 0` restores the old behaviour. Test is on the committed real calib: a foot through a homography 59 frames stale moves **9.39 / 7.92 / 10.80 m** — and that is the *gentle* broadcast case. **(2) The shot guard cannot tell a whip-pan from a cut.** Histogram distance 0.334 vs clip median 0.049, threshold 0.250 (the floor binds) — it truncated 60 frames to 38 on frames showing the same goal, players and stands. A pan or zoom is ONE homography and a cut is not: measured **0.995-1.000** for every camera move (including the whip) against **0.025** for the real cut at broadcast f236, a 40x gap. `find_shot_cuts(verify=...)` is a veto only, and a decode failure keeps the candidate. End to end: broadcast `[236] -> [236]`, fan clip `[38] -> []`. `tests/data/shots/` commits the three frame pairs (140 kB) so CI tests real pixels. **(3) One crop rect for a whole clip is a measurement of nothing.** The fan clip's grass band centre moves **177 px** and its height **354 px (59 %)** as the phone zooms; `broadcast_crop.py` now measures per window and emits one rect per framing — 4 segments at 82.4/91.3/91.7/90.3 % grass instead of one at 84.2 %, while a tripod still collapses to exactly 1 segment and the broadcast clip is unchanged. **(4) A solved plane is not a sane un-projection — gate the OUTPUT too.** Found by the pod run that was meant to validate fix 1, and it **refuted the theory fix 1 rested on**: after fixes 1–3 the scene was still 1050 m across, 248 of the 253 off-pitch frames were `interpolated` between just **six measured seeds**, and those six carried the *highest* calibration confidence in the run (0.546–0.575). Confidence was anti-predictive — **0 of 1299** frames below 0.5 landed off-pitch, **6 of 1339** above it did. It scores how well a homography fits the landmarks it can see, and says nothing about a foot pixel sitting near that homography's vanishing line. So the second gate tests the output physically: a footballer stands on a football pitch (±25 m, which keeps a keeper behind his line and a throw-in taker). The leverage is the striking part — the run refused **2** subject-frames and **268 off-pitch placements disappeared**, because those two were interpolation anchors and `_smooth_path` drags neighbours out with them. **Hard walls that no code fixes:** `fit_rigid_camera` assumes one focal + one centre (a tripod), and a handheld phone translates AND zooms — measured `realizable: False`, 142 px — so social clips give per-frame homographies but **no camera path to render a novel view from**; and once the zoom leaves no landmarks the plane is undetermined, which is now an honest refusal instead of kilometre-scale placement. Architecture that follows: **segment -> per-segment crop + calibrate -> keep only measured frames -> refuse the rest**, i.e. a set of usable windows, not one continuous reconstruction. |
| #137 | ~~Team assignment silently produced nothing on the fan clip~~ **CLOSED 2026-08-08 (W2, `6f4c270`)** | Not the clip and not the pixel scale. The **identity gate** (`--identity`, on in that pod run) blanks `team_id` in all three of its constructors under the comment *"let downstream re-assign"* — and there is no downstream, because `_assign_teams` runs inside the tracker, before the gate. `_restore_team_labels` now re-anchors each blanked tracklet against the ones that kept a label. Same clip, same 355 frames: **23 unlabelled of 27 → 0 of 29** (A 11 / B 18). 6 tests, mutation-checked. The controller now also prints any posed subject with no tracklet, because `team=None` + `role=PLAYER` were *defaults* indistinguishable from measurements — which is how this hid for a whole pod run. |
| #139 | **`team_id` on a short fragment is not trustworthy, and the handover merge depends on it** | opened 2026-08-08 out of W13. t77 has **3 measured frames**, is unmistakably a yellow Colombia player in the crop, and is labelled **team B** — the only mislabel in 24. `HandoverConfig.require_same_team` therefore passed a merge (`10→77`) that joins blue #8 to a yellow player; the geometric `suspect` check caught it anyway at 0.05 m from t5. Not fixed, deliberately: a "minimum frames before a team label is trusted" gate tuned on **n=1** is the kind of heuristic this repo has rejected before. Needs either a second instance or the user's verdict on the A/B first. |
| #103 | ~~OpenCV 5 changed YUV→RGB~~ **CLOSED 2026-08-08 (W10)** | The clip declares `color_space=bt709, color_range=tv`, so OpenCV 5 is **correct** and OpenCV 4 was not — R9's note had the polarity backwards, and it is the *pre*-migration numbers that were decoded wrong. Measured from the raw YUV planes: cv2 5.0.0 sits 0.65/255 from BT.709 and 3.43 from BT.601; the matrix moves 95.6 % of pixels by a mean 2.96/255, which on a shirt is 1.6° of hue; **18 of 1055 boxes change class and 0 of 32 tracks change their modal kit**. OCR half is moot — #109 measured 0 of 23 usable crops. [`work-plan`](work-plan-2026-08.md) |
| #138 | **The 2026 occlusion stack, reviewed against our own pixels** | **reviewed 2026-08-07**, 19 methods verified against primary sources — [`findings`](findings/occlusion-stack-review-2026-08-07.md). The survey's diagnosis is independently ours (detector 96 % ceiling · 23.5 % of crops carry two bodies · `max_gap` 12 frames). Its prescription mostly is not: **our median player box is 28 × 72 px on the phone clip and 41 × 86 on broadcast — ~573 px of shirt, the same 573 px for eleven teammates.** BMPv2's authors state mask refinement *"is not suitable"* below 100 px and that looping it *"pronounces the error"* — which is why BMPv2+ (the 55.8 OCHuman headline) is **worse** than BMPv2 on COCO, 78.1 vs 78.8. Sapiens filtered its training data to boxes >300 px (v1) / ≥384 px short side (v2), i.e. **11× above our subjects**. OCHuman selects by occlusion, never by size, and **no method publishes AP_S**. On ReID: KPR's prompt is worth **+7.0 R-1** where multi-person ambiguity exists and **+0.2 where it does not** (Market-1501 93.0 → 93.2) — so it disambiguates, it does not identify, and at 28 px with identical kits nothing identifies. **Measured today:** Deep-EIoU's geometric core (expansion IoU) on our own 4362 cached detections — raw ids 56 → 43, stitcher merges 14 → 8, worst seam speed **25.7 → 9.5 px/f**, but the post-stitch count stays in a **33–38 band** around the baseline 36. Work moves from stitcher to tracker and the seams get safer; identities do not fall. **The fifth cheap fix to hit the same plateau.** Worth taking: **GTA-Link** (MIT, offline, bolts onto our output) and **Selective Mask Propagation** (arXiv 2606.13033 — our #133 architecture with the VOS fired *only on uncertainty*, the direct answer to our 4.2 h/pass). Not now: NOOUGAT and SAM2MOT have **no code**; Sapiens/Multi-HMR 2/PromptHMR are non-commercial; Deep-EIoU's repo has **no licence file at all**. Two external criticisms do **not** apply to us: SMPLest-X's per-bbox focal instability (we never read its camera) and PromptHMR's shot-boundary drift (single-shot clip). |
| #125 | A run that solved no calibration frame still exported a finished scene | **CLOSED 2026-08-01**, gate *and* root cause: it reconstructed a different video (`PITCH3D_CLIP` unset); now required. Re-run `out/pod_0801b` = 23 subjects, 60/60 solved |
| #120 | Stored scenes declare a world frame they are not in | body mirror 2026-07-31; **corrections mirror 2026-08-01** (user saw it, measured 0.114→0.323); stale `handedness` labels remain |
| #109 | `jersey_numbers.py` must crop from the real camera at native resolution | pending, unblocked by #107 |
| #108 | R3's line-constraint path is a no-op on this clip | pending — needs a log line at the decision first, then a run |
| #45 | F2: raw video → frame range → auto `scene.json` behind the GUI | **BLOCKED on a user decision** (where it runs). Do NOT stub a fake generate button |

**Closed, detail in findings:** #107 measured camera (07-31) · #117 frame preprocessing (07-31, its
focal reading superseded by #119) · #122 expired session (07-31) · #124 undiscoverable drag (08-01).

**The calibration thread is closed.** #61, #119 and #60 all passed the user's eye on 2026-08-03,
after the #120 corrections-mirror fix made the overlay judgeable at all. What is left of #112 is
ergonomics, not geometry: #127, now fixed and awaiting the same eye. The layout has both editors
the orient panel has — drag with live preview, and typed metres/degrees — held together by
`scripts/check_layout_preview.py` (10/10, 0.0000 px). The toolbar is two wrapping rows because one
nowrap row overflowed the moment the calibration badges appeared.

**Not acted on, for the user to judge:** the hand-registration stored on frame 0 (11 drags) reads
`fit 3.0 px · ok 279 / off 0`, where the untouched solve reads `1.0 px · 278 / 25`. It may well be
deliberate — their eye is ground truth here, not the residual, which is scored against the same
lines that placed the model.

## 5. Health (measured 2026-08-01)

Honest baseline, so the next session does not mistake green for safe.

| Signal | Measured | Note |
|--------|----------|------|
| Test suite | **1168 passed / 19 skipped / 0 failed in 20.2 s** (re-measured 2026-08-07) | fakes-backed (`conftest.py` says so). The old ">5 min" here was wrong — no reason to avoid the full run |
| Real-measurement coverage | **1 file, 8 tests / 20 assertions** | `tests/e2e/test_golden_real_camera.py` over the committed 7 kB camera fit — the only non-fake evidence in the suite, and mutation-checked. Everything downstream of the camera (detection, pose, export) is still fakes-only |
| Untested user-facing paths | **4629 lines** | `app/controller.py`, `app/cli.py`, `app/anim_export.py`, `poseannot/app.py`, `poseannot/camera.py`, `scripts/blender_animate.py` |
| Lint | **153 ruff errors** (was 311) | 87 E501 · 45 E702 · 5 E402 · tail, measured 2026-08-07 under the CI pin. The earlier 130/152 were read off a stale local ruff 0.15.18 — see CLAUDE.md's four-places note. UP042 switched off (its fix changes enum serialisation) |
| CI | **pre-commit + GH Actions** | gate = `scripts/lint_changed.py`: a changed file may not *gain* violations. The 153 are reported, not gated — the backlog can shrink, not grow |
| Type checking | **mypy checks nothing** | numpy's stubs use 3.12 `type` syntax, rejected under `python_version = "3.11"`; mypy stops at error 1. Declared but dead |
| Declared deps | **fixed 2026-08-01** | `pyyaml`/`scipy` were imported by `core/` but undeclared — a clean `pip install -e ".[dev]"` could not collect a single test. CI now guards this |
| Pipeline entry points | **1, not 6** | re-measured 2026-08-02: `cli.py` and `mcp/server.py` both go through `controller.Application`. `anim_export.py` + `blender_animate.py` consume an exported `scene.json` and never reconstruct. The "6" was a miscount |
| Gate-chain mirrors | **2, now guarded** | `controller.run_reconstruction` (16 gates) vs `poseannot/rerun.py` (12 + 4 declared provider-blocked). In sync; `tests/unit/test_gate_chain_parity.py` fails if they drift |
| Calibration backends | **4 paths, 1 wired** | only `KeypointFieldCalibrator` is in `wiring.py`; `CameraModuleFieldCalibrator`, `PnLCalibBackend` and the `*_rigid_camera.py` scripts are parallel routes |

The 2026-08-01 remediation plan is closed out: steps 1–3 (agent entry point · pre-commit + CI · one
golden test on real data) done, step 4 (collapse the "6" entry points) **retired 2026-08-02, not
completed** — its premise did not survive measurement. Full write-up, and the lesson about this
file's own claims, in [`archive/status-log-2026-07.md`](archive/status-log-2026-07.md) under
2026-08-03.

Genuinely open, and *not* addressed by any of those steps:

- **9 exported stubs raise `NotImplementedError`** — see CLAUDE.md. Construct fine, fail on call.
- **`BOWL_*` stadium geometry** in `core/scene/cameras.py` has no config override path.
- **`anim_export.py` (882 lines) has no direct test** of its export logic, only a manifest
  contract check. It is the widest untested surface left in the user-facing path.
- **CI is a mechanical fence, not evidence.** It stops new lint debt and proves a clean checkout
  installs; it does not make the suite meaningful. The golden test proves the *camera* is real, not
  the export downstream of it — a green CI still largely means "the fakes agree with each other".

### Local GPU box (added 2026-08-07)

`demorig-pc` (172.16.10.203, `ssh demorig`) now runs the pipeline in Docker on an **RTX 4080
(16 GB, sm_89)** — Win 11 → WSL2 Ubuntu 24.04 → docker-ce 29.7.2 + nvidia-container-toolkit.
Measured 2026-08-07: `pod_real_e2e.sh` over 48 frames in **75 s, exit 0**, with all five real
backends (RF-DETR · ByteTrack · **PnLCalib** · **SMPLest-X-H** · **WASB**) → 16 gates →
`smplx_npz`; **peak VRAM 3930 MiB = 24 % of the 16 GB**, so reconstruction has ~12 GB spare and
16 GB is not a constraint for it. Suite in-container: 6.5 s against ~21 s on the laptop (~3×; the
in-container count was 1123/46 because it ran `origin/main` without SMPL-X mounted).

**The generative tail stays on the pod.** Measured the same day: SeedVR2 3B fp16 @720p
`batch_size=33` *does* complete here — at **97.5 % of the card**, and it OOMs the moment you cap it
at 95 %. `expandable_segments` does not help and `batch_size` is not the lever (the OOM is in the
VAE phase, which spans the whole sequence). ~400 MiB of margin on a workstation that also drives a
display is not a margin. The fp8 variant (3.39 GB vs 6.78) is the lever, untested — and a
by-eye quality change, so it is the operator's call. Blender rendering is also still pod-only.

**How to use it, and the WSL traps that each cost a run — [`local-gpu-box.md`](local-gpu-box.md).**
Staging is reproduced by [`../docker/stage_weights.sh`](../docker/stage_weights.sh).

### Ground truth (found on disk 2026-08-07)

`WorldPose/` holds **673 MB** of FIFA 2022 ground truth — `_poses-dev.7z` (per-player `betas`,
`global_orients`, `body_poses`, `transl`) and `_cameras-dev.7z` (**89 clips**, per-frame `K`, `R`,
`t`, distortion). Several docs still call these *pending*; they are not. The **video** is still
gated (separate FIFA agreement, `WorldPose/_README.md`), so end-to-end scoring of our pipeline
remains blocked — but the poses alone settle constants we currently guess: the kinematic ceilings
(10.5 m/s, 8 m/s²), whether our 0.08–0.23 m of root-Z is anywhere near a real pelvis, and whether
П4's 0.5 m twin threshold is ever legitimate. Local, CPU, no pod.
[`research-ledger §5.2`](findings/research-ledger-2026-08-07.md).

## 6. Key references

- **Cross-thread research verdicts (validated / refuted / never judged / owed):**
  [`findings/research-ledger-2026-08-07.md`](findings/research-ledger-2026-08-07.md)
- **How to work here (commands, rules, architecture):** [`../CLAUDE.md`](../CLAUDE.md)
- **Where code lives:** [`code-map.md`](code-map.md)
- **Open-item detail:** [`findings/open-items-2026-08-01.md`](findings/open-items-2026-08-01.md)
- **History (verbatim log, … 2026-08-01):** [`archive/status-log-2026-07.md`](archive/status-log-2026-07.md)
- **v0 defects + code root-causes:** [`archive/v0-geometry-defects.md`](archive/v0-geometry-defects.md)
- **Pipeline overview:** [`pipeline.md`](pipeline.md) · **Rejected approaches:** [`adr/0012-rejected-approaches-log.md`](adr/0012-rejected-approaches-log.md)
- **Historical build log (M0–M4 = plumbing, not result quality):** [`roadmap.md`](roadmap.md) ·
  **M1 live state:** [`archive/m1-status-and-plan.md`](archive/m1-status-and-plan.md)
