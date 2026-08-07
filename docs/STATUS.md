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

**Last updated:** 2026-08-05 · **Repo:** /home/chubuchnyi/AVATAR · **Target clip:**
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

**THE ACTIVE THREAD 2026-08-07 — player correctness, not rendering.** Standing user constraint:
**no full final render until poses and physics are settled**; judge in the `/app` pitch panel
(#134) instead. The user judged all 24 tracks of `out/cue/scene_off.json` by eye on 2026-08-07 and
asked for the criteria to be written down — done, and they now score **19/20 against that eye**:
[`findings/track-correctness-criteria-2026-08-07.md`](findings/track-correctness-criteria-2026-08-07.md),
probe `scripts/track_quality.py`, labels `findings/track-labels-2026-08-07.json`. Headline: an
`imputed` frame is a **frozen mannequin** (exactly 0.00 rad of limb motion while the root coasts
metres), so the phantom the eye sees is already flagged in the scene and needs no new model. Every
defect the user listed is **association or placement — none is a wrong pose on a correct crop**.
See #135 for the work that follows.

**Vertical fan clip, run 2026-08-03 — completed, and the output is unusable.** Detail + all numbers:
[`findings §3.3`](findings/open-items-2026-08-01.md). Short version: raw, the frame is 37% grass
starting at y=1088 and PnLCalib solves 0/8; the new `scripts/broadcast_crop.py` measures the grass
band and crops to it (84.2% grass) and the same frames solve 8/8 at conf 0.47–0.56. But past frame
~155 the fan zooms in until only the goal mouth is left, PnLCalib stops solving, and the calibrator
carries the stale homography onto zoomed pixels — **43% of the 355 frames**, roots 3080 m apart, one
subject at 100 416 m/s. Not fixable by cropping: #119's one-camera result is broadcast-specific
(tripod = pan/tilt only), and a handheld phone that translates *and* zooms has no one camera
(`realizable: False`, 142 px). Usable window ≈ frames 0–155 (~5 s) — **worth a render? user's call.**
Two real defects fell out of it and are worth more than the clip: #130 (fixed) and #131.

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
| #132 | Player crossings break ByteTrack IDs and fuse per-crop poses (occlusion) | **CLOSED 2026-08-05, both halves.** *Pose half: not reproduced* — SMPLest-X per-crop does not fuse on the clip's hardest real overlap (79%/90% of each mesh on its own player), so **PromptHMR is not adopted** and the SAM 3 branch stays out. *Tracking half: fixed* — kit colour was sampled from a track's first 8 frames only, making a mid-track player swap undetectable by construction; `split_on_kit_change` takes tracks carrying two humans **9 → 0**, and turns the dormant stitcher from 1 merge into 14. Three follow-ups then measured empty (stitch gates ≤2 ids, NMS ≤1 id) and one of my own claims was corrected (the identity gap is a third, not double). *Shot-cut detection built and then fixed* — the first threshold was bin-dependent and shredded a 2-shot clip into 36; now scale-free, CLI guard ON by default with `--no-shot-guard`. *Render cost checked 2026-08-05:* run both ways through the real detector+tracker over 48 frames, the split creates 6 extra fragments and the stitcher merges exactly 6 back — **same 21 subjects, identical 848 subject-frames, no short-lived subjects**, so it relabels without churn. **Still unseen in a novel-view render (needs a pod run).** Detail: [`findings`](findings/occlusion-pose-research-2026-08-04.md) |
| #134 | **Pitch-3D scrubber in the UI** — judge poses without rendering | **built 2026-08-06** at `/world`, linked from the app toolbar. Whole pitch in 3D: measured markings + goals + turf from `core/scene/pitch.py` (same geometry the overlay and render use, so the views cannot drift), every subject's 22-joint skeleton at the current frame coloured by team, frame scrubber with play, ids and pelvis trails, free orbit/zoom/pan. Two new endpoints: `/api/world/geometry` (static) and `/api/world/{n}/skeletons` (batched 3D — the per-subject joints endpoint is for the single-player editor and would spend the scrub in HTTP). **Folded into the app 2026-08-07**, which reversed the "kept off `index.html`" call above: the viewer's core is now `poseannot/static/world_view.js`, mounted by BOTH `/world` and the app's right panel via a `3D pose | pitch` switch, so neither carries a second copy. In the app the middle panel is already the source frame, so pitch mode is frame-vs-3D side by side, synced; clicking a body selects that track everywhere. Column widths drag (two grid gutters writing `--rail-w`/`--right-w`, persisted); `overlays: on/off` (H) strips every drawn layer for a bare frame and says so on the frame. `/world` keeps its own copy of the frame panel for the A/B overlay case. Navigation: right/middle-drag or WASD pans (0.6 m/frame, shift 2.0), single click selects a player everywhere, double click re-centres the orbit on him, F fits. Verified in a real browser, no console errors. Verified: 23 markings / 6 uprights, frame 0 gives 23 subjects (A 10 / B 13) with joint z 0.06-2.13 m — feet on the turf, heads at 2 m — and the client JS passes `node --check`. **Standing request: no full final renders until poses and physics are settled** — use this. |
| #133 | **Human physics: filter motion, solid bodies, no popping, limbs in sync** | opened 2026-08-06 by the user, who noted it had been discussed before and **lost**. Most of it is already built and switched off; `joint`+`orientation` turned on today (violations 118+11 → 0, 97.8%/98.5% of real angular travel kept, **passed the eye** on the pod A/B). Still off: `collision`, `pose_motion_sync`, `contact_probe`, `foot_plant`, `gravity_project`, `identity`. *Identity half diagnosed and attacked:* 78 mid-pitch births/deaths, **96%** with an unclaimed detection a median 6–23 px away — so the boxes exist and the missing thing is an association **cue**. Three cheap fixes measured **null** (match threshold both ways, detector threshold). **McByte's mask cue now built and measured:** Cutie propagates one mask per track (236 frames in 686 s on GPU, 90× CPU), `MaskCue` applies the mm1/mm2 discount by wrapping supervision's `matching.iou_distance`. Same detections, same stitch: **mid-pitch events 28 → 24**, kit changes 10 → 10 (no cost), 2 fewer ids allocated, 48 vs 52 gap-frames bridged. **Works but weak — 14% against a 96% ceiling**, most likely because the two-pass design seeds masks from pass-1's broken tracks; McByte is online for exactly that reason. NOT the default — enable with `PITCH3D_MASK_CUE=<track_masks.npz>`. Scenes for the eye: `out/cue/scene_{off,on}.json` via `POSEANNOT_SCENE_JSON=… uvicorn poseannot.app:app --port 800X` → `/world`. Detail: [`findings`](findings/human-physics-requirement-2026-08-06.md) |
| #135 | **When is a reconstructed player correct?** — criteria defined, measured, scored | **defined 2026-08-07** on the user's own per-track verdict (24 tracks, `out/cue/scene_off.json`). Six признаки: **П1 shape** (measured frames span the clip → `FULL` ⇒ correct, 12/12), **П2 in-frame imputed run = phantom / off-frame = keep showing** (the user's own display rule, made computable — needs `--camera`, since the scenes store the *invented* 772 px fallback under which everything is trivially in-frame), **П3 handover** (a HEAD dying where a TAIL is born, same team, ≤14 frames, ≤6 m = one human), **П4 twin** (every interpenetration in the scene is a phantom parked inside a measured player — first measurement of #133's "solid bodies"), **П5 vertical** (largest root-Z excursion in the WHOLE scene is 0.082 m, so *«17 не прыгнул»* is a missing degree of freedom, not a t17 defect), **П6** an invented prefix *holds* the first measured position instead of extrapolating — that is the t25 placement bug. Score **19/20**, recall on defects **7/7**. Three things left open and deliberately not smoothed over: t20 (eye says correct, criteria say phantom), `15→71` (eye) vs `15→25` (geometry: 0.85 m / 2-frame gap, and it would fix t25 for free), and "32" (no such track; 16 and 31 are the only off-frame ones). **Work list in §6 of the findings.** |
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
| Test suite | **1125 passed / 14 skipped / 0 failed in 71 s** (re-measured 2026-08-02) | fakes-backed (`conftest.py` says so). The old ">5 min" here was wrong — no reason to avoid the full run |
| Real-measurement coverage | **1 file, 8 assertions** | `tests/e2e/test_golden_real_camera.py` over the committed 7 kB camera fit — the only non-fake evidence in the suite, and mutation-checked. Everything downstream of the camera (detection, pose, export) is still fakes-only |
| Untested user-facing paths | ~6000 lines | `app/controller.py`, `app/cli.py`, `app/anim_export.py`, `poseannot/app.py`, `poseannot/camera.py`, `scripts/blender_animate.py` |
| Lint | **152 ruff errors** (was 311) | 87 E501 · 45 E702 · tail. 148 auto-fixed 2026-08-01; UP042 switched off (its fix changes enum serialisation) |
| CI | **pre-commit + GH Actions** | gate = `scripts/lint_changed.py`: a changed file may not *gain* violations. The 152 are reported, not gated — the backlog can shrink, not grow |
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
- **`anim_export.py` (845 lines) has no direct test** of its export logic, only a manifest
  contract check. It is the widest untested surface left in the user-facing path.
- **CI is a mechanical fence, not evidence.** It stops new lint debt and proves a clean checkout
  installs; it does not make the suite meaningful. The golden test proves the *camera* is real, not
  the export downstream of it — a green CI still largely means "the fakes agree with each other".

## 6. Key references

- **How to work here (commands, rules, architecture):** [`../CLAUDE.md`](../CLAUDE.md)
- **Where code lives:** [`code-map.md`](code-map.md)
- **Open-item detail:** [`findings/open-items-2026-08-01.md`](findings/open-items-2026-08-01.md)
- **History (verbatim log, … 2026-08-01):** [`archive/status-log-2026-07.md`](archive/status-log-2026-07.md)
- **v0 defects + code root-causes:** [`v0-geometry-defects.md`](v0-geometry-defects.md)
- **Pipeline overview:** [`pipeline.md`](pipeline.md) · **Rejected approaches:** [`adr/0012-rejected-approaches-log.md`](adr/0012-rejected-approaches-log.md)
- **Historical build log (M0–M4 = plumbing, not result quality):** [`roadmap.md`](roadmap.md) ·
  **M1 live state:** [`m1-status-and-plan.md`](m1-status-and-plan.md)
