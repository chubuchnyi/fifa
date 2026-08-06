# Open items — full detail (snapshot 2026-08-01)

Verbatim §3.1 of `docs/STATUS.md` before the 2026-08-01 split. The board in STATUS.md is
now one line per item and links here for the reasoning, measurements and root causes.

When you close an item: move its evidence to `docs/archive/status-log-2026-07.md`, delete
the row here, and drop the STATUS line.

---

## 3. Open board — the work right now

### 3.1 Open items (#1xx — poseannot UI, calibration, pipeline)

The CC task list does not survive a session, so this table is the **only** durable list of what is
still open. Add a row when you open an item, move it to §6 with the evidence when you close it.

| ID | Open item | Status | Why it is open |
|----|-----------|--------|----------------|
| #120 | **Stored scenes declare a frame they are not in** | **body mirror done 2026-07-31, corrections mirror done 2026-08-01**; the stale *labels* remain | Three halves now. **(c) The corrections were never mirrored — found by the user's eye 2026-08-01.** `mirror_subjects` flips `proposal.pose`, but a *replayed* scene keeps its motion in `scene.corrections`, which `resolve_subject_motion` lays over the proposal. The shipping default clip (`out/physics_debug/scene_replayed_v2_rigid.json`) therefore held right-handed proposals, a right-handed #119 camera and **677 left-handed corrections**, and drew all 23 players reflected about the halfway line. Scored on pixels (HSV grass mask, 44×85 px patch above each stance marker): mean non-grass **0.114 → 0.323**, markers off-frame **6/23 → 1/23**, markers standing on a body **2/23 → 8/23** — now level with the untouched `out/pod_0801b` (0.361). It was the *only* correction-bearing scene, so no other `_rigid` artifact was affected, and **today's pipeline never had the bug** (fresh runs carry 0 corrections). Fix: `mirror_corrections` in `scripts/apply_rigid_camera.py` maps `ROOT_TRANSLATION → v·M`, `ROOT_ORIENTATION → Mˡ·R·S`, `POSE_BODY_JOINT → (ωx,−ωy,−ωz)` **plus** a `_BP_SWAP` of `joint_index`, and **raises** on any kind/mode it has no rule for rather than passing it through unmirrored. Pinned by `test_corrections_mirror_with_the_subject`, which asserts the *resolved* motion mirrors — the invariant that was actually broken. **(a) The body mirror is fixed and pinned** (§6): `mirror_subjects` changed one parameter where it needed two, and used the *world* mirror on a **camera-frame** rotation. Correct is `global_orient → Mˡ·R·S` with `Mˡ = R_cam→worldᵀ·M·R_cam→world = diag(1,1,−1)`, `S = diag(−1,1,1)`, **plus** the SMPL-X left↔right `body_pose` flip; `transl → M·t` was always right. Measured against the real SMPL-X forward pass over 23 subjects × 4 frames: **1.01 m → 0.041 m**, and 0.041 m is the neutral template's own asymmetry (its rest hips differ by 0.011 m off-axis), i.e. the floor. `scene_rigid.json` regenerated: 100 % upright, feet at z = **+0.001 m**. `tests/unit/test_subject_mirror.py` pins it *and* pins the old transform as wrong. **(b) Still open:** `out/anim_full_realism/scene.json` and other pre-#118 artifacts declare `world_frame: {handedness: right}` while their contents are left-handed (measured: `plane_orientation` 0/60 right-handed). Nothing reads the label today, but it will mirror the first renderer that trusts it. Not flipped blind — 60 frames of user-validated poses. |
| #119 | **Re-solve the calibration as ONE camera, not 60 free homographies** | **CLOSED 2026-08-03 by the user's eye** | Shipped (§6): 184 parameters beat 480 on paint (1.28 vs 1.62 px) and on pixel motion (1.44 vs 7.18) *simultaneously*; steadiness is **0.55 vs 4.08 px** of gap-1 residual against 8.55 px of real camera motion (the `jitter` column, 5.11 vs 5.31, is too flat to discriminate and is a guard, not evidence). f = 4169 px @1920×1080 from four seeds spanning 2700–5200, centre (−2.3, −70.1, 17.2) m — a focal sweep brackets the paint minimum on both sides, so **±10 %**, and since `f/dist` is fixed at ≈57 the distance carries the same ±10 %. Written to `scene_rigid.json` and registered in the clip switcher **beside** the old scene — the alignment verdict is the user's, per the ground-truth rule. |
| #117 | **Frame preprocessing to feed auto-calibration** — markings, mowing stripes, other fixed field objects | **research done 2026-07-31**; its payoff #119 is now built, and it read the focal wrong | Its "focal = 2700 or 3903 or 4277" is **superseded** — #119 used the same pixel data as a direct reprojection residual instead of through `rotation_cost`'s singular-value spread and got a single-valued 4169 ±10 % at 1.44 px. The spread was the functionals disagreeing, not the data — and the #119 sweep says which functional was wrong: the pixel-motion residual is **monotone** in the focal (0.85 px at f=2700 rising to 1.64 at 5200, no interior minimum anywhere), so it can never name a focal on its own, while the paint's minimum is bracketed on both sides. #117 read the flat instrument. The rest stands: the ground plane is *not* where the error is (1.4–1.7 px against paint), so more per-frame evidence buys almost nothing. The real wins were #118 (done) and #119. Mowing stripes are real — 63% of the visible surface lies >30 px from any paint, with a 6–8% tone modulation — but a stripe has **no known world coordinate**, so it constrains a direction, never a position. Lens undistortion is **rejected on measurement** (residual flat 0.27–0.33 px from r=0 to r=1100). |
| #112 | **Drag the pitch layout to correct the homography** | **the geometry is CLOSED 2026-08-03** ("works" — user); the controls became #127 | Shipped (§6): the manual-override half of the standing request, alongside #111's per-player drags. The gesture must exist because **no residual we compute can see this error** — every one is scored against the same lines that placed the model, so the operator's eye is the only instrument. The design choice that makes it safe is composing the edit on the **world plane** (`H @ B`, a 4-DOF similarity) rather than in image space (`A @ H`): `K[r₁ r₂ t] @ B` is again `K[r₁' r₂' t']`, so the drag provably cannot turn a camera-realizable calibration unrealizable, and K is never touched so the focal is unchanged. Live drag: `fit_px` 1.0 → 19.38 with the camera badge held at `measured · 0.0 px · f 4169.3`; undo restores `edits.json` byte-identical. The `turn` handle is pinned by unit test and by curl but **not yet watched in a browser** (session died, #122). |
| #122 | **An expired session silently degrades the UI instead of asking for a re-login** | **done 2026-07-31** | The `poseannot_token` cookie lasts `jwt_expire_hours: 24`; when it lapses mid-session every layer just empties, because all ~50 call sites end `if (!r.ok) { clear the layer; return; }` — right for "no data on this frame", wrong for "your token died". The storm was live and measured: **~0.8 req/s of 401s on `/api/frame/1/ground` for hours**, 48 821 when first seen and 56 000+ still climbing a session later, while the frame the operator was judging alignment against kept rendering from cache. Fixed by wrapping `window.fetch` **once**: latch on the first 401, raise a blocking re-login card, short-circuit every later request. **Deliberately driver-agnostic** — no timer, retry, recursion or observer in `index.html` (nor anywhere in its git history) can re-issue that call, and the storming tab ran a build we can no longer read, so the repeat was never pinned to a line; interception at the one boundary every caller shares stops it regardless. The frame `<img>` bypasses `fetch`, so its `@error` re-requests through the wrapper — as a **GET**, since FastAPI's `APIRoute` answers HEAD with 405 and a HEAD probe would miss the very 401 it hunts. Mechanism verified in a real page: 502 calls → **1** network call, event latched once, callers still get a readable 401 `Response`. |
| #61 | Camera-calibration accuracy (offset + ~3× scale) | **CLOSED 2026-08-03 by the user's eye** | The defect was never on the ground plane (#110/#113/#114 measured 1.4 px against real paint), and it was not a *wrong* camera either. PnLCalib solves each frame as a free 8-DOF DLT with nothing tying it to a pinhole, and on this clip the result is **no camera at all**: swept over every focal from 200 to 12000 px, the nearest realizable pinhole is still **~525 px** away (#119's fit: 1.36 px). So `camera_from_calibration` refuses (`REALIZABLE_PX = 1.0`), `controller.py:296` substitutes a synthetic `Viewpoint.BROADCAST` at 1280×720 / fx 772, and the scene draws its pitch through the measured homography and its players through a camera **3.9× too small** — marks land, bodies do not. Fixed by applying the one measured fit (`calib/…npz`, committed) to **every** scene variant via `apply_rigid_camera.py`; live API agrees to `fit 1.0–1.4 px` on all of them. Close once the user has looked. |
| #125 | **A run that solved no calibration frame still exports a finished scene** | **fixed + root-caused 2026-08-01** | Opened on a wrong diagnosis, and the correction is the finding (§6). There was **no cache**: the "identical calibration" was measured on two `scene_rigid.json` files, i.e. the two files `apply_rigid_camera.py` had *overwritten* with #119's camera — bit-identical (`max\|dH\| = 0.0`) because the same script wrote both. Their RAW calibrations differ by **88 m on-pitch**. The run that failed was **2026-08-01's**, and its own file said so in a field nobody read: `confidence` is **0.0 on 60/60 frames**, which both calibrators document as "unsolved — carried the last good homography, or `eye(3)` if there was none". 34/60 frames are exactly `eye(3)`; the scene is built on *one pixel = one metre*, which is the 11-subjects-instead-of-23. **Fixed** by `require_solved_calibration` in `core/orchestration/pipeline.py` — called straight after CALIBRATE and again by `apply_rigid_camera.py`, which is what laundered the dead scene into a healthy-looking 1.0 px. `min_solved` is the dial; the default refuses only "no answer anywhere". 4 tests pin it. **Root cause: I reconstructed a different video.** `pod_real_e2e.sh` defaulted `CLIP=$VOL/clip.mp4` and I never set `PITCH3D_CLIP`, so it ingested a 59.94 fps stock clip with no pitch in it — **the same trap as 2026-07-03**, whose entry proposed exactly this guard and never built it. `PITCH3D_CLIP` is now **required** in `pod_real_e2e.sh` + `pod_make_video.sh` (exit 2, lists the volume's `*.mp4`). Re-run named explicitly → `out/pod_0801b`: **23 subjects, 60/60 solved, conf 0.54, worst teleport 0.98 m**, which also settles *11 vs 23* → **23**. **Last hole closed 2026-08-01:** the dead scene reached the clip switcher *paired with the Colombia video*, and a scene carries no pixels, so poseannot drew it over that footage and the overlay looked plausible. The scene had said so all along — `source_id: "clip"` where every good scene says `Colombia-1-0-Congo-DR1080p`. `poseannot/clips.py` now reads that field (bounded 4 KB head read + regex; a scene is 7 MB and this runs per clip per list request) and **marks, not refuses** (R-6): `⚠` in the switcher, an `⚠ wrong video` chip in the toolbar naming both videos. Uploads are exempt because `create_clip_from_upload` normalises every video to `video.<ext>`, so their filename is not evidence. Verified in-browser against the real `out/pod_0801` artifact — chip fires, and clears on a matching clip; 7 tests pin it. |
| #126 | **The confidence stored next to a carried homography was computed for a different matrix** | **new 2026-08-01**, measured but unresolved | Fell out of #125. `out/fresh60` (`CAMERA_CARRY=0`) and `out/pod_0801b` (wiring default `camera_carry=8`) are the same 60 frames of the same clip on the same pod, and their confidences are **bit-identical** (`max\|dc\| = 0.0`) while the homographies differ by **0.76 m median / 3.67 m max on-pitch**. That is not jitter — the calibrator is deterministic; it is the R2 carry (#94), cleanly isolated, and the first measurement of what that flag does to this clip. **(a)** Nobody has checked which track fits the paint better — score both with #119/#123's on-paint metric instead of guessing. **(b)** The likelier real bug: confidence is scored per-frame *before* `carry_on_motion` replaces the track (`calibration.py:586-593`), so the number ships attached to a matrix it does not describe — on **every** run that does not set `CAMERA_CARRY`, i.e. the default. Either re-score after the carry or mark carried frames as carried (R-6). |
| #124 | **The pitch-layout drag is unverifiable** | **done 2026-08-01** | The #112 gesture worked; nothing about it was legible (§6). Unlabeled handles among ~23 same-coloured player markers, a handle that springs back on release so a good drag reads as a failed one, instructions in a hover tooltip, and the numbers that prove it worked in 10 px grey type ~500 px away. Fixed with on-frame labels + an in-panel HUD carrying the gesture, the live `fit px` / on-paint score (red when worse than at open), the last delta, `undo` and `done`. Verified with real pointer events: drag → `32.1 px · 22/261` red, undo → `1.0 px · 278/25`. |
| #108 | R3's line-constraint path is a **no-op** on the target clip | pending, **needs instrumentation, not the pod** | Fresh post-R3 homographies are byte-identical to the pre-R3 export (max\|dH\| = 0.0 over 60 frames) — the line path self-disables via `_lines_agree` (`_LINE_FRAME_TOL_M = 3.0`). The plan was to read which branch trips out of `carry_ab_full.log`; checked on the live pod 2026-07-31 and **the log cannot answer it** — 1.4 MB with *zero* mentions of the agreement test, because the branch emits nothing. So this was never blocked on pod access: it needs a log line at the decision first, and only then a run. |
| #107 | Render the **measured** camera, not the synthetic one | **done 2026-07-31** | Was framed as a decision; it was a measurement, and it came out **neither** for the old scene (§6). `core/scene/plane_camera.py` decomposes the calibration into a real `CameraTrack` (focal from Zhang's constraint, no extra input) and **refuses** — `camera is None` — when no camera explains the homographies. Control: `scene_rigid.json` → focal 4169, reprojection **0.0003 px**; `scene.json`'s 480 free params → **5048 px**, so #107 was never an oversight, there was nothing to render. `AppController` now keeps the solved camera and falls back only when the refusal fires; the header badge names which you are looking at (`one camera · 0 px` vs amber `two cameras · 6220.64 px apart`). Unblocks #109. |
| #109 | `jersey_numbers.py` must crop from the real camera at native resolution | pending, **unblocked by #107** | 0/23 usable crops: it projects through `scene.camera`, which used to be synthetic *and* whose intrinsics are the 1280×720 **render** size, so the 1920×1080 source is downscaled 1.5× before cropping → subjects 18 px tall against a 45 px floor. #107 fixes the *identity* of the camera but not the resolution — on `rigid-camera` the recovered camera is native 1920×1080, on the old scene it is still the synthetic fallback. Fix = crop via `field.calibration.homographies` at native res, which works either way. |
| #60 | Re-run overlays + verify acceptable alignment | **CLOSED 2026-08-03** | The eye-check that closed the calibration thread — and it only became answerable once #120(c) stopped drawing every player on the wrong side of the halfway line. The user's verdict on 2026-08-03 closed #61, #119 and #60 together, and accepted #112's geometry. Note what the verdict was *not*: it says the overlay aligns, not that the export downstream of it does. |
| #127 | **The layout gizmo is twitchy and shows nothing until you let go** | **fixed 2026-08-03**, awaiting the user's hands | The half of #112 the user rejected: "очень высокая чувствительность и нет обратной связи по лэйауту — тянешь, отпускаешь, и только потом видишь результат". Both halves are one defect. `index.html` drew only the *handle* at the cursor (`_dragPitchUV`); the outline it was moving came from the server and was refreshed in `adjustLayout` **after `pointerup`**, so the gesture could not be steered while it was still open — and measured on this clip a 78×50 px cursor move is **18.05 m** of lawn, because at broadcast perspective a pixel near the horizon is metres. Fix: `/api/pitch/calibrated/{frame}` now returns the plane map both ways, and the browser previews the drag as `A = H·B·H⁻¹` applied to the pixels it already drew — markings, uprights, stance markers and both gizmo handles move together, with a live `m / ° / ×` readout on the frame and in the HUD. **Shift = ¼ gain**, damping the *angle* and the *log*-scale so half a gesture is half the turn and the square root of the zoom rather than a coupled mess. The gain is applied as a damped **world target point** projected back through the homography before the POST, so `camera.py:plane_similarity` stays the only definition of the arithmetic and the endpoint gains no parameter. Verified against the server over 5 drags (both handles, both gains): the previewed map and the committed one agree to **0.0000 px** across a 63-point grid on the frame — i.e. no jump on release — and the readouts match to rounding. In-browser: shift picked up *mid*-gesture gives ratio exactly 0.250, and the commit stores 4.51 m (what was previewed) rather than 18.05 m (where the cursor was). Uprights ride the same plane transform, which is exact at their feet and approximate at the crossbar — they stand off the plane the transform describes; the server's redraw on release is the truth. **Typed twin, 2026-08-03:** the user then asked for the orient panel's control surface on the layout too, and they were right that the drag alone is not enough — a gesture cannot express "25 cm along" when a pixel is metres. The panel shares `_layoutB` with the drag and differs only in how it picks `dst`: it projects a typed *world* target back through `w2i` and posts that pixel, so the server formula and the endpoint signature are still untouched. Slide and turn are mutually exclusive groups because one POST carries one handle — so what is previewed is exactly what one commit stores, and one gesture is still one undo. The control jogs back to neutral after commit, since `/api/pitch/adjust` composes rather than replaces. `check_layout_preview.py` now drives both paths: 10/10 cases (5 drag + 5 panel), 0.0000 px. **The panel's four numbers are absolute** (moved here from STATUS 2026-08-03): a drag is a gesture and appends, so it must zero itself; the panel states where the layout should *sit* and rewrites one correction in place, so it can hold a value. Springing back to 0/1 after each commit was the defect. It previews from `layout_basis`, not `w2i` — its correction keeps its slot among the drags, so replacing it is not a plain right-multiply. In-browser: typed `along = 2` previewed `2.00 m · 0.0° · x1.000`, committed `last: 2 m · 0° · x1` with no jump, and undo restored the operator's own registration byte-for-byte (`fit 2.8 px`, `ok 246 / off 65`). |
| #128 | A hand-registered pitch layout never reaches the render | **CLOSED 2026-08-03** | **Fix.** The read path moved to `core/correction/plane.py` (`has_plane_corrections` / `apply_plane_corrections`, plus the three functions lifted verbatim out of `poseannot/camera.py`, which now re-exports them) and `anim_export.main` calls it. It lives in core because `src/pitch3d` may not import `poseannot`, and it is re-exported rather than copied — a second implementation of an edit the operator judged by eye is two answers to one question, the trap the hand-maintained gate-chain mirror fell into. The decision the entry below flagged as real resolves against the *camera*: `poseannot/scene_state.py` already says "subjects are stored in world coordinates, so moving the pitch model moves what is drawn, not the bodies", so the export must reproduce that or double-count the correction. **A third half, found while verifying and not visible from the code the entry was written against:** the annotator never rewrites `scene.json` — `poseannot/edits.py` appends rows to a sibling `edits.json` and `build_scene_state` folds them in at load. Nothing outside `poseannot` opened that file, so the read path above would have been starved on every real scene and the fix would have been theatre. `core/correction/sidecar.py` now discovers it (`<stem>_edits.json`, then `edits.json`, both names minted by `poseannot/clips.py`) and merges sidecar rows *after* the scene's own, exactly as the annotator stacks them — with `--corrections` / `PITCH3D_CORRECTIONS_JSON` to override. `load_edits` moved there too, so there is one loader. Both steps print what they did, per #125: a run that silently ignored the registration read exactly like one that honoured it. **Measured on the user's own 11 drags** (`out/physics_debug/scene_replayed_v2_rigid.json` + its sidecar): sidecar found, 677 → 688 corrections, `B(f0)` = 1.45 m · −0.30° · ×1.0363, camera and calibration both move, subjects untouched, and the two halves of the camera project the centre spot and both far corners to **0.0000 px of each other on frames 0/30/59** — the #107 invariant survives the edit. The edit is worth 63–207 px at the pitch corners, i.e. plainly visible and previously discarded. Full export re-run end to end: `ANIM_EXPORT_OK (23 subjects)`. 9 new tests in `test_pitch_layout_adjust.py` (18 total), four-mutation checked — camera not moved, calibration not moved, the export dropping the call, and a disabled correction applied anyway are each caught. | 
| #128-note | The entry as originally written, kept for the reasoning | superseded | The user asked the right question — does the layout drag affect the calibration? — and the answer is yes inside poseannot and **no** past it. `FIELD_CALIBRATION` is declared in `core/scene/layers.py:35` with a documented `PlaneTransformPayload`, and `poseannot/camera.py` applies it to both halves of the camera. But nothing in `src/pitch3d/` consumes it: `core/correction/engine.py` resolves subject and ball layers only, `anim_export.py` calls `resolve_subject_motion` / `resolve_ball` and never asks for a plane transform, and the Blender adapter is downstream of that. The correction *is* persisted (`corrections_out`, default `out/physics_debug/edits.json`), so the data is not lost — it is simply never read back. Consequence: an operator can spend a session registering the pitch by eye, and the exported `scene.json` → Blender → render chain still uses the raw solve. That is the opposite of #107's rule that the scene holds one camera, and it silently discards the only correction no residual can compute for us. Fix is a read path, not new geometry: `plane_adjustment` already exists and is tested; the export needs to fold it into the camera it writes (or refuse, loudly, when a scene carries enabled FIELD_CALIBRATION corrections it cannot apply — the #125 pattern). Blast radius check first: whether the exporter should bake the adjustment into the camera or into the world points is a real decision, because #120's mirror work says the two are not interchangeable. |
| #129 | **The rigid one-camera solve is called by no pod script** | opened 2026-08-03, **not started** | Split out of #128, which was two unrelated corrections under one heading. `scripts/apply_rigid_camera.py` is what turned 60 free homographies into the single camera the user's eye approved on 08-03 (#119), and `grep` over `scripts/pod_*.sh` returns zero matches for it — so a pod render still reconstructs from the per-frame solve. Unlike #128 this is not a missing read path but a missing *step*: the artifact exists (`out/physics_debug/scene_replayed_v2_rigid.json`) and the chain does not produce it. Decide whether the pod chain should run it, or whether the pipeline should stop emitting free homographies at all — the second is the real fix and the larger one. |
| #130 | **A subject shorter than the clip sank the whole run** | **CLOSED 2026-08-03** | `app/cli.py` picked the observation frame as `scene.subjects[0].proposal.pose.frames[n // 2]` where `n = clip.n_frames` — the *clip's* midpoint read out of *subject 0's own* array. Tracks are routinely shorter than the clip (the identity gate splits one, or the player leaves frame), and on the 355-frame vertical clip subject 0 lived 167 frames: `IndexError: index 177 is out of bounds for axis 0 with size 167`, raised **after** the full reconstruction was paid for and **before** anything was exported — the worst possible place, 22 minutes of GPU thrown away. Fix is one line: index the middle of that subject's own track. It is not clip-specific — R-6 says a short track is kept, not dropped, so this was reachable on any clip whose first subject leaves early; the Colombia clip simply never had one. Regression test `test_the_observation_frame_comes_from_a_track_that_actually_has_it` truncates subject 0 to a third of its length through `Application.get_scene` and asserts the observed frame is one the subject actually has; it fails on the old line. Runs with `demo_edits=False`, matching the pod's `DEMO_EDITS=0` — the walkthrough's synthetic nudge trips on the same truncation and would aim the test at the wrong line. |
| #131 | **A run reports `confidence mean=0.28` and never says 43% of it is carried, not measured** | **CLOSED 2026-08-03** | Found by running the vertical fan clip end to end (below). `core/orchestration/pipeline.py:36-66` is explicit that **confidence exactly 0 means the calibrator carried the previous homography forward** — a deliberate fallback, not corruption — and that the #125 gate only refuses the "no answer anywhere" case (`min_solved_frames = 1`), leaving the drift judgement to the caller: *"How many carried frames are acceptable is a judgment about drift, so it is left to the caller rather than guessed here."* That is the right split. The gap is that **no caller is given the number.** The only calibration line a run prints is `field calibration confidence mean=0.28` — a mean over a bimodal distribution, i.e. the one statistic that hides bimodality. On this run the two modes separate perfectly: **0.421 on the frames whose roots are sane, 0.000 on the frames whose roots are kilometres out**, 153 / 355 = 43.1% carried, and `grep 'confidence\|solved' src/pitch3d/app/cli.py` shows nothing anywhere splits the two. **Fix: a report, not a gate** — the gating judgement is deliberately the caller's, so the fix is to give the caller the number, not to take the decision away. `describe_calibration_solve()` sits in `pipeline.py` beside `require_solved_calibration`, so what "solved" means is stated once, and `cli.py` prints it next to `== reconstructed`. Verified E2E on the dry run: `== calibration: 12/12 frame(s) measured, 0 carried (confidence 0) · over measured frames: mean 0.950, min 0.950, max 0.950`. On the fan clip the same line would have read `202/355 frame(s) measured, 153 carried` at minute one instead of 22 minutes and a 22 MB export later. 3 tests in `test_unsolved_calibration_gate.py`, two-mutation checked: counting carried frames as solved (`>=` for `>`) and taking the mean over all frames instead of measured ones are each caught — the second is the exact bug, and the test pins the real distribution (355 frames, 202 solved at 0.496, all-frames mean 0.28). The all-carried case is asserted too, because `describe` also runs on loaded scenes and a NaN in a status line is worse than silence. |
| #45 | F2: raw video → frame range → auto `scene.json` behind the GUI | **BLOCKED on a user decision** | This is the whole GPU pipeline (rfdetr+bytetrack+gvhmr+keypoints+physics+export) as a minutes-long async job — only runnable on the pod, not this CPU box. Needs a green light on *where it runs* before anything is built. **Do NOT stub a fake generate button.** |

Recently closed (evidence in §6): **#122** an expired session now says so instead of quietly emptying
every overlay · **#112** the pitch layout is draggable and the drag provably keeps
the one camera · **#119** the clip is ONE camera, and it beats 480 free parameters
on every metric at once · **#118** the world frame is measured, not assumed · **#116** goal frames +
corner flags overlaid, focal as a hand
control · **#115** the "D" was a 14 m phantom chord · **#114** the far field was never 37 px out ·
**#113** honest extrapolation marking · **#111** drag players onto their real feet · **#110** draw
the overlay through the solved calibration.

### 3.2 v0 punch-list (#2xx — all CLOSED, kept for the root causes)

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

### 3.3 Vertical fan clip `14604731_1080_1920_30fps.mp4` — ran to completion, output unusable (2026-08-03)

The user asked for this clip to be run on the pod. It ran, it exited 0, and **the result must not be
mistaken for a success** — this section exists so the next session does not re-spend the GPU hour to
rediscover it. Pod `ta1wvy4o8l7ti2`, `done in 1357s` (22.6 min), `OK — reconstruct → … → export
completed`, 23 subjects, `scene.json` 22.8 MB. Two real defects fell out of it (#130, #131 above) and
both are worth more than the clip was.

**Step 1 — the raw vertical frame cannot be calibrated at all.** 1080×1920, 355 frames; grass is
37.2% of the frame and does not start until y=1088 — above that is stand, sky and scoreboard.
PnLCalib solved **0 of 8** frames. `scripts/broadcast_crop.py` (new, committed) measures the grass
band per row and crops to it: `1080x608+0+1200`, 84.2% grass, framing the penalty box, both
goalposts and the goal line. Through that crop the same 8 frames solve **8/8 at confidence
0.473–0.558**. The crop is measured, not assumed — a fan holds the phone any way — and `--rect`
overrides it, per the auto-plus-manual rule. On an already-broadcast clip it is a no-op *by
construction* and measured as one: the target Colombia clip returns `1920x1080+0+0`, grass unchanged
at 53.2%.

**Step 2 — the full 355-frame run then fails for a reason no crop can fix.** Measured on the export:

| signal | value |
|---|---|
| calibration confidence | mean 0.282, min 0.000, max 0.632 |
| **frames at confidence exactly 0.000 (carried, not measured)** | **153 / 355 = 43.1%** |
| camera | `estimated=True`, fx 772, cx 640, cy 360, translation span [0,0,0] — the **synthetic fallback** |
| root spread | **3079.7 × 3079.7 m** (a sane run on this clip's smoke: 27.6 × 18.9 m) |
| pelvis z | 0.81–0.92 m (plausible — the bodies are fine, their *placement* is not) |
| speed | median 3.45 · p95 206.48 · **max 100 416 m/s** |
| physics gate | speed viol 2009→616, accel viol 4529→0, max dev 6393.35 m, +23 corrections, **393 teleports marked** |
| per-frame spread | 238/355 within 120 m (67%), median 30.1 m, **max 1822 m** |
| **confidence on sane frames vs blown frames** | **0.421 vs 0.000 — perfect separation** |
| first blown frame | **156** (= 5.2 s) |

**Root cause, confirmed by eye on cropped frame 260:** the fan zooms in until only the goal line,
part of the six-yard box and the goal frame remain — no penalty spot, no D-arc, no far touchline.
PnLCalib has nothing to solve with, returns nothing, and the calibrator carries the *pre-zoom*
homography forward. Applied to zoomed pixels it maps feet near the wrong horizon, where
un-projection diverges — the same geometry as #127's "a pixel near the horizon is metres", one
order of magnitude worse. The physics gate did its job (it marked 393 teleports and killed every
accel violation); it cannot invent a placement the calibration never had.

**Why this is not a bug to fix but a property to state.** #119's one-camera result is
*broadcast-specific*: a tripod camera only pans and tilts, so one camera exists and the Colombia
clip solves to 0.0003 px. A handheld phone translates *and* zooms, so one camera genuinely does not
exist — this clip's rigid fit reads `realizable: False` / 142 px. A monocular per-frame calibrator
cannot recover a pitch it cannot see. The honest usable window is **frames 0–155** (~5 s), which
calibrates cleanly and would reconstruct; whether 5 s is worth a render is the user's call, not
mine.

---


## #129 — the one fitted camera, A/B'd on the pod (2026-08-05)

`scripts/pod_129_ab.sh` ran `pod_finish_batch.sh` twice on `tn2gfx13mxu5c6`, same clip, same
`sideline` framing, `RIGID_CAMERA` the only difference. The two reconstructions came out
**identical to the digit** — 24 subjects, 60/60 calibration frames measured / 0 carried,
confidence 0.540 / 0.452 / 0.645, physics 12→0 speed and 1037→0 accel violations — so anything
that differs downstream is the camera and nothing else.

### What the scene actually carries, measured off both exported scenes

| | `RIGID_CAMERA=0` (control) | `RIGID_CAMERA=1` (#129) |
|---|---|---|
| focal fx = fy | **772.0 px** | **4169.3 px** |
| resolution | 1280×720 (the *render* size) | 1920×1080 (the *clip* size) |
| principal point | (640, 360) — dead centre | (960, 540) |
| `raw_frame_aligned` | `False` | `True` |
| root spread | 34.0 × 40.5 m | 34.0 × 40.5 m |
| pelvis height | 0.88 m (0.68–1.00) | 0.88 m (0.68–1.00) |

The control's camera is not a bad measurement, it is **not a measurement at all**:
`controller.py:654` builds "a static broadcast camera replicated over the whole clip (fallback
render path)" from `standard_viewpoints(Viewpoint.BROADCAST)` when `camera_from_calibration`
refuses — and it refuses on this clip because the closest realizable pinhole is 525 px from the
stored homographies (#119). A focal at render resolution with the principal point exactly at the
frame centre is the signature of that placeholder.

Normalising for resolution, 772/1280 = 0.60 against 4169/1920 = 2.17 — **a factor of 3.6**, which
is #61's documented "players through a synthetic camera 3.9× too small", now measured on both
sides of the fix rather than argued.

### The limitation, which the same measurement exposes

**The players do not move.** Root spread and pelvis height are bit-identical between the arms,
because world position is computed from the foot point through the pitch homography *during
reconstruction*, and `apply_rigid_camera.py` replaces the calibration *after* it. So as wired,
#129 corrects the camera and everything baked through it — the stadium backdrop, the crowd tile,
the per-vertex body texture and light-from-clip all sample the source video via `scene.camera`
(`anim_export.py:421, 648, 650, 738, 800`) — but it does **not** re-place the subjects.

Both arms already sit at the v0 bar of 34×40 m, so this is not a defect being hidden; it is a
statement of what the fix does and does not reach. Grounding the players through the rigid camera
would mean applying the fit *before* pose grounding, which is a pipeline change, not a post-pass.

Artifacts: `out/cmp_129/side_by_side.mp4` and `out/cmp_129/beauty_f*.png`
(`scripts/ab_compare_render.sh`).

### Positions: swapping the calibrator is NOT the fix — measured 2026-08-06

After #129 closed, the obvious next step looked like "ground the players through the rigid camera
too, since PnLCalib's homographies are 525 px from any realizable pinhole". **Measured first, and
the premise is wrong.**

Projecting the real ByteTrack foot points (frames 0–59, n=1053) through both calibrations:

| | x range | y range | inside the 105×68 pitch |
|---|---|---|---|
| PnLCalib | −50.6 … −16.8 m | −11.3 … 29.8 m | 100 % |
| rigid fit (#119) | −50.8 … −17.6 m | −11.5 … 29.1 m | 100 % |

Per-point displacement between them: **median 0.31 m**, p90 0.67 m, max 1.80 m; 30 % move more
than half a metre.

The reason is in `apply_rigid_camera.py`'s own docstring and I had read past it: *"The
homographies still fit the visible paint (that is why the marks land), they just extrapolate to a
pitch 9157 px wide in a 1920 px image."* On the ground plane — where the paint is and where feet
are — a homography fitted to that paint is fine. It is only wrong when **extrapolated off the
plane**, which is what a camera is. So #129 fixed the camera, and there is little left for it to
fix in ground placement.

0.31 m is also not a number anything here can adjudicate: both calibrations place every player on
the pitch, and we have no ground truth that says which is righter. Building a `RigidFieldCalibrator`
to chase it would be motion without evidence.

### Poses: the joint/orientation ceilings were measured and never enforced (2026-08-06)

Chasing "positions and poses more accurate", the position half came back null (above). The pose
half did not. `scripts/motion_stats.py` on the 2026-08-05 pod scene:

```
speed_viol=0  accel_viol=0  teleport_intervals=0  hover_frac_mean=0%
orient_viol=11   joint_viol_samples=118
```

Root motion is clean; the **angular** rates are not. Worst per subject: joint 2212 deg/s against a
600 ceiling, orientation 4514 against 720. That is what "poses are inaccurate" looks like in a
render — bodies twitching and snapping faster than a human can move.

**Why nothing fixed it.** `config/physics.yaml` ships `joint.enabled: false` and
`orientation.enabled: false`, both commented "NOT YET BUILT (schema reserved)" — but the gates
*are* built (`core/correction/joint_kinematics.py`, `orientation.py`); with `enabled: false` they
take a documented **measure-only** path and emit nothing. The same file carries a `safe_new`
profile described as **"Recommended for pod runs"** which turns them on, and no pod script ever
set `PHYSICS_PROFILE`, so every run went out on `default` — "Ships-today defaults … no future
gates". The comment was stale, not the code.

**Measured before switching it on**, because this repo has been bitten by a clamp before: an
iterative MA on HMR yaw removed 90 % of the jitter *and* flattened 100°+ real turns.
`scripts/pose_gate_ab.py` reports the fix and its cost together:

| | before | after |
|---|---|---|
| joint violations | 118 | **0** |
| orientation violations | 11 | **0** |
| worst joint rate | 2212 deg/s | 600 (the cap) |
| worst orientation rate | 4514 deg/s | 720 (the cap) |
| root angular travel kept | — | **97.8 %** |
| body-joint angular travel kept | — | **98.5 %** |

Removing every violation costs 1.5–2.2 % of the real angular motion, so this is a clamp landing on
noise, not the yaw low-pass eating turns. `VIDEO_PHYSICS_PROFILE_DEFAULT=safe_new` is now the pod
default; `PHYSICS_PROFILE=` still overrides.

**The obvious risk, checked before the pod run rather than after.** A joint clamp could buy
smoothness by introducing foot slide. `scripts/bench_subject_steadiness.py` over the two scenes:
displacement **0.0000 m** (median, p95 and max), frame-to-frame step **0.0527 m in both**. The
gates touch `body_pose` and `global_orient` only and never the translation, so they cannot move a
player at all — the smoothing is free of that trade.

**One measurement trap fixed on the way.** `motion_stats.py` counted violations with a strict `>`,
so a fully-clamped scene still read "89 joint violations" — every one of them the clamp's own
output at 600.0000000000017 against a 600.0 limit. It now compares against the limit plus one part
in a million. Without that, the next person to enable these gates would have concluded they do not
work.
