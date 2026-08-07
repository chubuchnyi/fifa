# Research ledger — what we validated, what came out, what is left

**2026-08-07.** Six research reports in `docs/research/`, five findings docs in `docs/findings/`,
and ADR-0012 now hold roughly 2 200 lines of measurement between them. This file is the index over
the *verdicts*: for every thread we opened, what we asked, what the measurement said, and what is
still owed.

It exists because the evidence had spread far enough that the same question was being asked twice,
and because two conclusions currently quoted in `STATUS.md` were superseded by their own findings
doc without the summary being updated (§5).

**Vocabulary.** **VALIDATED** — measured, and the result supports the claim. **REFUTED** — measured,
and it contradicts the claim, *including our own prior claims*. **NULL** — measured, and the effect
is not there. **NOT-RUN** — built, or planned, and never measured. **ASSERTED** — stated in a doc
with no number behind it; §4 lists these on purpose.

**What this does not duplicate.** ADR-0012 owns everything we *declined* from the incoming briefs,
with its own three tiers and re-open conditions. Rows here point at it rather than restating it.
Per-item detail lives in the findings docs; this is the map, not the territory.

---

## 0. The score

| | count | where |
|---|---|---|
| Research recommendations measured (R1–R10) | **10 of 10** | ADR-0012, `docs/research/football-3d-response.md:103` |
| — shipped | 4 (R4, R6, R7-metric, R8, R9) | |
| — rejected on our own measurement | 4 (R1, R3-edges, R5, R10) | |
| — partial / shipped-rescoped | 2 (R2, R3-salvage) | |
| Runnable evidence probes in `scripts/` | **19** of 66 scripts | `bench_*`, `track_quality`, `motion_stats`, `identity_*`, `pose_gate_ab`, `mutate_*`, `check_*` |
| Physics gates built | **~15** | `config/physics.yaml` |
| — on by the shipped default (`safe_new`) | **3** (`joint`, `orientation`, `foot_floor`) | verified 2026-08-07 |
| — of those, with a paired before/after measurement | **2** (`joint`, `orientation`) | `scripts/pose_gate_ab.py` |
| Test suite | **1168 passed / 19 skipped in 20.25 s** | re-measured 2026-08-07 (was 1125/14/71 s on 08-02) |
| Lint backlog | **130** (74 E501 · 42 E702 · tail) | re-measured 2026-08-07 (was 152 on 08-01) |

---

## 1. Validated — safe to build on

### 1.1 Camera and calibration

| Claim | The number | Repro |
|---|---|---|
| One camera beats 60 free homographies (#119) | steadiness **0.55 vs 4.08 px** against 8.55 px of real motion; f **4169 px @1920×1080**; closed by the user's eye 08-03 | `tests/e2e/test_golden_real_camera.py` |
| The rendered camera was 3.6× too small (#61/#129) | measured fit 4169.3 px vs invented fallback **772 px**; normalised **3.6×**; user: «стало лучше» | `scripts/bench_rigid_camera.py` |
| Camera propagation removes real swim (R2) | swim median **0.119 m**, carry removes **92 %**; E2E on the clip **91.3 %** | `scripts/bench_camera_swim.py` |
| Confidence must divide by residual DOF, not observation count (#105) | a metres-wrong 4-point fit scored **0.9999999999999791**; Spearman −0.059 → **−0.358** | `scripts/bench_calib_confidence.py` |
| A projection round-trip alone proves nothing (R6) | a mirrored convention closes to **1e-9** while putting a world point **40–66 m** out; 5-way mutation-checked | `scripts/mutate_projection_sign.py` |
| Grounding through a carried homography moves a foot metres (#136·1) | **9.39 / 7.92 / 10.80 m** on the committed real calibration — and that is the *gentle* broadcast case | `tests/unit/test_calibration_confidence_gate.py` |

### 1.2 Identity and association

| Claim | The number | Where |
|---|---|---|
| Kit colour sampled from the first 8 frames makes a mid-track swap undetectable (#132) | tracks carrying two humans **9 → 0**; stitcher **1 → 14** merges | occlusion doc §kit-split |
| The split costs no render churn | **21 vs 21** subjects, **848** subject-frames both arms, 0 short-lived | `scripts/ab_subject_stability.py` |
| Identity churn is the dominant failure, not pose | **60 spurious births/deaths** per 236 frames vs ~12 fused meshes — **5×** | physics doc + occlusion doc |
| 96 % of those events have an unclaimed detection nearby | median **6–23 px** from the extrapolation | `scripts/identity_failure_kind.py` |
| The detector was not the ceiling | score 0.30 → 0.10: detection-miss share **30 % → 4 %** | same |
| …but ByteTrack cannot use the extra boxes | 26 → **28** events. The information is present and unusable | same |
| Pose fusion **does** happen, and box IoU cannot find it | **2 / 40** pairs fused, both at cover 0.41/0.49, box IoU **0.126**; ~**12 per 236 frames** | occlusion doc §A2-redone — **supersedes the 08-05 verdict, see §5.1** |

### 1.3 Physics and motion

| Claim | The number | Repro |
|---|---|---|
| Angular rates violated human ceilings, and the gates fix it without eating real motion | joint violations **118 → 0** (worst 2212 → 600 °/s), orientation **11 → 0** (4514 → 720); **98.5 %** of body-joint and **97.8 %** of root angular travel kept; subjects moved **0.0000 m** | `scripts/pose_gate_ab.py`, `bench_subject_steadiness.py` |
| Passed the user's eye on a pod A/B | «стало лучше» (2026-08-06) | #133 |
| Footballers never hyperextend, so joint limits buy nothing (R5) | **0.0 %** of 1008 subject-frames past straight; knee min **+11.6°** | `scripts/bench_joint_limits.py` |
| Sparse FK sampling synthesised fake stances | cap 30 → 240 cut foot slide **15.4 m → 0.3 m (98 %)** | ADR-0012 Tier 1 |
| Contact-lock + momentum (Path 1) works on a real pod E2E | slide **5.62 → 0.99 m (−82 %)**; gravity dev **10.6 → 7.6 m/s²**; joint clamps 127 → 0 | `scripts/physics_diagnose.py` |

### 1.4 Correctness criteria (#135)

| Claim | The number | Repro |
|---|---|---|
| `imputed` is mechanically a frozen mannequin | **exactly 0.00 rad** of limb travel while the root coasts **1.68–3.66 m**, every imputed run in every track | `scripts/track_quality.py --explain-imputed` |
| `interpolated` is a different animal | 5.07–7.27 rad of real limb motion on comparable runs | same |
| The criteria match the user's own eye | **19/20** on 24 hand-labelled tracks | `--labels docs/findings/track-labels-2026-08-07.json` |
| The scene has no vertical DOF at all | largest root-Z excursion **0.082 m** (reference clip), **0.234 m** (fan clip); a jump moves a pelvis ~0.4 m | `track_quality.py` |
| The criteria transfer to a different camera, format, stadium and kit | on the fan clip they immediately named its dominant defect (one whip breaks six identities) | criteria doc §8 |

### 1.5 Social footage (#136)

| Claim | The number | Repro |
|---|---|---|
| A whip-pan is not a shot cut, and a homography can tell them apart | camera moves **0.995–1.000** inlier ratio vs a real cut **0.025** — a 40× gap | `tests/unit/test_shot_cut_verify.py` on committed real frames |
| One crop rect for a whole phone clip measures nothing | grass-band centre moves **177 px**, height **354 px (59 %)**; 4 segments at 82–92 % grass vs one at 84.2 %; a tripod still collapses to **1** | `tests/unit/test_broadcast_crop_segments.py` |
| Refusing to place what was never measured is worth more than any smoothing | root spread **3079.7 → 58.4 × 86.4 m**; max speed **100 416 → 405 m/s**; off-pitch placements **344 → 0** | criteria doc §9, `out/vert136`, `out/vert137` |
| Two refusals removed 268 bad placements | the two were interpolation anchors; `_smooth_path` drags neighbours out with them | same |

### 1.6 Infrastructure

| Claim | The number | Where |
|---|---|---|
| The full real-backend chain runs on the local RTX 4080 | 48 frames in **75 s**, all five real backends, peak **24 %** of 16 GB VRAM | `docs/local-gpu-box.md` |
| The generative tail does not | SeedVR2 completes at **97.5 %** of the card and OOMs at a 95 % cap | same |
| OpenCV 5 migration is code-neutral (R9) | 4.11 → 5.0.0.93, **zero code changes**, calibration bit-identical | ADR-0012 |

---

## 2. Refuted or measured null — do not re-try without new evidence

The expensive half of the ledger. Each of these looked right on paper.

| Idea | What we expected | What it measured | Where |
|---|---|---|---|
| USAC / MAGSAC++ for calibration (R10) | a better robust estimator | ours **0.07–0.12 m** vs USAC **28–180 m**; fails **0/20** with zero outliers | ADR-0012 T1 |
| Fitting the two edges of a painted line (R3) | a scale cue from a regulated width | paint is **2.00 px** median thick — a single-lobe PSF, not a band | ADR-0012 T1 |
| Joint limits (R5) | pose nets exceed human ranges | **0.0 %** of frames do; invented frames are *safer* than measured ones | ADR-0012 T1 |
| Unifying the SMPL-X→world constants (R1) | duplicated constants | both are correct — two different source frames. Chasing it found a real **206 → 18 mm** bug next door | ADR-0012 T1 |
| Iterative moving-average on HMR yaw | remove jitter | removes 90 % of α **and flattens 100°+ real turns** | ADR-0012 T1 |
| More per-frame cues for auto-calibration (#117) | markings/stripes/undistortion buy focal accuracy | residual is **monotone** 0.85→1.64 px with no interior minimum; undistortion flat **0.27–0.33 px** | open-items |
| Lowering the tracker match threshold | recover the 6–23 px orphans | it thresholds `1 − IoU`, so lowering it *tightened* matching: 56 → 104 tracks, 66 → 162 events | physics doc |
| Raising it instead | fewer ids, less churn | the gain is entirely eaten by the kit-split: **26 vs 28** events post-stitch | physics doc |
| Feeding ByteTrack more detections | the boxes exist, so use them | **26 → 28** events. No | physics doc |
| Widening the stitch gates | recover merged ids | 36 → 34 ids across the whole sweep | occlusion doc |
| **Expansion IoU** (Deep-EIoU's geometric core) | inflate boxes so the 6–23 px orphans clear the 0.8 threshold | raw ids 56 → 43, merges 14 → 8, worst seam **25.7 → 9.5 px/f** — but the post-stitch count **stays in a 33–38 band** around the baseline 36. Work moves, identities do not. Past 1.6 it fuses distinct players (1.8 → 21 ids) | `scripts/bench_expansion_iou.py`, [occlusion-stack review §5](occlusion-stack-review-2026-08-07.md) |
| NMS / duplicate suppression | the structural blocker | duplicates are **~1 %** of detections; ids move at most one; merges go *down* | occlusion doc |
| Camera→world rotation as the orientation defect | the missing rotation explains it | **68 %/36 % against a raw 63 %/35 %** — noise | open-items |
| Grounding feet through the rigid fit instead of PnLCalib | better placement | both put **100 %** of 1053 real foot points on the pitch; median disagreement 0.31 m, **unadjudicable** | open-items |
| The rigid camera re-places players (#129) | one camera fixes geometry | roots **bit-identical** (34.0 × 40.5 m) — the fit lands after pose grounding | open-items |
| R3's line path on this clip (#108) | a measurable gain | post-R3 homographies **byte-identical**, max\|dH\| = 0.0 over 60 frames — the path self-disables | open-items |
| Calibration confidence as an off-pitch predictor (#136·4) | low confidence ⇒ bad placement | **anti**-predictive: 0/1299 below 0.5 landed off-pitch, 6/1339 above did | criteria doc §9 |
| The body-facing-vs-travel metric | detect wrong poses | withdrawn same day — it measured football, not the reconstruction | physics doc |

Two of the entries above are **refutations of our own earlier claims**, kept deliberately: the
threshold sweep whose semantics were backwards, and the identity budget that read "2× gap" when the
cast is ≥28 humans, an error the doc notes was "in the flattering direction".

---

## 3. Built and never judged

The honest gap. All of this exists in the tree, imports, and has never been measured on the real
clip or seen by the user.

### 3.1 Physics gates — 3 on, ~12 off

Verified against `config/physics.yaml` on 2026-08-07: the shipped `safe_new` profile enables
`foot_floor`, `joint` and `orientation`. Everything below is built and **off**.

| Gate | Ever measured? | Note |
|---|---|---|
| `collision` (capsule repulsion) | never on the real clip | one capsule per player, so limbs still interpenetrate; needs `momentum_smooth` |
| `pose_motion_sync` | never | knees/hips only — **no arm-swing term**, which is what the user asked for |
| `contact_probe`, `foot_plant`, `gravity_project` | never | |
| `orient_verticality`, `jerk_clamp`, `inertia_smooth`, `joint_smooth` | never | `joint_smooth`'s nearest precedent is the *rejected* yaw low-pass |
| `identity` (DBSCAN split + merge) | built 07-06, never A/B'd against the #132 kit-split baseline | |
| `full_realism` / `full_realism_collide` presets | **never run on the real clip** | one measurement unlocks eight gates at once |

`foot_floor` is on **without** the paired measurement the physics doc's own rule demands.

### 3.2 Everything else

| Item | State |
|---|---|
| **#132 in a novel-view render** | the kit-split and stitcher have never been seen in a render |
| **PromptHMR arms 2–4** (occlusion doc Stage B) | cancelled on evidence later withdrawn — see §5.1 |
| **#109 jersey OCR** | **0/23** usable crops; the fix (crop at native res through the calibration) is not built |
| **#127 layout gizmo** | fixed, verified to 0.0000 px, still awaiting the user's hands |
| **#207 physics gate re-render** | gate takes 22/999 violations to 0/0; the eye-check was never done |
| **R7's metric on real data** | the number that retires the 0.35–0.45 m envelope is from synthetic fields only |
| **Mask cue (#133)** | works and is weak — **28 → 24** events, 14 % against a 96 % ceiling; not default |

---

## 4. Claimed without a measurement

R-6 applied to our own notes. None of these is necessarily wrong; none has a number.

- **`kinematic` and `foot_floor` are on** with no cited before/after, breaking the physics doc's own
  rule that a gate ships only with both halves measured.
- **`collision`'s "known accel-spike tradeoff"** — described as documented; no figure, no run.
- **Six of the eight "what will bite" items** in the occlusion doc are explicitly headed *found by
  reading their code, not by running it*.
- **"9 two-kit tracks is a floor, not the count"** — generalised from one referee example.
- **The 94 % vs 76 % crossing correlation** — the doc disclaims causality itself.
- **"Two-pass seeding explains the weak mask cue"** — pure hypothesis, zero measurement, and it is
  the reason the remaining 82 % of that ceiling is being left alone.
- **#202's "target ~22 bodies"** — no cited ground truth.
- **#119's camera distance ±10 %** — inferred from f/dist ≈ 57, not independently measured.
- **The frame-236 cut appears as both 0.775 and 0.9051** in the occlusion doc; never reconciled.
- **Mid-pitch event counts appear as 60 / 78 / 66 / 28 / 26 / 24** across the physics doc under
  different filters. Any future comparison must state its baseline — the headline "60 in 8 seconds"
  is pre-stitch and is **not** the number the render sees (26).

---

## 5. Three things this survey found

### 5.1 "Pose fusion does not happen" is superseded by its own findings doc

`STATUS.md` #132 says: *«Pose half: not reproduced — SMPLest-X per-crop does not fuse on the clip's
hardest real overlap (79 %/90 % of each mesh on its own player)»*. The occlusion doc marks that
verdict **WITHDRAWN as under-powered** (line 398) and its 40-pair sweep concludes, verbatim, *"The
2026-08-05 verdict was wrong and this supersedes it. Fusion happens."*

What actually holds: fusion occurs on **2 of 40** stratified pairs, both at cover 0.41–0.49; below
cover 0.31 every pair is clean; above 0.40 the rate is **20 %**, i.e. ~**12 fused meshes per 236
frames**. It was missed because the whole investigation ranked candidates by **box IoU**, and the
fusing case has box IoU **0.126** — a small distant player passing *behind* a near one.

**The decision survives, the reason does not.** We still should not swap the pose model: the typical
contaminated crop is fine (80 % of the mesh on its own player) and identity churn is 5× more
frequent. But "the failure does not occur" is false, and the cancellation of PromptHMR arms 2–4
rested on the withdrawn verdict, so that cancellation is currently unsupported. Either re-justify it
against the 40-pair numbers or treat the ~61 heavily-covered pair-instances per shot as an open
defect.

### 5.2 Ground truth arrived and nobody noticed

`WorldPose/` in the repo root holds **673 MB** of FIFA World Cup 2022 ground truth: `_poses-dev.7z`
(621 MB — per-player `betas`, `global_orients`, `body_poses`, `transl`) and `_cameras-dev.7z`
(**89 clips**, per-frame `K`, `R`, `t`, distortion). Every doc that touches ground truth still says
poses and cameras are *pending*.

The **video is not included** — FIFA requires a separate agreement (`WorldPose/_README.md`), so
scoring our pipeline end-to-end is still blocked. But the poses and cameras alone are enough to
answer, locally and on CPU, several questions that are currently **asserted constants**:

- Is our vertical range plausible? Ours: **0.082 m** / **0.234 m** peak root-Z. What does a real
  footballer's pelvis do? This settles П5 and tells us whether `gravity_project` would ever fire.
- Are the kinematic ceilings right? `10.5 m/s` and `8 m/s²` are guessed. GT gives the distribution.
- Is П4's 0.5 m twin threshold ever legitimate? How close do real players actually get?
- R5's hyperextension question, on humans rather than on our own network's output.

It also converts ADR-0012's factor-graph re-open condition from "blocked on ground truth" to
"blocked on the FIFA video agreement" — a narrower and differently-shaped blocker.

### 5.3 #103 is on no board at all

The OpenCV 5 migration (R9) shipped with one unresolved consequence: the YUV→RGB matrix changed
(BT.601 → BT.709), shifting **92 % of the frame**, and kit colours and OCR were never re-measured
against it. It is tracked as **#103** in `docs/roadmap.md`, and appears **zero times** in
`docs/STATUS.md` — neither open nor closed.

This is not bookkeeping. #135's П7 reads kit colour off the video pixels and the user's own words
were *«я просто брал цвет игроков в реконструкции за истину, но похоже, что там тоже ошибки»*. An
unmeasured 92 %-of-frame colour shift sitting under the kit reader is a candidate cause.

---

## 6. What remains, ordered by what it would visibly buy

1. **Stitch on the handover criterion (П3) and drop the mannequin half.** Closes the user's entire
   stitch list — three pairs become three players. Needs no new model: the signal is already in
   `provenance` plus the roots. *(criteria doc §6.1–6.2)*
2. **Run `full_realism` on the real clip and look at it.** One pod run measures eight gates that are
   currently built, off and unjudged, including `collision` — and the fan scene says collision is
   real: **32 twin pairs, 19 of them with both tracks measured**, i.e. two solid bodies inside 0.5 m.
   On the reference clip every twin was a parked phantom; on social footage it is not.
3. **Give the scene a vertical DOF (П5).** Until then «17 не прыгнул» cannot be fixed and no
   airborne moment can exist. WorldPose now says what the range should be (§5.2).
4. **Finish eye-review item (b).** Half of it is answered — the limb-activity metric exists and is
   `track_quality.py --explain-imputed`, and its answer is that zero-articulation frames are exactly
   the `imputed` runs. The open half is whether the mid-pitch identity events line up with an
   activity handover, which would make it a *stitching cue* and not just a detector.
5. **Re-measure kit colour under BT.709 (#103)** — cheap, and it sits under П7 and the user's own
   doubt about team colours. Related: the fan pod run assigned **`team=None` to 23 of 27 subjects**
   (measured 2026-08-07 in `out/vert137/run.log`) — team assignment silently produced nothing and no
   gate noticed.
6. **Re-justify or re-open PromptHMR arms 2–4** (§5.1), or fix the ~61 heavily-covered
   pair-instances per shot directly.
7. **#126** — confidence is still attached to a matrix it does not describe: homographies differ
   **0.76 m median / 3.67 m max** while confidences are bit-identical. #136's fix 1 now *gates* on
   that confidence, which makes the mismatch load-bearing rather than cosmetic.
8. **#108** — put a log line at `_lines_agree` before spending another run on R3.
9. **#109** — 0/23 jersey crops; the fix is known and unbuilt.

---

## 7. Ceilings — things no code fixes

- **A handheld phone has no single camera.** `fit_rigid_camera` assumes one focal and one optical
  centre; a phone translates *and* zooms. Measured `realizable: False`, **142 px** (vs the broadcast
  clip's 0.0003 px). Social clips therefore yield per-frame homographies but **no camera path to
  render a novel view from**. The architecture that follows is *segment → per-segment crop and
  calibrate → keep only measured frames → refuse the rest* — a set of usable windows, not one
  continuous reconstruction.
- **Once a zoom leaves no landmarks the plane is undetermined.** On the fan clip that is **153 of 355
  frames**. The pipeline now refuses them instead of placing them kilometres out, which is the
  correct behaviour and still means 22 of 27 tracks come back `OK_UNMEASURED`.
- **The target clip contains no hard occlusion.** Measured: the "hardest" pair fills **0.59** of its
  box against 0.30–0.45 unoccluded. Every pose verdict on this clip is clip-bound until we obtain
  footage that actually contains one.
- **No metric separates a wrong pose from unusual play.** The one attempt was withdrawn as invalid;
  ~86 px of player height may cap what is recoverable at all.
- **The physics stack is kinematic clamps, not a simulation.** Nothing enforces momentum, balance or
  ground reaction, and nothing measures how often a clamped-but-impossible pose survives.

---

**Related:** [`track-correctness-criteria-2026-08-07.md`](track-correctness-criteria-2026-08-07.md) ·
[`occlusion-pose-research-2026-08-04.md`](occlusion-pose-research-2026-08-04.md) ·
[`human-physics-requirement-2026-08-06.md`](human-physics-requirement-2026-08-06.md) ·
[`open-items-2026-08-01.md`](open-items-2026-08-01.md) ·
[`eye-review-2026-08-06.md`](eye-review-2026-08-06.md) ·
[`../adr/0012-rejected-approaches-log.md`](../adr/0012-rejected-approaches-log.md)
