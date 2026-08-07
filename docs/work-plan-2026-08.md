# Work plan — August 2026, and the running report against it

**What this is.** The ordered backlog that came out of the 2026-08-07 research sweep, plus a report
appended as each item lands. One file so nothing is lost between sessions: the plan and the result
sit next to each other, and an item is only closed by a number.

**Where it came from.** [`findings/research-ledger-2026-08-07.md`](findings/research-ledger-2026-08-07.md)
(what we validated / refuted / never judged, across every thread) and
[`findings/occlusion-stack-review-2026-08-07.md`](findings/occlusion-stack-review-2026-08-07.md)
(the 2026 occlusion stack measured against our own pixel scale). `STATUS.md` stays the board;
this is the queue and the log.

**How an item closes.** A measurement, on real data, written into §Report below with the number —
or a rendered/scrubbed frame the user has judged. Not "implemented". Not "tests pass".

**Where it runs.** `demorig` (RTX 4080, free, `docs/local-gpu-box.md`) for anything needing a GPU up
to the generative tail. The pod only for Blender and Wan/SeedVR2.

---

## The queue

Ordered by what each item unblocks, not by size. The reason column is the measured fact that put it
here — if that fact is wrong, the item leaves the queue.

| # | Item | Why it is here (measured) | Verified by |
|---|---|---|---|
| **W1** | **Detector input resolution** | RF-DETR resizes the frame to a `res × res` **square** and its default is **560**, which we never set. A measured 28 × 72 px player reaches the net as **14 × 21 px** on the portrait clip. Every cue downstream is capped by that. The SoccerTrack-2025 winner's small-target work moved HOTA **0.380 → 0.491** — larger than any association change in the review | players found per frame at 0.3 and 0.1, box sizes, s/frame, on both clips |
| **W2** | **#137 — `team=None` on 23 of 27** | Kit colour is the one appearance signal that survives at 28 px, and `StitchConfig.require_same_team` treats `None` as a **wildcard**, so a null label silently removes the stitcher's only working constraint. Clustering works locally on both clips (broadcast **56/56**, fan **A=11 / B=9** with measured white/red) — so this is the pod path, not a scale wall | a full 355-frame run that assigns teams, or the line that drops them |
| **W3** | **Stitch on the handover criterion (П3)** | Closes the user's entire stitch list — three pairs become three players — and needs no new model: the signal is already in `provenance` plus the roots. Must be an **assignment** (one partner each, nearest first) or t20 gets swept into t25's merge at 2.09 m | `track_quality.py` against the 24 eye labels: 19/20 → 20/21 |
| **W4** | **Drop the mannequin half of every merged pair (П2)** | `imputed` is mechanically a frozen mannequin — **0.00 rad** of limb travel while the root coasts 1.68–3.66 m. The duplicate half is what the eye calls a phantom; the man nobody measured stays, marked (R-6) | phantom count in `track_quality.py`, then the user's eye in `/app` |
| **W5** | **Selective mask propagation** | arXiv 2606.13033 is **training-free**, treats base tracker and VOS as black boxes, and its dispatch signal is the **assignment margin in the Hungarian cost matrix** — implementable over our own matrix at the same seam as the McByte cue, no weights. Answers our measured **686 s GPU / 4.2 h CPU** per pass by firing Cutie only where the tracker is unsure | mid-pitch identity events vs the 28 → 24 the always-on cue bought, and the propagation cost |
| **W6** | **GTA-Link** | **MIT**, offline, consumes and returns MOT-format text. Its Connector is appearance-clustered global re-association; we measured our own stitcher's *gates* are not the binding constraint, so the clustering is the genuinely new part | identity count and seam speeds against the 36 / 14-merge baseline |
| **W7** | **`full_realism` on the real clip** | **3 physics gates on, ~12 built and off**, and of the 3 only `joint` and `orientation` have a paired before/after. One run measures eight at once — including `collision`, which the fan scene says is real: **32 twin pairs, 19 of them with both tracks measured** | `motion_stats.py` + the user's eye on an A/B |
| **W8** | **Vertical DOF (П5)** | Largest root-Z excursion in a whole scene is **0.082 m** (broadcast) / **0.234 m** (fan); a jump moves a pelvis ~0.4 m. Until this exists «17 не прыгнул» cannot be fixed. WorldPose GT now says what the range should be | root-Z distribution vs WorldPose GT |
| **W9** | **WorldPose GT constants** | **673 MB on disk all along** — poses + cameras for 89 clips; every doc still said "pending". Video stays FIFA-gated, but the poses alone settle constants we currently guess: kinematic ceilings (10.5 m/s, 8 m/s²), plausible root-Z, and whether П4's 0.5 m twin threshold is ever legitimate | the measured distributions, replacing the guessed constants |
| **W10** | **#103 — kit colour under BT.709** | OpenCV 5 changed YUV→RGB, shifting **92 % of the frame**, and kit colour and OCR were never re-measured. Sits directly under П7 and under the user's own doubt about team colours | the kit scan under both matrices on the same clip |
| **W11** | **Eye-review (b), second half** | The limb-activity metric exists and is `track_quality.py --explain-imputed`; its answer is that zero-articulation frames are exactly the `imputed` runs. The open half — do the mid-pitch identity events line up with an activity handover — would make it a **stitching cue**, not just a detector | events vs activity handovers |
| **W12** | **#126 / #108 / #109** | #126: confidence is bit-identical while homographies differ **0.76 m median / 3.67 m max** — and #136's fix 1 now *gates* on that confidence, so the mismatch is load-bearing. #108: R3's line path is byte-identical to pre-R3, needs a log line at `_lines_agree` before another run. #109: **0/23** usable jersey crops | each has its own probe |

**Not in the queue, and why:** NOOUGAT and SAM2MOT (no code — NOOUGAT's numbers verified exactly,
SAM2MOT's repo is 6 files); BMPv2+ (its authors publish that the refinement loop *"pronounces the
error"* on small instances, and it is GPL-3.0); Sapiens (trained on ≥300–384 px people, **11×**
ours; v1 CC BY-NC, v2 forbids re-identification); Multi-HMR 2 and PromptHMR (non-commercial, and
neither emits SMPL-X natively). Full reasoning: occlusion-stack review §6, §8.

---

## Report

Appended as items land. Each entry states what was run, on what, and the number — including when
the number says the idea was wrong.

### W1 — detector input resolution → **the premise was right, the payoff is not. Default stays 560.**

Run on `demorig` 2026-08-07, RF-DETR at four input squares over 60 frames of each clip,
`scripts/bench_detector_resolution.py`:

| | | players/frame @0.3 | @0.1 | median box h | p05 h | s/frame |
|---|---|---|---|---|---|---|
| **Broadcast** 1920×1080 | 560 | **19.1** | 28.4 | 86.6 | 65.2 | 0.019 |
| | 728 | 19.5 | 29.5 | 86.4 | 65.8 | 0.021 |
| | 896 | **19.9** | 31.0 | 86.1 | 64.1 | 0.029 |
| | 1064 | 19.8 | 33.4 | 86.2 | 65.0 | 0.044 |
| **Fan clip** 1080×1920 | 560 | **15.1** | 29.4 | 76.5 | 59.5 | 0.020 |
| | 728 | 15.4 | 32.1 | 74.9 | 59.9 | 0.021 |
| | 896 | 15.7 | 33.8 | 73.3 | 58.0 | 0.030 |
| | 1064 | **16.0** | 34.7 | 72.4 | 56.5 | 0.041 |

**At the threshold the pipeline actually uses, quadrupling the pixel count buys ~5 %.** Broadcast
19.1 → 19.9 (+0.9 players/frame) at **1.52×** the time, and 1064 is no better than 896. Fan clip
15.1 → 16.0 (+0.9) at **2.06×**. That is not a lever; it is a rounding error with a compute bill.

At the 0.1 floor the gain is real — **+18 %** on both clips (28.4 → 33.4 and 29.4 → 34.7) — and
that is exactly the band we already measured as worthless downstream: doubling the detections at
0.1 moved identity churn **26 → 28**. More low-confidence boxes is a thing we have tried.

Two details worth keeping. **Median box height does not grow** (86.6 → 86.2; on the fan clip it
*falls* 76.5 → 72.4, and p05 falls 59.5 → 56.5) — so the extra detections are *smaller, more
distant* people, which is the mechanism working as advertised; there just are not many of them.
And the portrait clip, which is squashed 0.29× vertically at 560 and should have gained most, gains
the same +0.9 as the broadcast clip.

**Verdict: hypothesis refuted at the working threshold.** I claimed last session that this was "the
pixel-level attack neither of us proposed" and implied it was the large one. It is not. The
mechanism is real and the knob is worth having — `RFDETRBackend.resolution` stays, `None` keeps 560
so nothing changes silently, and a clip with genuinely distant subjects can raise it — but the
default does not move on a 5 % detection gain that our own prior measurement says does not convert
into identities.

**And it separates two things I had conflated.** GTATrack's small-target jump (HOTA 0.380 → 0.491)
came from **pseudo-labelling** — retraining the detector on small targets — with 1280 px input as
the carrier, not from the resolution alone. This measurement is the resolution alone, and it says
the resolution alone is not where their win came from. Retraining the detector is a different item
with a different cost, and it is not in this queue.

*Closed 2026-08-07. Cost: ~4 min of a free box.*

### W2 — #137, `team=None` on 23 of 27 → **narrowed to "assigned, then lost"**

Three measurements, cheapest first.

**1. Clustering is not the failure.** The same code on the same clips, locally:

| | tracklets | team split | measured kit colours |
|---|---|---|---|
| Broadcast, 236 f | 56 | **A 27 / B 29** | yellow `(0.76, 0.69, 0.28)` · blue `(0.35, 0.54, 0.63)` |
| Fan clip, 38 f | 20 | **A 11 / B 9** | white `(0.82, 0.89, 0.73)` · red `(0.76, 0.39, 0.26)` |

Both correct against the frames. So this is not the 28 px wall, and not k-means.

**2. The assembly path is not the failure either.** A full local CLI run on the fan clip
(`--frames 38 --coherence --export gltf`) carries the labels all the way into `scene.json`:
**A 11 / B 9**.

**3. The labels are assigned and then lost.** In `out/vert137/scene.json` the `teams` block holds
**both** teams with computed colours — `A (0.68, 0.75, 0.44)`, `B (0.71, 0.22, 0.52)` — and a team
colour is the mean HSV of *its members*, so members existed at `_assign_teams` time. Yet **no
subject carries `B`** and only 4 carry `A`.

**4. Root cause, and my step-3 inference was wrong.** I read `role=player team=None` in the pod log
as *both* defaults of `assemble_scene`'s missing-tracklet branch, and concluded `motions` carried
ids absent from `tracks.tracklets`. It does not. `role=player` was simply the measured class.

The pod log's own line 8 has it: **`== identity: --identity ON (GTA split + merge)`** — a flag none
of my local runs passed. Three constructors in `core/orchestration/identity.py` —
`_split_tracklet`, `_truncate_tracklet`, `_merge_tracklets` — each emit `team_id=None` under the
comment *"let downstream re-assign on the clean identity"*, and **there is no downstream**:
`ByteTrackTracker._assign_teams` runs inside the tracker, which is *before* the gate in
`ReconstructionPipeline.run`. Every tracklet the gate touched lost its team permanently.

Confirmed by exclusion on the same clip and the same 355 frames:

| | subjects | teams |
|---|---|---|
| identity gate **off** | 68 | **A 35 / B 33** — every one labelled |
| identity gate **on** (the pod run) | 27 | **23 with no team at all** |

**Fixed** (`6f4c270`). `_restore_team_labels` re-assigns each blanked tracklet against the ones
that *kept* a label — nearest centroid, the same cosine metric the split stage uses — rather than
re-clustering, so the result stays consistent with the `teams` block and with every id the tracker
already stamped. A split's two halves are re-derived **independently** and may land on different
teams, which is the whole point of splitting there. With nothing to anchor against it leaves
`None`: an unlabelled subject beats an invented team (R-6).

And the thing that let this hide for a whole pod run is fixed too — the controller now prints any
posed subject with no tracklet, because `team_id=None` + `role=PLAYER` were **defaults**
indistinguishable from measurements in the output.

6 tests, **mutation-checked**: 4 of them fail with the fix disabled and pass with it. Suite
**1175 passed / 19 skipped**.

**5. Verified end to end, on the clip and at the length that produced the defect.** Same
`vert_crop.mp4`, same 355 frames, `--identity --coherence`, i.e. the exact configuration that gave
the pod 23 unlabelled subjects out of 27:

| | subjects | team split | unlabelled |
|---|---|---|---|
| pod, before | 27 | A 4 | **23** |
| local, gate off | 68 | A 35 / B 33 | 0 |
| **local, gate on, fixed** | **29** | **A 11 / B 18** | **0** |

No `unmatched` line was printed either, which independently confirms the step-3 inference was
wrong: no posed subject lacks a tracklet.

*Closed 2026-08-07. The whole diagnosis and fix cost CPU time and no GPU at all — the pod log's
own line 8 was the evidence.*

Two things fell out on the way that are worth their own line:

* **`out/vert_full/vert_crop.mp4` is 1920 × 1080, not 1080 × 608.** `broadcast_crop` measured the
  grass band correctly, and the ffmpeg step then **upscaled 1.78×**. So the pod pipeline saw players
  at roughly **50 × 128 px**, not the 28 × 72 measured on the source. The occlusion review's §2
  numbers describe the *source*; the pod chain works on an upscaled copy. That does not add
  information, but it does mean the pod run is not the 28 px regime.
* A silent `None` team is not cosmetic: `StitchConfig.require_same_team` treats `None` as a
  **wildcard**, so every one of those 23 subjects is stitchable to anyone.

### W9 — WorldPose GT constants → **two of our four constants are wrong, in opposite directions**

`scripts/worldpose_constants.py`, run over all **89 clips** of WorldPose (1080p 50 Hz World Cup
broadcast, up to 22 players each): **2.4 M frame-to-frame root samples over 152 392 frames**, and
**3.8 M player-pairs**. Ground truth for our exact problem, on CPU, with no video — the FIFA gate
covers the footage, not the poses. Extracted from the 7z that had been sitting on disk since
2026-08-07.

| | p50 | p99 | p99.9 | max | our constant | verdict |
|---|---|---|---|---|---|---|
| horizontal speed | 2.04 | 7.25 | 8.68 | **9.74 m/s** | 10.5 m/s | **never fires** |
| horizontal accel, raw | 0.89 | 9.35 | 23.38 | 285.5 m/s² | 8 m/s² | — (jitter) |
| horizontal accel, 100 ms | 0.92 | **8.35** | 15.21 | 49.7 m/s² | 8 m/s² | **clips real football** |
| root range, whole clip | **0.23** | 0.97 | 1.13 | 1.52 m | our best scene: 0.234 m | **we are flat** |
| root rise, 0.5 s window | 0.12 | **0.67** | 0.84 | 0.96 m | "a jump is ~0.4 m" | **we understate it** |
| nearest neighbour | 1.80 | 10.73 | 23.15 | 36.7 m | twin radius 0.5 m | **validated** |

**1. The speed ceiling is a sanity net, not a gate.** The fastest root in 2.4 M samples of World
Cup football is **9.74 m/s**; p99.9 is 8.68. Our 10.5 m/s therefore never fires on real motion —
which is not a bug, but it does mean the gate has never rejected anything and is not evidence of
anything. A gate that could actually catch a reconstruction error sits near 9.8–10 m/s.

**2. The acceleration ceiling is too tight, and this one costs us.** On a 100 ms average — the
scale our own poses arrive at, so the fair comparison — real football reaches p99 **8.35 m/s²**.
Our gate is **8**. It therefore fires on roughly the top 1.5 % of *genuine* accelerations: every
hard cut, every sprint start. p99.9 is 15.2. Raw 50 Hz double differences peak at 285 m/s², and
smoothing takes that to 49.7 — so the raw tail is GT jitter and must not be used to set a
threshold, which is exactly why both rows are printed.

**3. The vertical finding is the strongest one.** A single real player's root moves **0.23 m**
vertically over a clip at the *median*, and 0.97 m at p99. The largest excursion in an entire
reconstructed scene of ours is **0.234 m** — one median player. And inside any half-second, a real
root rises p99 **0.67 m**, max 0.96 m, so the "~0.4 m for a jump" we have been citing is
conservative by nearly half. W8 was already in the queue on our own numbers; this says how far
short we are, in GT units.

**4. П4's 0.5 m twin radius is validated as a defect detector.** Two real pelvises come within
0.5 m in **0.018 %** of player-pairs — 703 of 3.8 M. It happens (2.19 % of frames contain some
such pair), so the criterion must not be a hard error. But our fan scene reported **32 twin pairs,
19 with both tracks measured**, which is orders of magnitude above the GT rate. Those are
reconstruction artefacts, not football.

*Caveat, so the numbers are not over-read:* `transl` is SMPL's root offset, so world pelvis is
`transl + R·J₀(β)`; ranges and pair distances are therefore accurate to the few-cm spread of J₀
across players, not exact. Nothing above turns on centimetres.

*Closed 2026-08-07. CPU only, ~3 min, no GPU and no video.*
