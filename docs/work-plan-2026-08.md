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
| **W1** ✅ refuted | **Detector input resolution** | RF-DETR resizes the frame to a `res × res` **square** and its default is **560**, which we never set. A measured 28 × 72 px player reaches the net as **14 × 21 px** on the portrait clip. Every cue downstream is capped by that. The SoccerTrack-2025 winner's small-target work moved HOTA **0.380 → 0.491** — larger than any association change in the review | players found per frame at 0.3 and 0.1, box sizes, s/frame, on both clips |
| **W2** ✅ fixed | **#137 — `team=None` on 23 of 27** | Kit colour is the one appearance signal that survives at 28 px, and `StitchConfig.require_same_team` treats `None` as a **wildcard**, so a null label silently removes the stitcher's only working constraint. Clustering works locally on both clips (broadcast **56/56**, fan **A=11 / B=9** with measured white/red) — so this is the pod path, not a scale wall | a full 355-frame run that assigns teams, or the line that drops them |
| **W3** ✅ refuted → merged into W4 | **Stitch on the handover criterion (П3)** | Closes the user's entire stitch list — three pairs become three players — and needs no new model: the signal is already in `provenance` plus the roots. Must be an **assignment** (one partner each, nearest first) or t20 gets swept into t25's merge at 2.09 m | `track_quality.py` against the 24 eye labels: 19/20 → 20/21 |
| **W4** ✅ built, OFF by default — needs the eye | **Drop the mannequin half of every merged pair (П2)** | `imputed` is mechanically a frozen mannequin — **0.00 rad** of limb travel while the root coasts 1.68–3.66 m. The duplicate half is what the eye calls a phantom; the man nobody measured stays, marked (R-6) | phantom count in `track_quality.py`, then the user's eye in `/app` |
| **W5** ✅ premise confirmed, not built | **Selective mask propagation** | arXiv 2606.13033 is **training-free**, treats base tracker and VOS as black boxes, and its dispatch signal is the **assignment margin in the Hungarian cost matrix** — implementable over our own matrix at the same seam as the McByte cue, no weights. Answers our measured **686 s GPU / 4.2 h CPU** per pass by firing Cutie only where the tracker is unsure | mid-pitch identity events vs the 28 → 24 the always-on cue bought, and the propagation cost |
| **W6** | **GTA-Link** | **MIT**, offline, consumes and returns MOT-format text. Its Connector is appearance-clustered global re-association; we measured our own stitcher's *gates* are not the binding constraint, so the clustering is the genuinely new part | identity count and seam speeds against the 36 / 14-merge baseline |
| **W7** | **`full_realism` on the real clip** | **3 physics gates on, ~12 built and off**, and of the 3 only `joint` and `orientation` have a paired before/after. One run measures eight at once — including `collision`, which the fan scene says is real: **32 twin pairs, 19 of them with both tracks measured** | `motion_stats.py` + the user's eye on an A/B |
| **W8** | **Vertical DOF (П5)** | Largest root-Z excursion in a whole scene is **0.082 m** (broadcast) / **0.234 m** (fan); a jump moves a pelvis ~0.4 m. Until this exists «17 не прыгнул» cannot be fixed. WorldPose GT now says what the range should be | root-Z distribution vs WorldPose GT |
| **W9** ✅ measured | **WorldPose GT constants** | **673 MB on disk all along** — poses + cameras for 89 clips; every doc still said "pending". Video stays FIFA-gated, but the poses alone settle constants we currently guess: kinematic ceilings (10.5 m/s, 8 m/s²), plausible root-Z, and whether П4's 0.5 m twin threshold is ever legitimate | the measured distributions, replacing the guessed constants |
| **W10** ✅ closed | **#103 — kit colour under BT.709** | OpenCV 5 changed YUV→RGB, shifting **92 % of the frame**, and kit colour and OCR were never re-measured. Sits directly under П7 and under the user's own doubt about team colours | the kit scan under both matrices on the same clip |
| **W11** | **Eye-review (b), second half** | The limb-activity metric exists and is `track_quality.py --explain-imputed`; its answer is that zero-articulation frames are exactly the `imputed` runs. The open half — do the mid-pitch identity events line up with an activity handover — would make it a **stitching cue**, not just a detector | events vs activity handovers |
| **W13** ✅ | **The kit reader called the grass yellow** | Not planned — it fell out of W3/W10 and retracted four of their claims. `18 <= H <= 48` for yellow vs a floodlit pitch at H 39-40: **64.9 % of every frame read "yellow kit"**. Fixed, shared, and pinned by 15 tests | the negative control nobody had run |
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

### W3 — stitch on the handover criterion → **refuted where the plan put it; the criterion does not survive being moved pre-pose**

The plan said П3 "needs no new model: the signal is already in `provenance` plus the roots". That
is true *after* pose and false *before* it, and the difference is measurable.

`scripts/bench_handover_stitch.py` replays the pipeline's own tracker over cached detections
(60 frames of the broadcast clip, `kit_split` ON as the wiring ships it), un-projects every
tracklet endpoint through the **measured** camera so the endpoint distance is in metres rather
than extrapolated pixels, and then reads each fragment's shirt off the video. No GPU, no pose.

**1. Why the shipped stitcher misses the pairs the eye named.** Three different reasons, and only
one of them is a threshold:

| pair | gap | extrapolated px | on the pitch | rejected by |
|---|---|---|---|---|
| t3 → t66 | +16 | 70.1 (budget 63.6) | — | gap **and** size 1.65 **and** centre — all three by <10 % |
| t10 → t77 | −14 | 329 (budget 74) | — | 14 overlapping measured frames |
| t15 → t71 | +28 | 240.8 (budget 71) | **1.65 m** | gap 28 > 12, centre 241 px > 71 px |

The last row is the mechanism. t15 and t71 stand **1.65 m apart on the grass**, which is a
handover — but the stitcher does not compare positions, it compares *a constant-velocity
extrapolation* of the head against the tail. Over a 28-frame gap a footballer does not keep his
velocity, so the prediction lands 240 px away and the gate fires. **The 2D stitcher is
structurally unable to span long gaps**, and no threshold fixes that.

**2. So run П3 pre-pose instead — and it is wrong 4 times in 6.** Endpoint distance in metres,
|gap| ≤ 14, ≤ 4 simultaneous measured frames, greedy nearest-first assignment: exactly the rule
that scores 20/21 against the eye post-pose. Pre-pose it accepts six merges, and the video pixels
say **four of them join two different shirts**:

> ⚠ **Re-measured 2026-08-07 after W13 found the kit reader broken.** The first version of this
> table said *four* clashes, and listed t17 → t79 as one of them. That row was the reader's fault,
> not the merge's. The corrected numbers are below; the verdict did not change.

| merge | distance | frame gap | head kit | tail kit | |
|---|---|---|---|---|---|
| t15 → t78 | 0.15 m | +1 | `B` | `Y` | **clash** |
| t9 → t77 | 0.15 m | +1 | 46× `B` | 14× `Y` | **clash** |
| t3 → t76 | 0.48 m | +1 | `B` | 24× `Y` | **clash** |
| t17 → t79 | 0.26 m | +1 | `B` | `B` | same kit ✓ |
| t73 → t74 | 0.04 m | +1 | `?` | `B` | unclear |
| t2 → t72 | 0.07 m | +1 | `?` | `B` | unclear |

**Three demonstrable clashes against one correct merge.** These are **not** the occlusion artefact
`track_quality.py` warns about — t78, t77 and t76 read yellow on every clean frame they have.

**Why it works after pose and not before.** Post-pose the scene holds 24 long subjects; pre-pose
the tracker hands over **32** tracklets, eight of them 4–13-frame fragments produced by the #132
kit split. Nearest-endpoint in a crowd of fragments picks whoever is closest, and at 0.15 m the
closest fragment is usually the man the id *jumped to*, not the man it left. The shipped
stitcher's extrapolation gate — the thing that makes it miss long gaps — is also what stops it
making these four mistakes. It is more accurate than П3-pre-pose, not less.

The clearest single case: **t77 (yellow, f46–59) is claimed by t9 (blue) at 0.15 m and by t5
(yellow) at 0.42 m.** The nearest-first assignment takes t9 and is wrong; the shipped stitcher
merges `[5, 77]` and is right.

**Verdict: do not relax the 2D stitcher on the handover criterion.** The item moves to where its
own evidence is — post-pose, on the scene, which is also where W4 lives. W3 and W4 are one change.

**~~Two things this turned up that were on no board.~~ Both were wrong — see W13.** The original
text claimed `team_id` was wrong on t3 (*"19 yellow against 10 blue"*) and that the #132 split cut
t17 nine frames late. Neither survives: **t3 is blue** until its box walks onto a yellow player at
~f32, where the split cuts it correctly, and **t17 is blue throughout and was never cut by the
split at all** — ByteTrack itself ended that id at f34. Both claims came out of a kit reader whose
yellow band contained the pitch. Left visible rather than deleted, because the wrong instrument
had already reached a committed findings file and the correction is the finding.

*Closed 2026-08-07 as measured-and-refuted. CPU only, no GPU.*

### W3 + W4 — the merge, where the criterion actually holds → **24 → 21 subjects, 183 mannequin frames gone**

W3 and W4 turned out to be one change, so they landed as one: `core/orchestration/handover.py`,
run from the controller straight after `add_temporal_coherence` (it needs `provenance`) and
before the physics gates (so they see one body, not two). `--handover`, **off by default**.

Measured on the scene the user judged track-by-track — not a fresh run, so every verdict in
`track-labels-2026-08-07.json` maps onto the result directly:

| | subjects | merges | mannequin frames | seam |
|---|---|---|---|---|
| `out/cue/scene_off.json` | 24 | — | 183 across the six halves | — |
| **after the merge** | **21** | **(3,66) (10,77) (15,25)** | **0 in the survivors** | 31 interpolated |

**The three merges are exactly the pairs the eye named** — with 15→25 rather than 15→71, which is
the pair the geometry prefers (0.85 m over 4.96 m) and the one the user had already withdrawn
certainty on. Each survivor comes out **49 measured / 11 interpolated / 0 imputed**, from two
halves that were each about half frozen mannequin. Re-running the criteria on the merged scene:
handover pairs remaining **none**, twin pairs **8 → 4**.

**Off by default, and that is not timidity.** This is the only pass in the pipeline that *deletes*
a subject. Every other correction marks; a wrong merge erases a real player, which is precisely
what R-6 forbids. The rule says the eye decides, so it decides:
`bash scripts/view_handover_ab.sh`.

**It flags its own weakest merge.** `HandoverReport.suspect` reports any survivor that ends up
inside another subject. On the reference scene it flags **t10, 0.05 m from t5 on 13 frames** — and
the video pixels agree: **t77 reads yellow on all 3 of its measured frames while t10 reads blue on
all 46.** The user's eye put 10+77 in the stitch list on 2026-08-07, and the label file's own note
already recorded that t77's `team_id` disagrees with the pixels. So the measurement and the eye
disagree here, and the honest thing at 2 a.m. is to say so rather than invent a gate that
overrules the user. Flagged, not rejected — WorldPose says real players genuinely do get that
close (39 pairs in 20 clips inside 0.5 m, one for **3.0 s** straight, `--clips 20`).

9 tests, **mutation-checked**: all 7 injected regressions caught. One is worth recording because
it did not fail at first — the fixture for "two humans measured at once are never merged" had an
11-frame overlap but a −15 gap, so `max_gap` rejected it before `max_both` was ever consulted and
deleting the simultaneity gate changed nothing. The fixture now sits at gap −10 and asserts the
pair *does* merge once `max_both` alone is relaxed.

Suite **1184 passed / 19 skipped**. Verified end to end through the CLI on the fakes path
(`--coherence --handover --export gltf`), which is what proves the wiring; the merge decisions are
verified on the real scene above.

*Closed 2026-08-07 for the code. **Open for the eye** — the A/B is staged and t10 is the question.*

### W5 — selective mask propagation → **the dispatch signal works, but only if you ask it the right question**

SMP (arXiv 2606.13033) fires the VOS only where the tracker is unsure, triggering on the
**assignment margin in the Hungarian cost matrix**. Our reason to want it is measured: the McByte
mask cue costs **686 s GPU / 4.2 h CPU** per pass and bought mid-pitch identity events 28 → 24.
Paying that everywhere for a 14 % gain is exactly what SMP claims to fix.

So the premise gets tested before anything is built — **on CPU, with no Cutie and no GPU**.
`scripts/bench_assignment_margin.py` replays the pipeline's own tracker over cached detections and
records every cost matrix, using the seam the mask cue already patches. No new monkey-patch: a
recorder that quacks like a `MaskCue` gets both the matrix and its frame number from the real
association loop, and returns the cost unchanged so it cannot perturb what it measures.

It reproduces **78 mid-pitch identity events** over 235 frames — independently matching #133's
own count of 78, which is a good sign the event definition is the same one.

**The naive question gives a null, and it looks convincing until you run the control.** Margins at
event frames really are lower — p10 **0.154** against 0.352 away, median ratio 0.92. But **73 % of
all rows already sit within ±2 frames of an event**, so a trigger firing at random hits event
frames anyway. Against a same-size random draw (200 repetitions):

| margin < | rows fired | of all | event frames hit | random | **lift** |
|---|---|---|---|---|---|
| 0.02 | 37 | 0.4 % | 11/65 | 10.0 | 1.10× |
| 0.05 | 91 | 1.0 % | 22/65 | 22.0 | 1.00× |
| 0.10 | 200 | 2.3 % | 34/65 | 38.6 | 0.88× |
| 0.20 | 566 | 6.4 % | 52/65 | 60.4 | 0.86× |
| 0.40 | 1012 | 11.4 % | 60/65 | 64.5 | 0.93× |

**No better than chance, and slightly worse at the useful thresholds.** The mechanism is real:
ambiguity *clusters*, so a fixed number of low-margin rows covers *fewer* distinct frames than a
random draw. Per frame, the signal is worthless — on a crowded frame something is always
ambiguous.

**The fair question is per track, and there the signal is strong.** Not "was something ambiguous
on the frame where an identity broke" but "was **the breaking track's own row** ambiguous":

| | p10 | p25 | median |
|---|---|---|---|
| breaking tracks (n=49) | **0.021** | **0.072** | **0.127** |
| every (frame, track) (n=4580) | 0.162 | 0.502 | 0.739 |

A breaking track's margin is **5.8× lower at the median**, and against the base rate:

| margin < | catches breaking tracks | fires on rows | **lift** |
|---|---|---|---|
| 0.05 | 22.4 % | 2.0 % | **11.3×** |
| 0.10 | 30.6 % | 4.4 % | **7.0×** |
| 0.20 | **65.3 %** | **12.3 %** | **5.3×** |
| 0.40 | **85.7 %** | 21.1 % | **4.1×** |

**Verdict: the premise holds.** Firing on ~12 % of rows catches two thirds of the tracks that
break; ~21 % catches 86 %. Against an always-on cue that pays 100 %, that is a **4–5× compute
saving at most of the coverage** — 686 s GPU becomes roughly 145 s.

**Two limits, both load-bearing, neither of which SMP fixes.**

* **29 of 78 events have no row at all for that track.** They are *births*: a track being born was
  not in the previous matrix, so there is nothing to be unsure about. Dispatch can catch a track
  about to **die**, never one about to appear. Since a mid-pitch birth is usually the other half of
  a death, catching the death may prevent it — but that is a hypothesis, not this measurement.
* **The ceiling is unchanged.** The cue itself converts 4 of 28 events (14 %). Perfect dispatch
  cannot beat that. **SMP makes the cue cheaper, not better** — which is only worth building if the
  saving is then spent on a *stronger* per-call propagation. That is the real opportunity and it is
  a separate item.

*Premise closed 2026-08-08 by measurement, CPU only, no GPU. The implementation is not built: it
should not be, until there is a reason to spend the saving.*

### W13 — **the kit reader called the grass yellow, and it had already reached a committed doc**

Not a queued item. It came out of chasing a suspected #132 bug and is the most important thing
measured tonight, because it invalidated three of my own findings from the same session and one
that predates me.

**The measurement.** `scripts/track_quality.py --kit` classified a shirt as yellow on
`18 <= H <= 48 and S > 90`, over the median of the whole patch. The floodlit pitch on this clip
sits at **H 39–40, S ≈ 150–170**. So:

> **64.9 % of every pixel in the frame classified as "yellow kit"**, and 51.8 percentage points of
> that sat in H 35–48 — the exact band the tracker's own appearance sampler has always rejected as
> grass.

Any box carrying a normal amount of turf read `Y`. That is most boxes.

**How it was caught: by looking.** The centroid path inside the tracker and my pixel reader
disagreed about t3, t11 and t17, so I dumped the crops and looked at them. t3 is a **light-blue**
Congo DR player for f0–31 whose box then walks onto a **yellow** Colombia player at ~f32; t11 and
t17 are blue for every frame they exist. **The tracker was right and the reader was wrong** — and
the tracker was right precisely because it rejects grass before taking a median, which this script
did not.

**What that overturns**, all corrected in place above with the originals left visible:

| claim | status |
|---|---|
| "t3 reads 19 yellow against 10 blue and is labelled B" (W3) | **false** — t3 is blue |
| "the #132 split cuts t17 at f35 but the kit flips at f26" (W3) | **false twice** — t17 is blue throughout, and the split never cut it; ByteTrack ended that id |
| "t11 and t13 carry an un-cut kit flip" (W10) | **false** |
| "`team_id` disagrees with the pixels on 2 of 28" (W10) | **false** — one of 24 |
| "П3-pre-pose: 4 of 6 merges join two shirts" (W3) | **3 of 6** — the verdict stands, one row was wrong |
| `track-labels-2026-08-07.json`: "22 of 24 … t3 and t77 disagree, t3 FLIPS kit mid-track" | **false, and it predates this session** — 23 of 24, t77 only, no flip |

**And one claim it strengthens.** t77 still reads yellow after the correction, so I looked at that
crop too: at f55–57 the box holds an unmistakable **yellow Colombia player, white shorts, red
socks**, while t10 is a **blue Congo DR player wearing number 8** and t5 is a **yellow Colombia
player wearing number 25**.

> **`team_id=B` on t77 is a genuine mislabel — the only one in 24 — and it is exactly what let the
> wrong merge through.** The handover pass's team gate would have refused t10 → t77 had the label
> been right, and its geometric `suspect` check flagged the merge anyway, independently, because
> the rebuilt t10 lands **0.05 m from t5** — the yellow player t77 almost certainly belongs to.
> Two unrelated signals, same conclusion.

**The fix.** `classify_kit()` now lives in one place, rejects grass with the tracker's own rule
before taking the median, keeps yellow *below* the grass band (H 15–34; the tracker's fitted
yellow centroid is at **H ≈ 25**, so 35–48 was never yellow, it was turf), and returns `-` rather
than guessing when a box is essentially all pitch. The three bench scripts import it instead of
carrying a copy — the copy is how this spread.

**The lesson, in one line:** a threshold nobody ever pointed at a *negative* control. Running that
band over the whole frame — two lines — would have shown 64.9 % of a football pitch reading as a
football shirt on day one.

*Closed 2026-08-07. CPU only. Cost: it invalidated four claims and saved a fifth.*

### W10 — #103, kit colour under BT.709 → **the migration was right and it changes nothing we read**

`ffprobe` answers half of it before a single pixel moves: the target clip declares
**`color_space=bt709`, `color_transfer=bt709`, `color_primaries=bt709`, `color_range=tv`.** So
BT.709 is what the file asks for — OpenCV 5 is *correct*, OpenCV 4 was decoding it wrong, and it
is the **pre-2026-07-29** kit numbers that were measured under the wrong matrix, not the current
ones. R9's note has the polarity backwards.

`scripts/bench_kit_colour_matrix.py` measures the rest. It pulls the raw `yuv420p` planes once
with ffmpeg (no matrix applied), converts them **both** ways in numpy with the limited-range
inverse, and reads every tracked shirt under each.

**What cv2 actually gives us**, on frame 5 against the two references: mean |Δ| **0.65 / 255**
versus BT.709 and **3.43** versus BT.601. So OpenCV 5.0.0 is on BT.709 and handles `tv` range —
that is now measured, not assumed.

| | |
|---|---|
| whole frame, BT.601 vs BT.709 | mean \|Δ\| **2.96 / 255**, max 26 |
| pixels changed at all | **95.6 %** (R9 said 92 %) |
| pixels changed by more than 4/255 | 54.0 % |
| **shirt patch**, hue | \|Δ\| mean **1.62** of 180, p95 4, max 7 |
| shirt patch, saturation | \|Δ\| mean 6.68, p95 10, max 13 |
| **per-box kit classification flips** | **18 of 1055 = 1.71 %** |
| **tracks whose modal kit changes** | **0 of 32** |

**So the scare does not materialise.** The matrix moves almost every pixel, but by ~3/255, which
is under 2° of hue — far inside the width of the yellow and blue bands the reader uses. Not one
track's team reading changes. The #135 kit reader was written this month, i.e. already under
BT.709, so its thresholds were never tuned against the old matrix in the first place.

**And that is what makes the W3/W4 flag stand.** t77 reading yellow while t10 reads blue is not a
decode artefact — it survives both matrices unchanged. The open question about that merge is a
question about the *tracker*, not about colour.

**~~Two things the same table shows, for free.~~ Also wrong — see W13.** The original text read
`team_id` as disagreeing with the pixels on t3 and t75, and reported un-cut kit flips on t11 and
t13. All four were the broken reader calling grass yellow. Re-measured with the grass-safe
classifier, on the scene: **`team_id` disagrees on exactly one track of 24 — t77 — and that one is
real** (see W13, where the crop is unambiguous).

The OCR half of #103 is moot for now — #109 measured **0 of 23** usable jersey crops, so there is
no OCR output for a colour matrix to have shifted.

*Closed 2026-08-07. CPU only, ~2 min.*

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
