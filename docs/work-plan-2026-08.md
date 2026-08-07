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
