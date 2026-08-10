# Reply on the occlusion stack — two of your four were already in, one is measured false, one lands

Answer to the review-of-the-review of
[`occlusion-stack-review-2026-08-07.md`](occlusion-stack-review-2026-08-07.md). Written 2026-08-10.
Every number below is from a run made today on the same cached detections that produced §5.

Reproduce: `.venv/bin/python scripts/bench_expansion_iou.py --frames 236` plus the two probes
quoted inline.

**The short version.** Your concessions I accept without argument — they were the doc's own
corrections. Of your four pushbacks: **two you already won** (they are in the doc, you read a
copy from before the revision), **one is measured false on the specifics but right in its
conclusion for a stronger reason than you gave**, and **one I concede and have fixed in the
headline**. Your SMP finding is the most useful thing in the reply and lands on a seam that
already exists. Your closing section is §8a, which was measured and retracted the same day it
was written — including on the portrait clip you name.

---

## 1 and 2 — you are answering the 19:21 version

Both pushbacks are already the doc's text:

| Your pushback | Where it already is |
|---|---|
| "the expansion-IoU verdict is stronger than the experiment shows; the right reading is *redundant with the stitcher*, and the measurement has almost no power" | §5, block **"Corrected 2026-08-07, same day, after review"** — both bullets, in your words |
| "the 573-px wall contradicts point 2; the wall stands before *identity*, not before *attribution*" | §4, **"The wall stands before *identity*, not before *attribution*"** — including the note that the first draft "contradicted itself by recommending a mask method while saying the win lay in appearance" |

Commit history, so this is checkable rather than asserted:

| commit | time | contains the corrections |
|---|---|---|
| `06d3f00` | 19:21 | **no** |
| `1250585` | 20:30 | **yes** (+48 lines) |

So there is no disagreement to resolve here — only a stale copy. That is **#141 again in its
purest form**: the fact was written down, and it did not reach the reader at the moment of need.
The cheap fix is procedural and I am adopting it: **a findings doc quoted in a message carries its
commit hash.** A doc is not a stable object; ours get corrected within the hour.

## 3 — measured false as a confound, and your conclusion is right anyway

> *"`team=None` is a confound of §5 … all the 'appearance does not help' measurements were taken
> with the only working appearance signal switched off."*

**Measured on the exact §5 path** — `ByteTrackTracker(kit_split=True)` over
`out/phmr_ab/dets_coco_0_236.npz`, 236 frames:

| | measured |
|---|---|
| player tracklets | **56** |
| of those `team_id is None` | **0 (0 %)** |
| label split | **B 29 / A 27** |

And the gate is not idle — it binds:

| `StitchConfig` | after stitch | merges |
|---|---|---|
| `require_same_team=True` (§5's setting) | **36** | 14 |
| `require_same_team=False` | **33** | 13 |

So §5 ran with kit colour **on and doing work**. The mechanism you have in mind is real but lives
one stage later: `team_id` is blanked by the **identity gate** (`_split_tracklet`,
`_truncate_tracklet`, `_merge_tracklets`), which `bench_expansion_iou.py` never calls — it goes
tracker → stitcher and stops.

**But your ordering was right, for a bigger reason than the confound would have given.** Turning
the kit constraint off costs **3 identities** — and the *entire measurable headroom* of the §5
experiment is ~8 (ceiling ≥28, baseline 36). The one appearance signal that survives 28 px is
therefore worth **~40 % of everything the expansion experiment could ever have shown**. A confound
would have made §5 unreliable; this makes §5 *small*, which is worse for the expansion trick and
better for your priority order.

**And #137 is closed.** Fixed 2026-08-08 (`6f4c270`), before your reply: `_restore_team_labels`
(`identity.py:478`) re-anchors each blanked tracklet against the ones that kept a label. Same fan
clip, same 355 frames: **23 unlabelled of 27 → 0 of 29** (A 11 / B 18), 6 tests, mutation-checked.

## 4 — conceded, and the headline is fixed

You are right that the doc's summary line was harder than the source. BMPv2's warning is about the
**refinement loop** (BMPv2+ vs BMPv2) and costs 0.7 AP on a COCO whose small instances are 30 % — a
soft penalty, never measured at 28 px. §3 and §8 both name **BMPv2+** specifically; only the
summary generalised it to "the two most cited methods". Fixed in place today, and the note says
what it used to say.

The decisive reason is **GPL-3.0 on code *and* HF weights** (§6), which needs no extrapolation from
COCO to 28 px. It should have been the stated reason, not the scale argument.

## A correction to my own correction

Your power argument is right and I supported it with a mechanism that does not exist. The
correction block said the 33–38 band *"wobbles between runs"*. Measured today, three independent
runs at `scale=1.0`:

```
run 0: raw 56  after stitch 36  merges 14
run 1: raw 56  after stitch 36  merges 14
run 2: raw 56  after stitch 36  merges 14
```

**There is no run noise.** The path is deterministic (`tracking.py:161` says so of the team
assignment, and it holds end to end). So a 3-identity change would be perfectly **visible**; what
it would not be is **attributable** — the band is a deterministic *non-monotonic* response, where a
small change to the cost matrix reshuffles which merges the stitcher subsequently makes. Your
conclusion stands, and it now rests on sensitivity rather than on noise. Both correction blocks in
§5 are updated.

## SMP — the useful part, and the seam is already built

Your finding upgrades item 2 from "code not stated as released" to **buildable without weights**,
and it lands somewhere we have already cut twice:

- `supervision.tracker.byte_tracker.matching.linear_assignment` runs scipy's
  `linear_sum_assignment` over the cost matrix returned by `matching.iou_distance`.
- `ByteTrackTracker._patch_matching` (`tracking.py:481`) already wraps **that exact function** for
  the McByte mask cue; `bench_expansion_iou.patch_expansion` is a second, independent user of the
  same seam.
- The dispatch signal you describe — the assignment margin — is *best minus second-best per row*,
  computable from the cost matrix **before** the assignment runs. No assignment result needed, no
  new weights, no fork of a validated tracker.

So the honest description of SMP's cost for us is **a probe at an existing seam**, not an
integration. That is a materially different item from the one §8 listed, and it is now the
cheapest thing on the list that has not already been measured null.

Your calibration numbers are taken and are the right ones to hold it to: GTA over the same
Deep-EIoU is 81.0 HOTA against SAM3-Deep-EIoU's 87.2, so the delta attributable to selective
propagation is what remains *over GTA-Link* — which is item 1 and cheaper. Plus SAM 3 over SAM 2 at
fixed logic is +1.7. Plus the abstract/body discrepancy (86.8 vs 87.2), single author, unreviewed
preprint, unreproduced. Worth a probe; not worth a plan.

## Your closing section is §8a, and it was measured and retracted

> *"Every move is about getting around 28 px. None attacks the pixel itself … on a portrait phone
> clip this may cost less than everything else on the list."*

That is §8a, written the same day, and it went further than proposing it. RF-DETR at
**560 / 728 / 896 / 1064**, 60 frames of **both** clips including the portrait fan clip, on
`demorig`:

| floor | gain | cost |
|---|---|---|
| 0.3 (the adapter's) | **+0.9 players/frame on both clips (~5 %)** — broadcast 19.1 → 19.9, fan 15.1 → 16.0 | 1.5–2× the time |
| 0.1 | +18 % | which is the band we had already measured worthless downstream (doubling detections moved churn 26 → 28) |

Median box height does not grow; the extra finds are smaller, more distant people. **The default
stays 560.** §8a also already separates GTATrack's 0.380 → 0.491 as coming from **small-target
pseudo-labelling**, with 1280 px as the carrier — the resolution alone is measured here, and the
resolution alone is not it.

**And there is a structural reason it was never going to move your wall.** RF-DETR's resize caps
**detection**; it does not touch the crop. The 573 px of shirt is cut from the **original** frame
at 28 × 72 regardless of what the detector ran at. Two different bottlenecks:

| bottleneck | set by | movable by detector resolution |
|---|---|---|
| can the net *find* the player | 560² square resize | yes — measured, +5 % |
| how much shirt the ReID/pose crop *contains* | source resolution, 28 × 72 | **no** |

So "attack the pixel" is right as an instinct and has exactly one lever, and it is not ours: the
source resolution of the footage. At 28 px the shirt is 573 pixels because the phone recorded it
that way.

---

**Net effect on §8's order:** unchanged at the top (GTA-Link first, MIT, lowest friction), but item
2 (SMP) is cheaper than listed — a probe on `_patch_matching`, not an integration — and item 3
(expansion IoU at 1.4) is now explicitly kept for **seam safety only** (max seam 25.7 → 14.5 px/f),
with its identity claim carrying the power caveat rather than a verdict.
