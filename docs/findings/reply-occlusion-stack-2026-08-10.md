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
integration. That is a materially different item from the one §8 listed.

> **Wrong, and corrected within the hour of writing it.** This paragraph originally ended *"and it
> is now the cheapest thing on the list that has not already been measured null."* **It had already
> been measured.** `scripts/bench_assignment_margin.py` exists, `docs/STATUS.md` **W5** carries the
> verdict, and I wrote a landmine about exactly this failure mode **one commit earlier** in this
> same session. Seventh recorded instance of #141, and mine. The numbers are below; they change the
> item's position again, and not in the direction I claimed.

### It is measured, and the premise splits in two

Re-run today, `PYTHONPATH=src .venv/bin/python scripts/bench_assignment_margin.py` — 62 tracklets,
235 cost matrices, 78 mid-pitch identity events:

**Per frame the margin is worthless.** Against a random trigger drawn to the same size, 200 draws:

| margin < | rows fired | of all rows | event frames hit | random would hit | lift |
|---|---|---|---|---|---|
| 0.02 | 37 | 0.4 % | 11/65 | 10.0 | 1.10× |
| 0.05 | 91 | 1.0 % | 22/65 | 22.0 | **1.00×** |
| 0.10 | 200 | 2.3 % | 34/65 | 38.6 | 0.88× |
| 0.20 | 566 | 6.4 % | 52/65 | 60.4 | 0.86× |
| 0.40 | 1012 | 11.4 % | 60/65 | 64.5 | 0.93× |

The reason is in the null: **73 % of rows already sit within ±2 frames of an event.** On a frame
holding 22 players something is always ambiguous, so "was this frame ambiguous" carries no
information.

**Per track it is strong.** The sharper question — was the *breaking track's own row* ambiguous —
separates cleanly:

| | n | p10 | p25 | median |
|---|---|---|---|---|
| breaking tracks | 49 | 0.021 | 0.072 | **0.127** |
| every (frame, track) | 4580 | 0.162 | 0.502 | **0.739** |

| margin < | catches of breaking tracks | rows fired | lift |
|---|---|---|---|
| 0.05 | 22.4 % | 2.0 % | **11.30×** |
| 0.10 | 30.6 % | 4.4 % | 7.01× |
| 0.20 | **65.3 %** | 12.3 % | **5.32×** |
| 0.40 | 85.7 % | 21.1 % | 4.06× |

So the dispatch signal is real: firing on 12 % of the work catches two thirds of the breaks. On our
measured Cutie cost that is **686 s GPU → ~85 s**, 4.2 h CPU → ~30 min.

### Two limits the probe names itself

- **It is blind to births.** 29 of 78 events had no recorded row for that track at all — a birth has
  no prior row to be ambiguous in. The probe takes `np.partition(a, 1, axis=1)`, i.e. best and
  second-best **per row = per track**. The symmetric **column** margin — per *detection* — is the
  birth-side signal and has never been measured. That is a ~10-line change to the same probe and it
  would say whether SMP's trigger can see 37 % of our events at all.
- **W5's verdict is "cost, not quality", and on its own terms it is right.** The thing being
  dispatched is our mask cue, whose measured ceiling is **14 %** (mid-pitch events 28 → 24 against
  a 96 % availability ceiling). Running a 14 % cue 5× more cheaply buys compute, not output.

### Where I think W5 is nonetheless incomplete

W5 assumes the propagation is unchanged and only its *frequency* moves. SMP as published also
changes **when the mask is seeded**, and seeding is the diagnosed cause of our cue's weakness.
`human-physics-requirement-2026-08-06.md`, on why 14 % and not McByte's own margin:

> *"masks are seeded from pass-1 tracks, which are the broken ones. A track already sitting on the
> wrong player propagates a mask that follows the wrong player and then confidently confirms the
> wrong pairing. McByte is online precisely because its masks are seeded as tracks are born."*

Its own fix list, item 3: *"Go online: seed each track at birth inside the association loop."*
**Firing the VOS at the ambiguous moment, from inside the association loop, is that item** — the
mask is seeded from a track that was still good a frame ago, not from a completed pass whose tracks
are already wrong.

So the open question is not "is the dispatch signal real" (measured: yes, 5.3×) nor "does it save
compute" (measured: yes, ~5×). It is **whether online seeding lifts the 14 % ceiling**, which
neither W5 nor this doc has measured, and which is the only reason to build SMP rather than shelve
it. That is a different experiment from the one already run, and it needs the GPU pass W5 avoided.

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

**Net effect on §8's order:** unchanged at the top (GTA-Link first, MIT, lowest friction). Item 2
(SMP) is **already measured** — the dispatch signal is real per track (5.3× lift at 12 % of the
work) and buys ~5× compute on a cue whose ceiling is 14 %, so W5 shelved it correctly *as a cost
optimisation*. What is unmeasured is whether firing inside the association loop also fixes the
seeding that caused the 14 %. Item 3 (expansion IoU at 1.4) is kept for **seam safety only**
(max seam 25.7 → 14.5 px/f), with its identity claim carrying the power caveat rather than a
verdict.

**And the process note, since it is now the seventh instance.** I wrote the landmine *"quote a
findings doc by commit hash — ours get corrected within the hour"* in commit `4258b16`, and in the
commit **before** it asserted that SMP was unmeasured while `bench_assignment_margin.py` sat in
`scripts/` and its verdict sat in `STATUS.md` line 67. The landmine addresses documents leaving the
repo. It does not address **the repo's own status board not being read before a claim about what
is unmeasured.** That is a different hole and it is the one that caught me.
