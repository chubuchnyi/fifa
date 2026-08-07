# The 2026 occlusion stack, reviewed against our own pixels

**2026-08-07.** A survey arrived recommending a specific pipeline for crossings and occlusion:
`RF-DETR → BBoxMaskPose loop → Deep-EIoU or NOOUGAT with KPR-ReID → GTA-Link`. Its framing is
right and independently matches what we measured on 2026-08-06. This doc records what survived
verification, what our own subject size rules out, and the one thing we measured today.

**The short version.** The survey's diagnosis is correct and already ours. Its prescription is
mostly aimed at people 3–11× larger than ours, and the two most cited methods carry the authors'
own warning about small instances. The parts that transfer are the two *offline, geometric* ones.

---

## 1. The premise is our own measured finding

The survey's key claim — *at crossings it is not the detector that breaks, but ambiguity inside
the box and the absence of global tracklet re-association* — is what
[`human-physics-requirement-2026-08-06.md`](human-physics-requirement-2026-08-06.md) concluded
from three measurements:

| Their claim | Our measurement | Result |
|---|---|---|
| Detection is not the bottleneck | detector threshold 0.30 → 0.10, boxes 4362 → **8810** | association ceiling 70 % → **96 %** |
| …and more boxes do not help | the same doubled detections through ByteTrack | mid-pitch events **26 → 28**. Verbatim: *"the information is present and ByteTrack cannot use it"* |
| Ambiguity lives inside the box | mask census over 3969 player crops | **23.5 %** carry >5 % of another player, **10.7 %** >20 %, **76 %** of frames hold at least one |
| No global re-association | we have one — `continuity.stitch_tracks` | but `max_gap = **12** frames`, and ByteTrack's own `lost_buffer = **30**` |

So this is confirmation from outside, not a new hypothesis. Worth having.

## 2. The measurement that decides everything: our subjects are tiny

Measured today from the cached detections of both clips:

| | median player box | p05 | torso patch actually available | upscale to a 256×128 ReID input |
|---|---|---|---|---|
| Fan clip (phone, portrait) | **28 × 72 px** | 20 × 59 | ~23 × 25 px = **573 px of shirt** | 3.6× |
| Broadcast clip | **41 × 86 px** | 27 × 63 | ~33 × 30 px = **981 px of shirt** | 3.0× |

Within one team those 573 pixels are the *same* 573 pixels for eleven people, and a shirt number
would be under 10 px tall. Keep that in front of every claim below.

## 3. What the authors themselves say about small instances

**BBoxMaskPose v2 (arXiv 2601.15200), §5.1, verbatim:**

> "COCO contains many small instances (more than 30 % of instances have a bbox size smaller than
> 100px). **Mask refinement is not suitable for such small instances** […] and looping
> SAM-pose2seg with PMPose **pronounces the error**."

That sentence exists to explain why **BMPv2+ — the 55.8 OCHuman headline — scores *worse* than
BMPv2 on COCO, 78.1 vs 78.8.** The extra refinement loop that buys +4.3 AP on large occluded
people costs accuracy on small ones. Their "unsuitable" threshold is 100 px; ours is 28–41.

**Sapiens is excluded by its own data curation.** v1 kept images with *"bounding box dimensions
exceeding 300 pixels"*; v2 keeps instances where *"at least one person is ≥384 pixels on the short
side"*. Our short side is 28–41 px — **11× below the training floor**. Its part segmentation is
also semantic, not instance (29 classes, per-pixel), so it cannot attribute a limb to a person in
a crowd — the survey's read is right.

**OCHuman measures the wrong axis.** Instances are selected by occlusion (MaxIoU > 0.5), never by
size, and **none of the six methods publishes AP_S / AP_M / AP_L**. A high OCHuman score is
evidence about crowding, not about scale. There is no published number for any of them at 35 px.

## 4. ReID: two questions with opposite answers

The survey's §4 merges two problems. For football they resolve differently:

* **"Which of the two people in this box do I embed?"** — real, and ours: 23.5 % of crops.
  **KPR (ECCV'24, arXiv 2407.18112) does solve this**, and every mechanism claim checks out:
  positive *and* negative keypoint prompts, part features with visibility scores, comparison
  restricted to mutually visible parts, keypoints optional at inference and pluggable from any
  pose model. Gain **+7.0 R-1 on Occluded-PoseTrack** (85.3 → 92.3).
* **"Which of the eleven identically-dressed teammates is this?"** — not available at our
  resolution. KPR's own table shows where its prompt pays: on **Market-1501, which has no
  multi-person ambiguity, 93.0 → 93.2**. The prompt buys disambiguation, not identity.

**The wall stands before *identity*, not before *attribution*** — a distinction the first draft of
this doc blurred, then contradicted itself on by recommending a mask method while saying the win
lay "in appearance, where the wall is". A mask answers *"these pixels are mine, those are someone
else's"*, which is solvable at 28 px and is what §8's item 2 buys. Only *"which of the eleven"*
is unreachable. Everything below keeps the two apart.

So the cue we need is **team + geometry + masks**, not an identity embedding. That is also what
the SoccerNet-2025 Game-State winners do: ReID is used for *tracklet association*, and identity
comes from jersey-number OCR (LLaMA-3.2-Vision, CLIP, PARSeq) — which needs digits we do not have.
For reference, SoccerNet-GSR's own module ablation puts JerseyNum at 56.75 against ReID's 87.42,
with the full baseline at **22.26 GS-HOTA**: identity is the hard part for everyone.

**On OSNet vs SOLIDER:** the numbers are real (0.491 vs 0.474 HOTA, GTATrack Table 3) but it is one
config pair in a challenge report — 1.7 HOTA, no seeds, no CI. KPR's own table contradicts the
generalisation: its SOLIDER-pretrained variant beats the ImageNet one by **+4.5 R-1 on
Occluded-Duke**. Read it as "OSNet was the better pick in that pipeline", not as a fact about
transformers.

## 5. What we measured today — the geometric half of Deep-EIoU, on our clip

Deep-EIoU's core idea needs no new weights: inflate both boxes before computing the IoU cost, so
two boxes that no longer overlap after a fast nonlinear move still clear the match threshold. That
attacks our measured failure directly — *96 % of identity events have an unclaimed detection a
median 6–23 px away, and 72 % of those score under 0.4 IoU against a 0.8 threshold.*

`scripts/bench_expansion_iou.py` patches the same seam the McByte mask cue uses, over the same 4362
cached detections, 236 frames, kit-split on:

| expand | player ids | after stitch | merges | seam speed px/f (median / max) |
|---|---|---|---|---|
| **1.00** (baseline) | 56 | **36** | 14 | 7.7 / 25.7 |
| 1.10 | 53 | 38 | 12 | 8.1 / 25.7 |
| 1.20 | 53 | 33 | 14 | 7.2 / 17.5 |
| 1.30 | 52 | 37 | 11 | 5.9 / 16.6 |
| 1.40 | 51 | 36 | 10 | 5.7 / 14.5 |
| 1.50 | 49 | 38 | 8 | 6.7 / 12.3 |
| 1.60 | 43 | 33 | 8 | 7.9 / **9.5** |
| 1.80 | **21** | 21 | 0 | — |
| 2.00 | **6** | 6 | 0 | — |

**Verdict: it moves work, it does not reduce identities.** Raw ids fall monotonically 56 → 43 as
the tracker holds through more crossings, and the merges the stitcher must then make fall 14 → 8 —
the tracker is doing the stitcher's job. But **the number that matters does not move**: after
stitching it bounces in a 33–38 band around the baseline 36, with no trend. What *does* improve is
safety — worst seam speed falls monotonically **25.7 → 9.5 px/frame**, so the merges that remain
are far less likely to be teleports.

Past 1.6 the wheels come off exactly as expected in a 22-player cluster: 1.8 gives **21** ids and
zero merges, i.e. it has begun fusing distinct players (we measured **≥28 distinct humans** in this
shot), and 2.0 collapses to 6.

**Corrected 2026-08-07, same day, after review.** The first draft of this section read the flat
column as *"the fifth cheap fix to hit the same plateau — the geometry does not break it"*. That
is a stronger claim than the experiment carries, on two counts, and both corrections came from
outside:

* **The right reading is "redundant with the stitcher we already have", not "does not work"** —
  and my own table already shows it. Without stitching, expansion removes **13 ids (56 → 43)**;
  with stitching it removes none. The tracker is holding exactly the crossings the stitcher was
  joining anyway. Those two readings make different predictions and only the second is supported.
* **The measurement has almost no power.** The identity ceiling is **≥28** distinct humans and the
  baseline is 36, so the entire measurable headroom is ~8 identities over 236 frames, against a
  band that already wobbles 33–38 between runs. A change worth 3 identities is indistinguishable
  from noise here. "No trend" is weak evidence of no effect, not evidence of no effect.

**What this does not test.** A single fixed scale, both sides inflated, is *not* Deep-EIoU: the
published method iterates the scale-up over rounds and pairs it with a sports-fine-tuned OSNet
embedder. This isolates the geometry, and what it shows is that geometry and our existing stitcher
are **substitutes** on this clip.

## 6. Licences — decisive for a repo with a commercial future

| Method | Licence | Verdict |
|---|---|---|
| **GTA-Link** | **MIT** | clean. Bundled OSNet weights are SportsMOT-trained = CC BY-NC, but a custom ReID is supported |
| **BUCTD** | **Apache 2.0** | clean, and the only pose method here whose design does not structurally degrade with subject size |
| **SOLIDER** / **OSNet** | Apache 2.0 / MIT | clean |
| **BBoxMaskPose / PMPose / MaskPose** | **GPL-3.0** (code *and* HF weights) | copyleft — commercial use allowed, but linking into a proprietary pipeline triggers source-disclosure |
| **Deep-EIoU** | **no LICENSE file at all** | all rights reserved by default. Weights SportsMOT-trained = CC BY-NC |
| **KPR** | **Hippocratic 3.0** (HL3-LAW-MEDIA-MIL-SOC-SV) | not OSI, not legally tested, open-ended ethics clauses |
| **SAM 3 / 3.1 / SAM 3D Body** | SAM License, HF weights **gated** | commercial permitted, non-OSI, request access |
| **Sapiens v1** | **CC BY-NC 4.0** | hard blocker |
| **Sapiens2** | custom | permits commercial use but **prohibits "biometric processing" and "attempting to identify or re-identify any individual"** |
| **Multi-HMR 2** | **NAVER non-commercial** | blocker |
| **PromptHMR** | non-commercial research only | our 494 MB of verified weights are research-only |

## 7. Two external criticisms we are already designed around

* **Patient4D (arXiv 2603.17178)** on our exact backbone: *"SMPLest-X […] exhibited the highest
  instability, likely due to its **per-bounding-box focal length estimation**, which produces
  inconsistent scale across frames"* — 32.5 % failure frames, 2.450 m temporal displacement.
  **This does not apply to us.** `smplestx_backend._infer_crop` reads only `smplx_root_pose`,
  `smplx_body_pose` and `smplx_shape`; we never touch the network's camera or translation. Position
  comes from the pitch homography in `_ground_root`. So the fan clip's 405 m/s is the plane, not
  the net's focal.
* **SMART (arXiv 2605.31551)** — SMPLest-X on broadcast soccer at our subject scale, CC BY 4.0.
  Its single largest ablation step is **foot-plane anchoring, 0.846 → 0.714**. That is already
  `_ground_root` + the T2 pelvis-above-foot anchor. Its remaining levers are depth-supervised
  domain fine-tuning (*"the dominant factor"*, global MPJPE 0.602 → 0.370 m) and two-pass
  smoothing — and it is the most relevant paper anyone has written for this pipeline.

## 8. What to take, in order

1. **GTA-Link** (arXiv 2411.08216, ACCV'24 workshop, **MIT**). Offline, tracker-agnostic, consumes
   MOT-format text and returns it. Its Splitter is a stronger version of our
   `split_on_kit_change`; its Connector is a stronger version of our stitcher, and we have already
   measured that our stitcher's *gates* are not the binding constraint — so the appearance
   clustering is the genuinely new part. Lowest-friction item on the whole list.
2. **Selective Mask Propagation** (arXiv 2606.13033, SAM-Deep-EIoU). This is **our #133
   architecture** — mask propagation as an association cue — with one change we do not have:
   the VOS model runs **only when the base tracker is uncertain**. That is the direct answer to our
   measured 686 s GPU / 4.2 h CPU per pass. It reports Deep-EIoU 51.7 → 59.7 HOTA (AssA **+10.9**)
   on DanceTrack val and SportsMOT 87.2 HOTA with GTA. Code not stated as released.
3. **Expansion IoU at ~1.4** — free, measured above, worth taking for the seam safety (max seam
   25.7 → 14.5 px/f) rather than for the identity count.
4. **Raise `RFDETRDetector.score_threshold` from 0.3 to 0.1** — we measured the 96 % ceiling it
   unlocks; it buys nothing on its own but every association cue above is capped by it.

**Not now:** NOOUGAT (numbers verified exactly — SportsMOT online 81.0 HOTA, offline 85.6 — but
**no code and no weights**, and its ReID is stock pedestrian FastReID). SAM2MOT (**no code**: the
repo is 6 files and the README has said "coming soon" for 9 months). BMPv2+ (the authors' own
small-instance warning). Sapiens (11× scale gap plus licence). Multi-HMR 2 and PromptHMR
(non-commercial, and neither outputs SMPL-X natively — Anny and MHR respectively).

**If we ever do want SAM in this loop, use SAM 3.1, not SAM 3.** SAM 3's video cost is linear in
object count and its own release notes cap "near real-time" at ~5 concurrent objects; we have
22–27. SAM 3.1's Object Multiplex tracks 16 per forward pass, so our scene is 2 passes instead of
25.

## 8a. The thing neither this doc nor the survey proposed: attack the pixel

Every move above works *around* 28 px. None of them changes 28 px. Measured 2026-08-07:

**`RFDETRBackend` never set `resolution`, and RF-DETR's default is 560.** It resizes the whole
frame to a `560 x 560` **square**, so aspect ratio is not preserved and a portrait phone clip is
squashed hardest:

| clip | source | scale to 560² | measured player box → what the net sees |
|---|---|---|---|
| Fan (portrait) | 1080 × 1920 | 0.52× across, **0.29× down** | 28 × 72 → **14 × 21 px** |
| Broadcast | 1920 × 1080 | 0.29× across, 0.52× down | 41 × 86 → **12 × 45 px** |

So the subject is a third of its measured size before any of §1–§8 applies, and **every
association cue downstream is capped by what survives this resize**. This is also the one place
the challenge winners spent effort: GTATrack ran **1280 px on the long side plus small-target
pseudo-labelling**, and that moved HOTA **0.380 → 0.491** (FN 40 046 → 16 186) — a larger jump
than any association change in this document.

Note this is *not* the same knob as the detector score threshold, which we measured null: that
test kept boxes the net had already found at 560. Raising the resolution changes what the net can
find at all.

`RFDETRBackend.resolution` now exists (must be divisible by 56; `None` keeps the 560 default so
nothing changes silently). **The default stays 560 until the A/B on `demorig` says otherwise** —
picking it by argument rather than by measurement is exactly the mistake this document is about.

## 9. For social footage specifically — the honest part

None of the above is what makes Instagram footage hard. Measured on both clips, the ambiguity rate
is **the same**:

| | boxes ≥20 % covered by another | frames with at least one |
|---|---|---|
| Fan clip (phone) | **16.0 %** | 92 % |
| Broadcast clip | **16.1 %** | 98 % |

What actually breaks on the phone clip is elsewhere, and all three are already on the board:
**153 of 355 frames solve no ground plane at all**; a handheld phone that translates *and* zooms
has **no single camera** to render a novel view from (`realizable: False`, 142 px); and the pod run
assigned **`team=None` to 23 of 27 subjects** (#137) — which, because `StitchConfig.require_same_team`
treats `None` as a wildcard, quietly removes the one appearance constraint that *does* work at
28 px.

So the order is: **#137 and calibration first, tracking second.** An occlusion stack that
never loses an identity still cannot place a player on a pitch it cannot see.

---

**Sources verified 2026-08-07:** [BMPv2 2601.15200](https://arxiv.org/abs/2601.15200) ·
[BMP ICCV'25 2412.01562](https://arxiv.org/abs/2412.01562) ·
[BUCTD 2306.07879](https://arxiv.org/abs/2306.07879) ·
[Sapiens 2408.12569](https://arxiv.org/abs/2408.12569) · [Sapiens2 2604.21681](https://arxiv.org/abs/2604.21681) ·
[KPR 2407.18112](https://arxiv.org/abs/2407.18112) · [Deep-EIoU 2306.13074](https://arxiv.org/abs/2306.13074) ·
[GTA-Link 2411.08216](https://arxiv.org/abs/2411.08216) · [GTATrack 2602.00484](https://arxiv.org/abs/2602.00484) ·
[NOOUGAT 2509.02111](https://arxiv.org/abs/2509.02111) · [SAM2MOT 2504.04519](https://arxiv.org/abs/2504.04519) ·
[Selective Mask Propagation 2606.13033](https://arxiv.org/abs/2606.13033) ·
[SAM 3 2511.16719](https://arxiv.org/abs/2511.16719) · [SAM 3D Body 2602.15989](https://arxiv.org/abs/2602.15989) ·
[Multi-HMR 2 2606.14841](https://arxiv.org/abs/2606.14841) · [PromptHMR 2504.06397](https://arxiv.org/abs/2504.06397) ·
[Patient4D 2603.17178](https://arxiv.org/abs/2603.17178) · [SMART 2605.31551](https://arxiv.org/abs/2605.31551) ·
[SoccerNet-GSR 2404.11335](https://arxiv.org/abs/2404.11335)
