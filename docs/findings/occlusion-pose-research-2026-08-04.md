# Occlusion survey: masks-not-boxes for tracking + pose (2026-08-04)

**Trigger.** User hypothesis: player crossings break ByteTrack IDs and corrupt per-crop HMR
(the occluded player's crop contains two bodies); would segmented players fix both? Plus a
sweep of what appeared late-2025 → 2026-08 for single-camera multi-person 3D pose.

**Provenance.** Web survey by two research agents on 2026-08-04. Every number below is *as
published*, nothing reproduced locally yet. Board item: **#132**.

## Verdict on the hypothesis

It holds — the 2025–26 literature converged on exactly this, in two separable halves:

1. **Masks as the association cue through crossings** (tracking) — biggest published sports-MOT
   gains of the period.
2. **Mask-prompted full-image HMR** instead of box crops (pose) — the occluded player's mesh is
   estimated from his own pixels with full-scene context.

And one counterpoint: the SoccerTrack 2025 winner used **no masks** — offline tracklet
split/merge (GTA-Link) is a cheap boxes-only complement, not a refutation.

## Tracking: association through crossings

| Method | What / evidence | Availability | Fit |
|---|---|---|---|
| **McByte** (CVPRW 2025) | ByteTrack + temporally-propagated mask cue, training-free. HOTA: SoccerNet-tracking **85.0 vs 72.1** ByteTrack, SportsMOT **83.9 vs 69.0**. 3–5 FPS on A100 | MIT, [code](https://github.com/tstanczyk95/McByte), [paper](https://arxiv.org/pdf/2506.01373) | our tracker *is* ByteTrack → most direct upgrade |
| **SAM 3 / 3.1** (Meta, 2025-11) | text-promptable detect+segment+track («soccer player» → all instances, identity through occlusion). In HF Transformers + Ultralytics | SAM License, [blog](https://ai.meta.com/blog/segment-anything-model-3/), [paper](https://arxiv.org/abs/2511.16719) | masklet source for both halves |
| **SAM-Deep-EIoU** (2026-06) | VOS invoked only on uncertainty windows; SportsMOT **87.2 HOTA** | code TBD, [arXiv](https://arxiv.org/html/2606.13033v1) | «masks only at crossings» = cheapest form |
| **GTA-Link** (boxes-only) | offline tracklet split/merge post-processor; SoccerNet **79.4→83.1 HOTA**. SoccerTrack-2025 winner GTATrack = Deep-EIoU + OSNet + this | [code](https://github.com/sjc042/gta-link), [paper](https://arxiv.org/pdf/2411.08216) | cheap complement, zero mask cost |
| SAM2MOT (AAAI 2026) | tracking-by-segmentation, DanceTrack 75.5 HOTA | Apache-2.0, [code](https://github.com/TripleJoy/SAM2MOT) | proves paradigm, heavier stack than McByte |
| SAMURAI | SAM2 + Kalman, **single-object** | Apache-2.0 | wrong shape for 20 players |

## Pose: HMR that survives a crossing

| Method | What / evidence | Availability | Fit |
|---|---|---|---|
| **PromptHMR** (CVPR 2025) | full-image HPS prompted by **masks**/boxes/text; SMPL-X out; video mode = world-frame multi-human; built for close-contact | code+ckpt (BEDLAM2), research-terms license, [repo](https://github.com/yufu-wang/PromptHMR), [paper](https://arxiv.org/abs/2504.06397) | closest drop-in for per-crop SMPLest-X, **no rig conversion** |
| **SAM 3D Body** (Meta, 2025-11) | promptable single-image HMR, accepts masks + 2D kp, ~8M-image training for occlusion | SAM License, [HF](https://huggingface.co/facebook/sam-3d-body-vith), [paper](https://arxiv.org/abs/2602.15989) | strongest raw robustness; outputs **MHR rig, not SMPL-X** → conversion tax |
| **SAM-Body4D** (2025-12) | training-free video chain: SAM-3 masklets → Diffusion-VAS amodal refinement → SAM 3D Body, multi-human | MIT, [repo](https://github.com/gaomingqi/sam-body4d) | our hypothesis pre-assembled; matches our `DiffusionVasOcclusionBackend` stub |
| **CoMotion** (Apple, ICLR 2025) | ONE model: concurrent multi-person SMPL + online tracking through occlusion; PoseTrack21 **+14% MOTA / +12% IDF1** | code open, weights **non-commercial**, [repo](https://github.com/apple/ml-comotion) | the architecture class our crop-HMR lacks |
| **MoRo** (2026-01, Tang+Meta) | generative motion recovery under occlusion (masked transformer), 70 FPS; beats SOTA occluded on EgoBody/RICH | [arXiv](https://arxiv.org/abs/2601.16079), code via project page | post-pass on tracks = R-6 «mark, never hide» made literal |
| Multi-HMR 2 (2026-06) | one-shot multi-person detect + metric mesh + tracking | code unconfirmed, [arXiv](https://arxiv.org/abs/2606.14841v1) | watch — could collapse 3 stages |
| GENMO (NVIDIA, ICCV 2025) | generative in-fill of occluded motion spans | **non-commercial**, [repo](https://github.com/NVlabs/GENMO) | heavier MoRo alternative |

## Benchmarks & adjacent

- **FIFA Skeletal Light 2026** ([Kaggle](https://www.kaggle.com/competitions/fifa-skeletal-light), WorldPose GT) — *literally our problem* (world 3D pose from the main broadcast camera). Only public write-up: **SMART**, 6th place — finetuned SMPLest-X + foot-plane anchoring → **global MPJPE 0.324 m / local 0.054 m** ([paper](https://arxiv.org/abs/2605.31551)). That is the reference bar for our pipeline class, and SMART is already in our chosen stub stack.
- **SoccerNet 2026** ([results](https://arxiv.org/html/2607.07320v1)): no 3D-pose task, but a new **Novel View Synthesis** task (winner DENSER, PSNR 29.89) — adjacent evidence base for our end goal.
- **Depth ordering**: MoGe-2 (MIT, [repo](https://github.com/microsoft/MoGe)) / Depth Anything 3 ([repo](https://github.com/bytedance-seed/depth-anything-3)) — metric depth as who-is-in-front prior in clusters.
- **YOLO26 / OpenCV 5 article** ([learnopencv](https://learnopencv.com/opencv-5-cpp-object-detection-yolo26/)): C++/ONNX/CPU deployment tutorial. Nothing for the occlusion problem — RF-DETR is already NMS-free; `yolo26n-seg` would only give cheap *per-frame* masks, identity needs video segmentation anyway. Not adopted.

## Proposed order (nothing started — user's pick)

1. **PromptHMR video/world mode** on the Colombia clip — replaces per-crop HMR wholesale;
   mask-promptable, SMPL-X native, world-frame output.
2. **Mask cue in ByteTrack** à la McByte (MIT), masklets from SAM 3 — or start even cheaper
   with **GTA-Link** offline merge (boxes-only, plug-and-play).
3. **MoRo** post-pass over occluded spans — lowest integration risk.

License flags before anything ships: CoMotion + GENMO weights non-commercial; PromptHMR
research terms; SAM License has restrictions; McByte / SAM-Body4D / MoGe-2 are MIT.

---

# PromptHMR — hands-on check (2026-08-04)

User picked candidate 1. Code pulled to `backends/PromptHMR` (gitignored; the box-local glue
pattern `backends/pnlcalib_backend.py` already uses, which also keeps the non-commercial code
out of the public `pitch3d` tree); weights to `models/prompthmr` with everything else we have
downloaded (`docs/models-dir.md`). Paths below are relative to the weights bundle — upstream
resolves its assets against the process cwd, so the scripts run from there.

## Weights, verified not assumed

| Artifact | Measured |
|---|---|
| `phmr_vid/phmr_b1b2.ckpt` | 494 744 294 B, sha256 `2a36132…b9e5e5` — **matches** the official `bedlam2_phmr.sha256` |
| video head structure | Lightning 2.4, epoch 499, 251 tensors, **41.4 M params**, all `pipeline.denoiser3d.*` (the GVHMR-lineage head) |
| `phmr_vid/prhmr_release_002.ckpt` | 251 tensors / 41.4 M — same shape as the BEDLAM2 ckpt, so the two are drop-in swaps (`phmr_vid.py:23` picks) |
| `phmr/checkpoint.ckpt` (2.2 GB) | 765 tensors, **574.5 M params**: `image_encoder` 346 · `prompt_encoder` 167 · `smpl_decoder` 219 · `smplx` 20 · `smpl` 9 · `cam_encoder` 3 |

**The mask prompt is real, not a paper claim.** Released `config.yaml` has `MASK_PROMPT: True`,
and the checkpoint carries trained weights for it: `prompt_encoder.mask_downscaling.{0,1,3,4,6}`
— `(4,1,2,2)` → `(1024,16,1,1)`, std 0.03–0.34 — plus `no_mask_embed (1,1024)` std 0.947 (the
learned "no mask given" fallback). A mask is embedded densely and added to the image features,
SAM-style; `mask_prompt=False` swaps in `no_mask_embed`, so the same checkpoint runs both ways
and an **A/B (boxes vs masks) is one flag**, not two pipelines.

## Code ↔ checkpoint: strict load on CPU

`scripts/check_prompthmr_weights.py` builds the image model from the *released* code and
`config.yaml`, then loads the *released* `state_dict`. Re-runnable, no GPU, ~1 min, exits
non-zero if the two ever stop matching:

```
.venv/bin/python scripts/check_prompthmr_weights.py
checkpoint tensors 765 · model tensors 765 · MISSING 0 · UNEXPECTED 0 · 569.6M params landed
mask_downscaling.6 loaded: match=True std=0.1461
```

Every learned tensor has a home; nothing in the code is left randomly initialised. (Stubs: the
gated classic SMPL, `pytorch_lightning`, and `xformers` → torch SDPA. SMPL-X and `smplx2smpl.pkl`
are the real files.) Two consequences that only a real load could show:

- **The MetaCLIP download is dead weight.** The run patches `pretrained=` away — "Model
  initialized randomly" — and still reports 0 missing, because the checkpoint carries all 152
  CLIP tensors / 124.4 M params under `prompt_encoder.clip_encoder.encoder.*`. Gotcha 3 below is
  therefore a one-line fix, not an offline blocker.
- **`dinov2_vitl14_pretrain.pth` is optional in the same way.** The run printed "Not using DINOv2
  weight"; `image_encoder.backbone` (343 tensors, 304.4 M params) comes from the checkpoint.
- The gated classic SMPL is 9 buffers / 4.9 M params (`shapedirs`, `v_template`, `J_regressor`,
  `posedirs`, `lbs_weights`, …) — pure body-model data, no learned weights. See below: it turns
  out not to block us at all.

## The gated SMPL does not block a run (`--forward`)

`data/body_models/smpl` does not exist on this machine, and the image model still runs:

```
.venv/bin/python scripts/check_prompthmr_weights.py --forward
SMPL_NEUTRAL.pkl present : False        forward pass : OK
rotmat (2,22,3,3) · transl (2,3) · betas (2,10) · features (2,2,1024)   all finite
mask on vs off : max |d rotmat| = 0.1666
```

Two independent reasons the gate is not load-bearing:

1. **Our path never reads the classic-SMPL outputs.** `PHMR.process_output` unconditionally
   computes `smpl_vertices/joints/j3d` (`phmr.py:142-153`), but the only consumers in the repo
   are `evaluator.py` and `datasets/{emdb,test}_dataset.py` — benchmark code. `phmr_vid.py`
   takes `rotmat`, `transl`, `betas`, `features` and nothing else.
2. **The checkpoint already carries the whole neutral SMPL** — `v_template (6890,3)`,
   `shapedirs (6890,3,10)`, `posedirs (207,20670)`, `J_regressor (24,6890)`,
   `lbs_weights (6890,24)`, `parents`, faces. `SMPL_NEUTRAL.pkl` is read only to *shape* buffers
   that `load_state_dict` then overwrites, so a stub sized from the checkpoint is exact.

`rotmat` is `(N, 22, 3, 3)` = global orient + **21** body joints — our `N_SMPLX_BODY_JOINTS`
(`core/scene/motion.py:19`) confirmed against a running model, not just against their source.

And the mask prompt is live end to end, not merely present in the weights: flipping
`mask_prompt` on the same image, boxes and keypoints moves the predicted rotations. The input
here is noise, so **0.1666 says "connected", not "better"** — the real A/B is on the clip.

**Licence caveat.** Technically unblocked ≠ licence-clear: the SMPL body model is MPI-licensed
wherever the bytes come from, and PromptHMR is research-terms anyway. Registration is a
ship-time task, not a prerequisite for measuring whether this helps.

## Fit to our seams

- **`HMRBackend` matches with no rig conversion.** `estimate_bodies(clip, tracks) → {track_id:
  RawBodyMotion}`; PromptHMR's `smplx_cam['pose']` is axis-angle `(T, 66+9)` → `[:3]` is
  `global_orient`, `[3:66]` reshapes to `body_pose (T,21,3)`, `shape` → `betas`.
- **Joint layout is 1:1.** Their `body_pose` is 63 = 21×3 axis-angle, and our
  `N_SMPLX_BODY_JOINTS = 21` (`core/scene/motion.py:19`). No remapping.
- **How the mask actually reaches the answer.** The final motion comes from the *video* head,
  but it is conditioned on the image model's per-frame `features` — which were computed *with*
  the mask prompt. So masks must be supplied per track per frame (SAM2 seeded from our boxes),
  and their benefit arrives through the features, not as a separate branch.
- **We keep our own world grounding.** `pose.py estimate()` places the root by field homography,
  so PromptHMR's SLAM/SPEC world is dead weight — setting `results['has_slam']=True` and
  injecting our PnLCalib R/T/focal/centre skips DROID-SLAM, Metric3D *and* SPEC. Their
  `phmr_vid.py` already builds its intrinsics from `results['camera']`, so the seam exists.

## The run on our own clip (2026-08-05) — masks do not beat boxes; the prompt was the wrong hypothesis

`scripts/prompthmr_find_crossing.py` ranks 60 frames of the Colombia clip by the strongest
player–player box IoU using **our own** RF-DETR + ByteTrack, so the boxes are the ones the real
pipeline produces. Winner: **frame 29, tracks 1 vs 2, IoU 0.511** (18 players in frame).

`scripts/prompthmr_mask_ab.py --frame 29` then runs the image model twice on that frame — once
`mask_prompt=True` with SAM masks prompted by those same boxes, once `False` — and scores each
player by the IoU between his projected mesh silhouette and his own SAM mask.

| | masks | boxes | delta |
|---|---|---|---|
| track 1 (crossing) | 0.614 | 0.620 | −0.006 |
| track 2 (crossing) | 0.413 | 0.392 | +0.021 |
| mean over 18 | 0.495 | 0.477 | +0.019 |

**Flat.** Nine players improved, nine got worse; the crossing pair — the entire point — moved by
less than the spread. Do not read this as "masks don't help". Read the scale:

- PromptHMR letterboxes the **whole frame** to 896², so 1920 → 896 is 0.467×. Our players are
  63–115 px tall at 1080p, i.e. **29–54 px** in the model's input.
- The mask prompt is 256², a further 0.286× — so each player's mask is **~10 px tall**. There is
  almost no shape in it to prompt with.
- The metric degrades with it: the reference is a SAM mask cut at that same size, and the mesh
  "silhouette" is projected vertices plus a 3×3 dilation, which at 40 px is a blob.

So the whole-frame A/B is a control that came out null for a reason that is about resolution, not
about the hypothesis. `--crop-pair` crops to the crossing pair first (frame 29 → a 124×370 window,
the taller player 112 px → **271 px** at 896) and re-runs the identical A/B in the regime the model
was trained for. Both runs are kept: the flat one is the honest control.

One thing the crop forces: the solved **4169 px** focal describes the 1920-wide frame, and on a
370 px window it is a ~5° FOV. The model has never seen one, and its first crop run put every
vertex off-canvas (IoU 0.000, no mesh drawn). The script now substitutes a plain `max(W, H)`
pinhole centred on the crop. Depth from that run is not metric — which costs nothing, because
per "We keep our own world grounding" above, `pose.py estimate()` places the root by field
homography. The A/B only asks whether the mesh lands on its own player's pixels.

### Crop result: also flat

| | masks | boxes | delta |
|---|---|---|---|
| track 1 (Colombia, behind) | 0.687 | 0.658 | +0.029 |
| track 2 (Congo DR, in front) | 0.573 | 0.593 | −0.021 |
| mean | 0.630 | 0.625 | **+0.004** |

`out/phmr_ab/ab_f29_crop_zoom.jpg`. The scale fix worked — IoU rose 0.50 → 0.63 and both meshes are
now legible — and **the two panels are indistinguishable by eye.** One player up, the other down,
mean inside the noise. That is the second null, and this one is not explained away by resolution.

**So the mask prompt is not what fixes #132.**

### Cross-person attention is opt-in, and off by default

The obvious follow-up was "then it must be the joint multi-person pass" — one forward with
cross-person attention, so two bodies cannot both claim the same pixels, where our
`GVHMRPoseEstimator` calls `HMRBackend.estimate_bodies` **per track**. `--joint-vs-solo` tests
exactly that: co-decode both players in one pass, versus one pass per player, same weights and
same image.

First run came back **bit-identical** (+0.000 on both tracks). That is not a null, it is a bug
report about the model:

- `SMPLDecoder.forward` takes a `crossperson` flag, and only under it does the BUDDI-style block
  in `twoway_transformer.py:256` flatten every person's two output tokens into one sequence.
- The flag is fed from `batch['interaction']` (`phmr.py:73`), and upstream's own `prepare_batch`
  leaves it `None` unless the caller asks (`inference.py:72`).

**With it off — the default — a PromptHMR multi-person pass is N independent decodes.** The
people share an image encoder, nothing more. Every earlier number on this page was measured that
way, including both arms of the masks-vs-boxes A/B.

### All four configurations, frame 29 crossing pair

| arm | `crossperson` | track 1 (behind) | track 2 (in front) | mean |
|---|---|---|---|---|
| masks | off | 0.687 | 0.573 | **0.630** |
| boxes | off | 0.658 | 0.593 | **0.625** |
| joint (both in batch) | **on** | 0.662 | 0.552 | **0.607** |
| solo (one per pass) | **on** | 0.666 | 0.522 | **0.594** |

With the flag on, co-decoding does beat solo (+0.013 mean, **+0.030** on the front player, and
`ab_f29_crop_solo_zoom.jpg` shows why — solo lets track 2's mesh slide right onto its neighbour's
shorts). But turning the flag on at all *costs* 0.018 against plain boxes. The whole family sits
in a 0.59–0.63 band.

### The honest limit of this experiment

`ranking` in `tracks.npz` says frame 29's box IoU **0.511 is the maximum over all 60 frames**
(runner-up 0.415). Zoom in and the pair is two *adjacent* players — the front one covers the back
one's legs, both are fully visible and separable. That is not the failure #132 describes.

So the null is real but narrow: **on the hardest overlap this 60-frame window contains, none of
PromptHMR's multi-person machinery moves the mesh measurably.** It does not yet say anything
about a true occlusion, because the clip window does not contain one.

## The widened search (2026-08-05) — and the clip is two shots, not one

Ranking 60 frames by box IoU was wrong twice over. Corrected in
`scripts/prompthmr_find_crossing.py`, which now also ranks by **cover** = `inter / area(back)`,
the back player being the one whose box bottom sits *higher* (players stand on a plane under a
raised camera, so lower feet = nearer = the occluder).

**The clip contains a hard cut at frame 236.** A colour-histogram scan over all 334 frames finds
exactly one boundary (L1 0.775; the largest delta anywhere else is the same frame). Frames 0–235
are the wide broadcast shot the calibration was solved on; **236–333 are a close-up replay from a
different camera.** Consequences, in order of how much they cost:

- Every hit in the naive full-clip ranking (frames 237–292, cover 1.000) lives in shot 2 and is
  unusable for #132.
- In shot 2, RF-DETR/COCO emits a **738×806 px** "person" box swallowing half the frame, plus
  crowd and touchline officials. `cover 1.000` was that blob containing a steward in a hi-vis bib.
  Box-prompted SAM then cut out the *wrong* person, so the `--verify-masks` "visible fill" column
  read 0.38–0.58 ("clear") for pairs that were not players at all. Sanity-bound the boxes.
- **The pipeline has no shot-cut detection.** Nothing stops `--frames 334` from running tracking
  and calibration straight through frame 236 and silently blending two cameras. Today we are only
  safe because every run takes 48–60 frames from the start. This is its own defect.

### The real #132 frame

Re-ranked inside shot 1, with boxes bounded to plausible players (h < 0.45·H, w < 0.30·W, h > 25):

| frame | cover | back ← front | h_back | h_front | box IoU |
|---|---|---|---|---|---|
| 121 | 0.867 | 33 ← 36 | 47 | 92 | — |
| **124** | **0.779** | **110 ← 97** | **86** | **86** | **0.649** |
| 34 | 0.708 | 5 ← 9 | 83 | 88 | 0.415 |
| 87 | 0.682 | 85 ← 15 | 86 | 94 | — |
| 207 | 0.657 | 126 ← 66 | 78 | 107 | — |

~~**Frame 124 is the case.**~~ **Withdrawn 2026-08-05 — I never looked at the pixels.** The
paragraph that stood here read "two Congo DR players in the *same light-blue kit*, one directly
behind the other, equal apparent height (86 px, so equal depth) … the back player reduced to a
head, a shoulder and one boot". Every word of it was inferred from box geometry. `out/phmr_ab/
f124_pair_boxes.png` shows what is actually there: **one** light-blue player, with both boxes
drawn around him 15 px apart. There is no hidden second player. See "What frame 124 really is".

### What frame 124 really is — and the measurement that settles it

The two teams wear light blue (Colombia) and yellow (Congo DR), so the **mean torso colour inside
a box says which player an id is on**. That is a ground truth no IoU can supply, it needs no model,
and `scripts/track_continuity.py --kit-scan` now reads it for every box in the clip.

Sampling the upper-middle of each box across the crossing:

| frame | track 97 | track 110 | track 85 |
|---|---|---|---|
| 121–123 | **BLU** | — | YEL |
| **124** | **BLU** | **BLU** ← duplicate | YEL |
| 125–127 | missing | BLU | YEL |
| **128–130** | **YEL** ← swapped | BLU | dead @127 |
| 131+ | YEL | dead @130 | — |

So the failure at frame 124 is two failures, and neither is the one the ticket assumes:

1. **A duplicate detection.** RF-DETR emits two boxes for the same blue player; ByteTrack accepts
   the second as a new identity. Track 110 is not a player, it is a second box on player 97.
2. **An identity swap.** Three frames later track 97 reappears **on the yellow player** and keeps
   that id to frame 180, while track 85 — the yellow player's real id — dies at 127. Nothing marks
   the handover: id 97 silently becomes a different human, carrying its avatar, kit assignment and
   motion history with it.

Extended to the whole of shot 1 (frames 0–235, 38 plausible player tracks):

```
38 of 38 tracks are broken somewhere in the window
 9 tracks change team  — 3, 7, 9, 14, 15, 36, 90, 97, 126   (track 15 flips three times)
 2 duplicate detections — f34 (5,9, IoU 0.415, YEL) and f124 (97,110, IoU 0.649, BLU)
```

Spot-checked by eye at ×6: `out/phmr_ab/swap_t3.png` shows track 3's box sitting on a blue player
at frames 34–35 and on the yellow player at 36–37. The swap is real, not a colour artefact.

**This is the strongest evidence for #132 so far, and it is entirely about tracking.** Nine of
thirty-eight identities in an 8-second shot end up on the wrong human. That defect needs no pose
model to demonstrate and no pose model can repair it downstream.

### The real occlusions, now that duplicates are excluded

Requiring the two boxes to hold **opposing** kits — the honest definition of "two different men" —
re-ranks shot 1 by cover:

| frame | cover | back ← front | kits |
|---|---|---|---|
| **87** | **0.682** | 85 ← 15 | YEL behind BLU |
| 134 | 0.542 | 104 ← 15 | YEL behind BLU |
| 140 | 0.504 | 104 ← 15 | YEL behind BLU |
| 226 | 0.500 | 130 ← 71 | YEL behind BLU |
| 13 | 0.494 | 21 ← 17 | YEL behind BLU |

Frame 87 replaces frame 124 as the pose test case, and it is a better one: opposing kits mean the
two players are separable by colour, so per-player masks can be built and *verified* rather than
assumed. `out/phmr_ab/occ_f85_88.png`.

## The plan from here

### Stage A — prove the defect exists (this baseline was never measured)

Every number on this page compares candidate *fixes* to each other. Nobody had shown our own
pipeline breaks on frame 124, so "PromptHMR fixes #132" was unfalsifiable.

#### A1, tracking — **DONE 2026-08-05, and it fails hard**

`scripts/track_continuity.py --window 115 135 --pair 97 110`, reading the same `tracks.npz`:

```
window 115-135: 19 tracks alive
  10 of 19 tracks are broken somewhere in the window
   33  DIED @123      99  DIED @124      85  DIED @127
  110  BORN @124, DIED @130           113  BORN @128
   97  gaps at 125,126,127
box pairs above IoU 0.5:  frame 124, tracks 97 & 110, IoU 0.649, centres 15 px apart
```

Frame by frame, the two players converge, merge, and the tracker comes apart:

| frame | 97 | 110 | |
|---|---|---|---|
| 115–123 | tracked, drifting right | — | one player visible to the tracker |
| **124** | (1080, 673) | (1081, 658) | **two ids, 15 px apart, IoU 0.649** |
| 125–127 | **MISSING** | (1077, 676) | 97 is gone for three frames |
| 128–130 | (1100, 667) | (1070, 667) | they re-separate, IoU back to 0.13 |
| 131+ | tracked | **dead** | 110 lasted 7 frames total |

**This is #132, with numbers** — but read the mechanism off the kit scan above, not off this
table. Track 110 is a *duplicate box on player 97*, not a phantom identity for a hidden player;
track 97 blinks out for three frames and comes back **on the other team's player**. Clip-wide,
9 of 38 identities in shot 1 end up on the wrong human and all 38 are broken somewhere.

The last paragraph here used to claim that at frame 124 "a per-crop estimator is handed
essentially the same pixels twice" and call that the second half of the ticket. It is handed the
same pixels twice — because it is the same man twice. That is a *tracking* fault presented to the
pose stage, not evidence that pose fusion happens. A2 has to be run on a real occlusion instead.

#### A2, pose — **DONE 2026-08-05, and it does NOT fail**

`scripts/smplestx_occlusion.py --frame 87 --pair 15 85`, SMPLest-X Huge on CPU, fed our own
ByteTrack boxes through the production `_infer_crop`. Both SAM masks were checked against kit
colour before anything was scored (15 → BLU, 85 → YEL, both as expected), so these numbers are
measured against the right two men:

| | track 15 (front, BLU) | track 85 (back, YEL) |
|---|---|---|
| own-mask IoU | 0.567 | 0.575 |
| mesh landing on **its own** player | **0.787** | **0.897** |
| mesh landing on **the other** player | 0.114 | 0.089 |
| solved depth | 39.58 m | 38.15 m |

**No fusion.** Two crops of an overlapping pair produce two distinct bodies on two distinct
players; 79 % and 90 % of each mesh sits on its own man and under 12 % strays onto the other.
`out/phmr_ab/a2_smplestx_f87_15_85.png` says the same by eye — the only visible defect is one
splayed right leg on the occluded player, which is a pose error, not an identity error.

Two things weaken this frame as a test, and both must be said rather than buried:

- **The depth order flips** (back player solves 1.42 m *nearer* than the front one) — but our
  pipeline never reads HMR depth. `SMPLestXBackend.estimate_bodies` returns orientation, body
  pose, betas and pelvis-above-foot and *no translation at all*; world position comes from the
  foot point through the pitch homography. So metric 3 measures a quantity our architecture
  already routes around. It is listed here because it was promised, not because it costs us.
- **The occlusion is mild.** `cover = 0.682` is box arithmetic. In pixels, SAM gives the back
  player 1973 px inside a 3354 px box — a fill of **0.59**, where an *unoccluded* standing player
  runs 0.30–0.45. He is barely hidden. This is the worst genuine two-player overlap in the shot,
  so the honest reading is that **this clip does not contain a hard occlusion at all**.

#### A2 setup notes (superseded plan text below kept for the run recipe)

**On frame 87, not 124**

Run the production per-crop path — SMPLest-X Huge, `--pose gvhmr --pose-backend
pitch3d.adapters.models.smplestx_backend:make` — on **frame 87 (track 85 YEL behind track 15 BLU,
cover 0.682)** and score it with the metrics below. This is the control every other arm is
measured against.

Ready to run locally, no pod. Weights verified by hand: `models/smplest-x/smplest_x_h.pth.tar`,
8.246 GB, 519 tensors / **687.2M params** under key `network`, plus optimizer state (which is what
makes a 687M model an 8.2 GB file). Staged into the layout the loader expects at
`backends/SMPLest-X/pretrained_models/smplest_x_h/` and
`backends/SMPLest-X/human_models/human_model_files/smplx/` — the latter is six relative symlinks:
the three MPI `SMPLX_{NEUTRAL,MALE,FEMALE}.npz` out of `models/smplx/`, plus `SMPLX_to_J14.pkl`,
`MANO_SMPLX_vertex_ids.pkl` and `SMPL-X__FLAME_vertex_ids.npy`, which the MPI zip does **not**
carry and which came from the ungated `camenduru/SMPLer-X`. The `SMPLX` singleton loads clean
under our numpy 2.4.6 / smplx 0.1.28 (10475 verts, 20908 faces, J14 regressor 14×10475).

**Two things it took to run locally**, both solved and both in `scripts/smplestx_occlusion.py`:

- SMPLest-X hardcodes `.cuda()` in ~15 places on the inference path (`models/SMPLest_X.py:25,47,54`,
  `utils/transforms.py`, `main/base.py`) and our torch is `2.12.1+cpu`. They are all `.cuda()`
  *method* calls, so making `torch.Tensor.cuda` / `torch.nn.Module.cuda` identity for the run
  clears every one without patching the vendored checkout. `DataParallel` already degrades to a
  plain call when there are no devices.
- **The shipped checkpoint froze this machine twice.** `smplest_x_h.pth.tar` is 8.25 GB of which
  two thirds is optimizer state; loading it plus the model plus three SMPL-X gender layers
  exceeded the ~12 GB free here. `models/smplest-x/smplest_x_h_slim.pth.tar` is the same 519
  tensors / 687.2M params with `optimizer` dropped — **2.75 GB**, and it is the script's default
  (`--ckpt`). Written with `torch.load(..., mmap=True)` so the strip itself never holds 8 GB.
  `--threads 6` keeps a ViT-H off all 16 cores; without it the laptop is unusable while it runs.

### Stage B — the head-to-head, same boxes, same frame, same metrics

Our ByteTrack boxes feed every arm, so nothing here measures a different detector or tracker.

1. **SMPLest-X per-crop** — production today.
2. **PromptHMR, box prompt** — full-frame, one pass, `crossperson` off (upstream default).
3. **PromptHMR, mask prompt** — arm 2 plus SAM masks.
4. **PromptHMR, box prompt, `interaction=True`** — the BUDDI cross-person block forced on.

Arms 2–4 are cheap once arm 1 runs; the honest comparison is 1 vs 2, and 3/4 only earn their
dependencies if they beat 2.

**Not run, and on this evidence it should not be — 2026-08-05.** Stage B exists to find a pose
model that fuses two crossing players less than arm 1 does. A2 measured arm 1 fusing by 0.09–0.11
at the hardest genuine overlap in the clip. There is no gap for arms 2–4 to close, and the three
of them together cost a 2.2 GB checkpoint, a 6.5 GB SAM branch and a new inference path. Running
them would produce a table, not an improvement. Revisit only if a clip with real occlusion
appears, or if A1's tracking fix changes which boxes reach the pose stage.

### What exactly gets compared

One number is not enough — own-mask IoU cannot tell a *bad fit* from a *fused* one.

1. **Own-mask IoU** — mesh silhouette ∩ that player's SAM mask, over their union. "Is the mesh on
   the right person?" Already implemented in `prompthmr_mask_ab.py`.
2. **Cross-contamination** *(to add — this is the actual fusion metric #132 describes)* — the
   fraction of player A's mesh silhouette landing inside player B's mask. Two per-crop meshes
   both collapsing onto the front player is exactly the failure, and metric 1 alone reports it
   only as a mild drop.
3. **Depth order** — does the back player's solved root stay *behind* the front player's? A
   per-crop estimator has no reason to preserve it, and a flipped pair is visible in the render.
4. **The eye, on a ×5 zoom panel.** Per `CLAUDE.md` this outranks 1–3 when they disagree.

Metrics 1–3 also run over the window 115–135, not just frame 124, so a lucky single frame cannot
carry the verdict.

### The tracking fix, built and measured 2026-08-05

`ByteTrackBackend` sampled a track's kit colour from **its first 8 frames only**, so a track that
picks up a different player mid-way was undetectable by construction — track 97 is scored on
frames 112–119 and keeps that label while wearing the other kit from 128. `RawTracklet` now also
carries `appearance_series (T, D)` sampled over the whole span (the same decode pass, just more
crops), and `split_on_kit_change` cuts a track at a sustained kit change, giving each piece its
own id. The first piece keeps the original id, so healthy tracks are untouched.

Measured on shot 1 with the **same detections and the same association** in both arms — only the
split differs, so the delta is the fix and nothing else (`--also-nosplit` writes the control):

| `kit_split_min_run` | player tracks | splits | tracks still changing team |
|---|---|---|---|
| off (control) | 38 | — | **9** |
| **3 (default)** | 56 | 18 | **0** |
| 4 | 53 | 15 | 1 |
| 5 | 49 | 11 | 2 |
| 6 | 49 | 11 | 2 |

Every split was audited back to its parent by exact frame tiling, not by eye-balling counts:

```
min_run=4, 15 splits from 9 parents -- 8 of them on the --kit-scan swap list, 1 not (track 2)
  parent   2 -> 6 pieces   parent  15 -> 4 pieces   every other parent -> 2 pieces
```

- **The one parent `--kit-scan` did not flag is a true positive.** `out/phmr_ab/t2_cut.png`: track
  2's box sits on a **referee** at frames 38–40, swells to hold both as a Colombia player crosses,
  and leaves on the player. `--kit-scan` is blind to it because a referee's dark kit classifies as
  neither team and those frames are skipped — so **9 is a floor, not the count**.
- **Track 2 shredding into 6–8 pieces is the honest reading of a box that really does wander
  between people**, not the split rule misfiring. Every other parent yields 2 pieces, and track 15
  yields 4, matching its three measured kit flips.
- The one track `min_run=4` misses is 90, whose yellow phase is **3 boxes** (frames 97, 98, 110)
  against a 104-frame blue phase — real evidence, just short. `min_run=3` catches it and cuts no
  parent that `min_run=4` did not already cut.

`kit_split=False` restores the previous behaviour (auto-detect plus manual override). Referees are
excluded from both the centre fit and the cut: their kit is a third colour that would pollute the
centres and split good officials.

**Verified end to end, not just in the probe.** The split fires inside the standard 48-frame window
too (21 → 27 player tracks), and `python -m pitch3d.app.cli --frames 48 --detector rfdetr --tracker
bytetrack --render overlay --export gltf` consumes the new ids without complaint: 21 subjects, both
teams assigned, every gate run, `scene.json` written. It also cannot be quietly undone downstream —
`core/orchestration/continuity.py` carries `require_same_team = True`, so the structural stitcher
may re-link fragments of one team but never bridge a kit change. The two compose: the split
separates humans the tracker fused, the stitcher re-joins fragments of the same human.

**What this does not do.** Splitting fixes *membership* — each piece is one human wearing one kit —
but it does not re-link. The yellow player in piece `97→144` and track 85 are the same man under
two ids.

#### The split turns out to be what makes the existing stitcher work

`scripts/identity_budget.py --frames 236`, all four arms over the same cached detections:

| `kit_split` | stitch | player ids | merges found |
|---|---|---|---|
| off | off | 38 | — |
| off | **on** | 37 | **1** |
| on | off | 56 | — |
| on | **on** | **36** | **14** |

`core/orchestration/continuity.py` has been in the tree all along and is on by default in the CLI,
and on the unsplit tracks it finds **one** merge in 38 tracks — effectively inert. On the split
tracks it finds **fourteen**. The reason is its own `require_same_team` gate: a track that covers
two humans has an averaged, meaningless team label, so it can match nothing. Give the fragments an
honest team and the stitcher starts doing its job. The two features are not in tension; the split
is a **precondition** for the stitch.

**Read the counts together, not alone.** 37 (unsplit + stitched) and 36 (split + stitched) look
alike, but 9 of the 37 are ids covering two humans, and none of the 36 are. Fewer ids is what the
broken tracker already scored well on, by fusing two players into one.

**The honest remaining gap.** Shot 1 shows a median of **17** plausible players per frame (min 13,
max 19), so a shot-long budget is ~17–19 identities. We are at **36** — about two ids per player.
That is the stitcher's conservatism, not a defect: `max_gap = 12` frames (0.4 s) and a strict
predicted-position gate, because per its own docstring "a missed merge leaves two fragments, but a
*wrong* merge teleports a body". Widening those gates is the next measurable step, and this table
is the harness to judge it by.

### Stage C — the decision, taken 2026-08-05

The first bullet is the one that landed:

> Arm 1 not measurably worse → #132's premise is wrong for this clip.

**#132 splits in two, and only one half survives contact with the pixels.**

- **"…and fuse per-crop poses (occlusion)" — not reproduced. Drop it.** At the clip's hardest
  genuine two-player overlap the production estimator keeps 79 % and 90 % of each mesh on its own
  player. The overlap itself is mild (back player fills 0.59 of his box). No pose model can be
  justified against a failure that does not occur, so **PromptHMR is not adopted**: no 2.2 GB
  image checkpoint, no 6.5 GB SAM 3 branch, no `interaction` flag, no second inference path. The
  work is not wasted — it is what proved the mask prompt, the joint pass and the cross-person
  attention all buy nothing here, and that is a permanent answer, not a deferral.
- **"Player crossings break ByteTrack IDs" — reproduced, and worse than the ticket says.** All 38
  player tracks in shot 1 break somewhere; **9 change team**; 2 duplicate detections put two ids
  on one man. That is where the effort goes.

The tracking work #132 actually needs, in the order it pays:

1. **Reject duplicate detections** — same kit, IoU ≥ 0.35, box heights within 25 % is already a
   working test (`--kit-scan` finds exactly 2 in shot 1, both confirmed by eye). NMS before
   ByteTrack, not after.
2. **Refuse to hand an id to a new kit.** Team colour is a one-line per-box feature that our two
   teams make trivially separable, and it catches the swap class that appearance re-ID is meant
   to catch, at no model cost. It cannot fix a swap on its own, but it can *mark* one — which is
   R-6, and better than a silent handover.
3. **Shot-cut detection** — a real defect found on the way, unrelated to occlusion but still open.

### Status of the pieces

- SMPLest-X Huge runs locally on CPU (`--ckpt smplest_x_h_slim`, 2.75 GB, `--threads 6`). No pod
  was needed for the download or the run.
- The 6.5 GB SAM 3 branch stays out. So does PromptHMR — see Stage C.
- SAM ViT-B (443 MB, already cached) earns its place as the *measurement* tool: it is what makes
  own-mask IoU and cross-contamination checkable, and the kit test is what makes SAM's own output
  checkable.

## What will bite (found by reading their code, not by running it)

1. ~~**Gated classic SMPL blocks any run.**~~ **Withdrawn — measured false.** `PHMR.__init__`
   does construct `SMPL('data/body_models/smpl')`, but the section above runs a forward pass
   with that directory absent. Costs ~10 lines of stub in our glue; needs no MPI registration.
2. **Their SAM2 tracker seeds identities from ONE frame** — first frame with detections, then
   `break`, then pure propagation. Players entering later are never tracked. Concrete reason to
   keep RF-DETR + ByteTrack and take only the mask-prompted HMR.
3. **`ClipEncoder` downloads MetaCLIP at construction** (`open_clip.create_model(...,
   pretrained='metaclip_fullcc')`) even though the checkpoint overwrites it — an offline pod
   fails here. Measured above as *provably* redundant: drop the `pretrained=` kwarg and the load
   is still 0-missing. DINOv2 does not download (vendored; `data/dinov2_vitl14_pretrain.pth`
   likewise optional).
4. `det_height_thresh=0.3` drops anyone shorter than 0.3× the tallest box — far-side players.
   Lower to ~0.1 for a wide broadcast frame.
5. Frames are downscaled to `max_height=896`; our 1080p loses detail before the model sees it.
6. **The 180° roll applies here too** — PromptHMR eats raw frames, so rotate first or every pose
   comes out inverted.
7. **`xformers` is a hard import**, not an optional accel: `models/components/{transformer,
   twoway_transformer}.py` import `memory_efficient_attention` at module level (vendored DINOv2
   guards its own use, these two do not). Add it to the pod install, or shim it onto
   `F.scaled_dot_product_attention` as `check_prompthmr_weights.py` does.
8. **Three stacked non-commercial licences**: Meshcapade (PromptHMR), ZJU/GVHMR (the vendored
   video head), NAVER CC BY-NC-SA 4.0 (the Multi-HMR image encoder). Research-fine; a product
   needs three conversations.
