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

**Frame 124 is the case.** `out/phmr_ab/f124_pair.jpg`: two Congo DR players in the *same
light-blue kit*, one directly behind the other, equal apparent height (86 px, so equal depth),
box IoU 0.649 — the highest in shot 1 — and the back player reduced to a head, a shoulder and one
boot. Same kit means appearance re-ID cannot separate them either. Frame 29, which all the
numbers above were measured on, is a far easier case that merely scored well on box IoU.

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

**This is #132, with numbers.** Track 110 is born at exactly the crossing frame and dies seven
frames later — a phantom identity that exists only for the duration of the occlusion. Track 97,
a real player who never left frame, blinks out for three frames. Around them 33, 99 and 85 die
and 113 is born, so the crossing churns five identities in a 21-frame window. Downstream each
new id is a *different person*: new avatar, new kit assignment, motion history reset.

And at frame 124 the two boxes are 15 px apart with IoU 0.649, so **a per-crop estimator is
handed essentially the same pixels twice** and has nothing to tell it which of the two players
it is supposed to fit. That is the mechanism behind the second half of the ticket, and it is
what A2 measures.

#### A2, pose — next

Run the production per-crop path — SMPLest-X Huge, `--pose gvhmr --pose-backend
pitch3d.adapters.models.smplestx_backend:make` — on frame 124 and score it with the metrics
below. This is the control every other arm is measured against. Weights verified by hand:
`models/smplest-x/smplest_x_h.pth.tar`, 8.246 GB, loads to 519 tensors / **687.2M params**
under key `network`, plus optimizer state (which is what makes an 687M model an 8.2 GB file).

### Stage B — the head-to-head, same boxes, same frame, same metrics

Our ByteTrack boxes feed every arm, so nothing here measures a different detector or tracker.

1. **SMPLest-X per-crop** — production today.
2. **PromptHMR, box prompt** — full-frame, one pass, `crossperson` off (upstream default).
3. **PromptHMR, mask prompt** — arm 2 plus SAM masks.
4. **PromptHMR, box prompt, `interaction=True`** — the BUDDI cross-person block forced on.

Arms 2–4 are cheap once arm 1 runs; the honest comparison is 1 vs 2, and 3/4 only earn their
dependencies if they beat 2.

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

### Stage C — the decision this buys

- Arm 1 not measurably worse → #132's premise is wrong for this clip. Close it, and the whole
  PromptHMR adoption is unnecessary. *This remains a live outcome.*
- Arm 2 beats arm 1 → adopt PromptHMR for the multi-person pass; the SAM/mask branch (6.5 GB) and
  the `interaction` flag both stay out unless arms 3/4 beat arm 2.
- Either way, shot-cut detection gets fixed, because it is a real defect found on the way.

### Status of the pieces

- SMPLest-X Huge (8.2 GB, `waanqii/SMPLest-X`, ungated) downloading straight to `models/smplest-x/`;
  code checkout in `backends/SMPLest-X`. No pod needed for either.
- The 6.5 GB SAM 3 branch is still unjustified on the evidence so far — do not wire it in yet.

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
