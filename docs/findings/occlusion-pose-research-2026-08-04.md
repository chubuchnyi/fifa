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

User picked candidate 1. Repo + weights pulled to `backends/PromptHMR` (gitignored; the
box-local glue pattern `backends/pnlcalib_backend.py` already uses, which also keeps the
non-commercial code out of the public `pitch3d` tree).

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
  `posedirs`, `lbs_weights`, …) — pure body-model data, no learned weights, so the blocker is a
  licence wall rather than a missing part of the network.

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

## What will bite (found by reading their code, not by running it)

1. **Gated classic SMPL blocks any run.** `PHMR.__init__` constructs `SMPL('data/body_models/smpl')`
   → needs `SMPL_NEUTRAL.pkl` from smpl.is.tue.mpg.de. We have SMPL-X locally, not this.
   **Needs the user's registration** (same MPI account family as the SMPL-X one already used).
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
