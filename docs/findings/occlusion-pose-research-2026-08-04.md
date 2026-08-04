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
