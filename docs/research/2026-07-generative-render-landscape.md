# Generative-model landscape vs. the AVATAR task (research, 2026-07-03)

**Question asked:** if we started from scratch today — with video character-swap models,
image→3D models, upscalers and general video generators available — how would we build
"broadcast clip → faithful novel-view video"? Fidelity target is a *stunt double*: poses,
emotions and scene essence must be right; identity likeness matters but "не до фанатизма".

**Method:** three parallel web-research passes (2026-07-03): (A) character replacement /
pose-driven human video, (B) camera-controlled re-rendering + 4D reconstruction,
(C) world-frame HMR + one-shot avatars + environment-to-3D + finishing.

---

## TL;DR — the verdict

**All three research passes converged on the same architecture: reconstruct-then-condition.**

1. **No end-to-end generative path exists for this task.** Direct "re-camera" of a broadcast
   clip tops out at ±10–30° from the source ray, ~480–720p, 81–121 frames — and its
   monocular-depth backbone fails precisely on telephoto pans, 5–50 px players, textureless
   grass and stochastic crowds. Character-swap models top out at 1–7 large-in-frame subjects;
   22 players at 50–150 px are out-of-distribution for every model surveyed.
2. **The measured 3D scaffold is the irreplaceable part** — and it is exactly what this
   project already builds (calibration → world-frame SMPL-X → virtual operator). No 2026
   model supplies world-frame pose truth; several *consume* it as conditioning.
3. **The photoreal-CG finish is the replaceable part.** The 2026 generation of
   structure-locked video-to-video (depth/pose-locked restyle), point-cloud-conditioned
   re-rendering, one-shot splat avatars and open video restoration turns "make Cycles look
   like broadcast" into a solved conditioning problem. The renderer's job shifts from
   *photorealism* to *clean conditioning maps*.

So: built from scratch today, the system would keep this project's measured core nearly
unchanged and replace the "make CG pretty" effort with a generative finishing layer.
ADR-0007 anticipated exactly this split (seam A: render; seam B: amplify+inpaint) — the
news is that seam B is now buildable with open weights.

---

## A. Video character replacement / pose-driven human video

| Model (org, date) | I/O | Weights | Multi-person | Limits |
|---|---|---|---|---|
| **Wan2.2-Animate-14B** (Alibaba, 2025-09) | char image + ref video → animate or **replace-in-video**; skeleton-aligned body; Relighting LoRA | **Open** | one char per pass | ~5 s clips; trained on large-in-frame subjects |
| **DreamActor-M1/V2** (ByteDance, 2025/2026) | image + driving video (3D skeleton + head sphere; V2 reads raw pixels) | API | single | identity drift on long clips |
| **OmniHuman-1.5** (ByteDance) | image + audio → >1 min avatar video | API (~$0.14/s) | yes (dialogue) | audio-driven, not pose-controllable |
| **Runway Act-Two** | performance video ≤10 s + character | API | **one per generation**; multi = crop+composite | gesture control lost with video refs |
| **Viggle JST-1 v3** | video + char refs, per-frame 3D | API | **up to 7 tracks** | meme-grade fidelity; 7 ≪ 22 |
| **Kling 3.0 Motion Control** (2026-03) | ref video + char image | API | single-oriented | short clips |
| **HunyuanVideo-Avatar** (Tencent) | image + audio, multi-char dialogue | Open (24 GB) | yes | audio-driven — wrong control channel |
| **Champ** (ECCV'24) | **SMPL depth+normal+semantic+skeleton** → animation | Open | single | SD1.5-era |
| Research 2026 (MotionWeaver, 3D-pose+view control, StableAnimator++) | pose/4D-anchored | mostly unreleased | **pairs** | benchmarks = 2 people large in frame |

**Verdict A:**
- No model handles 20+ small players in one pass. Product ceiling = Viggle's 7 tracks.
- Realistic route: **per-player crop → generate → composite back into our rendered scene**
  (Runway explicitly prescribes crop-and-composite). Upscale-before/downscale-after
  mitigates the small-figure OOD problem. Cost scales ×22.
- The conditioning these models accept (DWpose/skeleton, SMPL depth/normal/semantic) is
  **exactly what we can render from measured SMPL-X from any novel camera** — the measured
  scene stays the director; generation is a finishing layer.
- Picks: open = **Wan2.2-Animate-14B** (replace mode + relighting LoRA); API = Viggle
  (multi-track), Kling 3.0 / DreamActor V2 for single-hero shots.

## B. Camera-controlled re-rendering + 4D reconstruction

Generative re-camera of *existing* video:
- **ReCamMaster** (Kwai, ICCV'25) — new trajectories, but fixed 480×832 / 81 f.
- **TrajectoryCrafter** (Tencent, ICCV'25) — dual-stream diffusion on depth point-cloud
  renders; authors admit failure on large trajectories; inherits depth errors.
- **GEN3C** (NVIDIA, CVPR'25) — Cosmos-7B on an **explicit 3D point-cloud cache**;
  704×1280, 121-f chunks, ~43 GB VRAM; NVIDIA Open Model License (commercial gated).
- **Vista4D** (Netflix Eyeline, CVPR'26, 2026-04) — "video reshooting with 4D point
  clouds"; Wan2.1-based, **code+models released**, chunked long video, and the 4D point
  cloud is an **editable input** (insert/delete subjects). Best open candidate.
- **EX-4D** (ByteDance, open) — extreme viewpoints via watertight depth mesh (close-range).
- ReCapture / CAT4D (Google) — unreleased. Commercial APIs (Kling 3.0, Runway,
  Higgsfield) control cameras at *generation* time only — no re-camera of existing footage.
- 2026 wave (FreeOrbit4D, LaVR, SpaceTimePilot, PostCam, Track2View) — the field is
  converging on geometry-conditioned re-rendering.

4D reconstruction: MegaSAM (camera+depth from dynamic video), MoSca / Shape-of-Motion
(dynamic Gaussians, degrade sharply off the source view), feed-forward 4D line
(DynamicVGGT, PAGE-4D, St4RTrack, SpatialTrackerV2). Soccer-specific: no production
single-camera volumetric system; SoccerNet-v3D calls the broadcast→pitch-POV gap
"significant" even with multi-cam.

**Verdict B:**
- Direct 90–180° re-camera of a soccer broadcast is **not viable today** (resolution,
  clip length, and monocular depth all fail on our footage). Realistic: ±10–30°.
- But GEN3C / Vista4D / TrajectoryCrafter take an **explicit point cloud as conditioning**
  — we can render *our own reconstructed scene* (SMPL-X + calibrated pitch) into that
  slot, bypassing monocular depth (their main failure mode) entirely.
- **Winning recipe = reconstruct-then-condition:** our geometry defines the new view; the
  diffusion model becomes finishing/inpainting, not the source of 3D truth. Radical angle
  changes become a geometry problem we already solve.

## C. World-frame HMR, one-shot avatars, environment, finishing

**HMR:** **SAM 3D Body** (Meta, 2026-02, open, promptable with boxes/keypoints/masks —
plugs into our tracker; camera-frame → still needs our calib/world lift); SAM-Body4D
(video), Fast-SAM-3D-Body (real-time); PromptHMR (CVPR'25, world-frame variants);
OnlineHMR (CVPR'26, paper-stage); **SMART** (2026-05) = SMPLest-X + RAFT *for soccer* —
validates our pose pick.

**One-shot animatable avatars:** **LHM/LHM++** (Alibaba, open; single image →
SMPL-X-skinned 3DGS avatar in seconds, 8 GB), **IDOL** (CVPR'25, open, <1 s, 1K-res
SMPL-X-aligned Gaussians), AniGS. Limit: trained on ≥1K-px near-frontal refs — our
~100 px crops are OOD; achievable grade = **stunt double** (kit colour/number yes, face
no); super-resolve the best crop first.

**Environment:** **World Labs Marble** (commercial; image/video → 3D scene, exports 3DGS
.ply + collider mesh) — best single answer for the stadium bowl; **TRELLIS.2-4B**
(Microsoft, open, PBR, object-centric); Hunyuan3D 3.0 API-only (2.1 stays open);
Rodin Gen-2 commercial.

**Finishing:** **Wan 2.2 Fun-Control / VACE** (open) — v2v restyle with depth/canny/pose
locks: turns a clean CG render into broadcast-look footage while freezing layout and
poses — *the "generative renderer" piece*. **SeedVR2** (open, quality benchmark) and
**FlashVSR** (CVPR'26, ~17 fps) for restoration/VSR — both now rival Topaz. Relighting:
LightCtrl (ICLR'26, training-free), Light-A-Video (ICCV'25), LTX-2.3 IC-LoRA Day-To-Night
— the floodlit-night look is off-the-shelf.

**Verdict C:**
- First slot-in: **Wan 2.2 Fun-Control v2v (depth+pose-locked) over existing Cycles
  output + SeedVR2 pass** — biggest realism jump, zero pipeline-contract change.
- Next: **SAM 3D Body behind the PoseEstimator port** (promptable = feeds on our tracks);
  keep our calibration/world-lift — that's the moat.
- Avatars: LHM++/IDOL splat "doubles" replace gray meshes; stadium: Marble (or TRELLIS.2
  per-asset); measured pitch/calib stays the world anchor.
- Dead ends: end-to-end character swap at 22×100 px scale; audio-driven avatar models;
  waiting for open Hunyuan3D 3.0; pure text-to-video regeneration (loses measured poses —
  violates R-6).

---

## The three candidate paradigms (from-scratch design)

| # | Paradigm | 2026 status | Fit |
|---|---|---|---|
| 1 | **Pure generative** — re-camera or char-swap the clip end-to-end | ±10–30° max, sub-HD, ≤5 s, ≤7 subjects; loses measured poses | ✗ dead end for 22-player broadcast |
| 2 | **Pure geometry (CG finish)** — measured scene + photoreal Cycles | works (current v2), but CG-look ceiling is high-effort (PBR bodies, faces, cloth…) | ○ correct core, expensive finish |
| 3 | **Hybrid: measured scaffold + generative finish** — geometry defines every pixel's *where*, diffusion defines *how it looks* | every needed piece is open-weights as of mid-2026 | ✓ winner |

**From scratch I would build paradigm 3 — which is ~80 % the existing project.** The parts
I would *not* rebuild differently: perception seams, calibration + world lift, canonical
scene JSON, virtual operator, export contract, Blender as geometry host. The part I would
do differently *from the start*: never aim Cycles at photorealism — render **conditioning
passes** (RGB base + depth + normal + per-subject semantic/skeleton) and put a
structure-locked v2v model on top.

## What this changes for AVATAR

**Keep (validated by the research):** the entire measured core; ADR-0003 two-process
split; ADR-0011 contract + virtual operator; SMPLest-X pick (SMART confirms); R-6.

**Reframe:** the Blender render's product is no longer the deliverable — it is the
*conditioning stack* for a finishing model. Photoreal levers (v2 texture/grass/lighting)
stay useful as a better base plate, but stop being the quality ceiling.

**Add (priority order):**
1. **Finishing seam (ADR-0007 seam B):** Wan 2.2 Fun-Control/VACE v2v, depth+pose-locked,
   over the broadcast-camera Cycles output; then SeedVR2. Prototype = one ComfyUI/CLI pass
   over existing frames. Biggest visual jump, no contract change.
2. **Renderer emits conditioning passes** alongside RGB (depth/normal/semantic per camera)
   — cheap: Cycles/EEVEE AOVs into the existing npz/manifest contract.
3. **SAM 3D Body** behind the PoseEstimator port (promptable with our tracker's boxes).
4. **Splat doubles:** LHM++/IDOL one-shot avatars from best per-player crops (SR first),
   SMPL-X-driven; replaces gray/tinted meshes at stunt-double fidelity.
5. **Stadium:** Marble (commercial) or TRELLIS.2 from clip frames for the bowl.
6. **Experiment:** Vista4D/GEN3C with *our* point cloud in the conditioning slot — tests
   the radical-angle claim without trusting monocular depth.

**Explicit non-goals:** end-to-end char-swap of the raw clip; pure T2V regeneration;
chasing face-level identity at 100 px (stunt-double grade is the target per the user).

---

## Sources

**A (char swap / pose-driven):** [Wan2.2-Animate](https://humanaigc.github.io/wan-animate/)
([paper](https://arxiv.org/html/2509.14055v1), [GitHub](https://github.com/Wan-Video/Wan2.2)) ·
[DreamActor-M1](https://grisoon.github.io/DreamActor-M1/) ·
[OmniHuman-1.5](https://omnihuman-lab.github.io/v1_5/) ·
[Runway Act-Two multi-char workflow](https://help.runwayml.com/hc/en-us/articles/41748090660499-Creating-Multi-Character-Dialogues-with-Act-Two) ·
[Viggle](https://viggle.ai/tools/ai-body-swap) ·
[Kling Motion Control](https://kling.ai/quickstart/motion-control-user-guide) ·
[HunyuanVideo-Avatar](https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar) ·
[Animate Anyone 2](https://arxiv.org/abs/2502.06145) ·
[Champ](https://github.com/fudan-generative-vision/champ) ·
[MotionWeaver](https://arxiv.org/pdf/2602.13326) ·
[3D pose+view control](https://arxiv.org/pdf/2602.21188)

**B (re-camera / 4D):** [Vista4D](https://eyeline-labs.github.io/Vista4D/)
([GitHub](https://github.com/Eyeline-Labs/Vista4D), [arXiv](https://arxiv.org/abs/2604.21915)) ·
[GEN3C](https://research.nvidia.com/labs/toronto-ai/GEN3C/)
([weights](https://huggingface.co/nvidia/GEN3C-Cosmos-7B)) ·
[ReCamMaster](https://jianhongbai.github.io/ReCamMaster/) ·
[TrajectoryCrafter](https://arxiv.org/abs/2503.05638) ·
[EX-4D](https://arxiv.org/abs/2506.05554) ·
[ReCapture](https://generative-video-camera-controls.github.io/) ·
[MoSca](https://arxiv.org/abs/2405.17421) ·
[Shape of Motion](https://shape-of-motion.github.io/) ·
[St4RTrack](https://st4rtrack.github.io/) ·
[DynamicVGGT](https://arxiv.org/pdf/2603.08254) ·
[Dynamic NeRFs for Soccer](https://arxiv.org/pdf/2309.06802) ·
[SoccerNet-v3D](https://arxiv.org/html/2504.10106v1) ·
[Awesome-4D-Spatial-Intelligence](https://github.com/yukangcao/Awesome-4D-Spatial-Intelligence)

**C (HMR / avatars / env / finishing):** [SAM 3D Body](https://arxiv.org/abs/2602.15989)
([code](https://github.com/facebookresearch/sam-3d-body)) ·
[SAM-Body4D](https://arxiv.org/pdf/2512.08406) ·
[PromptHMR](https://github.com/yufu-wang/PromptHMR) ·
[SMART (soccer)](https://arxiv.org/pdf/2605.31551) ·
[WorldPose](https://eth-ait.github.io/WorldPoseDataset/) ·
[LHM](https://github.com/aigc3d/LHM) / [LHM++](https://lingtengqiu.github.io/LHM++/) ·
[IDOL](https://github.com/yiyuzhuang/IDOL) ·
[Marble](https://www.worldlabs.ai/blog/marble-world-model) ·
[TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) ·
[SeedVR2 vs FlashVSR](https://upsampler.com/blog/seedvr-vs-flashvsr-ai-video-super-resolution-2026) ·
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR) ·
[Wan 2.2 Fun-Control restyle](https://www.runcomfy.com/models/community/wan-2-2/fun-control/first-frame-restyle) ·
[VACE v2v](https://www.runcomfy.com/comfyui-workflows/vace-wan2-1-video-to-video-workflow) ·
[LightCtrl](https://github.com/GVCLab/LightCtrl) ·
[Light-A-Video](https://github.com/bcmi/Light-A-Video/) ·
[LTX Day-To-Night LoRA](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Day-To-Night)
