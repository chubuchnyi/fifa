# Player physics realism — SOTA survey + integration plan (2026-07-06)

Deep survey commissioned after the user's directive:

> «Задача номер один — построить реалистичную физику для игроков и мяча.
> Текущее решение несостоятельно. Человек должен физически бежать с инерцией,
> центром масс, коэффициентом трения и сцепления. Не менять цвет / команду
> (перманентен). Не исчезать. Не проникать сквозь других людей и предметы.»

Three orthogonal problems live under that mandate. Solving each requires its
own SOTA and integration point. **Physics gates alone can't fix identity
flicker; identity locks alone can't fix hovering; foot-planting alone can't
fix inertia.** This doc maps each to concrete open-source components and a
staged roadmap for our stack.

**Current stack:** `rfdetr detect → bytetrack → team-cluster (HSV kmeans) →
PnLCalib homography → SMPLest-X pose → M3-9 kinematic gate + coherence +
T1a/b/c + T2/T3/T4/T6a (this session's work) → anim_export → Blender/Cycles
beauty → Wan2.1-VACE video-to-video → SeedVR2 → hue/tone pin chain`.

**What today's ceiling is:** motion metrics improve (safe_new profile clamped
125 joint violations + 15 orientation violations on a real 60-frame scene);
motion in the final video is **noticeably better** (user's eye verdict). But
three visible failures remain, and they're outside what the correction seam
can reach with heuristic clamps:

1. Kit colour flickers per-player (aggregate team hue is stable to std 0.34°).
2. Root translation and gait are decoupled ("stands but moves").
3. All subjects hover in the air.

---

## 1. Problem A — permanent identity (colour / team / jersey)

**Root cause:** `_assign_teams` in `adapters/models/tracking.py` clusters
appearance per tracklet by k-means on mean HSV. When ByteTrack ID-swaps
mid-video (occlusion, sprint, cluster of players), the "new" tracklet gets
re-clustered — sometimes into the *other* team's centroid. The player's
mesh then renders with the wrong team mask; v2v propagates the colour.

### 1.1 SOTA — plug-and-play post-tracker association

**GTA — Global Tracklet Association (Sun et al., ACCV 2024)** is the
reference solution. It runs OVER any tracker (ByteTrack, Deep-EIoU, OC-SORT)
as pure post-processing:

* **Tracklet Splitter** — DBSCAN on deep feature embeddings inside each
  tracklet; intra-tracklet clusters ≥ 2 → the tracker fused two people → split.
* **Tracklet Connector** — iterative merge across tracklets whose feature
  centroids are close in cosine distance.

Sets SoccerNet HOTA 79.41 → 83.11 %; SportsMOT SOTA 81.04 %. Code:
[github.com/sjc042/gta-link](https://github.com/sjc042/gta-link) (~200 lines
numpy + sklearn.DBSCAN).

**Feature extractor:** OSNet from `torchreid` (2 M params, 20 ms/crop on
GPU), or **CLIP-ReIdent** (SoccerNet Player Re-ID 2023 winner, 93.51 % mAP
test) if we want max quality; both public.

### 1.2 Kit / team classification stability

**Multi-task Learning for joint Re-ID + Team Affiliation + Role**
(Somers et al., 2024, arXiv 2401.09942) — one backbone with three heads.
Learns kit *identity* (not raw pixel colour) so a shadowed jersey and a lit
jersey embed to the same team. Direct answer to "aggregate hue stable but
per-player flips."

Simpler pragma: **whole-tracklet voting** over per-frame kit descriptors
(Automatic Team Assignment paper, ResearchGate 369328672) — replaces our
per-tracklet mean-HSV with a temporal median.

### 1.3 Jersey number OCR

**Koshkina & Elder CVPR 2024** — Jersey Number Recognition via Keyframe
Identification, robust on 320×320 broadcast crops. Emits per-digit
confidence. Provides a hard cross-clip identity lock (a "10" is always "10")
and can seed the T4 `PlayerProfile` schema.

### 1.4 Integration for our stack

Insert **one new stage** between `continuity.stitch` and `assemble_scene`:

```
detect → track → continuity → [NEW identity_gate] → assemble → …
```

`identity_gate` = OSNet features + GTA split/merge + team lock in
`PlayerProfile.appearance.shirt_hue_deg` via `set_operator_field()` (T4
schema already has the field; just uses the operator-immutable path so
auto-tune can't drift it). Kit-inject stage downstream now reads a
temporally stable team mask.

**Cost:** ~1-2 days coding + one pod re-run. Weights ~10 MB.

---

## 2. Problem B — foot contact / no-slide (not gliding)

**Root cause:** SMPLest-X returns per-frame axis-angle pose + fixed
pelvis-above-foot ≈ 0.92 m from a nominal T-pose, without per-frame foot-FK.
The homography anchor puts the ROOT at (foot_xy, 0.92) but the ACTUAL
feet (varying with stride) are wherever SMPL-X FK puts them — usually
above Z=0. Two consequences: all subjects hover; root translates without
step pattern.

### 2.1 SOTA — post-hoc foot-plant on SMPL-X sequences

**HuMoR (Rempe et al., ICCV 2021)** —
[github.com/davrempe/humor](https://github.com/davrempe/humor). Conditional
VAE motion prior + test-time optimization over CVAE latents. Predicts per-
frame binary heel/toe contacts and constrains foot velocity to zero on
contact frames. 30-60 s per 60-frame sequence on GPU.

**LEMO (Zhang et al., ICCV 2021)** —
[github.com/sanweiliti/LEMO](https://github.com/sanweiliti/LEMO). "PhysCap-
lite" energy: (a) foot velocity when planted, (b) foot penetration into a
fitted ground plane, (c) unrealistic pose transitions. Runs as pure L-BFGS
trajectory-opt over existing SMPL-X, no video re-processing. ~5-15 s per
60-frame subject on CPU. Closest architectural match to our correction seam.

**WHAM contact head (Shin et al., CVPR 2024)** —
[github.com/yohanshin/WHAM](https://github.com/yohanshin/WHAM). Trained
per-frame heel/toe contact probability head. Weights + inference stack are
open; lift just the head and feed our SMPL-X sequence to get labels, then
run our own IK correction. ~5 ms/frame on GPU for the head alone.

**Adjacent worth naming:** PhysCap (heavier, PD-controller-in-the-loop),
NeMF (no explicit contact, weaker), TokenHMR (no contact — same failure
mode as SMPLest-X).

### 2.2 Ground plane / pelvis-above-foot per subject

Combine, don't choose:

* **Homography anchor (we have this)** — PnLCalib puts pitch at Z=0 in world
  coords. Ground truth for XY.
* **Per-frame FK on SMPL-X foot vertices** — hardcode 6-8 sole vertex indices
  per foot; run FK on β + pose; take min(z_toe, z_heel). This gives the
  actual pelvis-above-foot per frame (varies with stride/crouch).
* **Robust fusion (recommended default):** median over the whole track of
  `pelvis.z - min(foot_verts.z)` = per-subject standing offset. Replaces our
  nominal 0.92 m with a MEASURED value. This is what LEMO and WHAM do.

### 2.3 Integration — foot_plant_v2

Add `src/pitch3d/core/correction/foot_plant_v2.py` alongside the median-lock
already shipped (T6a). Two stages, ship independently:

**Stage A (cheap, first) — measured pelvis-above-foot.**
* Deps: `smplx` package (already installable), foot vertex indices.
* Cost: one FK pass per frame per subject ~2 ms CPU.
* Output: replace the constant 0.92 m in T6a with a per-subject measured
  offset. Kills the hover bias PROPERLY without touching stride variance.

**Stage B (next tier) — WHAM contact head + anti-slide IK.**
* Deps: torch, WHAM checkpoint (~10 MB).
* Cost: ~50 ms per subject per second of video, GPU.
* Output: per-frame contact labels → windowed L-BFGS over root_transl to
  zero foot velocity on contact frames, preserving pose. Kills the
  glide-walking.

Both fit the existing seam (KEYFRAME_INTERP on ROOT_TRANSLATION). No
Blender / v2v changes.

---

## 3. Problem C — physics-realism (inertia, CoM, friction, collision)

**Root cause:** our correction gates CLAMP violations. They never
INFER a physically-executable trajectory from a bad reference. SMPLest-X
gives noisy joint rotations that don't correlate with root velocity
direction; there's no rigid-body simulator anywhere in the loop; contact
is never solved between subjects.

### 3.1 Three architectural paths, ordered by (cost × quality)

**Path 1 — Momentum / contact-lock refinement (cheapest, ships fast)**

Enforce CoM + angular-momentum consistency as a differentiable loss over
the resolved motion (no simulator). Reference:
[Body Momentum, arXiv 2509.09496](https://arxiv.org/abs/2509.09496). Add
foot-contact zero-velocity constraint from § 2.

* Deps: numpy + scipy L-BFGS-B; no new package.
* Cost: 1-2 iterations of code, ~seconds per subject.
* What it fixes: aggregate CoM drift, foot slide.
* What it doesn't fix: pose still standing while translating (needs walk-
  cycle inference — path 2).

**Path 2 — PACER+ trajectory-conditioned walk-cycle**
[arXiv 2404.19722, CVPR 2024](https://arxiv.org/abs/2404.19722)

Trajectory-conditioned SMPL controller: consumes a 2D XY path and outputs
physically-consistent full-body SMPL motion. Directly solves "root
translates but pose doesn't step." Trained on massive locomotion data
(AMASS + LaFAN + Motion Matching); does NOT need per-clip training.

* Deps: torch + PACER+ weights.
* Cost: modest (~seconds/subject on GPU).
* What it fixes: walk-cycle synthesis from a bad HMR pose sequence with a
  good XY trajectory (which we have from homography).
* Integration: emits corrected `body_pose` (T, K, 3) as
  `POSE_BODY_JOINT` keyframes per subject.

**Path 3 — MuJoCo + PHC simulation-in-the-loop (highest quality ceiling)**

The full physics answer. MuJoCo (with MJX for GPU) simulates 22 SMPL-X
humanoids on a pitch plane with friction; a Perpetual Humanoid Controller
[PHC, Luo et al. 2023,
github.com/ZhengyiLuo/PULSE](https://github.com/ZhengyiLuo/PULSE)] policy
tracks each reference motion. Output is physically valid by construction:
inertia + CoM + friction + subject↔subject collision all enforced by the
solver.

* Deps: mujoco, mjx, PHC weights, SMPL2MJCF converter.
* Cost: 22 subjects × 60 frames ≈ 10-20 s on an A100 (MJX batched worlds).
  Weights + first-load ~30 s. Adds ~$0.05 to the pod cost.
* Integration: emits `ROOT_TRANSLATION` + `POSE_BODY_JOINT` keyframes,
  R-6 tagged with low `subject_frame_conf` for refined frames.

**Concrete precedent — SMPLOlympics
[arXiv 2407.00187,
github.com/SMPLOlympics/SMPLOlympics](https://github.com/SMPLOlympics/SMPLOlympics)**:
their pipeline is `TRAM (HMR from video) → PHC (physics tracker) →
physics-refined SMPL`. This IS our problem, already solved by that team.
Ships with a **soccer 1v1/2v2 physics environment** built on PHC-MJX.

**MaskedMimic (NVIDIA, SIGGRAPH Asia 2024,
[arXiv 2409.14393](https://arxiv.org/abs/2409.14393))** is a stronger
alternative for OUR use case: unified controller trained via masked motion
inpainting → gracefully handles low-confidence / occluded / extrapolated /
teleport-interpolated frames (which we have). Better fit than raw PHC for
a noisy HMR upstream.

### 3.2 Sports-specific baselines and datasets

* **SMART — FIFA Skeletal Tracking Challenge 2026
  [arXiv 2605.31551](https://arxiv.org/abs/2605.31551)** — current SOTA on
  broadcast soccer: SMPLest-X + RAFT camera tracker + foot-plane anchoring
  + 2-pass smoothing. **Starter kit is public
  ([FIFA-Skeletal-Tracking-Starter-Kit-2026](https://github.com/FIFA-Skeletal-Light-Tracking-Challenge/FIFA-Skeletal-Tracking-Starter-Kit-2026))**;
  adopt the anchoring + smoothing directly.
* **PhysicsFC [arXiv 2504.21216](https://arxiv.org/abs/2504.21216)** —
  user-controlled physics-based football player controller. Shows SMPL-
  shaped humanoids executing football skills under rigid-body physics with
  ball contact. Not a reconstructor but the reference for what "correct
  football physics" looks like.
* **SoccerNet Game State Reconstruction (CVPR-W 2024)** — pipeline for
  identity persistence via YOLOv5m + DeepSORT + Re-ID + orientation +
  jersey OCR. Reuse the identity components (§ 1) but its physics is basic.
* **Datasets for eval:** WorldPose (ECCV 2024), SoccerNet GSR,
  SoccerNet-ReID (340k thumbs), SoccerNet-Jersey.

### 3.3 Post-hoc physics correctors (drop-in idea)

* **LARP (Learned Articulated Rigid body Physics, ECCV 2024,
  [arXiv 2410.12023](https://arxiv.org/abs/2410.12023))** — neural surrogate
  simulator, ~10× faster than analytical; differentiable; SMPL-compatible
  after joint remap.
* **Plug-and-Play Physical Motion Restoration
  [arXiv 2412.17377, Dec 2024](https://arxiv.org/abs/2412.17377)** — self-
  declared post-hoc restorer for high-difficulty motions; inserts between
  any HMR and rendering. Explicitly built for sports dynamics. Verify code
  release before committing.
* **Contact and Human Dynamics
  [arXiv 2007.11678](https://arxiv.org/abs/2007.11678) / DiffPhy trajectory
  opt [arXiv 2205.12292](https://arxiv.org/abs/2205.12292)** — offline
  trajectory-opt over rigid-body dynamics + predicted foot contact.
  Minutes/clip, higher final quality than any RL approach.

---

## 4. Integration architecture — proposed

```
[EXISTING]  detect → track → continuity.stitch
[ADD § 1]                      → identity_gate (OSNet + GTA + Team lock)
[EXISTING]  → PnLCalib calib → SMPLest-X pose → assemble Scene
[EXISTING]  → coherence + M3-9 + T1a/b/c + T2 + T4a-c + T6a (median-lock)
[ADD § 2A]                      → foot_plant_v2 stage A (measured pelvis)
[ADD § 2B]                      → foot_plant_v2 stage B (WHAM contact + IK)
[ADD § 3]                       → physics_refine_gate
                                    mode: off | momentum | pacer | phc | cio
[EXISTING]  → anim_export → beauty render → Wan-VACE → SeedVR2 → pins
```

All additions land through the ADR-0002 Correction seam
(`KEYFRAME_INTERP` on `ROOT_TRANSLATION` / `POSE_BODY_JOINT`) — inspectable,
disable-able, R-6 tagged. YAML-parametric under `config/physics.yaml`.
Named profiles: `identity_lock` (§ 1 only), `foot_v2` (§ 2 only),
`sim_phc` (§ 3 highest quality), `full_realism` (everything).

---

## 5. Recommended roadmap (best-first by cost × quality)

1. **§ 1 identity_gate — GTA + PlayerProfile team lock.** ~1-2 days coding,
   ~$0.30 pod re-run to validate. Kills the visible colour flicker. Uses
   `PlayerProfile` schema shipped this session (T4).

   **STAGE 1a + 1b DONE 2026-07-06.** Full identity gate, end-to-end
   through the CLI.

   * ``src/pitch3d/core/orchestration/identity.py`` — split (DBSCAN over
     per-frame appearance features via injected ``AppearanceProvider``,
     ``min_split_gap_frames`` flicker guard) + cross-track merge (iterative
     greedy over disjoint tracklets whose mean features are within
     ``merge_cosine_threshold``, capped by ``merge_max_gap_frames``).
     Split→merge feature plumbing is internal so post-split truncated
     tracklets don't trip the shape guard.
   * ``src/pitch3d/adapters/models/appearance_hsv.py`` — starter numpy-only
     Re-ID adapter. Samples HSV histograms from bbox crops (dedup-decoded
     frames), returns dense (T, D) features. Enough for team-A vs team-B
     splits; a torch OSNet/CLIP-ReIdent swap is the next tier.
   * Pipeline wire: ``ReconstructionPipeline`` accepts ``identity_cfg`` +
     ``appearance_provider``; runs the gate BETWEEN stitch and CALIBRATE so
     POSE sees clean tracks. ``controller.run_reconstruction`` and the CLI
     both forward it.
   * CLI: ``--identity`` flag (requires ``--physics`` for the config).
     Pod-script env-var ``IDENTITY=1`` in ``scripts/pod_real_e2e.sh``.

   28 unit tests total (15 split + 7 merge + 6 HSV provider). 822 in the
   full suite, 12 skipped, 0 regressions.

   Stage 1c (next): swap the HSV provider for OSNet from ``torchreid`` /
   CLIP-ReIdent for real Re-ID quality; add ``PlayerProfile.appearance``
   operator-lock via ``set_operator_field`` so kit-inject uses per-player
   hue instead of the team aggregate.
2. **§ 2A foot_plant_v2 stage A — measured pelvis-above-foot.** ~1 day.
   Kills hovering PROPERLY. Uses `smplx` package. Pod re-run confirms.
3. **§ 3 momentum + contact-lock (Path 1).** ~2-3 days. Cheapest physics
   step; kills foot-slide + CoM drift. No new dep.
4. **§ 2B foot_plant_v2 stage B — WHAM contact head + IK anti-slide.**
   ~2 days once § 3 is validated.
5. **§ 3 physics_refine via PACER+ (Path 2).** ~1 week. Adds walk-cycle
   inference. torch weights.
6. **§ 3 physics_refine via PHC / MaskedMimic (Path 3).** ~2 weeks (MuJoCo
   + MJX + PHC weights + SMPL2MJCF integration). Highest quality ceiling;
   solves inertia + CoM + friction + collision by construction. Uses
   SMPLOlympics as reference pipeline.
7. **Adopt SMART FIFA 2026 anchoring/smoothing as an alternate baseline
   before PHC** (small, drop-in, sports-tuned).

Steps 1-3 close the three visible complaints (colour flicker, hover, glide).
Steps 4-6 raise the physical realism ceiling toward "physically executable
by construction." SMPLOlympics's TRAM→PHC pipeline is the proof-of-existence
that our approach converges.

---

## 6. Datasets to add for eval (0 code, immediate use)

* [WorldPose (ECCV 2024)](https://eth-ait.github.io/WorldPose/) — real
  football broadcast with GT SMPL-X (already local per memory).
* [SoccerNet-ReID](https://github.com/SoccerNet/sn-reid) — 340k player
  thumbs, kit identity ground truth.
* [SoccerNet Jersey Number](https://github.com/SoccerNet/sn-jersey) — jersey
  OCR benchmark.
* [SoccerNet GSR (CVPR-W 2024)](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.pdf) — game state reconstruction leaderboard.
* [SportsMOT / SoccerTrack](https://arxiv.org/pdf/2506.01373) — multi-object
  tracking benchmarks.

---

## 7. Sources

Physics motion reconstruction:
[LARP](https://arxiv.org/abs/2410.12023) ·
[Plug-and-Play Restoration](https://arxiv.org/pdf/2412.17377) ·
[Contact and Human Dynamics](https://arxiv.org/pdf/2007.11678) ·
[DiffPhy trajectory-opt](https://arxiv.org/pdf/2205.12292) ·
[PhysDiff](https://nvlabs.github.io/PhysDiff/) ·
[PhysCap](https://arxiv.org/pdf/2008.08880) ·
[WHAM](https://arxiv.org/pdf/2312.07531) ·
[UnderPressure](https://arxiv.org/pdf/2208.04598) ·
[PACER+](https://arxiv.org/pdf/2404.19722) ·
[PULSE / PHC code](https://github.com/ZhengyiLuo/PULSE) ·
[MaskedMimic](https://research.nvidia.com/labs/par/maskedmimic/) ·
[Body Momentum](https://arxiv.org/pdf/2509.09496) ·
[SMPLOlympics](https://arxiv.org/pdf/2407.00187) ·
[SMPLOlympics code](https://github.com/SMPLOlympics/SMPLOlympics) ·
[PhysicsFC](https://arxiv.org/pdf/2504.21216) ·
[SMART FIFA 2026](https://arxiv.org/abs/2605.31551) ·
[SMART starter kit](https://github.com/FIFA-Skeletal-Light-Tracking-Challenge/FIFA-Skeletal-Tracking-Starter-Kit-2026).

Identity persistence:
[GTA ACCV 2024](https://arxiv.org/abs/2411.08216) ·
[gta-link code](https://github.com/sjc042/gta-link) ·
[CLIP-ReIdent SoccerNet 2023](https://arxiv.org/pdf/2309.06006) ·
[Multi-task Re-ID+Team+Role](https://arxiv.org/pdf/2401.09942) ·
[Jersey OCR (Koshkina & Elder CVPR 2024)](https://arxiv.org/pdf/2309.06285) ·
[SoccerNet 2024 challenges](https://arxiv.org/pdf/2409.10587) ·
[SoccerNet 2025 challenges](https://arxiv.org/pdf/2508.19182) ·
[SoccerNet-ReID](https://github.com/SoccerNet/sn-reid) ·
[SoccerNet Jersey](https://github.com/SoccerNet/sn-jersey).

Foot contact IK:
[HuMoR](https://github.com/davrempe/humor) ·
[LEMO](https://github.com/sanweiliti/LEMO) ·
[WHAM](https://github.com/yohanshin/WHAM) ·
[TokenHMR](https://tokenhmr.is.tue.mpg.de/).

Simulation frameworks:
[MuJoCo Multi-Agent Soccer](https://arxiv.org/pdf/2105.12196) ·
[Genesis](https://genesis-world.readthedocs.io/) ·
[Isaac Gym](https://arxiv.org/pdf/2108.10470) ·
[Humanoid Locomotion Survey 2025](https://arxiv.org/pdf/2501.02116) ·
[Contact-Implicit Trajectory Optimization](https://arxiv.org/pdf/1809.06436).
