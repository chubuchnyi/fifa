# Competitive landscape & differentiation — pitch3d

**What this is:** an honest map of who else turns sports video into 3D, where the genuine white
space is, and what *defensible* superiority we can claim (and what we must **not** claim). Built from
web research 2026-06-22 (four parallel surveys: commercial products, academic/open SOTA, the
human-in-the-loop/LLM moat, multi-sport extensibility). Every claim cites a source; anything we could
not verify from a primary source is flagged **[uncertain]** — we position on measured reality, not
marketing.

**Our one-line position:** *world-space metric 3D athlete pose (SMPL-X) from a **single broadcast
camera**, delivered as an **editable, correction-first** scene that a human or an LLM agent refines
through one seam — with honest per-frame confidence.* No single piece of that is novel; the
**combination, in the single-broadcast-camera sports framing, is the moat.**

---

## 1. The problem and why it's hard

A single TV camera → every player as an articulated 3D mesh standing in **real metres on the pitch**.
This is hard for three compounding reasons: (a) monocular depth/scale is ambiguous; (b) the camera
is unknown and moving (must be calibrated from pitch lines); (c) "world-grounding" error dominates —
off-the-shelf global HMR lands **metres** off on broadcast soccer (GLAMR ~18.9 m, SLAHMR ~8.3 m
Global MPJPE on WorldPose). The benchmark for *exactly* our problem is **WorldPose / FIFA Skeletal
Tracking** (arXiv [2501.02771](https://arxiv.org/abs/2501.02771)); metric = Global + Local MPJPE in
metres, no Procrustes.

---

## 2. Commercial landscape

**Legend — Input:** BC = single broadcast cam · MC = multi-camera in-venue rig · PH = phone/webcam.
**3D:** 2D = positions only · SK = 3D skeleton · MESH. **WM:** world-metric (real metres on the
field) vs body-relative.

| Vendor | Input | 3D | WM | Open | Editable | Positioning |
|---|---|---|---|---|---|---|
| **Track160** | **BC** | **SK** | **yes** (matched 6-cam in FIFA EPTS test) | closed | [uncertain] | analytics/coaching |
| Hawk-Eye (Sony) | MC | SK+ball | yes | closed | no | officiating, broadcast |
| Second Spectrum / Genius "Dragon" | MC | SK→mesh | yes | closed | no | analytics, broadcast |
| Tracab / ChyronHego | MC | 2D/3D pos | yes (~7 cm) | closed | no | broadcast, analytics |
| Stats Perform (AutoStats) | BC | 2D (+pose bolt-on) | 2D; pose WM [uncertain] | closed | no | analytics, scouting |
| SkillCorner | BC | **2D only** | pitch 2D | closed | no | analytics, scouting |
| Sportlogiq | BC | 2D + events | pitch 2D | closed | no | analytics (hockey/football) |
| Metrica Sports | BC | 2D (~10 cm) | pitch 2D | closed (+`codeball` OSS) | manual annot. | analytics, video |
| Bepro | MC (panoramic) | 2D→3D joints (OSS pose) | pitch 2D | closed | no | analytics |
| Pixellot | MC | — (auto-production) | n/a | closed | no | auto-broadcast/OTT |
| Move.ai | MC **or** PH | MESH/SK | body-rel; metric [uncertain] | closed | yes (export+cleanup) | animation/VFX |
| Theia Markerless | MC | SK (IK) | yes (lab biomech) | closed | yes (research pipe) | biomechanics |
| DeepMotion | BC/PH | SK/MESH | body-rel | closed | **yes** (Rotoscope Pose Editor) | animation/consumer |
| Plask AI | BC/PH | SK | body-rel | closed | **yes** (cleanup/retarget) | animation |
| Rokoko Vision | PH | SK | body-rel | closed | yes (Studio cleanup) | animation |
| Sportsbox AI | PH | SK (30+ pts) | body-rel biomech | closed | [uncertain] | golf biomechanics |

**Two findings that define our white space:**

1. **Single-broadcast-cam → world-metric 3D articulated pose is almost unoccupied.** Exactly one
   confirmed match — **Track160** (single camera → 3D skeletons on the pitch, reportedly matched
   6-camera systems in FIFA's EPTS test;
   [NVIDIA](https://blogs.nvidia.com/blog/ai-soccer-track160/),
   [PitchBook](https://pitchbook.com/profiles/company/234936-82)). Everyone else is either
   **single-cam but 2D-only** (SkillCorner, Stats Perform/AutoStats, Sportlogiq, Metrica, Bepro) or
   **3D but a multi-camera in-venue rig** (Hawk-Eye, Second Spectrum/Dragon, Tracab). The
   markerless-mocap vendors (Move.ai, DeepMotion, Plask, Rokoko, Sportsbox) do single/few-camera 3D
   but output **body-relative** motion for animation/biomechanics, **not** athletes grounded in world
   metres on a field. **No vendor was confirmed to emit a single-broadcast-cam → world-metric SMPL-X
   mesh.**

2. **Editable 3D and sports-tracking are disjoint markets.** Correction tooling lives only in the
   **animation** vendors (DeepMotion's Rotoscope Pose Editor, Plask, Rokoko, Move.ai cleanup). The
   **sports-tracking** vendors all ship **fixed, black-box** data/API. An *editable world-metric 3D
   sports scene* sits in the gap between the two.

---

## 3. Academic / open SOTA — our building blocks, not our competitors

Most of these are candidate **backends** behind our seam (ADR-0006), not rivals. The map matters
because it shows **no end-to-end open system exists** for our task.

| Bucket | SOTA verdict | Open? |
|---|---|---|
| **World-grounded mono HMR** | **TRAM** (arXiv [2403.17346](https://arxiv.org/abs/2403.17346), EMDB WA-MPJPE 76.4 mm) best open world-trajectory; **GVHMR** ([2409.06662](https://arxiv.org/abs/2409.06662)) strong, gravity-stable; **WHAM** ([2312.07531](https://arxiv.org/abs/2312.07531)); **WHAC** ([2403.12959](https://arxiv.org/abs/2403.12959)) is whole-body SMPL-X | yes (all body-only SMPL except WHAC) |
| **Whole-body SMPL-X (per-crop)** | **SMPLest-X** (arXiv [2501.09782](https://arxiv.org/abs/2501.09782); our primary, what SMART builds on); **SAM 3D Body** ([2602.15989](https://arxiv.org/abs/2602.15989), Meta, newest in-the-wild SOTA, commercial-friendly licence) | yes (both open weights) |
| **Soccer-specific world 3D** | **SMART** (arXiv [2605.31551](https://arxiv.org/abs/2605.31551)) — **verified Global 0.324 m / Local 0.054 m / overall 0.593** on WorldPose. **Code/weights NOT released.** | recipe only |
| **Broadcast calibration** | **PnLCalib** (arXiv [2404.08401](https://arxiv.org/abs/2404.08401); ours) and **"No Bells, Just Whistles"** co-SOTA; both beat **TVCalib** ([2207.11709](https://arxiv.org/abs/2207.11709)) | yes |
| **2D minimap GSR** | **SoccerNet Game-State-Reconstruction** ([2404.11335](https://arxiv.org/abs/2404.11335)) — *2D minimap*, not 3D pose | yes |

**The opening:** there is **no end-to-end open system** (calibration + tracking + world-3D pose) for
broadcast soccer — only disjoint open pieces. **SMART** is the closest published *recipe* gluing them,
and it is **unreleased**. Building the open, reproducible, *editable* end-to-end pipeline is precisely
our project's reason to exist. A 2023 precedent for the base pipeline exists — monocular 3D HPS for
sports broadcasts via partial field registration (arXiv [2304.04437](https://arxiv.org/abs/2304.04437))
— but it is neither world-metric-grade nor editable.

---

## 4. Where we win — honestly

A skeptical audit of our four hypothesized differentiators. **None is individually novel.** The
honest moat is the **bundle**, scoped to single-broadcast-camera metric sports reconstruction.

| Differentiator | Who else does it | Honest verdict |
|---|---|---|
| **Editable / correction-first** output | Everyone exports to Blender; Cascadeur/MotionBuilder are full non-destructive correction surfaces; research interactive SMPL refinement (arXiv [2403.11634](https://arxiv.org/abs/2403.11634)) | **Weakest claim — do not claim editing itself.** What's rarer: making the edit the *canonical source of truth* with review/approve semantics for **sports HPS** (ADR-0002). Uncommon, not unique. |
| **LLM/agent edits the 3D scene over MCP w/ visual feedback** | Blender-MCP ([ahujasid](https://github.com/ahujasid/blender-mcp)); **official Blender Foundation MCP server**; SceneCraft ([2403.01248](https://arxiv.org/abs/2403.01248)), Motion-Agent, Keyframer ([2402.06071](https://arxiv.org/abs/2402.06071)) | **Already prior art.** The "human ≡ LLM over MCP with visual feedback" mechanism is established. Our novelty survives **only** scoped to *correcting metric sports reconstructions* — nobody has shown that specifically **[uncertain]**. |
| **Honest per-frame confidence/uncertainty surfaced to users** | Research exists (CUPS conformal [2412.10431](https://arxiv.org/abs/2412.10431); ProHMR [2110.00990](https://arxiv.org/abs/2110.00990)); a 2026 paper notes existing methods give "no mechanism to flag unreliable frames or joints" ([2603.26844](https://arxiv.org/abs/2603.26844)). **Products surface none of it.** | **Strongest standalone claim — as productization, not invention.** R-6 honesty (inlier-scored calibration confidence, confidence-fading markers) is already wired. |
| **Open + swappable SOTA backends** (hexagonal) | **Pose2Sim** ([GitHub](https://github.com/perfanalytics/pose2sim)) is open + swappable, but multi-cam → OpenSim joint angles, not single-BC → world SMPL-X, and not a hexagonal backend-swap | **Rare, not unprecedented.** Open + swappable *for broadcast world-HPS* is genuinely uncommon. |

**The defensible thesis:** every commercial 3D sports system is a **closed, multi-camera, black-box
number**; every editable 3D tool is **animation, body-relative, off-field**. We are the only one
combining **single broadcast camera + world-metric SMPL-X + editable-as-truth + human≡LLM correction
+ surfaced uncertainty + open swappable SOTA**. Claim the **integration and the framing**, never the
individual pillars.

---

## 5. Where competitors are ahead — threats to track

- **Track160** already ships single-camera 3D skeletons in production. We must verify whether it is
  world-metric SMPL-X *mesh* or skeleton-only, and whether it is editable **[uncertain]** — it is the
  closest thing to a direct competitor.
- **SMART beats us on numbers today** (0.324 m Global) — but it is unreleased and we can reproduce it
  (all components open). Speed of execution is the race.
- **Animation vendors have mature correction UX** (DeepMotion, Plask, Cascadeur). Our editing surface
  must not try to out-animate them; it wins only by being *metric, on-field, and agent-drivable*.
- **LLM-over-MCP 3D editing is commoditizing fast** (Blender Foundation's official server). Our edge
  is the *domain* (metric sports correction), not the mechanism — do not over-invest in the plumbing.

---

## 6. Multi-sport extensibility (basketball / hockey / tennis)

Our architecture isolates the sport-specific pieces behind four swappable ports: the
**`FieldCalibrator`** keypoint set, the **`WorldFrame` Z = 0** ground plane, the **`core/correction`
motion prior**, and the **`BallTracker`/object** model. The SMPL-X core, FK, layer-resolve and
grounding math stay fixed. How cleanly each sport drops in:

| | Calibration | Ground plane | 3D-pose data | Object | Verdict |
|---|---|---|---|---|---|
| **Tennis** | classic court-line + net, ~px-accurate | **strongest** (flat, players grounded) | **CalTennis** (arXiv [2606.20542](https://arxiv.org/abs/2606.20542)) — monocular-to-3D, ≈our exact task | fast small ball, ballistic lift reuses | **cheapest next** — all four ports near-trivial; thin multi-person value is the only downside |
| **Basketball** | solved — **KaliCalib** (arXiv [2209.07795](https://arxiv.org/abs/2209.07795)), same keypoint-grid paradigm | holds, breaks on **jumps/dunks** (airborne) — but solved by NBA reconstruction (ECCV'20, [2007.13303](https://arxiv.org/abs/2007.13303), predicts jump + court position) | NBA2K19 meshes, SportsMOT/MultiSports | standard ball | **moderate** — needs an **airborne** motion prior + dense-contact occlusion handling; no new object model |
| **Ice hockey** | harder — **boundary-aware segmentation** beats keypoints (sparse rink markings); HockeyRink dataset | holds, but **boards/glass** occlude lower body | **3D-pose is the gap** — only 2D (16-joint CNN); no SMPL-X-grade set **[uncertain]** | **puck** tiny/fast/occluded — breaks the projectile model | **most new work** — 3 of 4 ports need real R&D, not config |

**Shared vs sport-specific:**
- **Reuse as-is:** SMPL-X core, FK, layer-resolve, corrections, grounding math; the keypoint-grid
  calibration *paradigm*; ballistic 3D-lift for ball sports.
- **Swap (designed-for):** calibration template + keypoints; court-plane dimensions; ball model.
- **Beyond a clean swap:** the **motion prior** for airborne (basketball) and skating (hockey) is new
  modelling, not retuned smoothing; the **puck** breaks the ball-as-projectile object model; hockey
  may force a **segmentation-based** calibration adapter (new adapter, not a template swap).

**Recommended ordering: tennis → basketball → hockey.** Tennis validates the multi-sport seam at
lowest risk and comes with a matching monocular-to-3D benchmark; hockey is the stress test that
reveals where the architecture genuinely breaks (motion prior + object model) rather than configures.

---

## 7. Research directions to deepen the moat

The "unique superiority" worth building, in leverage order — each turns a *rare* differentiator into
a *defended* one:

1. **Ship the open, reproducible, editable end-to-end pipeline SMART hasn't released.** All its
   components are open; being first to a *correctable* world-metric soccer system is the clearest win.
2. **Productize honest uncertainty** (our strongest standalone): calibrated per-joint/per-frame
   confidence (conformal, CUPS-style) → the M3-5 "needs attention" review UX. No product surfaces
   this; research says it's missing.
3. **Calibration is the #1 quality lever** (memory: field-markings-only 548 mm vs +player-keypoint
   bundle-adjust 80 mm Global MPJPE, ≈7×) — add player-keypoint bundle-adjust to PnLCalib, and
   foot-plane anchoring (T2, the SMART recipe's single biggest jump) once B3 yields foot/pelvis FK.
4. **Make "edit = source of truth, human ≡ LLM" real on metric sports** — the one framing of the
   editable/agent loop that isn't already prior art.
5. **Prove the multi-sport seam** with tennis (CalTennis eval) — turns "swappable backends" from an
   architecture claim into a demonstrated capability across sports.

---

## Sources

Commercial: [Hawk-Eye/SkeleTRACK (SVG)](https://www.sportsvideo.org/2025/09/22/wrexham-afc-select-hawk-eye-to-deliver-skeletal-tracking-and-video-review-system-services-for-home-matches/) ·
[NBA Dragon (SVG)](https://www.sportsvideo.org/2023/03/09/nba-taps-sonys-hawk-eye-for-data-tracking-beginning-with-2023-24-season/) ·
[AutoStats](https://www.statsperform.com/press/stats-launches-autostats-the-first-patented-ai-powered-technology-to-capture-sports-tracking-data-via-broadcast-video/) ·
[Tracab Gen5](https://www.vision-systems.com/cameras-accessories/article/14178534/tracab-optical-sports-tracking-system-gen-5-uses-teledyne-dalsa-genie-nano-gige-cameras) ·
[SkillCorner XY](https://skillcorner.com/products/football/xy-tracking-data) ·
[Track160 (NVIDIA)](https://blogs.nvidia.com/blog/ai-soccer-track160/) ·
[Move.ai](https://move.ai/) · [Theia](https://www.theiamarkerless.com/) ·
[DeepMotion Animate 3D](https://www.deepmotion.com/animate-3d) · [Plask](https://plask.ai/en-US) ·
[Rokoko Vision](https://www.rokoko.com/products/vision) · [Sportsbox AI](https://www.sportsbox.ai/) ·
[Metrica codeball](https://github.com/metrica-sports/codeball) ·
[Bepro Cerberus](https://www.soccerscene.com.au/bepro-cerberus-revolutionising-football-data-with-optical-tracking/).

Academic & moat: arXiv ids inline above. SMART [2605.31551](https://arxiv.org/abs/2605.31551) ·
WorldPose [2501.02771](https://arxiv.org/abs/2501.02771) · SMPLest-X
[2501.09782](https://arxiv.org/abs/2501.09782) · SAM 3D Body
[2602.15989](https://arxiv.org/abs/2602.15989) · PnLCalib
[2404.08401](https://arxiv.org/abs/2404.08401) · CalTennis
[2606.20542](https://arxiv.org/abs/2606.20542) · KaliCalib
[2209.07795](https://arxiv.org/abs/2209.07795) · NBA reconstruction
[2007.13303](https://arxiv.org/abs/2007.13303) · Pose2Sim
[github](https://github.com/perfanalytics/pose2sim) · Blender-MCP
[github](https://github.com/ahujasid/blender-mcp) · CUPS
[2412.10431](https://arxiv.org/abs/2412.10431).

> Research-only snapshot, 2026-06-22. Numbers and access status drift — re-verify before any
> external/strategic use. **[uncertain]** flags are load-bearing, not hedges.
