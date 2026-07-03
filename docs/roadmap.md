# pitch3d — Roadmap (M0 → M4)

Maps the TZ milestones (§10) to concrete packages/tickets. **Mono through every milestone.**
Each milestone ends with a **working end-to-end artifact on one real clip**. Real multi-camera
(M4) is out of scope — only the data-model seam exists.

> **Positioning & competitors:** [`competitive-landscape.md`](competitive-landscape.md) — where we
> sit vs. commercial/academic systems, the honest moat (combination + single-broadcast-cam framing,
> not any single pillar), and the multi-sport (tennis → basketball → hockey) extension plan.

Legend: ✅ exists in this scaffold · 🟡 stub/contract only · ⬜ not started.

---

## ⚑ Status reset — results over process (2026-06-27)

> **Living tracker / source of truth: [`STATUS.md`](STATUS.md)** — updated & committed every step.
> This roadmap is the historical build log; `STATUS.md` holds the durable, current state.

**The bar changed.** Despite M0–M3 being marked 🟢, we had never produced and *looked at* a full
reconstruction of a real clip as a deliverable. The green milestones measure **plumbing** — pure
cores, contracts, injection seams, tests on **fake** adapters — not a good real-clip result. The
heavy/generative halves (photoreal pose/ball, avatars, view-synth) are still **injection-seam stubs**.

**The only deliverable that counts now:** from a source broadcast clip, a *realistic novel-view video
of the same episode* — players like the originals (kit + numbers), the same stadium — **judged by
eye**. Confirmed target clip: `samples/video/Colombia-1-0-Congo-DR1080p.mp4`.

**Staged bar (do in order):**
- **v0 — correct geometry (CURRENT FOCUS).** Stable ~22 players, correct world placement/scale,
  correct poses, virtual cameras that frame the action, pitch lines. Output: a clean *geometric*
  novel-view video. This is the first "good" result; everything builds on it.
- **v1 — recognizability.** Team kit colors; numbers (OCR where readable, else roster); simple
  stadium backdrop.
- **v2 — photoreal.** Textured/Gaussian avatars + photoreal stadium + view-synth (the gated
  `avatars`/`viewsynth` heavy halves). The full stated goal; a long research stage.

Approximations are acceptable for exact numbers / exact stadium, backstopped by **manual Blender
editing** and **generative prompt-editing** (ADR-0008 LLM-over-MCP).

**The first 300-frame render of the real clip exposed concrete geometry defects** — see
[`v0-geometry-defects.md`](v0-geometry-defects.md). Tracked as **#202–#205**, root causes now located
in code:
- **#202 too many bodies** — ByteTrack fragments; `min_track_frames` defaults to 1 and the
  fragment-stitch pass (`core/orchestration/continuity.py`) is wired but **off by default**.
- **#203 depth collapse / wrong scale** — the field homography `H` collapses to its
  identity/degenerate fallback, so `image_to_world` piles every foot point into one spot / pixel-space.
- **#204 cameras don't frame** — `action_centroid` averages the collapsed roots (downstream of #203);
  the render also uses a single static broadcast camera frozen at frame 0.
- **#205 bare pitch** — pitch-line code exists but the video used a render path that skips it; **goal
  geometry is genuinely missing**.

**Working rule:** no new milestone flips or breadth expansion until v0 looks right on the real clip.
The M0–M4 sections below are the historical build log of the *platform* — accurate about plumbing,
not about result quality.

---

## M0 — Skeleton (this scaffold)

**Goal:** the hexagonal skeleton compiles, the scene model round-trips, a dry-run runs end-to-end
on fakes, glTF export path exists.

| Ticket | Package | State |
|---|---|---|
| M0-1 Canonical scene model + JSON serialization | `core/scene` | ✅ |
| M0-2 Port ABCs (all of §6) | `core/ports` | ✅ |
| M0-3 Correction engine (4 modes) + rotations | `core/correction` | ✅ |
| M0-4 Orchestration: Stage enum, Pipeline, cache-key, ball lift | `core/orchestration` | ✅ |
| M0-5 Fakes (incl. FakeViewSynthesizer A+B), in-proc queue, cache | `adapters/fakes` | ✅ |
| M0-6 Real-adapter + exporter skeletons behind every port (pure half real; heavy half real or injection-seam stub) | `adapters/{models,viewsynth,blender,render,export}` | 🟢 |
| M0-7 CLI dry-run + wiring | `app` | ✅ |
| M0-8 Core tests green without GPU/Blender | `tests` | ✅ |
| M0-9 Observation port + viewpoint math + scene summary (LLM feedback) | `core/ports`, `core/agent` | ✅ |
| M0-10 MCP tool catalog (import-free) + `FakeSceneObserver` | `adapters/mcp`, `adapters/fakes` | ✅ |
| M0-11 Live MCP `serve()` (needs `mcp` extra + app controller) | `adapters/mcp` | ✅ |

**Artifact:** `python -m pitch3d.app.cli` runs source → stages(fakes) → scene → edit →
propagate → render(fake) → export, and `pytest` is green. The LLM-feedback loop
(`observe → reason → Correction → resolve → observe`) is exercisable on fakes — no GPU/LLM.

---

## M1 — Editable loop (the vertical slice) 🟢

**Goal (TZ M1):** a 3D scene on a real clip that can be **edited**, with edits propagated.
Proxy only — no photoreal yet. This is the first milestone that replaces fakes with real models.

**Status — GOAL MET, closed 2026-06-27** (live state, blockers and next steps live in
[`m1-status-and-plan.md`](m1-status-and-plan.md)). The editable loop runs end to end on a real clip:
every perception/render/export port has a real adapter behind the same split (pure half unit-tested
via an injected stub; heavy half lazy-imported behind its extra), selectable per-port via
`default_ports(...)` and the `--detector/--tracker/--calibrator/--pose/--ball/--render/--export`
flags. Suite green (480 passed / 5 display-gated skips). **Documented ceiling** (R-6, not narrowed):
the photoreal-grade pose/ball *heavy* nets ship as **injection-seam stubs** (real backends inject by
dotted path, ADR-0006 — pod-validated, not boxed); the **GUI** live-Blender placement session
(step 10) is **display-gated** (host-side seam real + socket-tested, but no display in CI); and the
metric-XY reprojection floor is measured on lined footage (B1, below) — an unlined clip grounds Z
and topology but not absolute pitch XY.

- **Real, no GPU, today:** overlay render (+ confidence highlighting, UX-3/FR-16), glTF/`.npz`
  export, the camera-free top-down **radar** VIEW, the live MCP `serve()` dispatch (SDK stdio loop
  behind the `mcp` extra), and the Blender proxy `.blend`/`SCENE_3D` builder (needs a Blender
  binary, Workbench/CPU).
- **Real on CPU, validated on a real clip:** RF-DETR detection (~15–17 players/frame) → ByteTrack →
  teams A/B (HSV), plus the pose adapter's *pure* root-grounding (16 tracks grounded to world metres
  from the field homography). The interactive-placement seam (a live-Blender drag → `ROOT_TRANSLATION`
  `Correction` through the **same** `apply_offset` the LLM drives, ADR-0008/0010) is host-side real
  and unit-tested over a socket pair.
- **Device knob:** `--device {cpu,cuda}` forwards to every real adapter, defaulting to `cpu` (the
  local validation profile, ADR-0009); production flips to `cuda` with no code change.
- **Still box-gated (stubs):** the pose and ball *heavy* backends. The chosen nets are **SMPLest-X**
  (pose) and **WASB** (ball); they inject on the box by dotted path
  (`--pose-backend`/`--ball-backend pkg.module:Factory` — config-not-fork, research code stays out
  of core, ADR-0006). The injection seam itself is wired + headlessly tested. Also pending: the live
  **GUI** Blender session (`launch_live_session` + depsgraph watcher, step 10) — it needs a display.

### Vertical slice (one clip, end to end)
1. **Ingest** a real broadcast clip via the FFmpeg adapter → `Source` (fps/res/timecode). `adapters/models` (io)
2. **Episode** select (manual range first; action-spotting later). `core/scene`, `app`
3. **Detect** players/keepers/refs/ball per frame — RF-DETR adapter. `adapters/models` (replaces `FakeDetector`) — 🟢 *CPU-validated (P2.3)*: `RFDETRDetector` (pure map/threshold/assembly, unit-tested) over an injected `RFDETRBackend` (torch/cv2/rfdetr, `cv` extra). The `--detector-classes {coco,sports}` knob picks the class map: `coco` (default — the free, downloadable base weights; person→player, sports_ball→ball) or `sports` (the Roboflow checkpoint via `--detector-weights`, which splits keepers/refs apart). Same adapter=production / root=validation default as `device` (ADR-0009). Real run on CPU detected ~15–17 players/frame on a broadcast clip.
4. **Track + teams** — ByteTrack/BoT-SORT + team clustering. `adapters/models` — 🟢 *CPU-validated (P2.3)*: `ByteTrackTracker` (pure association + appearance team clustering, unit-tested via injected stub); the live `ByteTrackBackend` (`cv` extra) needs **no learned weights or GPU** (supervision's ByteTrack + HSV torso sampling). Real run on CPU: detections → 16 stable tracks across 8 frames → teams A/B.
5. **Field homography** — keypoint model + robust homography + temporal smoothing → world anchor. `adapters/models` — 🟢 *wired + robustified + B1-measured*: `KeypointFieldCalibrator` fits each frame with **RANSAC + confidence-weighted normalized DLT** (rejects mislocalised landmarks, the dominant real-broadcast failure) and scores confidence on the inliers × inlier-fraction (R-6 honesty); pure numpy, unit-tested. Validated on real PnLCalib output (messi frame: rejecting 2/18 outliers cut reprojection RMS 3.87 m → 0.36 m). Live keypoint net is injected by dotted path with `--calibrator-backend pkg.module:Factory` (ADR-0006) — the PnLCalib HRNet, now **benchmarked end-to-end on SoccerNet `calibration-2023`** (B1): completeness 0.745, **median 0.236 m** on the frames it locks on. A 2026-06-24 kp-threshold sweep (400 frames) showed the heatmap gate is a **precision/recall dial** — lowering it lifts completeness 0.745→0.918 but net line-accuracy stays flat ~0.61, so the real lever is **better landmarks**, not the gate (portrait / drone footage is still out-of-distribution for the net). A 2026-06-25 A/B of PnLCalib's full camera module (`--solver camera`: points **and** lines) vs the bare DLT confirmed the *solver* is not the lever — median a wash (0.170→0.179 m), completeness unchanged (~0.74) — but it does tighten line registration (line_acc@5px 0.757→0.811) and the error tail (p95 6.41→1.15 m); the remaining lever is **completeness** (the ~¼ of frames the detector drops).
6. **HMR → SMPL-X** — GVHMR/WHAM; **root from homography**, foot-contact (anti-foot-slide). `adapters/models` — 🟢 *pure half CPU-validated (P2.4)*: `GVHMRPoseEstimator` grounds each subject's world root on the pitch from the field homography + assembles SMPL-X `SubjectMotion` + applies geometric refit (numpy, unit-tested). Validated on CPU on REAL ByteTrack tracks — 16 subjects grounded into world metres (Z = pelvis height). The **live `GVHMRBackend` is an unwired, GPU-bound stub**: GVHMR is a research repo (not pip), so the `hmr` extra ships only torch/smplx/chumpy, not the network/weights; `estimate_bodies` raises an actionable `NotImplementedError` (use `--pose fake`, or inject a vendored network by dotted path with `--pose-backend pkg.module:Factory` — the on-box config-not-fork seam, ADR-0006).
7. **Ball** — TrackNet 2D + **core ballistic 3D lift** (already implemented) with height confidence. `adapters/models` + `core/orchestration` — 🟢 *pure half CPU-validated* (parity with pose, step 6): `TrackNetBallTracker` (threshold + linear gap-fill with honest zero-confidence fills, unit-tested) runs end to end on the fake. The **live `TrackNetBackend` is an unimplemented stub** — unlike the rfdetr/bytetrack reals, the `ball` extra ships only torch (no TrackNet weights/decoder), so `detect_ball` raises an actionable `NotImplementedError` (use `--ball fake`, or inject a vendored network by dotted path with `--ball-backend pkg.module:Factory`, ADR-0006) until the network is wired. **Option to bench (2026):** *"Where Is The Ball"* (CVPR'25 CVsports) — a **calibration-free** monocular 2D→3D ball lift (predicts height; trained in sim, soccer-tested) as an alternative to our hand-rolled ballistic lift; evaluate behind the same seam once code releases.
8. **Assemble Scene** — proposal layer + confidence map. `core/orchestration` — ✅
9. **Proxy + overlay** — SMPL proxy in Blender; **reprojection overlay**; confidence highlighting. `adapters/render`, `adapters/blender` — ✅ *all real, no GPU*: `ReprojectionOverlayRenderPass` (pure pinhole projection + visibility + stdlib PNG, no extra) and the Blender proxy (`build_proxy_plan` → out-of-process `blender --background` → editable `.blend` / proxy `SCENE_3D` PNGs, Workbench/CPU). **Confidence highlighting** is real (`confidence_to_color` blends each marker toward a red warning colour as per-frame confidence drops; full confidence is a no-op so existing scenes render unchanged — UX-3/FR-16).
10. **Edit** — pose bones/β, ball/root as F-curves, 2D radar VIEW + placement. `adapters/blender` ↔ `core/correction` — 🟡 *editing surface + radar VIEW are real*: `BlenderProxyBuilder` writes the editable `.blend` with root/ball location + axis-angle **F-curves** and β + per-joint body pose as keyframed channels (baked from the resolved proposal ⊕ corrections); edits there map back to `Correction`s, the source of truth (ADR-0002). The top-down **radar** is real on the read side — `render_radar` (pure numpy + stdlib PNG) draws a pitch + team-coloured subject dots + ball from the resolved world XY, surfaced as an `ObservationImage(kind=RADAR)` via `observe(include_radar=…)`. The *interactive* **placement** drag is now 🟡 *seam real, GUI pending (P1)*: the host side is complete and unit-tested over a socket pair — `radar_to_world` (the inverse of `world_to_radar`, turning a dragged radar pixel back into world XY), the `subject_<id>` id↔name contract (`subject_object_name`/`parse_subject_name`, one source of truth shared by the builder and the bridge), and `serve_edits`/`apply_drag`, which diff a dropped root against the *resolved* position and commit the offset through the **same** `apply_offset` use-case the MCP agent calls (human ≡ LLM, ADR-0008/0010). The live `launch_live_session` (GUI Blender, no `--background`) + the in-Blender `_live.py` depsgraph watcher stay ⬜ — they need a display + a Blender binary, so they are not run in CI (the same honest limitation as the GVHMR/TrackNet reals).
11. **Propagate** — offset / interp / re-fit / smoothing with preview + undo (re-fit calls `PoseEstimator.refit`). `core/correction` — ✅
12. **Export** — glTF + SMPL-X `.npz` + intermediate JSON. `adapters/export` — 🟢 *real + tested*: `GltfExporter` — SMPL-X `.npz` (resolved per subject) + canonical JSON are real (numpy/stdlib, no extra); glTF/GLB assembly (Z-up→Y-up) is real and unit-tested, the `pygltflib` serialization gated behind the `export` extra (the standard heavy-half optional-dep pattern, not a stub). Pod E2E exports a full scene.

**Exit criteria (TZ AC-1, AC-2, AC-3) — all MET (2026-06-27):**
- **AC-1** *reprojection overlay matches the source on wide shots* — **MET.** Quantified on real
  frames by the **B1 SoccerNet `calibration-2023`** reprojection benchmark (median **0.236 m**,
  completeness 0.745) — a measured overlay-to-source floor, stronger than an eyeball; the overlay
  pass itself is pure-pinhole and unit-tested. *Caveat:* B1 is lined footage; an unlined clip aligns
  topology/Z but not absolute pitch XY.
- **AC-2** *operator fixes a pose and propagates it three ways with preview+undo* — **MET.** All four
  correction modes (offset / keyframe-interp / temporal-smoothing / re-fit) are unit-tested in
  `tests/unit/test_corrections.py` with `preview_subject_motion` and non-destructive resolve
  (ADR-0002); exercised on a real pod run (corr-1).
- **AC-3** *ball/player curves edit and show up in 3D and export* — **MET.** Root/ball/axis-angle
  F-curves are baked into the editable `.blend`, edits map back to `Correction`s, and the resolved
  scene exports (glTF assembly tested + SMPL-X `.npz`/JSON; pod E2E exports a full scene).

**Engineering notes**
- Replace one fake at a time; the dry-run wiring is the integration harness.
- Each real adapter must satisfy the same port test the fake passes.
- **Pattern (set by the detector):** split each real adapter into a *pure* half (maps the model's
  raw output into the canonical types — tested with no GPU via an injected backend) and a *heavy*
  half (decode + inference, lazy torch/cv2, gated behind its extra). Select it in wiring per-port
  (`default_ports(detector="rfdetr")`) so the swap is isolated.
- `re-fit` is the first place a model is called from inside a correction — keep it behind the port.

**Progress — M1 GOAL met, flipped 🟡→🟢 (2026-06-27):** the editable loop *closes* on a real clip.
Ingest → detect (RF-DETR, ~15–17 players/frame) → track + teams (ByteTrack, 16 stable tracks) →
calibrate (PnLCalib HRNet behind the dotted-path seam, RANSAC+DLT, **B1 median 0.236 m**) → ground
SMPL-X roots on the pitch → assemble proposal → edit via `Correction`s (the sole edit path, ADR-0002)
→ propagate (4 modes + preview + non-destructive resolve) → overlay/radar/proxy render → export
(glTF + `.npz` + JSON). **All three exit criteria are MET** (AC-1 B1 reprojection; AC-2 four-mode
propagation tested + pod corr-1; AC-3 F-curve edits + export). Suite **480 passed / 5 display-gated
skips**. **Honest ceiling (R-6, not narrowed):** (1) the photoreal-grade pose/ball *heavy* nets
(GVHMR/SMPLest-X, TrackNet) ship as **injection-seam stubs** — real backends inject by dotted path
(ADR-0006, pod-validated), the pure halves are real + CPU-validated, the boxed heavy halves raise an
actionable `NotImplementedError`; (2) the **GUI** live-Blender placement session (step 10) is
**display-gated** — host-side seam real + socket-tested, but no display in CI; (3) the metric-XY
floor is measured on **lined** footage (B1) — an unlined clip grounds Z + topology, not absolute
pitch XY. These are delivery/environment seams, not missing capability. **The loop is real,
editable, and in sync, so M1 is 🟢. Photoreal is M2 (🟢); broadcast fidelity is M3.**

---

## M2 — Photoreal layer 🟢  (+ ViewSynthesizer **seam A**)

**Goal (TZ M2):** photoreal render of the edited scene; edit↔render stays in sync.

**Scope decision (2026-06-25, option A — measured photoreal):** M2's photoreal target is a *real
renderer* (Blender/Cycles) of the **measured** scene — textured, LBS-posed SMPL-X avatars on a grass
pitch under scene lighting, with unmeasured regions honestly marked. **Broadcast-convincing** fidelity
(clothing geometry, full-body texture inpaint, stadium) needs *generative* fill and is **M3** (R-8,
per M2-0). The numpy vertex-splat pass (M2-3) is debug-grade scaffolding, not the photoreal deliverable.

| Ticket | Package |
|---|---|
| **M2-0 Realism approach spike** — decide the realism strategy before building avatars: textured SMPL-X via *measured pixel-projection* (primary) vs generative avatars (#2); explicitly assess & rule on image upscaler + image-to-3D (see note below); honesty gate = measured over hallucinated. Feeds M2-2. | `adapters/models`, doc |
| M2-1 Env reconstruction (3DGS/NeRF by camera motion; generative stadium fallback) | `adapters/models` (`EnvReconstructor`) |
| M2-2 Avatars strategy #1 (textured SMPL-X) + #2 (generative, Rodin-class API) | `adapters/models` (`AvatarBuilder`) |
| M2-3 `SplatAvatarRenderPass` — assemble a measured **vertex-splat** debug frame from `resolved` (scaffolding, not photoreal — see M2-7) | `adapters/render` |
| M2-4 Edit↔render sync wiring (resolved drives every render rep) | `app`, `adapters/render` |
| **M2-5 ViewSynthesizer seam A** — `render_orbit` as a `RenderPass` for limited orbits | `adapters/viewsynth`, `adapters/render` |
| M2-6 Render preview (fast low-q) for both paths | `adapters/render` |
| **M2-7 Real photoreal renderer ✅ (2026-06-25)** — `CyclesRenderPass`: Blender/Cycles `RenderPass` over the *resolved* scene (ADR-0003 external + import), the photoreal path alongside the no-dep splat debug viz. Rigid-root placement (LBS=M2-8), neutral ground (material=M2-9), R-6 tint intact | `adapters/render`, `adapters/blender` |
| **M2-8 Posed, textured avatar** — **8a ✅ (2026-06-27): per-frame SMPL-X LBS** (geometry follows the edited pose, incl. limbs), pure-numpy (no torch/GPU), posed through Cycles; **8b ✅ (2026-06-27): measured surface texture** — the scene camera + decoded source frames thread into `observe`, the subject is posed at its *measured* pose into a capped, evenly-spread set of reference frames, and the pure projection/z-buffer sampler paints per-vertex colour from real pixels (R-6 marks unmeasured grey) | `adapters/models`, `adapters/render`, `adapters/blender` |
| **M2-9 Measured env materials + lighting ✅ (2026-06-27)** — procedural grass PBR (mowing stripes + bump) for the measured pitch, the standard line markings rendered as flat white ribbons, and a physically-based (multiple-scattering) sky + matched key-light sun; stadium stays gated/marked (R-8 → M3) | `core/scene`, `adapters/render`, `adapters/blender` |
| **M2-10 Edit↔render sync + photoreal `observe` ✅ (2026-06-27)** — a pose edit (root *and* limbs via LBS) re-projects into the Cycles frame with no avatar rebuild (AC-4: gated test renders the *same* PLY twice, clean vs after a ROOT_TRANSLATION+POSE_BODY_JOINT correction, and the forearm/root move in pixels); `observe` returns **photoreal** multi-view via `CyclesSceneObserver` (A-8, `--observer cycles`); seam-A orbit is a real **re-render of the resolved 3D scene** at the orbit cameras via `CyclesViewSynthesizer` (A-9, `--viewsynth cycles`, non-generative, cached, `editable=False`) | `app`, `adapters/render` |

**Exit criteria (TZ AC-4, AC-5a — measured-photoreal reading, option A) — ✅ MET (2026-06-27):** a
*real renderer* (Blender/Cycles) yields a photoreal frame of the resolved scene — textured, LBS-posed
avatars on a grass pitch under scene lighting, unmeasured regions honestly marked (**not** the numpy
vertex-splat viz); editing an SMPL pose (root *and* limbs) re-projects into that photoreal frame with
no manual redo; `observe` returns photoreal multi-view; seam A yields a photoreal limited-orbit by
re-rendering the 3D scene at orbit cameras, **cached**, clearly flagged "video, not editable". All
four are gated-tested on real Blender + verified in the integrated CLI E2E (below). **Honest ceiling
(R-6, holds at 🟢):** measured-only ≈ textured mannequins (single broadcast cam ⇒ half the body
unmeasured; naked SMPL-X ⇒ no clothing geometry; no stadium); and the *texture* is only as good as the
pose alignment — on the **fake** pose pipeline coverage is 0% (synthetic bodies don't sit on the real
players, all honest R-6 grey), so genuine per-player colour waits on the still-gated GVHMR pose (M1).
The *renderer and all seams are real*; broadcast-convincing fidelity is M3.

**Boundary reminder (R-14/R-15):** seam A only for moderate moves; arbitrary free camera stays on
the splat/avatar path.

**Realism approach — investigation (M2-0, R-6 honesty, added 2026-06-24):** the realism gap is *3D
appearance*, not pixel count, so the levers by leverage are: **(1)** texture/appearance on the
existing tracked SMPL-X by **projecting the player's real broadcast pixels** onto the mesh (M2-2 #1)
— keeps temporal coherence + rigging and is *measured* (honest); **(2)** Blender materials/lighting
(grass, PBR kit, shadows, HDRI) — cheap, high payoff; **(3)** clothed-human geometry beyond the naked
SMPL-X body (ECON/ICON-class), *not* generic image-to-3D. **Assessed & de-prioritized:** *image
upscalers* (a 2D-pixel tool — they hallucinate detail, can hurt the geometric perception nets, and
output resolution is set in Blender anyway) and *image-to-3D-object* models (single-shot, un-rigged,
not temporally coherent → they break our tracked SMPL-X; useful only for static props like
goals/stands, which broadcast rarely shows cleanly). **Honesty gate (R-6):** generative
upscale/image-to-3D *hallucinate unmeasured* detail, conflicting with the project's mark-don't-
fabricate stance — prefer pixel-projection (measured) over generative synthesis, and if generative
avatars (#2) are used, flag them as synthesized. Realism is a *presentation* layer, sequenced after
M1's measurement accuracy (calibration/pose), which is the core product value.

**Cross-input — SAM-Body4D / photoreal-rendering brief** (`upscail/sam-body4d-and-photoreal-
rendering-brief.md`, reviewed 2026-06-25): three takeaways adopted, one subordinated. **Adopt:**
**(a)** homography as a *validator-anchor*, not just calibration — the PnLCalib ground plane (M1)
bounds every heavier appearance/HMR layer (a measured texture or an inferred body part that violates
pitch geometry is rejected — "a leg can't pass through the pitch"), operationalising the measured-
over-hallucinated gate; **(b)** a failure-mode eval harness *first* — quantify error by mode
(occlusion / athletic pose / small scale / blur) on real clips before building avatars, then
fine-tune only the dominant-error component; **(c)** the brief's ports table matches ours — keep each
backend a swappable adapter. **Subordinate (R-6):** the brief defaults to *generative* avatars
(feed-forward Gaussian LRMs PF-LHM/IDOL/LHM, multi-view diffusion, kit-number hallucination); M2-0
keeps measured pixel-projection **primary** and routes those candidates to M3 (#3 / seam-B), flagged
synthesized. (The brief's stated "current stack" assumes GVHMR/WHAM + ByteTrack; the actually-wired
calibration is PnLCalib — it is a *target* design, not the repo state.)

**Progress — M2-2 #1 wired E2E (2026-06-25):** the measured textured-SMPL-X `AvatarBuilder`
(projection → front-face/z-buffer visibility → per-vertex colour averaging, R-6: never-seen verts
stay `measured=0`) is now wired through the pipeline, not just unit-tested. New controller stage
`Application.build_avatars` consumes the *resolved* scene's subjects and attaches one
`RenderAssetRef` per subject to the scene; the CLI dry-run runs it as stage 9 (`reconstruct → … →
resolve → observe → **avatar** → render → export`). An injectable, dependency-free
`SyntheticAvatarMeshBackend` (no SMPL-X, no GPU) drives the *real* measured path E2E so CI/`--avatar
textured` exercises projection without weights — `--avatar-backend
pitch3d.adapters.models.avatar:SyntheticAvatarMeshBackend` yields a genuine 2/3-coverage PLY (one
vertex off-frame, honestly unmeasured). The heavy SMPL-X meshing half stays gated behind the
`avatar` extra.

**Progress — M2-3 `SplatAvatarRenderPass` wired E2E (2026-06-25):** the first render pass that
consumes the M2-2 assets. It reads each avatar's vertex-coloured PLY (new paired
`read_vertex_colored_ply`), places the canonical mesh at the subject's *resolved* per-frame root,
projects every vertex through the camera and paints **z-buffered colour splats** onto per-frame
PNGs (pure numpy + the stdlib PNG encoder shared with the overlay pass — no Blender/GPU). R-6
travels into the picture: a `measured=0` vertex renders in a distinct *unmeasured tint*, not its
fabricated placeholder, so unobserved body regions are *visibly* marked. Wired as `--render splat`
(`default_ports(render="splat")`); the dry-run `--avatar textured … --render splat` renders the
measured meshes E2E (decoded frame: measured colour + black + the R-6 tint, never a faked colour).
Honest limits, deferred to heavier upgrades: vertex splats (no triangle fill) and root-translation
placement only (no per-frame limb re-posing — needs the SMPL-X model). **Next: M2-4** (edit↔render
sync) / **M2-1** env reconstruction.

**Progress — M2-4 edit↔render sync wired E2E (2026-06-25):** the splat pass now places the
canonical mesh by the subject's **full resolved root rigid transform** — `global_orient` (axis-angle
→ rotation matrix via the existing `axis_angle_to_matrix`) **⊕** root translation — instead of
translation only. Because the placement reads the *resolved* pose, a `ROOT_ORIENTATION` **or**
`ROOT_TRANSLATION` correction re-projects straight into the rendered frame with **no avatar rebuild**
(TZ AC-5a "editing an SMPL pose re-projects into photoreal with no manual redo"). Proven at the
framebuffer level: a 90° root-orientation edit swings an off-root vertex to its new pixel and vacates
the old one while the PLY's mtime is unchanged (asset never rebuilt); a translation edit slides the
same vertex — both through the `resolved → render` path only. Audit confirms *resolved drives every
render rep*: splat via `_resolved_roots`→`resolve_subject_motion`, the overlay reprojects resolved
roots/ball, and observe snapshots the resolved scene — M2-4 closed the one gap (root orientation was
previously dropped). Honest deferred limit: `POSE_BODY_JOINT` and `SHAPE_BETA` edits still need the
heavy SMPL-X LBS model and are *not* faked — the dependency-free pass honours the rigid root only.
**Next: M2-1** env reconstruction.

**Progress — M2-1 measured-pitch env wired E2E (2026-06-25):** the environment we genuinely
*measure* is the **pitch plane**, so M2-1 ships exactly that and gates the hallucinated rest. New
pure-numpy core geometry `core/scene/pitch.py` (`pitch_line_world_points`) generates the standard
Laws-of-the-Game markings — outer rectangle, halfway line, centre circle/spot, both penalty &
goal areas, penalty spots and the "D" arcs — in world meters on `Z = plane_z`. A new
`MeasuredPitchEnvReconstructor` (`--env pitch`) emits them as a vertex-coloured PLY with **every
vertex `measured=1`** (`coverage=1.0`): it is a measured template, anchored by the M1 calibration,
and is the M2-0 *validator anchor* ("a leg can't pass through the pitch"). It is labelled with a new
honest `ENV_PITCH_MESH` asset kind — **not** the `env_splat` it is not. The same `SplatAvatarRenderPass`
now splats world-space **env meshes** by identity (no per-subject root) under the *shared* z-buffer,
so the pitch grounds the avatars (a near player occludes a far line); `measured=0` env verts would
ride the same R-6 tint, though the pitch is fully measured. New controller stage
`Application.build_env` attaches one `RenderAssetRef` to the scene; the dry-run runs it as stage 9b
(`… → avatar → **env** → render → export`). E2E: `--env pitch --render splat` renders the markings —
a decoded frame shows ~24k white pitch pixels over the background (manifest `env_vertices=1444`),
while `--env fake` returns a placeholder marker the splat pass honestly skips (`env_meshes=0`).
Honest deferred (R-8): a photoreal 3DGS/NeRF stadium from camera motion, or a *generative* stadium
when motion is insufficient, stay gated in `SplatEnvReconstructor` until their milestone — those
hallucinate unmeasured stands. **Next: M2-5** (ViewSynthesizer seam A).

**Progress — M2-5 ViewSynthesizer seam A wired E2E (2026-06-25):** seam A feeds the *eye*, never
the reconstruction — it re-shoots the source clip along a bounded orbit and returns a photoreal
**video, not editable**. New pure camera math `core/agent/viewpoints.py:bounded_orbit_camera`
pans the BROADCAST framing by a *bounded* azimuth arc around the action centroid (one prescribed
`estimated=False` pose per source frame, ground radius + height preserved so it is an orbit not a
dolly), **hard-capped at ±45°** (R-15: a moderate re-aim, never free-viewpoint). The `adapters/render`
stub is now real: `ViewSynthOrbitRenderPass` delegates to `ViewSynthesizer.render_orbit` and wraps
the `SynthViewRef` as a `RenderResult(is_video=True)` via the pure `orbit_render_result` (which
carries `frustum_overlap` + the non-editable flag into the note, so an orbit re-shoot can never
masquerade as an editable result). New controller use-case `Application.render_orbit` is the
authoritative path: it builds the orbit over the **registered clip** (uri/fps the resolved-scene
contract can't carry), calls the synthesizer, **content-addresses + caches** the ref (ADR-0004 — a
second call is a cache hit, no recompute), and attaches it to `scene.synth_views` (deduped on id).
Wiring adds the `--render orbit` selector (sharing one synthesizer instance with the use-case) and
the dry-run runs seam A as stage 10b (`… → render → **seam-A** → export`). E2E: stage 10b reports
`A_render overlap=0.85 editable=False — video, not editable`, `cached + deduped: 1 synth_view after
2 render_orbit call(s)`; `--render orbit` makes the main render `is_video=True`. Honest deferred
(R-8, ADR-0007): the real generative backends (ReCamMaster/GEN3C/TrajectoryCrafter-class) stay gated
in `GenerativeViewSynthesizer`; only the dependency-free fake re-shoots here. **Next: M2-6**.

**Progress — M2-6 fast low-q preview wired E2E (2026-06-25):** "preview before the expensive
final" (UX-9) is implemented as one honest lever — **resolution**. New `RenderQuality.scale`
(preview = 0.5 → ¼ the pixels, final = 1.0) drives new pure camera math `CameraIntrinsics.scaled`
/ `CameraTrack.scaled`, which downscale **only the intrinsics**: per-frame pose, frame indices and
frame *count* are preserved, so the same content rasterises at fewer pixels (no geometry faked,
just cheaper). `scale(1.0)` returns `self`, so a FINAL render keeps the caller's exact camera
(identity intact). Both TZ paths consume it: `SplatAvatarRenderPass.render` renders into the
downscaled framebuffer (manifest/note report the actual size), and the seam-A pair
(`ViewSynthOrbitRenderPass.render` + `Application.render_orbit`) downscales the bounded orbit for a
cheaper generative re-shoot and threads `quality` into both the **scene hint** (a real synthesizer
can also drop steps) and the **cache key** — so the preview and the final are *distinct* ADR-0004
entries and neither expensive pass recomputes for the same inputs. The debug `ReprojectionOverlay`
pass is deliberately left full-res (it is the reprojection inspector, not a TZ render path). E2E:
`--render splat` writes the preview manifest `size=640x360 quality=preview` (the 1280×720 broadcast
camera at ¼ the pixels); `--render orbit` re-shoots the orbit at 640×360 with the 6-frame count
preserved and `is_video=True`. Honest scope (R-6): the lever changes *resolution only* — final-grade
fidelity (texture, AA, real generative steps) still rides the heavy backends, which stay gated.
**Honest status (re-assessed 2026-06-27): the M2 GOAL — a photoreal render of textured, LBS-posed
avatars on grass, edit↔render/observe in sync — is MET (🟢).** M2-0…M2-6 delivered the render-pass
*architecture* (seam, edit↔render sync on the rigid root, content-addressed cache, preview-downscale,
measured pitch-env, seam-A contract) plus a measured **vertex-splat** visualization — debug-grade,
pure-numpy. **M2-7 wired a real Blender/Cycles renderer, M2-8a LBS-poses the avatars through it,
M2-8b paints them from measured pixels, M2-9 puts them on the measured grass pitch, and M2-10 closes
the loop** (below) — root *and* per-joint limbs follow the resolved pose (a bent elbow renders bent),
in-frustum verts take the player's real broadcast colour (unseen verts stay honest R-6 grey), the
ground is a **grass-PBR pitch with the standard line markings under a physical sky**, and now an
**edit re-projects into the Cycles frame with no avatar rebuild** (AC-4), `observe` returns
**photoreal** multi-view (A-8, `CyclesSceneObserver`), and seam A is a real **re-render of the 3D
scene** at the orbit cameras (A-9, `CyclesViewSynthesizer`) — cached, `editable=False`, no generative.
All four exit clauses are gated-tested on real Blender and verified in the integrated CLI E2E
(`--render cycles --observer cycles --viewsynth cycles`: 6 photoreal observe views, 2 rendered frames,
a cached seam-A orbit at overlap 0.78). **R-6 honesty, unchanged at 🟢:** the *measured* ceiling is
textured mannequins (single broadcast cam ⇒ half the body unmeasured; naked SMPL-X ⇒ no clothing
geometry; no stadium), and texture quality is only as good as the *pose alignment* — on the **fake**
pose pipeline the synthetic bodies don't sit on the real players, so measured coverage is ~0–1%
(genuine per-player colour waits on the still-gated GVHMR pose, M1). The renderer and all seams are
real and in sync; **broadcast-convincing fidelity is M3.** Next: **M3**.

**Progress — M2-7 `CyclesRenderPass` wired E2E (2026-06-25):** the first *real* renderer — a
Blender/Cycles `RenderPass` over the **resolved** scene, the photoreal path alongside the no-dep
splat debug viz (ADR-0003: Blender hosts editing, photoreal rendering runs external behind the port).
The error-prone maths is a new **dependency-free** `adapters/blender/cycles_plan.py` (mirroring
`proxy.py`): the OpenCV→Blender camera conversion (world→cam optical `R,t` → Blender `matrix_world`
camera→world, optical flip `diag(1,-1,-1)`, centre `-Rᵀt`), the BlenderProc-style `K`→lens mapping
(`fx`/`fy` → `pixel_aspect`, principal point → lens shift, HORIZONTAL fit, `lens = fx·sensor/width`),
and rigid-root placement identical to the splat formula (`axis-angle → R` **⊕** root translation) —
all unit-tested with strong invariants (`flip @ X_c == Rᵇᵀ(P−C)`, placement vs the verbatim splat
formula), no Blender. The heavy half (`_cycles_script.py`) runs **out of process** like the proxy
builder (`blender --background --factory-startup --python … -- --plan … --mesh-dir … --render-dir …`,
sentinel `PITCH3D_BLENDER_OK`): it loads each avatar NPZ (verts/faces/**baked R-6-tinted** vertex
colours), builds a Cycles mesh with a `ShaderNodeVertexColor → Principled` material, adds a sun + sky
world + a neutral ground plane, and renders `frame_{i:05d}.png` per camera frame. `CyclesRenderPass`
(`adapters/render/cycles.py`, `--render cycles`) is the orchestrator: it resolves the scene's avatar
PLYs, bakes the R-6 tint into per-vertex `rgb01`, writes the NPZ meshes + JSON plan into a tempdir and
drives the subprocess, consuming `RenderQuality.scale` (M2-6) so a PREVIEW renders at ¼ the pixels.
**E2E proof (R-6 honest):** the Blender-gated integration test takes a real measured PLY triangle all
the way to a **non-empty lit Cycles PNG** (FINAL 160×120); the full `--render cycles` dry-run drives
`reconstruct → … → render → Blender/Cycles → 2 real 640×360 frames` (decoded pixel stats confirm lit
sky+ground content, not black). The manifest honestly reports `avatars=0` on the *fake* pipeline —
the fake avatar builder emits no real mesh, exactly as the splat pass does; real avatars need
`--avatar textured` (SMPL-X, the M2-8 path). **Honest scope (R-6):** rigid-**root** placement only —
per-frame limb **LBS is M2-8**, untextured beyond vertex colour and the **grass/line materials are
M2-9**, edit↔render/observe through Cycles is M2-10. So this wires the *renderer*, not yet the
photoreal *avatar on grass* — M2 stays 🟡. **Next: M2-8** (LBS-posed textured avatars through Cycles).

**Progress — M2-8a posed geometry (LBS) wired E2E (2026-06-27):** the avatars now **articulate** —
geometry follows the resolved pose, root *and* limbs. The skinning is a new **dependency-free**
`adapters/models/smplx_lbs.py` (the "model-independent maths runs with no torch/GPU" split again):
shape + pose-corrective blendshapes → forward kinematics down the kintree → rest-pose removal →
linear-blend skinning, all pure numpy reading the SMPL-X `.npz` directly (located like every gated
asset). Correctness is not on faith — a gated unit test **cross-checks it against the reference
`smplx` package** (posed match ~1e-4) plus invariants (a zero pose is the shaped template; one joint
articulates only its kinematic subtree). `SmplxTextureBackend.observe` is now wired for its
**geometry half**: the subject's `betas` → the canonical shaped mesh (10475 verts) + faces, returned
*geometry-only* (`frames=[]`), so every vertex stays `measured=0` (honest R-6 grey) until M2-8b
samples real pixels. `CyclesRenderPass` poses each SMPL-X avatar per frame (`model.pose_sequence` →
`(T,V,3)`), writes the posed-vertex NPZ, and the plan carries **identity placements + a per-frame
`vert_index`**; `_cycles_script.py` swaps that frame's vertices into the shared mesh before each
render. A non-SMPL-X mesh or a missing model **falls back to rigid-root** (M2-7) — an explicit,
honest limit, not a silent fake. **E2E proof (R-6 honest):** a Blender+SMPL-X-gated test renders rest
vs a hard left-elbow bend sharing root, camera and Cycles seed → **only the forearm pixels change**
(≈183 px, tightly localized; a rigid bug would give 0). The full `--avatar textured --render cycles`
dry-run builds a **real 10475-vert SMPL-X avatar** (coverage 0.00 — all R-6 grey, honest) and renders
2 posed Cycles frames (`avatars=1 posed=1 vertices=10475 unmeasured=10475`). **Honest scope (R-6):**
the geometry follows the pose, but the surface is **untextured** (R-6 grey) until **M2-8b** threads
the scene camera + decoded frames into `observe`, and the ground is neutral (grass is **M2-9**). So
this delivers the *posed* half of M2-8, not the *textured* half — **M2 stays 🟡. Next: M2-8b.**

**Progress — M2-8b measured texture wired E2E (2026-06-27):** the avatars now take their **real
broadcast colour**. The plumbing keeps the hexagon intact — core holds only *references*, so the scene
`camera` (`CameraTrack`, already a core type) and source `clip` (`ClipRef`) thread through
`AvatarBuilder.build` → `AvatarMeshBackend.observe` as keyword-only optionals; pixel decoding stays on
the adapter side. A shared `adapters/io/frames.py` decoder (`iter_clip_frames` — directory or video
seek, `file://` strip; reused by the detector) hands frames to `SmplxTextureBackend._sample_frames`,
which poses the subject at its **measured** (proposal) pose into a **capped, evenly-spread** set of
reference frames (`max_ref_frames=8`, `_even_subset` for distinct viewing angles per decode), then the
existing pure `vertex_normals → sample_vertex_colors (front-facing + in-frustum + z-buffer) →
aggregate_observations` path paints per-vertex colour. The colour is keyed off **canonical vertex
ids** so it is pose-invariant; `cv2` BGR is flipped to stored RGB. **R-6 throughout:** no camera/clip,
a synthetic URI with no pixels behind it, or no frame common to pose∩camera∩clip → geometry-only
(`frames=[]`, every vertex `measured=0` grey), never a fabricated colour. **E2E proof:** a
SMPL-X+cv2-gated unit test poses the real 10475-vert body into three solid-colour frames and asserts
every *measured* vertex carries exactly that colour (BGR→RGB), with genuine partial coverage in
(0, 1) (the back is never seen); the geometry-only fallback is asserted for a synthetic URI. The real
`--clip … --avatar textured` dry-run on the Colombia broadcast decodes real frames and samples real
pixels onto both subjects (coverage ~1%, 140/107 of 10475 verts) — **honest and expected**: the
*fake* pose places synthetic bodies that don't sit on the real players, so the texturer samples
mostly background; genuine coverage waits on the wired GVHMR pose, not on the texturer. **Honest scope
(R-6):** this delivers the *textured* half of M2-8 (measured, not hallucinated), but the ground is
still neutral (grass/line materials are **M2-9**) and edit↔render/observe through Cycles is **M2-10**,
so the photoreal *avatar on grass* still does not exist — **M2 stays 🟡. Next: M2-9.**

**Progress — M2-9 measured grass pitch + sky wired E2E (2026-06-27):** the avatars now stand on the
**measured pitch**, not a neutral plane. Three pieces, R-6 honest throughout. (1) The standard line
markings become real geometry: a new pure `pitch_line_ribbons` (`core/scene/pitch.py`) turns each
measured marking polyline (the same FIFA template the calibration anchors to) into width-wide white
quad ribbons on the pitch plane, lifted 1 cm to dodge z-fighting and wound +Z-up — crisp continuous
lines with **nothing fabricated** (no texture, no hallucinated stadium). (2) `CyclesPlan` gains a
`pitch_npz` ref (round-tripped through the JSON subprocess boundary); `CyclesRenderPass` emits the
ribbons into the mesh tempdir and the manifest reports `env=grass+lines+sky`. (3) `_cycles_script.py`
swaps the neutral matte ground for a **procedural grass PBR** (banded mowing stripes quantised to two
greens + a noise bump for blade micro-texture, high roughness), loads the ribbon NPZ as a flat
matte-white mesh, and replaces the flat-tint world with a **physically-based multiple-scattering sky**
(Blender 5.x's Nishita successor; sun *disc* off, a matched SUN lamp carries the crisp key light so it
converges clean at 48 spp). **E2E proof (R-6 honest):** the Blender-gated render test now asserts
`env=grass+lines+sky` in the manifest; new pure tests pin the ribbon geometry (valid indexed quads,
+Z winding, on the lifted plane, inside the ground bounds) and the `pitch_npz` JSON round-trip. The
full `--avatar textured --render cycles --env pitch` dry-run renders 2 lit 640×360 frames whose pixels
are **76% green (grass), 2% near-white (line paint), 0% black** under a bright sky — a measured grass
pitch with markings, lit. **Honest scope (R-6):** the *environment* is now measured-real, but
edit↔render/observe *through Cycles* and the seam-A re-render are still **M2-10**, so the full photoreal
loop (edit a pose → see it re-render; `observe` returns photoreal views) does not yet exist — **M2
stays 🟡. Next: M2-10.**

**Progress — M2-10 edit↔render/observe sync through Cycles wired E2E (2026-06-27) — M2 GOAL met,
flipped 🟡→🟢:** the photoreal loop now *closes* through the real renderer. Three seams, no new
heavy machinery — each reuses `CyclesRenderPass` so what the agent edits, observes and orbits are the
same photoreal pixels. **(AC-4) Edit re-projects with no avatar rebuild:** `CyclesRenderPass` already
resolves `scene.corrections` at render time (`_pose_avatar` → `resolve_subject_motion` → SMPL-X LBS),
so an edit is layered, not baked — proven by a Blender+SMPL-X gated test that renders the *same* PLY
twice (clean vs after a `ROOT_TRANSLATION` **and** a `POSE_BODY_JOINT` correction) and asserts the
root move + forearm swing change a localized block of pixels (a rebuild-required pipeline would render
two identical frames). **(A-8) Photoreal `observe`:** `CyclesSceneObserver` (`adapters/render/observe.py`,
`--observer cycles`) renders each canonical viewpoint as its own single-frame Cycles pass and copies
the frame out to a stable per-view URI before the next overwrites it; the camera-free overlay/radar/UI
delegate to the fake (same split as the proxy observer). **(A-9) Seam-A orbit = real re-render:**
`CyclesViewSynthesizer` (`adapters/render/cycles_orbit.py`, `--viewsynth cycles`) makes the bounded
orbit honest — it re-renders the *resolved 3D scene* (passed as `scene_hints["scene"]`) at the orbit
cameras instead of hallucinating frames; `frustum_overlap` falls with the re-aim, the result stays
`editable=False` "video, not editable", and the generative seam B stays gated (R-8). The controller's
orbit cache now keys on a content `_orbit_fingerprint` of the resolved poses + assets, so an edit
busts the cache (the orbit would show different geometry) while an unedited re-call still hits it.
**E2E proof (R-6 honest):** `--avatar textured --env pitch --render cycles --observer cycles
--viewsynth cycles` ran the whole loop — reconstruct → **observe (6 photoreal views)** → edit →
resolve → **observe again** → render (2 frames) → **seam-A orbit (overlap 0.78, editable=False,
cached + deduped over 2 calls)** → export. Tests: 3 new Blender-gated tests pass in 23.5 s; pure
wiring/contract tests pin the selectors, the scene-hint requirement, the gated seam B, and the
cache-bust-on-edit. **Honest ceiling (unchanged, R-6):** the avatars are textured *mannequins* and on
the fake-pose path coverage is 0% (bodies don't sit on real players → all R-6 grey) — genuine colour
waits on the gated GVHMR pose (M1); broadcast fidelity is M3. **The renderer + all seams are real and
in sync, so M2 is 🟢. Next: M3.**

---

## M3 — Quality & polish 🟢  (+ ViewSynthesizer **seam B**)

**Goal (TZ M3):** raise reconstruction quality and operator efficiency.

| Ticket | Package |
|---|---|
| ✅ **M3-1 Per-subject Gaussian avatars (#3)** selectively — candidates: feed-forward Gaussian LRMs (**IDOL** <1 s, **LHM**, **PF-LHM** pose-free/multi-image) for the bulk; per-subject **GaussianAvatar/GART** for hero shots. All flagged *synthesized* (R-6). | `adapters/models` (`AvatarBuilder`) |
| ✅ **M3-2 Constraint-guided re-fit hardened (PromptHMR-class)** — root XY locked to the **measured homography anchor** (`core.correction.anchor`, validated; off-anchor frames flagged R-6, never silently trusted); **cluster-occlusion robustness** option — amodal occlusion completion (**Diffusion-VAS**) + pixel-level identity (**SAM-3** masklets) gated to occluded segments (R-8), validated against the homography anchor (from the brief) | `adapters/models` (`PoseEstimator.refit`) |
| ✅ **M3-3 ViewSynthesizer seam B** — `amplify` (mono → pseudo-multi-view) feeding env/avatar recon | `adapters/viewsynth`, `core/orchestration` |
| ✅ **M3-4 ViewSynthesizer seam B** — `inpaint_occlusions` for unseen player sides (multi-view human-diffusion candidates: **PSHuman** / **SiTH** / **AniGS**; text-conditioned **TeCH** for kit+number) | `adapters/viewsynth`, `adapters/models` |
| ✅ **M3-5 Confidence map + "needs attention" prioritization UI** | `adapters/render`, `core/scene` |
| ✅ **M3-6 Versioning / named snapshots / rollback** — pure-core `SnapshotStore` (`core/scene/versioning.py`): a named `Snapshot` is a **deep, independent copy** of the `Scene` tagged with a **content-addressed SHA-256 fingerprint** (ADR-0004, over the canonical-JSON codec) so a redundant snapshot is detectable and a rollback to current state is a provable no-op; both `take` and `restore` copy, so a checkpoint stays pristine. Exposed as `snapshot`/`list_snapshots`/`rollback` use-cases on the shared controller API (the same surface the operator and MCP agent drive, ADR-0008/0010) — the checkpoint primitive an agent uses to bracket a risky edit. | `core/scene`, `app` |
| ✅ **M3-7 Web export (three.js / R3F)** — `WebViewerExporter` (`adapters/export/web.py`): a **dependency-free** self-contained `index.html` (no build step, no server — scene data inlined) + standalone `scene.json`; shows the **resolved** subject roots + ball as animated team-coloured markers on a metric pitch in glTF **Y-up** (reuses the glTF Z-up→Y-up conversion, opens without scale/coord loss, AC-6). Honest scope (R-6): markers, not SMPL-X meshes. | `adapters/export` |
| ✅ **M3-8 Learned motion-prior smoothing (option)** — a learned denoiser/motion-prior as an alternative to moving-average, behind the existing `smoothing`/`refit` Correction seam. Leading candidates: **HTD-Refine**, **StableMotion** (purpose-built denoisers); MotionBricks = generative fallback | `adapters/models` ↔ `core/correction` |
| ✅ **M3-9 Kinematic plausibility gate (player physics)** — DONE 2026-07-03. `core/correction/kinematics.py`: per-subject speed/accel limits (shared ceilings `HUMAN_MAX_SPEED` 10.5 m/s, `HUMAN_MAX_ACCEL` 8 m/s²) on the root XY track. (b) **feasible-set projection** (velocity clamp + bounded fw/bw accel sweeps + guaranteed-feasible final forward sweep; both endpoints anchored) emitted as ONE dense `KEYFRAME_INTERP` correction per subject through the ADR-0002 seam — inspectable, disableable, supersedes the MA(5) basis it resolved through. (c) **teleports MARKED, not erased** (R-6): interval speed >2× limit → `TeleportEvent`; out-and-back spikes demoted to clampable jitter by a velocity-reversal test; **consecutive runs collapse into ONE preserved region** (no feasible anchored path covers their displacement — clamping would invent motion). **Worst measured slide fixed at SOURCE:** the coherence edge-coast inherited a dying track's 43 m/s edge velocity — now capped (`CoherenceConfig.coast_max_speed=HUMAN_MAX_SPEED`). Real-scene result: speed/accel viols 22/999→**0/0**, 10 raw teleports→**1 region event**; CLI `--physics`, deliverable path defaults `PHYSICS=1`, env `PITCH3D_KIN_MAX_SPEED/MAX_ACCEL/TELEPORT`. M3-8's learned prior plugs the same seam as a stronger (b). | `core/correction` (`kinematics`/`coherence`/`engine`) ↔ `app/controller`, `app/cli` |

**Exit criteria (TZ AC-5b, AC-6, AC-7):** seam B emits N synthetic views accepted by
reconstruction as multi-view input; export + three.js viewer open externally without scale/coord
loss; core still passes tests with fakes (incl. `FakeViewSynthesizer`) **without GPU/Blender**.

**On M3-8 (option, not committed):** the *denoiser*-style models fit our smoothing/refit seam best —
**HTD-Refine** (arXiv 2605.26879, post-hoc SMPL-X velocity/accel refiner, −58–77% foot-slide/jitter)
and **StableMotion** (arXiv 2505.03154, diffusion mocap cleanup from *unpaired* corrupt data,
**soccer-proven** −68% pops / −81% frozen frames). NVIDIA **MotionBricks** (SIGGRAPH 2026) is a
*generative* model (constraints→motion), not a from-video estimator — a weaker fit as a denoiser, a
fallback only. None replaces the HMR backend (B2 pick stays SMPLest-X + SMART); all inject behind the
`smoothing`/`refit` port (ADR-0006) as an isolated swap. Open before adopting: licences (NVIDIA-
research; HTD-Refine / StableMotion code pending), **SMPL-X** in/out compatibility, and football-
motion coverage. Tracked as research, off the M1 critical path.

**Progress — M3-7 web viewer (three.js) wired E2E (2026-06-27):** M3 opens with the cleanest
fully-shippable win (real today, no GPU, no generative gating). `ExportFormat.THREEJS` — already on
the port — is now implemented dependency-free: `GltfExporter` (also `--export threejs`) emits a
self-contained bundle (`index.html` + `scene.json`) via `adapters/export/web.py`. The viewer shows
the **resolved** subject roots + ball as animated, team-coloured markers on a metric pitch, **reusing
`build_gltf_scene`'s Z-up→Y-up conversion** so the web view inherits the glTF export's exact
axis/scale — it can't drift (**AC-6**, "without scale/coord loss", pinned by a parity test against
the glTF tracks). The data is **embedded** in the page (opens off the filesystem, no server); three.js
is pinned from a CDN at view time (the only view-time dep). **E2E proof:** `--export threejs --format
threejs` ran the golden path → `scene.json` (up=Y, pitch 105×68, 6 nodes = 5 subjects + ball, 8
samples each); the generated viewer JS passes `node --check`. **Tests:** 6 new (`test_export_web.py`)
+ the supports-matrix updated; suite **486 passed / 5 skipped**. **Honest scope (R-6):** markers, not
SMPL-X meshes — the full textured-mesh web viewer rides the gated `.glb` path (`export` extra) in a
later increment; the WebGL render itself isn't headlessly testable here (open `index.html` to view).

**Progress — M3-6 named snapshots + rollback (2026-06-27):** the operator-efficiency primitive, pure
core. Corrections are the sole edit path (ADR-0002) and the proposal is never mutated, so the state
worth checkpointing is the whole `Scene` (correction stack + assembled assets). New
`core/scene/versioning.py`: a `Snapshot` is a **deep, independent copy** tagged with a
**content-addressed SHA-256 fingerprint** (`scene_fingerprint`, reusing the canonical-JSON
`serialization.encode` codec, ADR-0004) — identical content → identical digest, so a redundant
snapshot is detectable and a rollback to the *current* state is a **provable no-op**. `SnapshotStore`
is in-memory, no I/O (last-write-wins per name); **both `take` and `restore` deep-copy**, so a
checkpoint stays pristine even as the live scene keeps being edited. Wired as `snapshot` /
`list_snapshots` / `rollback` use-cases on the shared `app` controller — the same surface the operator
and the MCP agent drive (ADR-0008/0010), i.e. the checkpoint an agent can use to bracket a risky edit
and roll back if the attention list gets worse. **Tests:** 9 (`test_versioning.py`) — fingerprint is
content-addressed + stable across deep copies, snapshots are independent (mutating the live scene or a
restored copy leaves the snapshot untouched), rollback-to-current is a no-op, last-write-wins on name,
empty name rejected. Pure core, **no GPU/Blender** (**AC-7**).

**Progress — M3-3/M3-4 ViewSynthesizer seam B wired E2E (2026-06-27):** the **headline M3 exit
(AC-5b)**. The seam-B contract was *defined* in M0 (`ViewSynthesizer.amplify` /
`inpaint_occlusions`, `SynthViewRef`, and `synth_views=` params already on `EnvReconstructor` /
`AvatarBuilder`); this increment makes it **flow** — two controller use-cases plus the orchestration
that routes the synthesized views into reconstruction. `Application.amplify_views(scene_id, n_views,
deviation)` turns the mono broadcast camera into N pseudo-multi-views (FR-30), content-addressed +
cached (ADR-0004); `inpaint_subject(scene_id, track_id)` hallucinates one subject's unseen sides
(FR-31). Both attach to the scene's `synth_views`. **The wiring that closes AC-5b:** `build_env`
feeds the scene-shared `B_AMPLIFY` views to the reconstructor, and `build_avatars` feeds each subject
the shared amplify views **plus that subject's own `B_INPAINT`** view — and "accepted by
reconstruction" is now **observable**, the fakes record how many views they consumed
(`extra["synth_views"]`). **Frustum overlap falls as the synthetic camera strays** (R-14/R-16), so a
caller can still gate trust. **R-8 honesty:** the real diffusion backend (`GenerativeViewSynthesizer`,
`--viewsynth generative`) is importable but every method raises an actionable `NotImplementedError`
pointing at the `viewsynth` extra + the fake — it never silently runs; the dry-run skips it
gracefully and reconstruction proceeds on the mono view. **E2E proof:** `--amplify-views 3
--amplify-deviation 0.35` ran the golden path → both avatars **and** the env report `synth_views=3
(seam B)`; `--viewsynth generative` skips honestly (exit 0). **Tests:** 15 new
(`test_viewsynth_seamb.py`) — amplify cardinality/overlap, cache hit + param-distinct entries,
inpaint per-subject routing, the AC-5b "consumed-by-reconstruction" assertions, and the R-8 gate;
suite **503 passed / 12 skipped**, no GPU/diffusion (**AC-7**).

**Progress — M3-5 confidence map + "needs attention" UI wired E2E (2026-06-27):** closes the
**visual half of UX-4/FR-17**. The *ranking* (`core.scene.review.attention_list`) and the per-marker
`confidence_to_color` ramp already existed; what was missing was **the picture the operator/LLM
reads**. New `adapters/render/attention.py` (pure numpy + the stdlib PNG encoder — **no
GPU/Blender, no font engine**) renders two panels: a dense **per-subject × per-frame confidence
heatmap** (each row the team colour pulled toward warning-red as confidence drops; a ball-height
row) and a ranked **"needs attention" bar chart** (most-urgent on top, bar length ∝ severity, hue ∝
reason — red = low confidence, orange = high reprojection, gold = ball height). `render_attention_ui`
composites both into the PNG the `SceneObserver` returns as the `UI` observation, so the agent
*literally sees what to fix* (ADR-0008). **R-6 honesty:** a subject/scene with no measured
confidence stays flat background — never a falsely-green "all good"; and meaning is carried by rank,
length and hue, so it is **deterministic and pixel-testable** rather than relying on rendered text.
`FakeSceneObserver.capture_ui` now writes the real panel (lazy-import, mirroring `capture_radar`);
headless-with-no-scene still returns the flat placeholder. **E2E proof:** `--frames 6 --subjects 3`
ran the golden path → the `ui` observation present, "Needs attention (2, highest first)", and a real
`scene-1_ui.png` artifact written. **Tests:** 8 new (`test_attention_ui.py`) — heatmap red-pull +
honest-background + ball row, panel rank/colour + all-clear bar, valid-PNG composite, observer
wiring (bytes match the renderer) + headless placeholder; suite **511 passed / 12 skipped** (**AC-7**).

**Progress — M3-1 per-subject Gaussian (3DGS) avatar wired E2E (2026-06-27):** strategy #3, built
the **measured-over-generative** way (M2-0) the textured builder (#1) is. New
`adapters/models/gaussian_avatar.py` **anchors one 3D Gaussian on every measured SMPL-X vertex** —
centre from the canonical mesh, colour from the player's real broadcast pixels (it **reuses the
exact measured sampling** the textured builder uses, now extracted as `measured_vertex_texture`),
scale from local surface spacing, identity rotation. A vertex the cameras never saw stays
``measured=0`` with a *faint* (not invisible, not confident) opacity rather than a fabricated splat
(**R-6**). `write_gaussian_splat_ply` emits the **standard INRIA-3DGS `.ply`** (SH band-0 colour,
log scale, logit opacity) so off-the-shelf splat viewers load it, plus our per-Gaussian ``measured``
uchar (the honesty channel carried in the asset). So the **pure half is a real, deterministic,
GPU-free 3DGS init** — not a stub. **R-8 gating:** the generative densify/inpaint refiner
(IDOL/LHM/PF-LHM/GART), `FeedForwardGaussianRefiner`, is importable (no torch) but every call raises
an actionable `NotImplementedError` pointing at the `avatar` extra; it is **optional** — the builder
runs without it, and only engages the gated path when a refiner is injected. Selected with
`--avatar gaussian`; inject a real mesh backend by dotted path (`--avatar-backend`, ADR-0006) exactly
like `textured`. **E2E proof:** `--avatar gaussian --avatar-backend …:SyntheticAvatarMeshBackend`
ran the golden path → 3 real splat `.ply` assets at a genuine 2/3 coverage; the default SMPL-X
backend (no real pixels) honestly yields a 10 475-Gaussian geometry-only avatar at coverage 0 (knows
the shape, not the look). **Tests:** 10 new (`test_gaussian_avatar.py`) — SH/`.ply` round-trips,
mesh→Gaussian anchoring + faint-unmeasured opacity, the synthetic measured build, geometry-only
honesty, the R-8 refiner gate, and the wiring; suite **521 passed / 12 skipped**, no GPU (**AC-7**).

**Progress — M3-2 hardened re-fit on the measured homography anchor + gated occlusion seam (2026-06-27):**
the **measured-over-generative** (M2-0) half of M3-2, built real. New `core/correction/anchor.py` is a
pure, deterministic, GPU-free **homography-anchor validator**: `anchor_residuals` (per-frame horizontal
XY distance, Z ignored — mono depth is ambiguous, **R-4**), `blend_to_anchor` (pull root XY toward the
measured ground track), and `validate_against_anchor` → `AnchorReport` that **surfaces off-anchor frames
(R-6)** for the attention list / a confidence dip rather than silently accepting them. The re-fit is
hardened with two constraints shared by **both** pose adapters (fake + `GVHMRPoseEstimator`, same
contract): `foot_anchor` (lock root XY to the bbox-foot→world homography track, per-frame `(M,2)` or a
single `(2,)`) and `anchor_blend` (partial pull); **Z is left untouched** so the mono height stays a
single source of truth. **R-8 gating:** the generative cluster-occlusion completer
(`OcclusionBackend` Protocol + `DiffusionVasOcclusionBackend` — Diffusion-VAS amodal masks + SAM-3
masklets) is importable (no torch) but raises an actionable `NotImplementedError` pointing at the
`occlusion` extra and the measured `--coherence` gap-fill alternative; it engages **only** when injected
(`--occlusion-backend pkg:Factory`, ADR-0006, gated to `--pose gvhmr`), runs inside `refit` **before** the
anchor lock is applied on top, and any completion it produces is meant to be validated against the same
anchor. **E2E proof:** the dry-run re-fits the first subject's first 7 frames locked to its measured
ground track → `on-anchor 12/12 (max residual 0.000 m, 0 off-anchor, R-6)`; golden path completes (exit
0), Z-offset preserved, corrections bake empty on resolve (no regression). **Tests:** 14 new
(`test_occlusion_anchor.py`) — pure anchor math (residuals/blend/validate + ragged rejection), the XY-only
lock (parametrized fake+gvhmr, Z untouched, input intact), per-frame & blended anchors, refit→validate
all-on-anchor, the R-8 occlusion gate + actionable no-backend error, an injected-stub proving the
completion output flows then the anchor locks on top, and the wiring (inject + `requires --pose gvhmr`);
suite **535 passed / 12 skipped**, no GPU/torch (**AC-7**).

**Progress — M3-8 learned motion-prior through the existing smoothing seam (2026-06-27):** the seam
half, built real. A learned denoiser is offered as a **new `method="learned"` on the existing
`TEMPORAL_SMOOTHING` correction** (reuses `make_smoothing`/`SmoothingPayload` — no new mode or payload),
routed through an injected **`MotionPrior` port** (`core/ports/motion_prior.py`, `denoise(values, frames,
*, is_rotation)`), exactly mirroring how REFIT routes through `refit_port`. The engine threads
`motion_prior` through `resolve_subject_motion`/`preview_subject_motion`/`resolve_scene` and the
controller/CLI, so a learned model stays a normal, inspectable **correction** (ADR-0002), never a hidden
edit. `FakeMotionPrior` (`adapters/fakes`) is a **fake-real, GPU-free** zero-phase gaussian denoiser
(reuses the engine smoothers, rotation-aware via `is_rotation`) — the seam runs and is tested now, not a
stub; the pure `moving_average`/`gaussian` methods still need **no** prior. **R-8 gating:**
`LearnedMotionPrior` (`adapters/models/motion.py`, HTD-Refine/StableMotion-class) is importable (no torch)
but `denoise` raises an actionable `NotImplementedError` pointing at the `motion` extra, the GPU-free fake,
and — crucially — that any learned completion must be **validated against the homography anchor** (off-prior
drift is hallucinated, not measured, **R-6**); it engages only when wired (`--motion-prior learned` or a
dotted-path BYO, ADR-0006). **E2E proof:** the dry-run *previews* (FR-23, not committed) a learned-smooth of
the first subject's stepped root path → `MotionPrior 'FakeMotionPrior' denoised subject 0 root over 12
frame(s) → max_abs_change 0.0365 m`; with `--motion-prior learned` the gate **degrades gracefully**
(`skipped — learned model not wired (NotImplementedError)`) and the golden path still completes (exit 0),
corrections bake empty on resolve (no regression). **Tests:** 13 new (`test_motion_prior.py`) — the fake
prior is a `MotionPrior` + softens a step (total-variation drops, input not mutated) + deterministic and
rotation-aware; the engine routes `method="learned"` through the port (== `fake.denoise`, differs from the
proposal, proposal intact), raises actionably with no prior, passes `is_rotation` per target (translation
euclidean, orientation rotation), and touches only its frame range; pure smoothing still needs no prior;
preview threads the prior; the gated model raises; wiring defaults to the fake, selects the gated learned,
accepts a dotted-path BYO, and rejects a bad spec. Suite **548 passed / 12 skipped**, no GPU/torch
(**AC-7**).

**Progress — A-10 bounded, attention-driven autonomy + eval harness (2026-06-27):** the agent's
*hands*, kept honest — and with it the **M3 agent exit criterion** ("an agent fixes a seeded-wrong
pose end-to-end, verified by the attention list clearing") is met. New `core/agent/autonomy.py` is the
closed loop as **pure core** (numpy + `resolve_scene`, no LLM/port/GPU), so it is deterministic and
unit-tested. The hard problem was that `subject_frame_conf` is *static* — a loop can't re-check its own
work against a frozen score; the fix is to make the attention signal **recomputable from a measured
ground truth**: `residual_to_confidence` bridges the **anchor residual** (M3-2's
`core/correction/anchor.py`) to confidence as `m/(m+r)` (1.0 on-anchor, exactly **0.5 at the
`max_residual_m` tolerance** — lined up with the attention threshold), and `rescore_from_anchors`
re-derives `subject_frame_conf` after every edit. So "fixed" means the **resolved** root is back on its
measured anchor (R-6), not the agent's say-so. **Targeting** (`attention_targets`) keeps only the
`low_confidence` items for subjects we hold an anchor for, worst-first — ball-height / reprojection
signals are a different concern the loop deliberately does **not** touch (R-4), and the
`AutonomyReport` measures only the attention it *owns*, so before/after/`cleared` stay coherent.
**Bounded** two ways via `EditBudget`: `max_edits` (count) + `max_abs_change_m` (per-edit XY
magnitude); each `propose_anchor_offset` is the mean ground displacement over the off-anchor frames,
**clipped** (not rejected) to the cap, so a large error converges over several honest steps instead of
one teleport — applied as a normal `CONSTANT_OFFSET` `Correction` layered on top (ADR-0002, proposal
never mutated). **E2E proof:** the dry-run seeds subject 0 **2 m off** its measured anchor on a *local*
scene copy → the loop clears attention **12→0 in 2 bounded edits (≤1.0 m), `cleared=True`**, export
untouched, exit 0. **Tests:** 9 new (`test_autonomy.py`) — the confidence crossover at tolerance;
rescore flags off-anchor / passes on-anchor; the proposal is clipped to budget (4 m → capped 1.0 m) and
`None` when on-anchor; the **eval harness** (seed 2 m wrong → cleared in one edit, root back on anchor,
input untouched, proposal intact); targeting fixes the worst (3 m) before the lesser (1 m); the budget
*bites* (1 edit, 0.5 m cap → not cleared, clipped) then *converges* (2 m → ≤0.5 m in exactly 3 bounded
pulls); an on-anchor scene needs no edits; deterministic. Suite **557 passed / 12 skipped**, no
GPU/Blender/LLM (**AC-7**).

**M3 GOAL met, flipped 🟡→🟢 (2026-06-27):** quality & operator-efficiency — every M3 seam closed end
to end. **M3-1** per-subject Gaussian avatars, **M3-2** anchor-hardened re-fit + occlusion robustness,
**M3-3/M3-4** ViewSynthesizer seam B (`amplify` + `inpaint_occlusions`), **M3-5** confidence map +
"needs attention" UI, **M3-6** named snapshots + rollback, **M3-7** dependency-free three.js viewer,
**M3-8** the learned motion-prior seam, and **A-10** the bounded autonomy loop — all run through the
*real* path with pure, GPU-free halves that are unit-tested. Every generative/learned heavy half stays
an **honest gated stub** behind an extra or dotted-path injection (**R-8**): the avatar refiner, amodal
occlusion completion, view-synth diffusion, and the learned denoiser each raise an actionable
`NotImplementedError` until wired — never silently faked. **Exit criteria met:** seam B feeds N
synthetic views into reconstruction (**AC-5b**), the web viewer opens externally without scale/coord
loss (**AC-6**), and the whole suite — **557 passed / 12 skipped** — runs on fakes with **no
GPU/Blender/LLM** (**AC-7**). The agent track closes with it (A-1…A-10 all ✅): the M3 agent exit — a
seeded-wrong pose fixed end to end, verified by the attention list clearing — is proven in the dry-run.
**R-6 honesty (unchanged at 🟢):** 🟢 means every seam is real and *measured* where it can be, with
generative realism deferred behind gates — not that the heavy models ship. **Next: M4 (optional, real
multi-camera), or wiring the gated reals on a GPU box.**

---

## Agent / MCP automation track (cross-cutting, ADR-0008) 🟢  (through M3; extends with M4)

The north-star: **automate operator work with an LLM** that drives the same use-cases a human
does, over MCP, with visual feedback. This is not a milestone — it *rides on top of* each one,
gaining capability as the render path matures. The contracts (port, catalog, viewpoint math,
fakes) ship in **M0**; the loop gets sharper feedback as fakes are replaced.

| Capability | Lands with | Package | State |
|---|---|---|---|
| A-1 `SceneObserver` port + `Observation` (3D/overlay/UI + summary) | M0 | `core/ports/observation` | ✅ |
| A-2 Pure viewpoint math: `look_at`, `standard_viewpoints`, centroid | M0 | `core/agent` | ✅ |
| A-3 `scene_summary` from the UX-4 attention list (text feedback) | M0 | `core/agent` | ✅ |
| A-4 MCP `tool_catalog` (import-free, 12 use-cases as data) | M0 | `adapters/mcp` | ✅ |
| A-5 `FakeSceneObserver` (stdlib PNGs, no renderer) | M0 | `adapters/fakes` | ✅ |
| A-6 Live MCP `serve()` over the app controller (`mcp` extra) | M1 | `adapters/mcp`, `app` | ✅ dispatch (tool→use-case→text/image blocks) real + unit-tested on fakes; SDK stdio loop lazy-imported, gated behind the `mcp` extra |
| A-7 `observe` returns real `FRAME_OVERLAY` (reprojection) + proxy `SCENE_3D` | M1 | `adapters/blender`, `adapters/render` | ✅ reprojection `RenderPass` real (`ReprojectionOverlayRenderPass`); proxy `SCENE_3D` real via `BlenderSceneObserver` (out-of-process `blender --background`, Workbench/CPU, gated on a Blender binary) — `--observer blender` |
| A-8 `observe` returns **photoreal** `SCENE_3D` from canonical viewpoints | M2 | `adapters/render` | ✅ `CyclesSceneObserver` renders each viewpoint through Cycles (`--observer cycles`); 2D overlay/radar/UI delegate to the fake (M2-10) |
| A-9 Orbit viewpoints via ViewSynthesizer seam A in `observe` | M2 | `adapters/render` | ✅ `CyclesViewSynthesizer` re-renders the resolved 3D scene at the orbit cameras (`--viewsynth cycles`, non-generative, cached, `editable=False`); generative seam B stays gated (M2-10) |
| A-10 Agent autonomy hardening: bounded edits, attention-driven targeting, eval harness | M3 | `app`, `core/agent` | ✅ pure-core `auto_correct` (`core/agent/autonomy.py`): re-scores confidence from the **measured anchor residual** (M3-2), targets the **worst off-anchor** subject first, applies **bounded** anchor-pull `Correction`s (per-edit XY cap — clipped, never teleported) until attention clears (R-6 measured proof; input never mutated, ADR-0002); eval harness seeds a 2 m wrong pose → cleared in bounded edits, no GPU/Blender/LLM, run in the dry-run E2E |

**Invariants (hold at every milestone):** the agent edits *only* via `Correction` tools
(`apply_offset|keyframes|smoothing|refit`), never raw geometry; `resolve()` is the sole path to
pixels; feedback images come from the *same* `RenderPass` a human sees. The loop is testable on
fakes with **no GPU, no Blender, no LLM** (TZ AC-7 extends to the agent seam).

**Exit criteria (per stage):** *(M0)* catalog + observer satisfy their contracts in tests and the
dry-run emits inspectable snapshots; *(M1)* an MCP client opens an episode, calls `observe`, applies
an offset, and sees the overlay change; *(M2)* `observe` returns photoreal multi-view of the edited
scene; *(M3)* an agent fixes a seeded-wrong pose end-to-end, verified by the attention list clearing.

---

## M4 (optional) — Real multi-camera ⬜

Out of scope. The data model already treats cameras as a **list** and homography as a degenerate
calibration, so adding synchronized calibrated sources is an adapter + orchestration change, not a
core rewrite. Note: seam-B amplification already exercises the same multi-view input path with
*synthetic* cameras.

---

## Open dependencies / decisions to revisit per milestone

- **M2:** which generative avatar API is primary (realism vs cost vs rig control)? Resolve via the
  **M2-0 realism spike** (measured pixel-projection over generative hallucination, R-6 honesty gate);
  the generative path itself (deferred to M3) ranks per the SAM-Body4D brief — PF-LHM (pose-free,
  multi-image) default, IDOL/LHM alternates. Splat render inside Blender vs external renderer
  (ADR-0003 default: external + import).
- **M3:** which ViewSynthesizer backend per seam (seam B favors 3D-consistency, e.g.
  GEN3C/TrajectoryCrafter; seam A favors orbit fidelity, e.g. ReCamMaster) — see ADR-0007.
- **Cross-cutting:** SMPL-X (hands/face) vs SMPL/SMPL-H (body only) — affects pose dimensions;
  generative-API budget per episode bounds the share of per-subject/ViewSynthesizer work.
- **Agent track (ADR-0008):** which MCP host(s) to target first (Claude CLI/Desktop); how many
  viewpoints per `observe` is the right cost/feedback trade-off; whether the agent eval harness
  (A-10) seeds known-wrong scenes or replays operator-fixed episodes.
